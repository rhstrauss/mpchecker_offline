"""
Read, parse, and cache the MPC orbit catalog (MPCORB.DAT) and comet elements.

MPCORB format: https://www.minorplanetcenter.net/iau/info/MPOrbitFormat.html
Comet format:  https://www.minorplanetcenter.net/iau/info/CometOrbitFormat.html

Orbital elements are stored as a structured numpy array for fast vectorized
access during the Keplerian pre-filter step.
"""

import gzip
import re
import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from .config import MPCORB_FILE, MPCORB_GZ, COMET_FILE, CACHE_DIR

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MPCORB packed-epoch decoder
# ---------------------------------------------------------------------------

_CENT = {'I': 1800, 'J': 1900, 'K': 2000}
_MONTH_CHARS = {'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,
                'A':10,'B':11,'C':12}
_DAY_CHARS   = {str(i): i for i in range(1, 10)}
_DAY_CHARS.update({chr(ord('A')+i): 10+i for i in range(22)})  # A=10 … V=31


def unpack_epoch_mjd(s: str) -> float:
    """
    Convert a 5-character packed epoch (e.g. 'K244A') to MJD (TT).
    """
    c = s[0]
    if c not in _CENT:
        raise ValueError(f'Unknown century letter {c!r} in epoch {s!r}')
    year  = _CENT[c] + int(s[1:3])
    month = _MONTH_CHARS[s[3]]
    day   = _DAY_CHARS[s[4]]

    # Julian Day Number for YYYY-MM-DD (Gregorian)
    a = (14 - month) // 12
    y = year + 4800 - a
    m = month + 12*a - 3
    jdn = day + (153*m + 2)//5 + 365*y + y//4 - y//100 + y//400 - 32045
    return (jdn - 0.5) - 2400000.5  # MJD at 0h TT


# ---------------------------------------------------------------------------
# Numpy dtype for the asteroid catalog
# ---------------------------------------------------------------------------

ASTEROID_DTYPE = np.dtype([
    ('desig',   'U20'),   # readable designation
    ('packed',  'U8'),    # packed designation (first 7 chars of MPCORB line)
    ('a',       'f8'),    # semi-major axis (AU)
    ('e',       'f8'),    # eccentricity
    ('i',       'f8'),    # inclination (deg)
    ('Omega',   'f8'),    # longitude of ascending node (deg)
    ('omega',   'f8'),    # argument of perihelion (deg)
    ('M',       'f8'),    # mean anomaly (deg)
    ('epoch',   'f8'),    # epoch (MJD TT)
    ('H',       'f8'),    # absolute magnitude
    ('G',       'f8'),    # slope parameter
    ('U',       'U1'),    # uncertainty parameter
])

COMET_DTYPE = np.dtype([
    ('desig',   'U40'),   # readable designation
    ('packed',  'U12'),
    ('q',       'f8'),    # perihelion distance (AU)
    ('e',       'f8'),    # eccentricity
    ('i',       'f8'),    # inclination (deg)
    ('Omega',   'f8'),    # longitude of ascending node (deg)
    ('omega',   'f8'),    # argument of perihelion (deg)
    ('Tp',      'f8'),    # time of perihelion (MJD TT)
    ('H',       'f8'),    # absolute magnitude
    ('G',       'f8'),    # slope parameter (K in comet format)
])


# ---------------------------------------------------------------------------
# MPCORB.DAT parser
# ---------------------------------------------------------------------------

def _unpack_number(s: str) -> str:
    """Unpack 5-char packed minor planet number."""
    s = s.strip()
    if not s:
        return ''
    if s[0].isdigit():
        return str(int(s))
    b62 = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
    val = b62.index(s[0]) * 10000 + int(s[1:])
    return str(val)


def _unpack_provisional(s: str) -> str:
    """Best-effort unpack of 7-char packed provisional designation."""
    s = s.strip()
    if not s:
        return ''
    cent_map = {'I': '18', 'J': '19', 'K': '20'}
    if s[0] in cent_map and len(s) >= 5:
        year = cent_map[s[0]] + s[1:3]
        half = s[3]
        letter2 = s[4] if len(s) > 4 else 'A'
        sub = s[5] if len(s) > 5 else '0'
        if sub == '0':
            return f'{year} {half}{letter2}'
        elif sub.isdigit():
            return f'{year} {half}{letter2}{sub}'
        else:
            n = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'.index(sub)
            return f'{year} {half}{letter2}{n}'
    return s


def _make_desig(packed7: str) -> str:
    """Convert 7-char packed field to human-readable designation."""
    num_part  = packed7[0:5]
    prov_part = packed7[0:7]

    s = num_part.strip()
    if s and s[0].isdigit():
        return str(int(s))
    if s and s[0].isalpha() and s[0] not in 'IJKCPD':
        return _unpack_number(s)
    # Try provisional
    return _unpack_provisional(prov_part)


def parse_mpcorb_line(line: str) -> Optional[np.void]:
    """Parse one MPCORB data line and return a record, or None to skip."""
    if len(line) < 103:
        return None
    packed  = line[0:7].strip()
    if not packed:
        return None
    # Skip header lines (start with '-------' or similar)
    if packed[0] == '-':
        return None

    try:
        H      = float(line[8:13])   if line[8:13].strip()  else 99.0
        G      = float(line[14:19])  if line[14:19].strip() else 0.15
        epoch  = unpack_epoch_mjd(line[20:25])
        M      = float(line[26:35])
        omega  = float(line[37:46])
        Omega  = float(line[48:57])
        i_deg  = float(line[59:68])
        e      = float(line[70:79])
        a      = float(line[92:103])
    except (ValueError, KeyError, IndexError):
        return None

    U = line[105].strip() if len(line) > 105 else ''

    # Readable designation: prefer number from cols 167-194 if present
    desig_long = line[166:194].strip() if len(line) > 166 else ''
    if desig_long:
        desig = desig_long
    else:
        desig = _make_desig(packed)

    rec = np.zeros(1, dtype=ASTEROID_DTYPE)[0]
    rec['desig']  = desig
    rec['packed'] = packed
    rec['a']      = a
    rec['e']      = e
    rec['i']      = i_deg
    rec['Omega']  = Omega
    rec['omega']  = omega
    rec['M']      = M
    rec['epoch']  = epoch
    rec['H']      = H
    rec['G']      = G
    rec['U']      = U
    return rec


def load_mpcorb(path: Optional[Path] = None,
                cache: bool = True,
                H_limit: float = 35.0) -> np.ndarray:
    """
    Load MPCORB.DAT (or .gz) into a structured numpy array.

    Caches the parsed result as a .npy file so subsequent loads are fast.
    H_limit: skip objects fainter than this absolute magnitude (default 35).
    """
    if path is None:
        path = MPCORB_FILE if MPCORB_FILE.exists() else MPCORB_GZ

    cache_file = CACHE_DIR / f'mpcorb_H{int(H_limit)}.npy'
    if cache and cache_file.exists():
        src_mtime = path.stat().st_mtime if path.exists() else 0
        if cache_file.stat().st_mtime > src_mtime:
            log.info('Loading MPCORB from cache %s', cache_file)
            return np.load(cache_file, allow_pickle=False)

    if not path.exists():
        raise FileNotFoundError(
            f'MPCORB not found at {path}. Run: mpchecker --download-data')

    log.info('Parsing MPCORB from %s …', path)
    records = []
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt', encoding='ascii', errors='replace') as fh:
        for line in fh:
            line = line.rstrip('\n')
            rec = parse_mpcorb_line(line)
            if rec is not None and rec['H'] <= H_limit:
                records.append(rec)

    arr = np.array(records, dtype=ASTEROID_DTYPE)
    log.info('Loaded %d asteroids (H ≤ %.1f)', len(arr), H_limit)

    if cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.save(cache_file, arr)
        log.info('Saved cache to %s', cache_file)

    return arr


# ---------------------------------------------------------------------------
# AllCometEls.txt parser
# ---------------------------------------------------------------------------
# MPC comet elements single-line format (as of 2024):
# Cols 1-4:   Periodic comet number (or blank)
# Col  5:     Orbit type (P/C/D/X/A/I)
# Cols 6-12:  Provisional designation
# Cols 15-18: Perihelion year
# Cols 20-21: Month
# Cols 23-29: Day (decimal)
# Cols 31-39: Perihelion distance q (AU)
# Cols 42-49: Eccentricity
# Cols 52-59: Argument of perihelion (deg, J2000)
# Cols 62-69: Long. of ascending node (deg, J2000)
# Cols 72-79: Inclination (deg, J2000)
# Cols 92-95: Absolute magnitude H
# Cols 97-100: Slope parameter K (= G)
# Remaining:  Name

def parse_comet_line(line: str) -> Optional[np.void]:
    """Parse one MPC comet element line."""
    if len(line) < 80:
        return None
    line = line.rstrip('\n')

    try:
        # Perihelion epoch
        yr   = int(line[14:18])
        mon  = int(line[19:21])
        day  = float(line[22:29])
        day_i = int(day)
        frac  = day - day_i
        a_t = (14 - mon) // 12
        y_t = yr + 4800 - a_t
        m_t = mon + 12*a_t - 3
        jdn = day_i + (153*m_t+2)//5 + 365*y_t + y_t//4 - y_t//100 + y_t//400 - 32045
        Tp = (jdn + frac - 0.5) - 2400000.5

        q      = float(line[30:39])
        e      = float(line[41:49])
        omega  = float(line[51:59])
        Omega  = float(line[61:69])
        i_deg  = float(line[71:79])
        H      = float(line[91:95]) if line[91:95].strip() else 99.0
        G      = float(line[96:100]) if line[96:100].strip() else 0.15
    except (ValueError, IndexError):
        return None

    # Designation
    num   = line[0:4].strip()
    otype = line[4].strip()
    prov  = line[5:12].strip()
    name_long = line[102:].strip() if len(line) > 102 else ''
    if name_long:
        desig = name_long
    elif num:
        desig = f'{num}{otype}/{prov}'
    else:
        desig = f'{otype}/{prov}'

    packed = (line[0:12]).strip()

    rec = np.zeros(1, dtype=COMET_DTYPE)[0]
    rec['desig']  = desig
    rec['packed'] = packed
    rec['q']      = q
    rec['e']      = e
    rec['i']      = i_deg
    rec['Omega']  = Omega
    rec['omega']  = omega
    rec['Tp']     = Tp
    rec['H']      = H
    rec['G']      = G
    return rec


def load_comets(path: Optional[Path] = None,
                cache: bool = True) -> np.ndarray:
    """Load AllCometEls.txt into a structured numpy array."""
    if path is None:
        path = COMET_FILE
    cache_file = CACHE_DIR / 'comets.npy'

    if cache and cache_file.exists() and path.exists():
        if cache_file.stat().st_mtime > path.stat().st_mtime:
            log.info('Loading comets from cache')
            return np.load(cache_file, allow_pickle=False)

    if not path.exists():
        log.warning('Comet file not found: %s', path)
        return np.zeros(0, dtype=COMET_DTYPE)

    log.info('Parsing comets from %s …', path)
    records = []
    with open(path, 'r', encoding='ascii', errors='replace') as fh:
        for line in fh:
            rec = parse_comet_line(line)
            if rec is not None:
                records.append(rec)

    arr = np.array(records, dtype=COMET_DTYPE)
    log.info('Loaded %d comets', len(arr))

    if cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.save(cache_file, arr)

    return arr


# ---------------------------------------------------------------------------
# Observatory codes
# ---------------------------------------------------------------------------

OBS_DTYPE = np.dtype([
    ('code',    'U3'),
    ('lon_deg', 'f8'),   # geodetic longitude East (degrees)
    ('rcos',    'f8'),   # rho * cos(phi')
    ('rsin',    'f8'),   # rho * sin(phi')
    ('name',    'U60'),
])


def parse_obscode_html(text: str) -> np.ndarray:
    """
    Parse MPC ObsCodes.html or ObsCodes.txt into a numpy array.
    The HTML file has table rows; the plain text has fixed-width columns.
    """
    records = []
    # Try plain-text format first: "Code Longitude  Cos     Sin     Name"
    for line in text.splitlines():
        m = re.match(
            r'^([A-Z0-9]{3})\s+([\d.]+)\s+([\d.]+)\s+([+-]?[\d.]+)\s+(.*)',
            line.strip())
        if m:
            code = m.group(1)
            try:
                lon  = float(m.group(2))
                rcos = float(m.group(3))
                rsin = float(m.group(4))
                name = m.group(5).strip()
            except ValueError:
                continue
            rec = np.zeros(1, dtype=OBS_DTYPE)[0]
            rec['code']    = code
            rec['lon_deg'] = lon
            rec['rcos']    = rcos
            rec['rsin']    = rsin
            rec['name']    = name
            records.append(rec)
    return np.array(records, dtype=OBS_DTYPE)


def load_obscodes(path: Optional[Path] = None) -> dict:
    """
    Load observatory codes into a dict keyed by 3-char code.
    Returns: {code: (lon_deg, rcos, rsin, name)}
    """
    from .config import OBSCODE_FILE
    if path is None:
        path = OBSCODE_FILE

    # Fall back to the copy bundled with pyoorb if no local file
    if not path.exists():
        from .config import OORB_DATA
        fallback = OORB_DATA / 'OBSCODE.dat'
        if fallback.exists():
            path = fallback
        else:
            log.warning('ObsCodes not found; topocentric corrections disabled')
            return {}

    with open(path, 'r', encoding='ascii', errors='replace') as fh:
        text = fh.read()

    arr = parse_obscode_html(text)
    return {r['code']: (r['lon_deg'], r['rcos'], r['rsin'], r['name'])
            for r in arr}
