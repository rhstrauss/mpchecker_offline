"""
Parse astrometric observation records in multiple formats:

1. MPC 80-column format (default)
   https://www.minorplanetcenter.net/iau/info/OpticalObs.html

2. ADES PSV (pipe-separated values)
   https://minorplanetcenter.net/iau/info/ADES.html

3. hldet (HelioLinC detection CSV, e.g. parseout.csv)
   Comma-separated with columns for MJD, RA, Dec, mag, band, obscode, etc.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import csv
import io
import re


@dataclass
class Observation:
    """A single parsed observation record."""
    line: str
    designation: str          # human-readable designation
    packed_desig: str         # raw packed designation
    discovery: bool
    note1: str
    note2: str
    epoch_mjd: float          # MJD UTC
    ra_deg: float             # J2000 RA degrees
    dec_deg: float            # J2000 Dec degrees
    mag: Optional[float]
    band: Optional[str]
    obscode: str
    obj_type: str             # 'minor_planet', 'comet', 'satellite'


# ---------------------------------------------------------------------------
# Designation unpacking
# ---------------------------------------------------------------------------

# Characters used in base-62 packed designations
_B62 = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'

def _b62val(c: str) -> int:
    return _B62.index(c)


def unpack_number(s: str) -> str:
    """Unpack a 5-char packed minor planet number to integer string."""
    s = s.strip()
    if not s:
        return ''
    if s[0].isdigit():
        return str(int(s))
    # Numbers >= 100000: first char encodes the upper digits in base-62
    # (A=10, B=11, ..., Z=35, a=36, ..., z=61)
    val = _b62val(s[0]) * 10000 + int(s[1:])
    return str(val)


def unpack_provisional(s: str) -> str:
    """
    Unpack a 7-char packed provisional designation to readable form.
    e.g. 'J95X00A' -> '1995 XA'
         'K07Tf8A' -> '2007 TA8' (order digit >0 appended as subscript)
    """
    s_raw = s
    s = s.strip()
    if not s:
        return ''

    # Identify type by first character
    cent_map = {'I': '18', 'J': '19', 'K': '20'}
    if s[0] in cent_map:
        # Standard provisional: CCYYx##s
        # CC = century letter, YY = 2-digit year, x = half-month letter
        # ## = order (base-62 pair), s = optional second letter suffix
        if len(s) < 5:
            return s_raw.strip()
        year = cent_map[s[0]] + s[1:3]
        half_month = s[3]
        # Order encoded in chars 4-5
        # char 4: first letter A-Z (skip I) gives letter for designation
        # char 5: subscript (0 = none, 1-9, A-Z = 10-35)
        second_letter = s[4] if len(s) > 4 else 'A'
        sub_raw = s[5] if len(s) > 5 else '0'

        if sub_raw == '0':
            subscript = ''
        elif sub_raw.isdigit():
            subscript = sub_raw
        else:
            subscript = str(_b62val(sub_raw))

        return f'{year} {half_month}{second_letter}{subscript}'

    # Comet-style or survey designation
    if s[0] in 'ABCDEGHIJKL':  # survey programs
        return s_raw.strip()

    return s_raw.strip()


def unpack_designation(cols_1_5: str, cols_6_12: str) -> Tuple[str, str]:
    """
    Return (human_readable, packed) for an observation line's designation fields.
    cols_1_5  : characters 1-5  (0-indexed: 0:5)
    cols_6_12 : characters 6-12 (0-indexed: 5:12)
    Also determines object type: 'minor_planet', 'comet', 'satellite'
    """
    num  = cols_1_5.strip()
    prov = cols_6_12.strip()

    # Satellite: col 1 is planet letter, cols 2-4 are satellite number, col 5 = 'S'
    if len(cols_1_5) >= 5 and cols_1_5[4] == 'S':
        planet_letter = cols_1_5[0].strip()
        sat_num = cols_1_5[1:4].strip()
        packed = cols_1_5.strip()
        readable = f'{planet_letter}{sat_num}S'
        return readable, packed

    # Comet: col 5 is orbit type letter (C/P/D/X/A/I)
    if len(cols_1_5) >= 5 and cols_1_5[4] in 'CPDXAI':
        comet_num = cols_1_5[0:4].strip()
        orbit_type = cols_1_5[4]
        desig = prov if prov else ''
        packed = (cols_1_5 + cols_6_12).strip()
        readable = f'{comet_num}{orbit_type}/{desig}' if comet_num else f'{orbit_type}/{desig}'
        return readable, packed

    # Standard minor planet
    packed = (num or prov)
    if num:
        return unpack_number(num), num
    elif prov:
        return unpack_provisional(prov), prov
    return '', ''


# ---------------------------------------------------------------------------
# Date/coordinate parsing
# ---------------------------------------------------------------------------

def parse_date_mjd(date_str: str) -> float:
    """
    Parse MPC date string "YYYY MM DD.dddddd" to MJD UTC.
    """
    # Format: "YYYY MM DD.dddddd"  (columns 16-32, 0-indexed 15:32)
    parts = date_str.split()
    if len(parts) != 3:
        raise ValueError(f'Bad date string: {date_str!r}')
    year  = int(parts[0])
    month = int(parts[1])
    day_f = float(parts[2])
    day   = int(day_f)
    frac  = day_f - day

    # Julian Day Number for integer date
    # Using the standard calendar → JDN formula
    a = (14 - month) // 12
    y = year + 4800 - a
    m = month + 12 * a - 3
    jdn = day + (153*m + 2)//5 + 365*y + y//4 - y//100 + y//400 - 32045
    jd = jdn + frac - 0.5  # JD at 0h UT = JDN - 0.5
    return jd - 2400000.5  # MJD


def parse_ra_deg(ra_str: str) -> float:
    """
    Parse RA string "HH MM SS.ddd" to degrees.
    """
    parts = ra_str.split()
    if len(parts) != 3:
        raise ValueError(f'Bad RA string: {ra_str!r}')
    h = float(parts[0])
    m = float(parts[1])
    s = float(parts[2])
    return 15.0 * (h + m/60.0 + s/3600.0)


def parse_dec_deg(dec_str: str) -> float:
    """
    Parse Dec string "sDD MM SS.dd" (s=+/-) to degrees.
    """
    s = dec_str.strip()
    if not s:
        raise ValueError('Empty Dec string')
    sign = -1.0 if s[0] == '-' else 1.0
    parts = s[1:].split()
    if len(parts) != 3:
        raise ValueError(f'Bad Dec string: {dec_str!r}')
    d = float(parts[0])
    m = float(parts[1])
    sec = float(parts[2])
    return sign * (d + m/60.0 + sec/3600.0)


# ---------------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------------

def parse_line(line: str) -> Optional[Observation]:
    """
    Parse a single 80-column MPC observation line.
    Returns None if the line should be skipped (header, comment, blank).
    Raises ValueError on malformed lines.
    """
    if len(line) < 80:
        line = line.rstrip('\n').ljust(80)
    if not line.strip():
        return None
    # Skip lines that start with obvious header/comment markers
    if line[0] in ('#', '!') or line.startswith('COD') or line.startswith('OBS'):
        return None

    # Extract fixed-width fields (0-indexed)
    f_num     = line[0:5]     # cols 1-5:  packed number
    f_prov    = line[5:12]    # cols 6-12: packed provisional
    f_disc    = line[12]      # col 13:    discovery
    f_note1   = line[13]      # col 14
    f_note2   = line[14]      # col 15
    f_date    = line[15:32]   # cols 16-32
    f_ra      = line[32:44]   # cols 33-44
    f_dec     = line[44:56]   # cols 45-56
    # cols 57-65 blank
    f_mag     = line[65:70]   # cols 66-70  (magnitude value)
    f_band    = line[70]      # col 71      (photometric band)
    # cols 72-77 blank
    f_obs     = line[77:80]   # cols 78-80

    # Require non-blank date and RA/Dec fields
    if not f_date.strip() or not f_ra.strip() or not f_dec.strip():
        return None

    try:
        epoch_mjd = parse_date_mjd(f_date)
        ra_deg    = parse_ra_deg(f_ra)
        dec_deg   = parse_dec_deg(f_dec)
    except (ValueError, IndexError):
        return None

    # Magnitude
    mag: Optional[float] = None
    band: Optional[str]  = None
    mag_s = f_mag.strip()
    if mag_s:
        try:
            mag = float(mag_s)
            band = f_band.strip() or None
        except ValueError:
            pass

    # Designation and type
    desig_hr, packed = unpack_designation(f_num, f_prov)
    if not desig_hr:
        desig_hr = packed

    # Determine object type from packed fields
    if len(f_num) >= 5 and f_num[4] == 'S':
        obj_type = 'satellite'
    elif len(f_num) >= 5 and f_num[4] in 'CPDXAI':
        obj_type = 'comet'
    else:
        obj_type = 'minor_planet'

    obscode = f_obs.strip()
    if not obscode:
        obscode = '500'  # geocenter default

    return Observation(
        line=line,
        designation=desig_hr,
        packed_desig=packed,
        discovery=(f_disc == '*'),
        note1=f_note1.strip(),
        note2=f_note2.strip(),
        epoch_mjd=epoch_mjd,
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        mag=mag,
        band=band,
        obscode=obscode,
        obj_type=obj_type,
    )


def parse_observations(text: str) -> List[Observation]:
    """Parse all valid observations from a multi-line string."""
    obs = []
    for line in text.splitlines():
        try:
            o = parse_line(line)
            if o is not None:
                obs.append(o)
        except (ValueError, IndexError):
            continue
    return obs


def parse_file(path: str) -> List[Observation]:
    """Parse MPC 80-column observations from a file."""
    with open(path, 'r', errors='replace') as f:
        return parse_observations(f.read())


# ---------------------------------------------------------------------------
# ADES PSV format parser
# ---------------------------------------------------------------------------

def _parse_ades_obstime(obstime: str) -> float:
    """Parse ADES obsTime (ISO 8601) to MJD UTC.

    Accepts formats like:
      2024-01-15T12:34:56.789Z
      2024-01-15T12:34:56Z
      2024-01-15T12:34:56
    """
    s = obstime.strip().rstrip('Z')
    # Split date and time
    if 'T' in s:
        date_part, time_part = s.split('T', 1)
    else:
        date_part = s
        time_part = '00:00:00'

    parts = date_part.split('-')
    year = int(parts[0])
    month = int(parts[1])
    day = int(parts[2])

    time_parts = time_part.split(':')
    hour = int(time_parts[0])
    minute = int(time_parts[1]) if len(time_parts) > 1 else 0
    sec = float(time_parts[2]) if len(time_parts) > 2 else 0.0

    frac_day = (hour + minute / 60.0 + sec / 3600.0) / 24.0

    # JDN formula
    a = (14 - month) // 12
    y = year + 4800 - a
    m = month + 12 * a - 3
    jdn = day + (153 * m + 2) // 5 + 365 * y + y // 4 - y // 100 + y // 400 - 32045
    jd = jdn + frac_day - 0.5
    return jd - 2400000.5


def parse_ades_psv(text: str) -> List[Observation]:
    """Parse observations from ADES PSV (pipe-separated values) text.

    ADES PSV files have metadata lines starting with # or !, and data lines
    with pipe-separated fields.  The header row defines column names.
    """
    obs = []
    lines = text.splitlines()

    # Find the header row (first line with pipes that isn't a comment)
    header_idx = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith('#') or stripped.startswith('!'):
            continue
        if '|' in stripped:
            header_idx = i
            break

    if header_idx is None:
        return obs

    # Parse header
    header_line = lines[header_idx]
    columns = [c.strip() for c in header_line.split('|')]

    # Build column index
    col_idx = {}
    for j, name in enumerate(columns):
        col_idx[name] = j

    # Required columns
    required = {'obsTime', 'ra', 'dec', 'stn'}
    if not required.issubset(col_idx.keys()):
        # Try alternate column names
        alt_map = {'stn': 'mpcCode', 'obsTime': 'obsTime'}
        for req_col in list(required):
            if req_col not in col_idx:
                for alt in alt_map.get(req_col, []):
                    if alt in col_idx:
                        col_idx[req_col] = col_idx[alt]
                        break
        if not required.issubset(col_idx.keys()):
            return obs

    for i in range(header_idx + 1, len(lines)):
        line = lines[i]
        stripped = line.strip()
        if not stripped or stripped.startswith('#') or stripped.startswith('!'):
            continue
        if '|' not in stripped:
            continue

        fields = [f.strip() for f in line.split('|')]
        if len(fields) < len(columns):
            fields.extend([''] * (len(columns) - len(fields)))

        def _get(name: str) -> str:
            idx = col_idx.get(name)
            if idx is not None and idx < len(fields):
                return fields[idx].strip()
            return ''

        obstime_s = _get('obsTime')
        ra_s = _get('ra')
        dec_s = _get('dec')
        stn = _get('stn')

        if not obstime_s or not ra_s or not dec_s:
            continue

        try:
            epoch_mjd = _parse_ades_obstime(obstime_s)
            ra_deg = float(ra_s)
            dec_deg = float(dec_s)
        except (ValueError, IndexError):
            continue

        if not stn:
            stn = '500'

        # Magnitude
        mag: Optional[float] = None
        band: Optional[str] = None
        mag_s = _get('mag')
        if mag_s:
            try:
                mag = float(mag_s)
            except ValueError:
                pass
        band_s = _get('band')
        if band_s:
            band = band_s

        # Designation
        perm_id = _get('permID')
        prov_id = _get('provID')
        trk_sub = _get('trkSub')
        designation = perm_id or prov_id or trk_sub or ''

        # Mode -> note2
        mode = _get('mode')

        obs.append(Observation(
            line=line,
            designation=designation,
            packed_desig=perm_id or prov_id or '',
            discovery=False,
            note1='',
            note2=mode or '',
            epoch_mjd=epoch_mjd,
            ra_deg=ra_deg,
            dec_deg=dec_deg,
            mag=mag,
            band=band,
            obscode=stn,
            obj_type='minor_planet',
        ))

    return obs


def parse_ades_file(path: str) -> List[Observation]:
    """Parse ADES PSV observations from a file."""
    with open(path, 'r', errors='replace') as f:
        return parse_ades_psv(f.read())


# ---------------------------------------------------------------------------
# hldet (HelioLinC detection CSV) parser
# ---------------------------------------------------------------------------

# Common column name aliases for hldet CSV files
_HLDET_ALIASES = {
    'mjd': ['MJD', 'mjd', 'FieldMJD', 'fieldMJD', 'obsMJD', 'epoch_mjd',
            'JD', 'jd'],  # JD handled with offset
    'ra': ['RA', 'ra', 'RAdeg', 'ra_deg', 'AstRA(deg)', 'meanRA'],
    'dec': ['Dec', 'dec', 'DEC', 'Decdeg', 'dec_deg', 'AstDec(deg)', 'meanDec'],
    'mag': ['mag', 'Mag', 'VMag', 'Vmag', 'trailedSourceMag', 'PSFMag',
            'magV', 'filt_mag', 'ApMag', 'flux'],
    'band': ['band', 'Band', 'filt', 'filter', 'Filter', 'optFilter'],
    'obscode': ['obscode', 'obsCode', 'stn', 'observatory', 'Observatory',
                'MPC', 'mpcCode'],
    'objid': ['objID', 'ObjID', 'object_id', 'clusterID', 'trackletID',
              'ssObjectId', 'det_id', 'detID', 'Name', 'name',
              'provID', 'designation'],
}


def _resolve_hldet_columns(header: List[str]) -> dict:
    """Map canonical column names to indices using alias lookup."""
    mapping = {}
    for canonical, aliases in _HLDET_ALIASES.items():
        for alias in aliases:
            if alias in header:
                mapping[canonical] = header.index(alias)
                break
    return mapping


def parse_hldet(text: str) -> List[Observation]:
    """Parse observations from HelioLinC detection CSV (hldet) text.

    Expects a CSV with a header row.  Columns are matched flexibly using
    common aliases (e.g. 'RA', 'ra', 'RAdeg' all map to right ascension).
    At minimum needs MJD/epoch, RA, and Dec columns.
    """
    obs = []
    reader = csv.reader(io.StringIO(text))

    try:
        header = next(reader)
    except StopIteration:
        return obs

    # Strip whitespace from header names
    header = [h.strip() for h in header]
    col = _resolve_hldet_columns(header)

    if 'mjd' not in col or 'ra' not in col or 'dec' not in col:
        return obs

    # Check if the "mjd" column is actually JD (heuristic: alias was 'JD'/'jd')
    mjd_alias = header[col['mjd']]
    is_jd = mjd_alias.lower() == 'jd'

    for row in reader:
        if not row or not row[0].strip():
            continue
        # Skip comment rows
        if row[0].strip().startswith('#'):
            continue

        try:
            epoch_val = float(row[col['mjd']].strip())
            if is_jd:
                epoch_val = epoch_val - 2400000.5  # JD → MJD
            ra_deg = float(row[col['ra']].strip())
            dec_deg = float(row[col['dec']].strip())
        except (ValueError, IndexError):
            continue

        # Optional fields
        mag: Optional[float] = None
        band: Optional[str] = None
        obscode = '500'
        designation = ''

        if 'mag' in col:
            try:
                mag = float(row[col['mag']].strip())
            except (ValueError, IndexError):
                pass
        if 'band' in col:
            try:
                band = row[col['band']].strip() or None
            except IndexError:
                pass
        if 'obscode' in col:
            try:
                obscode = row[col['obscode']].strip() or '500'
            except IndexError:
                pass
        if 'objid' in col:
            try:
                designation = row[col['objid']].strip()
            except IndexError:
                pass

        obs.append(Observation(
            line=','.join(row),
            designation=designation,
            packed_desig='',
            discovery=False,
            note1='',
            note2='CCD',
            epoch_mjd=epoch_val,
            ra_deg=ra_deg,
            dec_deg=dec_deg,
            mag=mag,
            band=band,
            obscode=obscode,
            obj_type='minor_planet',
        ))

    return obs


def parse_hldet_file(path: str) -> List[Observation]:
    """Parse HelioLinC detection CSV observations from a file."""
    with open(path, 'r', errors='replace') as f:
        return parse_hldet(f.read())


# ---------------------------------------------------------------------------
# Format auto-detection
# ---------------------------------------------------------------------------

def detect_format(text: str) -> str:
    """Auto-detect observation file format from content.

    Returns one of: 'mpc80', 'ades', 'hldet'
    """
    lines = text.splitlines()

    for line in lines[:30]:
        stripped = line.strip()
        if not stripped:
            continue
        # ADES PSV metadata markers
        if stripped.startswith('# version=') or stripped.startswith('! mpcCode'):
            return 'ades'
        # ADES pipe-separated header row
        if '|' in stripped and any(kw in stripped for kw in
                                   ('obsTime', 'permID', 'provID', 'trkSub')):
            return 'ades'

    # Check for CSV with recognized header
    for line in lines[:5]:
        stripped = line.strip()
        if not stripped or stripped.startswith('#'):
            continue
        if ',' in stripped:
            cols = [c.strip() for c in stripped.split(',')]
            # Check for hldet-like column names
            col_lower = {c.lower() for c in cols}
            if ('ra' in col_lower or 'radeg' in col_lower) and \
               ('dec' in col_lower or 'decdeg' in col_lower):
                return 'hldet'
            # Also match on specific hldet aliases
            all_aliases = set()
            for aliases in _HLDET_ALIASES.values():
                all_aliases.update(aliases)
            if len(set(cols) & all_aliases) >= 3:
                return 'hldet'

    # Default to MPC 80-column
    return 'mpc80'


def parse_auto(text: str) -> List[Observation]:
    """Auto-detect format and parse observations."""
    fmt = detect_format(text)
    if fmt == 'ades':
        return parse_ades_psv(text)
    elif fmt == 'hldet':
        return parse_hldet(text)
    else:
        return parse_observations(text)


def parse_file_auto(path: str) -> List[Observation]:
    """Auto-detect format and parse observations from a file."""
    with open(path, 'r', errors='replace') as f:
        return parse_auto(f.read())
