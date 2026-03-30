"""
Planetary satellite checking via JPL SPICE kernels (SpiceyPy).

Downloads the required SPICE kernels on first use and caches them locally.
Uses DE440s for planetary positions and planet-specific SPK files for satellites.
"""

import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import numpy as np

from .config import (SPICE_DIR, SPICE_URLS, SPICE_LSK, SPICE_DE, SPICE_PCK,
                     SATELLITE_NAIF_IDS, SATELLITE_KERNEL_MAP,
                     SPICE_MAR, SPICE_JUP, SPICE_JUP2, SPICE_JUP3,
                     SPICE_SAT, SPICE_URA, SPICE_NEP, SPICE_PLU,
                     DWARF_PLANET_SATELLITES)

log = logging.getLogger(__name__)

_DEG2RAD = np.pi / 180.0
_RAD2DEG = 180.0 / np.pi

# Obliquity constants (J2000) needed for dwarf-planet satellite propagation
_EPS_J2000  = 23.43928111 * _DEG2RAD
_COS_EPS    = np.cos(_EPS_J2000)
_SIN_EPS    = np.sin(_EPS_J2000)
# Gaussian gravitational constant squared: GM_sun in AU^3 day^-2
_GM_SUN_SAT = 0.01720209895 ** 2

# Map key name → Path for satellite SPK files
_SAT_SPICE_FILES = {
    'mar':  SPICE_MAR,
    'jup':  SPICE_JUP,
    'jup2': SPICE_JUP2,
    'jup3': SPICE_JUP3,
    'sat':  SPICE_SAT,
    'ura':  SPICE_URA,
    'nep':  SPICE_NEP,
    'plu':  SPICE_PLU,
}

# Track which kernels are currently loaded
_loaded_kernels: set = set()
_spice_base_loaded = False


# ---------------------------------------------------------------------------
# Kernel management
# ---------------------------------------------------------------------------

def _download_kernel(url: str, dest: Path, show_progress: bool = True) -> None:
    """Download a SPICE kernel file with progress display."""
    import requests
    from tqdm import tqdm

    dest.parent.mkdir(parents=True, exist_ok=True)
    log.info('Downloading %s → %s', url, dest)
    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()
    total = int(r.headers.get('content-length', 0))
    with open(dest, 'wb') as f:
        if show_progress and total:
            with tqdm(total=total, unit='B', unit_scale=True,
                      desc=dest.name) as bar:
                for chunk in r.iter_content(65536):
                    f.write(chunk)
                    bar.update(len(chunk))
        else:
            for chunk in r.iter_content(65536):
                f.write(chunk)


def ensure_kernel(key: str, dest: Path) -> bool:
    """Ensure a SPICE kernel exists locally. Download if needed. Returns True if available."""
    if dest.exists():
        return True
    url = SPICE_URLS.get(key)
    if not url:
        return False
    try:
        _download_kernel(url, dest)
        return True
    except Exception as exc:
        log.warning('Failed to download %s (%s): %s', key, url, exc)
        return False


def _load_base_kernels():
    """Load leapseconds, planetary ephemeris, and PCK kernels."""
    global _spice_base_loaded
    if _spice_base_loaded:
        return
    import spiceypy as spice

    ok_lsk = ensure_kernel('lsk', SPICE_LSK)
    ok_de  = ensure_kernel('de',  SPICE_DE)
    ok_pck = ensure_kernel('pck', SPICE_PCK)

    if not (ok_lsk and ok_de):
        raise RuntimeError('Required SPICE kernels (LSK, DE440s) unavailable.')

    spice.furnsh(str(SPICE_LSK))
    _loaded_kernels.add(str(SPICE_LSK))
    spice.furnsh(str(SPICE_DE))
    _loaded_kernels.add(str(SPICE_DE))
    if ok_pck:
        spice.furnsh(str(SPICE_PCK))
        _loaded_kernels.add(str(SPICE_PCK))

    _spice_base_loaded = True
    log.info('SPICE base kernels loaded')


def _load_satellite_kernel(key: str) -> bool:
    """Load a satellite SPK kernel if not already loaded. Returns True on success."""
    import spiceypy as spice

    path = _SAT_SPICE_FILES.get(key)
    if path is None:
        return False
    k = str(path)
    if k in _loaded_kernels:
        return True
    if not ensure_kernel(key, path):
        return False
    try:
        spice.furnsh(k)
        _loaded_kernels.add(k)
        log.info('Loaded satellite kernel: %s', path.name)
        return True
    except Exception as exc:
        log.warning('Failed to furnsh %s: %s', path, exc)
        return False


# ---------------------------------------------------------------------------
# Satellite position computation
# ---------------------------------------------------------------------------

def _mjd_utc_to_et(mjd_utc: float) -> float:
    """Convert MJD UTC to SPICE ephemeris time (TDB seconds past J2000)."""
    import spiceypy as spice
    # JD UTC → UTC string → ET
    jd_utc = mjd_utc + 2400000.5
    utc_str = f'JD {jd_utc:.6f} UTC'
    return spice.utc2et(utc_str)


def _get_observer_itrf(obscode: str, obscodes: dict) -> Optional[np.ndarray]:
    """
    Return observer's position in ITRF93 (km) for topocentric correction,
    or None to use geocenter.
    """
    info = obscodes.get(obscode)
    if info is None or obscode == '500':
        return None
    lon_deg, rcos, rsin = info[0], info[1], info[2]
    R_earth_km = 6378.1363
    lon_rad = lon_deg * _DEG2RAD
    # ITRF: x/y in equatorial plane, z along north pole
    # (rcos * cos(lon), rcos * sin(lon), rsin) * R_earth
    x = R_earth_km * rcos * np.cos(lon_rad)
    y = R_earth_km * rcos * np.sin(lon_rad)
    z = R_earth_km * rsin
    return np.array([x, y, z])


def get_satellite_positions(
    naif_ids: List[int],
    t_mjd: float,
    obscode: str,
    obscodes: dict,
) -> Dict[int, Tuple[float, float, float, float]]:
    """
    Compute RA, Dec (J2000, degrees), geocentric distance (AU), and
    approximate V-magnitude for each satellite NAIF ID.

    Returns dict: {naif_id: (ra_deg, dec_deg, delta_au, vmag_approx)}
    """
    import spiceypy as spice

    _load_base_kernels()

    et = _mjd_utc_to_et(t_mjd)

    # Observer position in J2000 frame relative to solar system barycenter
    # For topocentric: observer is Earth + surface offset
    obs_itrf = _get_observer_itrf(obscode, obscodes)
    if obs_itrf is not None:
        # Get Earth body-fixed frame at this epoch
        # Approximate: rotate ITRF (body-fixed) to J2000 via Earth's rotation
        try:
            xform = spice.pxform('ITRF93', 'J2000', et)
            obs_j2000 = spice.mxv(xform, obs_itrf)   # km, J2000 relative to Earth center
        except Exception:
            obs_j2000 = None
    else:
        obs_j2000 = None

    AU_KM = 1.495978707e8   # km per AU
    results = {}

    for naif_id in naif_ids:
        sat_key = SATELLITE_KERNEL_MAP.get(naif_id)
        if not _load_satellite_kernel(sat_key):
            continue

        try:
            # Satellite position relative to Earth, J2000, light-time corrected
            state, lt = spice.spkez(naif_id, et, 'J2000', 'LT+S', 399)
            pos_km = np.array(state[:3])   # km from Earth geocenter

            if obs_j2000 is not None:
                pos_km = pos_km - obs_j2000  # topocentric

            dist_km = np.linalg.norm(pos_km)
            if dist_km < 1e-6:
                continue
            delta_au = dist_km / AU_KM

            # RA/Dec from unit vector
            ux, uy, uz = pos_km / dist_km
            ra_rad  = np.arctan2(uy, ux) % (2*np.pi)
            dec_rad = np.arcsin(np.clip(uz, -1, 1))

            # Rough V magnitude (use 10 as default for bright satellites)
            vmag = _satellite_vmag(naif_id, delta_au)

            results[naif_id] = (
                ra_rad  * _RAD2DEG,
                dec_rad * _RAD2DEG,
                delta_au,
                vmag,
            )
        except Exception as exc:
            log.debug('Cannot compute position for NAIF %d: %s', naif_id, exc)

    return results


# Approximate V-magnitudes for well-known satellites at 1 AU (very rough)
_SAT_VMAG_APPROX = {
    401: 11.4,  # Phobos
    402: 12.4,  # Deimos
    501: 5.0,   # Io
    502: 5.3,   # Europa
    503: 4.6,   # Ganymede
    504: 5.7,   # Callisto
    505: 14.1,  # Amalthea
    514: 15.7,  # Thebe
    515: 19.1,  # Adrastea
    516: 17.5,  # Metis
    # Classic Jovian irregulars (V at ~5.2 AU heliocentric)
    506: 14.6,  # Himalia
    507: 16.3,  # Elara
    508: 17.0,  # Pasiphae
    509: 18.0,  # Sinope
    510: 18.4,  # Lysithea
    511: 17.9,  # Carme
    512: 18.8,  # Ananke
    513: 19.5,  # Leda
    # Outer irregulars (typically V~20-23 at Jupiter distance)
    517: 20.7,  # Callirrhoe
    518: 20.2,  # Themisto
    546: 23.0,  # Carpo
    553: 22.0,  # Dia
    601: 12.9,  # Mimas
    602: 11.7,  # Enceladus
    603: 10.2,  # Tethys
    604: 10.4,  # Dione
    605: 9.7,   # Rhea
    606: 8.4,   # Titan
    607: 14.2,  # Hyperion
    608: 10.2,  # Iapetus (varies 10-12)
    609: 16.5,  # Phoebe
    701: 13.7,  # Ariel
    702: 14.8,  # Umbriel
    703: 13.9,  # Titania
    704: 14.2,  # Oberon
    705: 16.5,  # Miranda
    801: 13.4,  # Triton
    802: 18.7,  # Nereid
    901: 16.8,  # Charon
}


def _satellite_vmag(naif_id: int, delta_au: float) -> float:
    """
    Very approximate V magnitude from delta (geocentric distance, AU).
    Uses pre-tabulated opposition magnitudes scaled by distance.
    Phase angle correction is neglected (good to ~1 mag).
    """
    base_mag = _SAT_VMAG_APPROX.get(naif_id, 20.0)
    # Scale with 1/delta^2 (reflected light from planet ~constant illumination)
    # Most planetary satellites are far enough that delta ≈ planet distance
    # This is extremely rough but sufficient for a limiting-magnitude cut
    return base_mag + 5 * np.log10(max(delta_au, 0.001))


# ---------------------------------------------------------------------------
# Dwarf planet satellite position helpers
# ---------------------------------------------------------------------------

def _solve_kepler_scalar(M: float, e: float) -> float:
    """Newton-Raphson scalar Kepler solver for eccentric anomaly E."""
    E = M
    for _ in range(50):
        dE = (M - E + e * np.sin(E)) / (1.0 - e * np.cos(E))
        E += dE
        if abs(dE) < 1e-12:
            break
    return E


def _sat_offset_equatorial(
    a_au: float,
    e: float,
    i_deg: float,
    Omega_deg: float,
    omega_deg: float,
    P_days: float,
    t_peri_tt: float,
    t_tt: float,
) -> np.ndarray:
    """
    Satellite position offset (AU) in J2000 equatorial frame, relative to its
    primary.  Orbital elements are in J2000 equatorial (as published for TNO
    satellite systems).

    Parameters
    ----------
    a_au      : semi-major axis (AU)
    e         : eccentricity
    i_deg     : inclination (degrees, equatorial J2000)
    Omega_deg : longitude of ascending node (degrees, equatorial J2000)
    omega_deg : argument of periapsis (degrees)
    P_days    : orbital period (days)
    t_peri_tt : time of periapsis passage (MJD TT)
    t_tt      : observation time (MJD TT)

    Returns
    -------
    np.ndarray shape (3,) in AU
    """
    i_rad     = i_deg     * _DEG2RAD
    Omega_rad = Omega_deg * _DEG2RAD
    omega_rad = omega_deg * _DEG2RAD

    n = 2.0 * np.pi / P_days
    M = (n * (t_tt - t_peri_tt)) % (2.0 * np.pi)

    E  = _solve_kepler_scalar(M, e)
    nu = 2.0 * np.arctan2(np.sqrt(1.0 + e) * np.sin(E / 2.0),
                          np.sqrt(max(1.0 - e, 0.0)) * np.cos(E / 2.0))
    r  = a_au * (1.0 - e * np.cos(E))

    x_orb = r * np.cos(nu)
    y_orb = r * np.sin(nu)

    # Standard rotation: orbital plane → J2000 equatorial
    cO, sO = np.cos(Omega_rad), np.sin(Omega_rad)
    co, so = np.cos(omega_rad), np.sin(omega_rad)
    ci, si = np.cos(i_rad),     np.sin(i_rad)

    X = ( co*cO - so*sO*ci)*x_orb + (-so*cO - co*sO*ci)*y_orb
    Y = ( co*sO + so*cO*ci)*x_orb + (-so*sO + co*cO*ci)*y_orb
    Z = ( so*si             )*x_orb + ( co*si             )*y_orb

    return np.array([X, Y, Z])


def check_dwarf_planet_satellites(
    ra_obs: float,
    dec_obs: float,
    t_mjd_utc: float,
    obscode: str,
    obscodes: dict,
    asteroids: np.ndarray,
    primary_idx_cache: dict,
    search_radius_deg: float,
    mag_limit: float = 25.0,
) -> List[dict]:
    """
    Check known dwarf planet satellites (Dysnomia, Hi'iaka, Namaka, Weywot,
    Vanth, Xiangliu, Actaea, MK2) for a match at (ra_obs, dec_obs).

    Uses hardcoded Keplerian orbital elements in J2000 equatorial from
    published discovery papers (no SPICE coverage for TNO satellites).
    The primary's current position is looked up in the asteroid catalog via
    pyoorb-style propagation, and the satellite's offset is added.

    Parameters
    ----------
    primary_idx_cache : pre-built dict {packed_str: array_index} for the
                        dwarf planet primaries, built once per check_observations
                        call (avoids repeated O(N) catalog searches).
    """
    from astropy.time import Time
    from .propagator import get_observer_helio, kep_to_radec, _DEG2RAD as _D2R, _RAD2DEG as _R2D

    t_tt = Time(t_mjd_utc, format='mjd', scale='utc').tt.mjd

    # Observer heliocentric position (AU, J2000 equatorial)
    try:
        obs_helio = get_observer_helio(t_mjd_utc, obscode, obscodes)
    except Exception as exc:
        log.debug('check_dwarf_planet_satellites: get_observer_helio failed: %s', exc)
        return []

    matches = []

    for sat in DWARF_PLANET_SATELLITES:
        if sat['vmag_approx'] > mag_limit:
            continue

        primary_i = primary_idx_cache.get(sat['primary_packed'])
        if primary_i is None:
            continue

        try:
            obj = asteroids[primary_i]

            # Primary heliocentric equatorial XYZ via kep_to_radec
            ra_p, dec_p, delta_p = kep_to_radec(
                np.array([float(obj['a'])]),
                np.array([float(obj['e'])]),
                np.array([float(obj['i'])])     * _D2R,
                np.array([float(obj['Omega'])]) * _D2R,
                np.array([float(obj['omega'])]) * _D2R,
                np.array([float(obj['M'])])     * _D2R,
                np.array([float(obj['epoch'])]),
                t_tt,
                obs_helio,
            )
            # Geocentric unit vector of primary → reconstruct helio XYZ
            ra_r  = ra_p[0]  * _D2R
            dec_r = dec_p[0] * _D2R
            u_geo = np.array([
                np.cos(dec_r) * np.cos(ra_r),
                np.cos(dec_r) * np.sin(ra_r),
                np.sin(dec_r),
            ])
            primary_helio = obs_helio + delta_p[0] * u_geo

            # Satellite offset in J2000 equatorial
            sat_off = _sat_offset_equatorial(
                sat['a_au'], sat['e'],
                sat['i_deg'], sat['Omega_deg'], sat['omega_deg'],
                sat['P_days'], sat['t_peri_mjd_tt'], t_tt,
            )

            sat_helio = primary_helio + sat_off
            sat_geo   = sat_helio - obs_helio
            dist      = float(np.linalg.norm(sat_geo))
            if dist < 1e-15:
                continue

            u_sat  = sat_geo / dist
            ra_sat  = float(np.arctan2(u_sat[1], u_sat[0]) % (2.0 * np.pi)) * _R2D
            dec_sat = float(np.arcsin(np.clip(u_sat[2], -1.0, 1.0))) * _R2D

            sep = ang_sep_scalar(ra_sat, dec_sat, ra_obs, dec_obs)
            if sep > search_radius_deg:
                continue

            matches.append({
                'name':            sat['name'],
                'naif_id':         None,
                'ra_deg':          ra_sat,
                'dec_deg':         dec_sat,
                'sep_deg':         sep,
                'delta_au':        dist,
                'vmag':            sat['vmag_approx'],
                'type':            'satellite',
                'primary_packed':  sat['primary_packed'],
            })

        except Exception as exc:
            log.debug('check_dwarf_planet_satellites %s: %s', sat['name'], exc)

    matches.sort(key=lambda m: m['sep_deg'])
    return matches


# ---------------------------------------------------------------------------
# Angular separation helper (scalar)
# ---------------------------------------------------------------------------

def ang_sep_scalar(ra1_deg: float, dec1_deg: float,
                   ra2_deg: float, dec2_deg: float) -> float:
    """Great-circle angular separation in degrees."""
    r1, d1 = ra1_deg * _DEG2RAD, dec1_deg * _DEG2RAD
    r2, d2 = ra2_deg * _DEG2RAD, dec2_deg * _DEG2RAD
    dra = r1 - r2
    x = np.cos(d1)*np.sin(dra)
    y = np.cos(d2)*np.sin(d1) - np.sin(d2)*np.cos(d1)*np.cos(dra)
    z = np.sin(d2)*np.sin(d1) + np.cos(d2)*np.cos(d1)*np.cos(dra)
    return float(np.arctan2(np.sqrt(x*x + y*y), z) * _RAD2DEG)


# ---------------------------------------------------------------------------
# Main satellite check function
# ---------------------------------------------------------------------------

def check_satellites(
    ra_deg: float,
    dec_deg: float,
    t_mjd: float,
    obscode: str,
    obscodes: dict,
    search_radius_deg: float,
    mag_limit: float = 25.0,
    naif_ids: Optional[List[int]] = None,
) -> List[dict]:
    """
    Check all planetary satellites for a match at (ra_deg, dec_deg).

    Returns a list of match dicts, one per satellite within search_radius_deg.
    """
    if naif_ids is None:
        naif_ids = list(SATELLITE_NAIF_IDS.keys())

    try:
        positions = get_satellite_positions(naif_ids, t_mjd, obscode, obscodes)
    except Exception as exc:
        log.warning('Satellite check failed: %s', exc)
        return []

    matches = []
    for naif_id, (ra_s, dec_s, delta_au, vmag) in positions.items():
        if vmag > mag_limit:
            continue
        sep = ang_sep_scalar(ra_s, dec_s, ra_deg, dec_deg)
        if sep <= search_radius_deg:
            name = SATELLITE_NAIF_IDS.get(naif_id, f'NAIF:{naif_id}')
            matches.append({
                'name':     name,
                'naif_id':  naif_id,
                'ra_deg':   ra_s,
                'dec_deg':  dec_s,
                'sep_deg':  sep,
                'delta_au': delta_au,
                'vmag':     vmag,
                'type':     'satellite',
            })

    matches.sort(key=lambda m: m['sep_deg'])
    return matches
