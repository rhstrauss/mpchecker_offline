"""
Orbit propagation for the mpchecker pipeline.

Two stages:
  1. Fast Keplerian (two-body) propagation — used to pre-filter the entire
     MPCORB catalog down to a manageable candidate list. Uses a Numba-JIT
     parallel kernel when numba is available, falling back to vectorized NumPy.
  2. Precise N-body ephemeris via pyoorb — used for confirmed candidates to
     get accurate RA/Dec, magnitude, and sky motion.

Observatory topocentric offsets use the MPC ObsCodes parallax constants.
Earth/planet heliocentric positions are fetched via SPICE (de440s.bsp) when
loaded, falling back to astropy otherwise.
"""

import logging
import os
from functools import lru_cache
from typing import Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

# Physical constants
_DEG2RAD = np.pi / 180.0
_RAD2DEG = 180.0 / np.pi
# Gaussian gravitational constant squared: GM_sun in AU^3 day^-2
_GM_SUN = 0.01720209895 ** 2   # k^2
# Earth's equatorial radius in AU
_R_EARTH_AU = 6378.1363 / 1.495978707e8   # km → AU
# J2000 obliquity in radians
_EPS_J2000 = 23.43928111 * _DEG2RAD


# ---------------------------------------------------------------------------
# Obliquity rotation matrices (ecliptic ↔ equatorial)
# ---------------------------------------------------------------------------

def _ecl2equ_matrix() -> np.ndarray:
    eps = _EPS_J2000
    c, s = np.cos(eps), np.sin(eps)
    return np.array([[1, 0,  0],
                     [0, c, -s],
                     [0, s,  c]])


_ECL2EQU = _ecl2equ_matrix()
_COS_EPS = _ECL2EQU[1, 1]
_SIN_EPS = _ECL2EQU[2, 1]


# ---------------------------------------------------------------------------
# Numba parallel Keplerian propagation kernel (Optimization 2)
# Falls back to NumPy if numba is unavailable.
# ---------------------------------------------------------------------------

def _make_numba_kernel():
    """
    Build and return the Numba-JIT Keplerian kernel, or None if numba is not
    available. Called once at module load time.
    """
    try:
        from numba import njit, prange
    except ImportError:
        return None

    _gm   = float(_GM_SUN)
    _ceps = float(_COS_EPS)
    _seps = float(_SIN_EPS)
    _2pi  = 2.0 * np.pi
    _pi   = np.pi

    @njit(parallel=True, cache=True, fastmath=True)
    def _kep_kernel(a, e, i_rad, Omega_rad, omega_rad, M0_rad, epoch_tt,
                    t_tt, obs_x, obs_y, obs_z):
        """
        Parallel Keplerian propagation over n orbits.
        Returns (ra_rad, dec_rad, dist_au) each of length n.
        """
        n_obj = len(a)
        ra_out   = np.empty(n_obj, dtype=np.float64)
        dec_out  = np.empty(n_obj, dtype=np.float64)
        dist_out = np.empty(n_obj, dtype=np.float64)

        for j in prange(n_obj):
            aj = max(a[j], 1e-6)
            ej = e[j]

            # Mean motion (rad/day)
            nj = (_gm / (aj * aj * aj)) ** 0.5

            # Mean anomaly at t_tt
            Mj = M0_rad[j] + nj * (t_tt - epoch_tt[j])

            if ej < 1.0:
                # Elliptic: Newton-Raphson for eccentric anomaly
                Mj = Mj % _2pi
                Ej = Mj
                for _ in range(50):
                    sin_E = np.sin(Ej)
                    cos_E = np.cos(Ej)
                    dE = (Mj - Ej + ej * sin_E) / (1.0 - ej * cos_E)
                    Ej += dE
                    if abs(dE) < 1e-10:
                        break
                cos_E = np.cos(Ej)
                sin_E = np.sin(Ej)
                # True anomaly via half-angle formula (matches NumPy reference)
                sq_1pe = (1.0 + ej) ** 0.5
                sq_1me = max(1.0 - ej, 0.0) ** 0.5
                half_E = Ej * 0.5
                nu = 2.0 * np.arctan2(sq_1pe * np.sin(half_E),
                                      sq_1me * np.cos(half_E))
                r  = aj * (1.0 - ej * cos_E)
            else:
                # Hyperbolic: Newton-Raphson for hyperbolic anomaly
                Fh = min(max(Mj, -10.0), 10.0)
                for _ in range(50):
                    Fh = min(max(Fh, -20.0), 20.0)
                    sinh_F = np.sinh(Fh)
                    cosh_F = np.cosh(Fh)
                    denom = ej * cosh_F - 1.0
                    if abs(denom) < 1e-12:
                        denom = 1e-12
                    dF = (ej * sinh_F - Fh - Mj) / denom
                    dF = min(max(dF, -1.0), 1.0)
                    Fh -= dF
                    if abs(dF) < 1e-10:
                        break
                Fh = min(max(Fh, -20.0), 20.0)
                cosh_F = np.cosh(Fh)
                sinh_F = np.sinh(Fh)
                sq = (ej * ej - 1.0) ** 0.5
                nu = 2.0 * np.arctan2(sq * sinh_F, ej - cosh_F + 1e-300)
                r  = max(abs(aj) * (ej * cosh_F - 1.0), 1e-4)

            r = max(r, 1e-4)
            x_orb = r * np.cos(nu)
            y_orb = r * np.sin(nu)

            # Rotation: orbital plane → ecliptic J2000
            cO = np.cos(Omega_rad[j]); sO = np.sin(Omega_rad[j])
            co = np.cos(omega_rad[j]); so = np.sin(omega_rad[j])
            ci = np.cos(i_rad[j]);     si = np.sin(i_rad[j])

            Px = ( co*cO - so*sO*ci)*x_orb + (-so*cO - co*sO*ci)*y_orb
            Py = ( co*sO + so*cO*ci)*x_orb + (-so*sO + co*cO*ci)*y_orb
            Pz = (so*si)*x_orb + (co*si)*y_orb

            # Ecliptic → equatorial (J2000 obliquity)
            X =  Px
            Y =  _ceps*Py - _seps*Pz
            Z =  _seps*Py + _ceps*Pz

            # Observer-relative direction
            dx = X - obs_x
            dy = Y - obs_y
            dz = Z - obs_z
            dist = (dx*dx + dy*dy + dz*dz) ** 0.5

            if dist < 1e-15:
                ra_out[j] = 0.0; dec_out[j] = 0.0; dist_out[j] = dist
                continue

            inv = 1.0 / dist
            ra_out[j]   = np.arctan2(dy*inv, dx*inv) % _2pi
            dz_n = min(max(dz*inv, -1.0), 1.0)
            dec_out[j]  = np.arcsin(dz_n)
            dist_out[j] = dist

        return ra_out, dec_out, dist_out

    return _kep_kernel


# Compiled at import time; None if numba unavailable
_NUMBA_KEP_KERNEL = _make_numba_kernel()
if _NUMBA_KEP_KERNEL is not None:
    log.debug('Numba Keplerian kernel loaded (parallel JIT)')
else:
    log.debug('Numba not available; using NumPy Keplerian propagation')


# ---------------------------------------------------------------------------
# Kepler equation solver (vectorized — used by NumPy fallback path)
# ---------------------------------------------------------------------------

def solve_kepler(M: np.ndarray, e: np.ndarray,
                 max_iter: int = 50, tol: float = 1e-10) -> np.ndarray:
    """
    Solve M = E - e*sin(E) for E (eccentric anomaly), vectorized.
    Handles hyperbolic orbits (e >= 1) with the hyperbolic Kepler equation.
    """
    elliptic = e < 1.0
    E = M.copy()

    # Elliptic orbits: Newton-Raphson
    if np.any(elliptic):
        Ee = E[elliptic]
        Me = M[elliptic]
        ee = e[elliptic]
        for _ in range(max_iter):
            dE = (Me - Ee + ee * np.sin(Ee)) / (1 - ee * np.cos(Ee))
            Ee = Ee + dE
            if np.max(np.abs(dE)) < tol:
                break
        E[elliptic] = Ee

    # Near-parabolic / hyperbolic orbits.
    # For the fast pre-filter we only need a rough position, so clamp F to
    # avoid overflow and accept lower accuracy for these rare objects.
    hyp = ~elliptic
    if np.any(hyp):
        # Hyperbolic Kepler: M_h = e*sinh(F) - F  →  solve for F
        Fh = np.clip(M[hyp], -10.0, 10.0)
        Mh = M[hyp]
        eh = e[hyp]
        for _ in range(max_iter):
            sinh_F = np.sinh(np.clip(Fh, -20.0, 20.0))
            cosh_F = np.cosh(np.clip(Fh, -20.0, 20.0))
            denom  = eh * cosh_F - 1.0
            denom  = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
            dF = (Mh - eh * sinh_F + Fh) / denom
            Fh = Fh + np.clip(dF, -1.0, 1.0)
            if np.max(np.abs(dF)) < tol:
                break
        E[hyp] = Fh

    return E


# ---------------------------------------------------------------------------
# Fast Keplerian propagation (vectorized over entire catalog)
# ---------------------------------------------------------------------------

def _kep_to_radec_numpy(
    a: np.ndarray,
    e: np.ndarray,
    i_rad: np.ndarray,
    Omega_rad: np.ndarray,
    omega_rad: np.ndarray,
    M0_rad: np.ndarray,
    epoch_mjd,
    t_mjd: float,
    observer_helio_au: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorized NumPy Keplerian propagation. Used when Numba is unavailable.
    """
    dt = t_mjd - epoch_mjd  # days

    # Mean motion (rad/day)
    n = np.sqrt(_GM_SUN / np.maximum(a, 1e-6)**3)

    # Current mean anomaly (mod 2*pi for elliptic)
    M = M0_rad + n * dt
    M = M % (2 * np.pi)

    # Solve Kepler
    E = solve_kepler(M, e)

    # True anomaly and radius
    elliptic = e < 1.0
    nu = np.where(
        elliptic,
        2 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2),
                       np.sqrt(np.maximum(1 - e, 0)) * np.cos(E / 2)),
        2 * np.arctan(np.sqrt((e + 1) / np.maximum(e - 1, 1e-10))
                      * np.tanh(E / 2))  # hyperbolic
    )
    cosh_E = np.cosh(np.clip(E, -20.0, 20.0))
    r = np.where(
        elliptic,
        a * (1 - e * np.cos(E)),
        np.abs(a) * (e * cosh_E - 1)   # hyperbolic: r = |a| * (e*cosh(F) - 1)
    )
    # For near-parabolic handle robustly
    r = np.maximum(r, 1e-4)

    # Position in orbital plane
    x_orb = r * np.cos(nu)
    y_orb = r * np.sin(nu)

    # Rotation to ecliptic J2000 (3×N matrix multiply)
    cO, sO = np.cos(Omega_rad), np.sin(Omega_rad)
    co, so = np.cos(omega_rad), np.sin(omega_rad)
    ci, si = np.cos(i_rad),     np.sin(i_rad)

    # Position vector components in ecliptic J2000
    Px = (co*cO - so*sO*ci)*x_orb + (-so*cO - co*sO*ci)*y_orb
    Py = (co*sO + so*cO*ci)*x_orb + (-so*sO + co*cO*ci)*y_orb
    Pz = (so*si)*x_orb             + (co*si)*y_orb

    # Rotate ecliptic → equatorial
    X =  Px
    Y = _ECL2EQU[1, 1]*Py + _ECL2EQU[1, 2]*Pz
    Z = _ECL2EQU[2, 1]*Py + _ECL2EQU[2, 2]*Pz

    # Topocentric/geocentric direction
    dx = X - observer_helio_au[0]
    dy = Y - observer_helio_au[1]
    dz = Z - observer_helio_au[2]
    dist = np.sqrt(dx*dx + dy*dy + dz*dz)

    ra_rad  = np.arctan2(dy, dx) % (2*np.pi)
    dec_rad = np.arcsin(np.clip(dz / np.maximum(dist, 1e-15), -1, 1))

    return ra_rad * _RAD2DEG, dec_rad * _RAD2DEG, dist


def kep_to_radec(
    a: np.ndarray,
    e: np.ndarray,
    i_rad: np.ndarray,
    Omega_rad: np.ndarray,
    omega_rad: np.ndarray,
    M0_rad: np.ndarray,
    epoch_mjd,
    t_mjd: float,
    observer_helio_au: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Propagate Keplerian orbits to time t_mjd and return (ra_deg, dec_deg, dist_au).
    Dispatches to Numba parallel kernel when available, else NumPy fallback.
    """
    if _NUMBA_KEP_KERNEL is not None:
        ra_rad, dec_rad, dist = _NUMBA_KEP_KERNEL(
            np.asarray(a,         dtype=np.float64),
            np.asarray(e,         dtype=np.float64),
            np.asarray(i_rad,     dtype=np.float64),
            np.asarray(Omega_rad, dtype=np.float64),
            np.asarray(omega_rad, dtype=np.float64),
            np.asarray(M0_rad,    dtype=np.float64),
            np.asarray(epoch_mjd, dtype=np.float64),
            float(t_mjd),
            float(observer_helio_au[0]),
            float(observer_helio_au[1]),
            float(observer_helio_au[2]),
        )
        return ra_rad * _RAD2DEG, dec_rad * _RAD2DEG, dist
    else:
        return _kep_to_radec_numpy(
            a, e, i_rad, Omega_rad, omega_rad, M0_rad,
            epoch_mjd, t_mjd, observer_helio_au)


# ---------------------------------------------------------------------------
# Observer heliocentric position
# ---------------------------------------------------------------------------

_AU_KM = 1.495978707e8   # km per AU

# NAIF body IDs for solar system bodies used in position lookups
_NAIF_IDS = {
    'earth': 399, 'sun': 10,
    'jupiter': 5, 'saturn': 6, 'uranus': 7, 'neptune': 8,
    'mars': 4,
}


def _spice_helio(naif_id: int, et: float) -> np.ndarray:
    """
    Return heliocentric position (AU, J2000) for a body using loaded SPICE kernels.
    Raises on failure (caller should fall back to astropy).
    """
    import spiceypy as spice
    state, _ = spice.spkez(naif_id, et, 'J2000', 'NONE', 10)
    return np.array(state[:3]) / _AU_KM


def _mjd_to_et(t_mjd: float) -> float:
    """Convert MJD UTC to SPICE ephemeris time (ET). Requires LSK kernel loaded."""
    import spiceypy as spice
    return spice.unitim(t_mjd + 2400000.5, 'JDUTC', 'ET')


@lru_cache(maxsize=512)
def get_planet_helio(planet: str, t_mjd: float) -> tuple:
    """
    Return a planet's heliocentric position (AU, J2000 equatorial) at MJD UTC.
    Uses SPICE (de440s.bsp) if loaded, otherwise astropy. Cached by (planet, t_mjd).
    Returns tuple for lru_cache hashability.
    """
    naif_id = _NAIF_IDS.get(planet)
    if naif_id is not None:
        try:
            import spiceypy as spice
            if spice.ktotal('ALL') > 0:
                et = _mjd_to_et(t_mjd)
                pos = _spice_helio(naif_id, et)
                return (pos[0], pos[1], pos[2])
        except Exception:
            pass
    # Astropy fallback
    from astropy.coordinates import get_body_barycentric
    from astropy.time import Time
    t = Time(t_mjd, format='mjd', scale='utc')
    body_bary = get_body_barycentric(planet, t)
    sun_bary  = get_body_barycentric('sun',  t)
    helio = body_bary - sun_bary
    return (float(helio.x.to('au').value),
            float(helio.y.to('au').value),
            float(helio.z.to('au').value))


@lru_cache(maxsize=512)
def _get_earth_helio_cached(t_mjd: float) -> tuple:
    """Cached Earth heliocentric position as a tuple (for lru_cache)."""
    try:
        import spiceypy as spice
        if spice.ktotal('ALL') > 0:
            et = _mjd_to_et(t_mjd)
            pos = _spice_helio(399, et)
            return (pos[0], pos[1], pos[2])
    except Exception:
        pass
    from astropy.coordinates import get_body_barycentric
    from astropy.time import Time
    t = Time(t_mjd, format='mjd', scale='utc')
    earth_bary = get_body_barycentric('earth', t)
    sun_bary   = get_body_barycentric('sun',   t)
    helio = earth_bary - sun_bary
    return (float(helio.x.to('au').value),
            float(helio.y.to('au').value),
            float(helio.z.to('au').value))


def get_earth_helio(t_mjd: float) -> np.ndarray:
    """
    Return Earth's heliocentric position (AU, J2000 equatorial) at MJD UTC.
    Uses SPICE (de440s.bsp) if loaded, otherwise astropy. Result is cached.
    """
    return np.array(_get_earth_helio_cached(t_mjd))


def get_observer_helio(t_mjd: float,
                       obscode: str,
                       obscodes: dict) -> np.ndarray:
    """
    Return the observer's heliocentric position in AU (J2000 equatorial).
    Applies topocentric correction from MPC observatory parallax constants.
    """
    earth_helio = get_earth_helio(t_mjd)

    obs_info = obscodes.get(obscode)
    if obs_info is None or obscode == '500':
        return earth_helio  # geocenter

    lon_deg, rcos, rsin = obs_info[0], obs_info[1], obs_info[2]
    lon_rad = lon_deg * _DEG2RAD

    # GAST (approximate: OK for topocentric offset purposes)
    # t_mjd is UTC; convert to UT1 ≈ UTC for sub-arcsec purposes
    jd_ut1 = t_mjd + 2400000.5
    T = (jd_ut1 - 2451545.0) / 36525.0
    # GMST in seconds
    gmst_sec = (67310.54841
                + (876600.0*3600 + 8640184.812866)*T
                + 0.093104*T*T
                - 6.2e-6*T*T*T)
    gast_rad = (gmst_sec % 86400.0) / 86400.0 * 2 * np.pi
    lst_rad = (gast_rad + lon_rad) % (2*np.pi)

    # Observer offset from geocenter (AU, equatorial)
    off_x = _R_EARTH_AU * rcos * np.cos(lst_rad)
    off_y = _R_EARTH_AU * rcos * np.sin(lst_rad)
    off_z = _R_EARTH_AU * rsin

    # Rotate to equatorial (above is already in equatorial frame)
    return earth_helio + np.array([off_x, off_y, off_z])


# ---------------------------------------------------------------------------
# Topocentric parallax correction for pyoorb geocentric ephemerides
# ---------------------------------------------------------------------------

def _apply_topocentric_correction(
    eph_geo: np.ndarray,
    t_mjd_tt,
    obscode: str,
    obscodes: dict,
) -> np.ndarray:
    """
    Shift a pyoorb geocentric ephemeris to topocentric coordinates.

    pyoorb is always called with obscode '500' (geocenter) to avoid its
    brittle OBSCODE.dat parser failing on newer MPC entries.  This function
    applies the resulting parallax correction using the observer position
    from get_observer_helio(), which reads our own ObsCodes.txt directly.

    Parameters
    ----------
    eph_geo   : [n_orbits, 11] (single-epoch) or [n_orbits, n_epochs, 11]
    t_mjd_tt  : scalar TT MJD (single-epoch) or list of TT MJDs (multi-epoch)
    obscode   : MPC observatory code
    obscodes  : dict from mpcorb.load_obscodes()

    Columns modified: 1 (RA deg), 2 (Dec deg), 8 (delta AU).
    All other columns (vmag, rates, r_helio, phase) are from geocentric
    ephemeris; the corrections to those are negligible (<0.001 mag, <0.1"/hr).
    """
    if obscode == '500':
        return eph_geo

    single_epoch = (eph_geo.ndim == 2)
    if single_epoch:
        work = eph_geo[:, np.newaxis, :]   # [n, 1, 11]
        t_list = [t_mjd_tt]
    else:
        work = eph_geo
        t_list = list(t_mjd_tt)

    result = work.copy()

    for k, t_tt in enumerate(t_list):
        # Observer offset from geocenter (AU, J2000 equatorial).
        # We use TT as a proxy for UTC here; the 69-second difference
        # introduces a position error of ~1e-8 AU, which is negligible.
        obs_helio   = get_observer_helio(t_tt, obscode, obscodes)
        earth_helio = get_earth_helio(t_tt)
        d = obs_helio - earth_helio

        if np.allclose(d, 0.0):
            continue   # geocenter equivalent (space telescope, etc.)

        ra_rad  = result[:, k, 1] * _DEG2RAD
        dec_rad = result[:, k, 2] * _DEG2RAD
        delta   = result[:, k, 8]

        cos_ra  = np.cos(ra_rad)
        sin_ra  = np.sin(ra_rad)
        cos_dec = np.cos(dec_rad)
        sin_dec = np.sin(dec_rad)

        # Topocentric parallax: angular shift (radians)
        dra_rad  = -(d[0]*sin_ra  - d[1]*cos_ra) / (delta * cos_dec)
        ddec_rad = -(d[0]*cos_ra*sin_dec + d[1]*sin_ra*sin_dec - d[2]*cos_dec) / delta

        result[:, k, 1] += dra_rad  * _RAD2DEG
        result[:, k, 2] += ddec_rad * _RAD2DEG

        # Topocentric distance: subtract projection of observer offset onto LOS
        result[:, k, 8] -= d[0]*cos_dec*cos_ra + d[1]*cos_dec*sin_ra + d[2]*sin_dec

    if single_epoch:
        return result[:, 0, :]
    return result


# ---------------------------------------------------------------------------
# Angular separation (vectorized)
# ---------------------------------------------------------------------------

def ang_sep_deg(ra1: np.ndarray, dec1: np.ndarray,
                ra2: float, dec2: float) -> np.ndarray:
    """
    Great-circle angular separation in degrees (Vincenty-like formula).
    ra1/dec1: arrays (degrees); ra2/dec2: scalar (degrees)
    """
    r1 = ra1  * _DEG2RAD
    d1 = dec1 * _DEG2RAD
    r2 = ra2  * _DEG2RAD
    d2 = dec2 * _DEG2RAD
    dra = r1 - r2
    x = np.cos(d1)*np.sin(dra)
    y = np.cos(d2)*np.sin(d1) - np.sin(d2)*np.cos(d1)*np.cos(dra)
    z = np.sin(d2)*np.sin(d1) + np.cos(d2)*np.cos(d1)*np.cos(dra)
    return np.arctan2(np.sqrt(x*x + y*y), z) * _RAD2DEG


# ---------------------------------------------------------------------------
# Phase angle and V magnitude
# ---------------------------------------------------------------------------

def vmag_HG(H: float, G: float, r_helio: float, delta: float,
            phase_deg: float) -> float:
    """
    Compute predicted V magnitude using the H-G phase function.
    H, G  : absolute magnitude and slope parameter
    r_helio : heliocentric distance (AU)
    delta   : geocentric distance (AU)
    phase_deg : phase angle (degrees)
    """
    phi1 = np.exp(-3.33 * np.tan(phase_deg * _DEG2RAD / 2)**0.63)
    phi2 = np.exp(-1.87 * np.tan(phase_deg * _DEG2RAD / 2)**1.22)
    return H + 5*np.log10(r_helio * delta) - 2.5*np.log10((1-G)*phi1 + G*phi2)


def phase_angle(obj_helio: np.ndarray,
                obs_helio: np.ndarray) -> float:
    """Compute Sun-Object-Observer phase angle in degrees."""
    # Vector from object to sun: -obj_helio
    # Vector from object to observer: obs_helio - obj_helio
    to_sun  = -obj_helio
    to_obs  = obs_helio - obj_helio
    cos_ph  = (np.dot(to_sun, to_obs)
               / (np.linalg.norm(to_sun) * np.linalg.norm(to_obs) + 1e-30))
    return np.arccos(np.clip(cos_ph, -1, 1)) * _RAD2DEG


# ---------------------------------------------------------------------------
# pyoorb precise ephemeris
# ---------------------------------------------------------------------------

_OORB_INIT = False


def _init_oorb(force: bool = False):
    """Initialise (or re-initialise) the pyoorb library.

    Parameters
    ----------
    force : if True, call oorb_init even if pyoorb was already initialised.
            Needed after heavy N-body use: the integrator accumulates internal
            state (step-size caches, error estimates) that can make single-orbit
            N-body calls fail with NaN for close-approach objects.  Forcing a
            re-init resets this state without affecting cached catalog data.
    """
    global _OORB_INIT
    if _OORB_INIT and not force:
        return
    import pyoorb as oo
    from .config import OORB_EPHEM
    ephem = str(OORB_EPHEM)
    if not os.path.exists(ephem):
        raise FileNotFoundError(f'pyoorb ephemeris not found: {ephem}')
    oo.pyoorb.oorb_init(ephem)
    _OORB_INIT = True
    if not force:
        log.info('pyoorb initialized with %s', ephem)


def build_oorb_orbits_kep(
    a: np.ndarray,
    e: np.ndarray,
    i_rad: np.ndarray,
    Omega_rad: np.ndarray,
    omega_rad: np.ndarray,
    M_rad: np.ndarray,
    epoch_mjd: np.ndarray,
    H: np.ndarray,
    G: np.ndarray,
) -> np.ndarray:
    """
    Pack orbital elements into the pyoorb [n, 12] Fortran-order array.
    Element type 3 = KEP (a, e, i, Omega, omega, M).
    """
    n = len(a)
    orbits = np.zeros((n, 12), dtype=np.double, order='F')
    orbits[:, 0]  = np.arange(1, n+1)   # orbit ID
    orbits[:, 1]  = a
    orbits[:, 2]  = e
    orbits[:, 3]  = i_rad
    orbits[:, 4]  = Omega_rad
    orbits[:, 5]  = omega_rad
    orbits[:, 6]  = M_rad
    orbits[:, 7]  = 3          # KEP element type
    orbits[:, 8]  = epoch_mjd
    orbits[:, 9]  = 3          # TT timescale
    orbits[:, 10] = H
    orbits[:, 11] = G
    return orbits


def build_oorb_orbits_com(
    q: np.ndarray,
    e: np.ndarray,
    i_rad: np.ndarray,
    Omega_rad: np.ndarray,
    omega_rad: np.ndarray,
    Tp_mjd: np.ndarray,
    H: np.ndarray,
    G: np.ndarray,
) -> np.ndarray:
    """
    Pack cometary orbital elements (q, e, i, Omega, omega, Tp) for pyoorb.
    Element type 2 = COM.
    """
    n = len(q)
    orbits = np.zeros((n, 12), dtype=np.double, order='F')
    orbits[:, 0]  = np.arange(1, n+1)
    orbits[:, 1]  = q
    orbits[:, 2]  = e
    orbits[:, 3]  = i_rad
    orbits[:, 4]  = Omega_rad
    orbits[:, 5]  = omega_rad
    orbits[:, 6]  = Tp_mjd
    orbits[:, 7]  = 2          # COM element type
    orbits[:, 8]  = Tp_mjd     # epoch = perihelion time
    orbits[:, 9]  = 3          # TT
    orbits[:, 10] = H
    orbits[:, 11] = G
    return orbits


def reepoch_high_e_asteroids(
    asteroids: np.ndarray,
    t_target_tt: float,
    dynmodel: str = 'N',
    e_threshold: float = 0.5,
    cache_dir=None,
) -> np.ndarray:
    """
    Re-epoch high-eccentricity asteroids to t_target_tt via N-body integration.

    Uses pyoorb.oorb_propagation to integrate the high-e subset (~30K objects)
    from the catalog epoch to t_target_tt.  This greatly reduces Phase 1
    Keplerian propagation errors for close-approach NEOs when the catalog
    epoch is months away from the observation epoch.

    Results are cached in cache_dir (if provided) keyed by the catalog epoch,
    the target epoch rounded to the nearest 5 days, and the dynmodel.  The
    cache is invalidated when MPCORB is updated (catalog epoch changes).

    Returns a copy of the asteroids array with updated elements for the
    high-e subset.  On any pyoorb failure the original array is returned
    unchanged (the wide pre-filter in Phase 1 acts as a safety net).

    Parameters
    ----------
    asteroids   : structured array from mpcorb.load_mpcorb()
    t_target_tt : target epoch, MJD TT
    dynmodel    : 'N' (N-body) or '2' (two-body); use 'N' for accuracy
    e_threshold : eccentricity cut; default 0.5 (~30K objects in MPCORB)
    cache_dir   : directory for caching re-epoched elements (Path or str)
    """
    from pathlib import Path

    mask = asteroids['e'] > e_threshold
    if not np.any(mask):
        return asteroids

    # Round target epoch to nearest 5 days for cache key
    t_rounded = round(t_target_tt / 5.0) * 5
    cat_epoch = int(round(float(np.median(asteroids['epoch']))))
    n_high_e  = int(mask.sum())   # included in key so H-filtered catalogs don't collide

    # ---- Try cache ----
    cache_path = None
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_path = cache_dir / f'reepoch_{cat_epoch}_{t_rounded}_{dynmodel}_{n_high_e}.npz'
        if cache_path.exists():
            try:
                cached = np.load(cache_path)
                result = asteroids.copy()
                idx = np.where(mask)[0]
                for col in ('a', 'e', 'i', 'Omega', 'omega', 'M', 'epoch'):
                    result[col][idx] = cached[col]
                log.info('Loaded re-epoched high-e elements from cache: %s', cache_path.name)
                return result
            except Exception as exc:
                log.warning('Re-epoch cache load failed (%s); recomputing', exc)

    # ---- Compute ----
    import pyoorb as oo
    _init_oorb()

    sub = asteroids[mask]
    orbits = build_oorb_orbits_kep(
        sub['a'], sub['e'],
        sub['i']     * _DEG2RAD,
        sub['Omega'] * _DEG2RAD,
        sub['omega'] * _DEG2RAD,
        sub['M']     * _DEG2RAD,
        sub['epoch'], sub['H'], sub['G'],
    )

    epoch_arr = np.array([[t_target_tt, 3]], dtype=np.double, order='F')  # TT
    new_orbits, err = oo.pyoorb.oorb_propagation(
        in_orbits=orbits,
        in_epoch=epoch_arr,
        in_dynmodel=dynmodel,
    )

    if err != 0:
        log.warning('pyoorb re-epoch failed (err=%d); using original elements', err)
        return asteroids

    # oorb_propagation returns angular elements (i, Omega, omega, M) in degrees
    result = asteroids.copy()
    idx = np.where(mask)[0]
    result['a'][idx]     = new_orbits[:, 1]
    result['e'][idx]     = new_orbits[:, 2]
    result['i'][idx]     = new_orbits[:, 3]    # degrees
    result['Omega'][idx] = new_orbits[:, 4]    # degrees
    result['omega'][idx] = new_orbits[:, 5]    # degrees
    result['M'][idx]     = new_orbits[:, 6]    # degrees
    result['epoch'][idx] = t_target_tt

    log.info('Re-epoched %d high-e (e>%.1f) asteroids to MJD TT %.1f',
             len(idx), e_threshold, t_target_tt)

    # ---- Save cache ----
    if cache_path is not None:
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            save = {col: result[col][idx]
                    for col in ('a', 'e', 'i', 'Omega', 'omega', 'M', 'epoch')}
            np.savez(cache_path, **save)
            log.info('Saved re-epoch cache: %s', cache_path.name)
        except Exception as exc:
            log.warning('Re-epoch cache save failed: %s', exc)

    return result


def oorb_ephemeris(
    orbits: np.ndarray,
    t_mjd: float,
    obscode: str,
    dynmodel: str = '2',        # '2'=two-body, 'N'=N-body
    obscodes: dict = None,
) -> np.ndarray:
    """
    Compute ephemerides for a batch of orbits using pyoorb.

    Returns array of shape [n, 11]:
      0: MJD, 1: RA(deg), 2: Dec(deg), 3: dRA/dt(deg/day), 4: dDec/dt(deg/day),
      5: phase angle(deg), 6: elongation(deg), 7: r_helio(AU), 8: delta(AU),
      9: V-mag, 10: position angle

    pyoorb is always called with geocenter ('500') to avoid its brittle
    OBSCODE.dat parser failing on recently-added MPC codes.  When obscodes
    is provided the topocentric correction is applied by this function.

    Problem objects that pyoorb cannot integrate are returned with V-mag=99.
    """
    import pyoorb as oo
    _init_oorb()

    n = len(orbits)
    epochs = np.array([[t_mjd, 3]], dtype=np.double, order='F')   # TT
    eph, err = oo.pyoorb.oorb_ephemeris_basic(
        in_orbits=orbits,
        in_obscode='500 ',
        in_date_ephems=epochs,
        in_dynmodel=dynmodel,
    )
    if err == 0:
        result = eph[:, 0, :]   # [n_orbits, 11]
        if obscodes is not None:
            result = _apply_topocentric_correction(result, t_mjd, obscode, obscodes)
        return result

    # Non-zero error: split into sub-batches and retry after reinit.
    log.warning('pyoorb batch error %d; retrying %d orbits in sub-batches', err, n)
    _init_oorb(force=True)
    _SUBBATCH = 200
    result = np.full((n, 11), np.nan)
    result[:, 9] = 99.0   # default V-mag = 99 (marks failure)
    for start in range(0, n, _SUBBATCH):
        chunk = orbits[start:start + _SUBBATCH]
        try:
            e2, err2 = oo.pyoorb.oorb_ephemeris_basic(
                in_orbits=chunk,
                in_obscode='500 ',
                in_date_ephems=epochs,
                in_dynmodel=dynmodel,
            )
            if err2 == 0:
                result[start:start + len(chunk)] = e2[:, 0, :]
            else:
                _init_oorb(force=True)
                for j in range(len(chunk)):
                    try:
                        e3, err3 = oo.pyoorb.oorb_ephemeris_basic(
                            in_orbits=chunk[j:j+1],
                            in_obscode='500 ',
                            in_date_ephems=epochs,
                            in_dynmodel=dynmodel,
                        )
                        if err3 == 0:
                            result[start + j] = e3[0, 0]
                    except Exception:
                        pass
        except Exception:
            pass
    if obscodes is not None:
        result = _apply_topocentric_correction(result, t_mjd, obscode, obscodes)
    return result


def oorb_ephemeris_multi_epoch(
    orbits: np.ndarray,
    t_mjd_list: list,
    obscode: str,
    dynmodel: str = '2',
    obscodes: dict = None,
) -> np.ndarray:
    """
    Compute ephemerides for a batch of orbits at multiple epochs in one pyoorb call.

    Returns array of shape [n_orbits, n_epochs, 11].
    Columns same as oorb_ephemeris: MJD, RA, Dec, dRA, dDec, phase, elong, r, delta, V, PA.

    pyoorb is always called with geocenter ('500'); topocentric correction is
    applied via _apply_topocentric_correction when obscodes is provided.
    """
    import pyoorb as oo
    _init_oorb()

    n_epochs = len(t_mjd_list)
    epochs = np.array([[t, 3] for t in t_mjd_list], dtype=np.double, order='F')

    eph, err = oo.pyoorb.oorb_ephemeris_basic(
        in_orbits=orbits,
        in_obscode='500 ',
        in_date_ephems=epochs,
        in_dynmodel=dynmodel,
    )
    if err == 0:
        if obscodes is not None:
            eph = _apply_topocentric_correction(eph, t_mjd_list, obscode, obscodes)
        return eph  # [n_orbits, n_epochs, 11]

    # Non-zero error: split into sub-batches and retry.
    # The batch may have failed because: (a) too many orbits (pyoorb internal
    # limit, error 35), or (b) a corrupted integrator state. Re-initialise
    # before retrying so that close-approach orbits (which fail silently when
    # state is dirty) are processed correctly.
    log.warning('pyoorb multi-epoch batch error %d; retrying %d orbits in sub-batches',
                err, len(orbits))
    _init_oorb(force=True)
    _SUBBATCH = 200   # well below pyoorb's internal limit
    n = len(orbits)
    result = np.full((n, n_epochs, 11), np.nan)
    result[:, :, 9] = 99.0
    for start in range(0, n, _SUBBATCH):
        chunk = orbits[start:start + _SUBBATCH]
        try:
            e2, err2 = oo.pyoorb.oorb_ephemeris_basic(
                in_orbits=chunk,
                in_obscode='500 ',
                in_date_ephems=epochs,
                in_dynmodel=dynmodel,
            )
            if err2 == 0:
                result[start:start + len(chunk)] = e2
            else:
                # Sub-batch also failed: retry one-by-one within this chunk
                _init_oorb(force=True)
                for j in range(len(chunk)):
                    try:
                        e3, err3 = oo.pyoorb.oorb_ephemeris_basic(
                            in_orbits=chunk[j:j+1],
                            in_obscode='500 ',
                            in_date_ephems=epochs,
                            in_dynmodel=dynmodel,
                        )
                        if err3 == 0:
                            result[start + j] = e3[0]
                    except Exception:
                        pass
        except Exception:
            pass
    if obscodes is not None:
        result = _apply_topocentric_correction(result, t_mjd_list, obscode, obscodes)
    return result


def precise_ephemeris(
    asteroid_arr: np.ndarray,
    candidate_idx: np.ndarray,
    t_mjd: float,
    obscode: str,
    dynmodel: str = 'N',
) -> np.ndarray:
    """
    Run pyoorb N-body ephemeris on a subset of the asteroid array.
    Returns [n_candidates, 11] ephemeris array.
    """
    sub = asteroid_arr[candidate_idx]

    i_rad     = sub['i']     * _DEG2RAD
    Omega_rad = sub['Omega'] * _DEG2RAD
    omega_rad = sub['omega'] * _DEG2RAD
    M_rad     = sub['M']     * _DEG2RAD

    orbits = build_oorb_orbits_kep(
        sub['a'], sub['e'], i_rad, Omega_rad, omega_rad, M_rad,
        sub['epoch'], sub['H'], sub['G'],
    )

    # pyoorb uses MJD TT; convert t_mjd (UTC) → TT
    from astropy.time import Time
    tt_mjd = Time(t_mjd, format='mjd', scale='utc').tt.mjd

    return oorb_ephemeris(orbits, tt_mjd, obscode, dynmodel=dynmodel)


def precise_ephemeris_comets(
    comet_arr: np.ndarray,
    candidate_idx: np.ndarray,
    t_mjd: float,
    obscode: str,
    dynmodel: str = 'N',
) -> np.ndarray:
    """Run pyoorb N-body ephemeris on a subset of the comet array."""
    sub = comet_arr[candidate_idx]

    i_rad     = sub['i']     * _DEG2RAD
    Omega_rad = sub['Omega'] * _DEG2RAD
    omega_rad = sub['omega'] * _DEG2RAD

    orbits = build_oorb_orbits_com(
        sub['q'], sub['e'], i_rad, Omega_rad, omega_rad,
        sub['Tp'], sub['H'], sub['G'],
    )

    from astropy.time import Time
    tt_mjd = Time(t_mjd, format='mjd', scale='utc').tt.mjd

    return oorb_ephemeris(orbits, tt_mjd, obscode, dynmodel=dynmodel)
