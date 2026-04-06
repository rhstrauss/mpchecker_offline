"""
Main mpchecker matching logic.

Pipeline for each observation:
  1. Fast Keplerian pre-filter — propagate ALL asteroids/comets (vectorized
     NumPy) and keep those within a generous angular buffer of the target.
     Uses a per-observation H limit derived from field geometry (Sun elongation
     + outer planet proximity) to reduce the catalog before propagation.
  2. Precise ephemeris (pyoorb) — observations are grouped by observatory code
     and processed in a single batched pyoorb call covering all epochs at once,
     then post-filtered per observation.
  3. Satellite check (SpiceyPy) — test all known planetary satellites.

Results are returned as a list of Match dataclasses, one per observation.
"""

import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .obs_parser import Observation
from .propagator import (
    kep_to_radec, ang_sep_deg, get_observer_helio, get_planet_helio,
    build_oorb_orbits_kep, build_oorb_orbits_com,
    oorb_ephemeris_multi_epoch, oorb_ephemeris_multi_epoch_split,
    reepoch_high_e_asteroids,
    _init_oorb,
    _DEG2RAD, _RAD2DEG,
)
from .satellites import check_satellites, check_dwarf_planet_satellites
from .index import SkyIndex, MultiSkyIndex

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level state for fork-based multiprocessing workers
# Set by check_observations before Pool creation; inherited copy-on-write.
# ---------------------------------------------------------------------------
_PHASE1_STATE: dict = {}
_PHASE2_STATE: dict = {}

# ---------------------------------------------------------------------------
# Phase 1 field candidate cache (P1-C)
# Caches Phase 1 (Keplerian pre-filter) results by coarse (ra, dec, epoch, h_cut)
# bins.  Within a 6-hour / 0.5° bin, MBA motion is < 0.18° — well below the
# 0.5° margin used in the bin.  NEO safety: the wide-NEO and close-approach
# passes always run on the CACHED candidates (cheap, < 1 ms), so no NEO is lost.
#
# Effective primarily in the daemon (single long-lived process) and for surveys
# that revisit the same field multiple times per night (ATLAS, ZTF, etc.).
# In fork-based multi-worker mode the cache is per-process (copy-on-write);
# parent pre-populates on unique bins before forking.
#
# Max 512 entries; simple FIFO eviction to cap memory.
# ---------------------------------------------------------------------------
_FIELD_CACHE: dict = {}
_FIELD_CACHE_MAX = 512


@dataclass
class Match:
    """A single matched object for one observation."""
    name:          str
    packed:        str
    obj_type:      str          # 'minor_planet', 'comet', 'satellite'
    ra_deg:        float        # predicted RA (degrees)
    dec_deg:       float        # predicted Dec (degrees)
    sep_arcsec:    float        # angular separation (arcsec)
    ra_rate:       float        # dRA*cos(Dec) (arcsec/hr)
    dec_rate:      float        # dDec (arcsec/hr)
    r_helio:       float        # heliocentric distance (AU)
    delta:         float        # geocentric distance (AU)
    vmag:          float        # predicted V magnitude
    phase_deg:     float        # phase angle (degrees)
    orbit_quality: str  = ''    # MPC U parameter ('0'–'9', 'E', 'D', …); '' for satellites


@dataclass
class CheckResult:
    """Results for one input observation."""
    obs:     Observation
    matches: List[Match] = field(default_factory=list)


@dataclass
class Identification:
    """
    Orbit-level identification: a single catalog candidate whose predicted
    orbit fits all (or min_obs) input observations simultaneously.

    Returned by identify_tracklet().  An Identification represents a high-
    confidence association between an input multi-observation tracklet and a
    known catalog object: the candidate's orbit explains the observed positions
    at every input epoch, with per-observation O-C residuals all within the
    requested threshold.

    Fields
    ------
    match            : the candidate, as it appeared in Phase 2 results; None
                       when no Phase 2 candidate was found (orbit_fit only)
    residuals        : per-observation O-C in arcsec (same order as input obs)
    rms_arcsec       : RMS of residuals across all input observations
    n_obs            : total number of input observations
    method           : 'ephemeris'  — residuals from catalog orbit via pyoorb
                       'orbit_fit'  — independent orbit fit via fo; residuals
                                      are O-C from the fo solution
    fo_elements      : fitted orbit (1-element ASTEROID_DTYPE array); only set
                       when method='orbit_fit'
    fo_catalog_name  : best catalog match by (a, e, i) element similarity; set
                       when method='orbit_fit' and a match is found in MPCORB
    fo_catalog_score : orbital element similarity score for fo_catalog_name
                       (dimensionless; smaller is better)
    """
    match:            Optional[Match]
    residuals:        List[float]
    rms_arcsec:       float
    n_obs:            int
    method:           str
    fo_elements:      Optional[np.ndarray] = None
    fo_catalog_name:  Optional[str]        = None
    fo_catalog_score: Optional[float]      = None
    fo_rms_internal:  Optional[float]      = None   # fo's own RMS (arcsec); reliable for close-approach objects
    fo_n_obs:         Optional[int]        = None   # observations used in fo fit
    fo_earth_moid_au: Optional[float]      = None   # Earth MOID from fo solution (AU)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utc_to_tt_mjd(mjd_utc: float) -> float:
    """MJD UTC → MJD TT."""
    from astropy.time import Time
    return Time(mjd_utc, format='mjd', scale='utc').tt.mjd


def _utc_to_tt_mjd_vec(mjd_utc_iterable) -> np.ndarray:
    """Vectorized UTC MJD → TT MJD: single astropy.Time call for an iterable."""
    from astropy.time import Time
    arr = np.asarray(list(mjd_utc_iterable), dtype=np.float64)
    if len(arr) == 0:
        return arr
    return Time(arr, format='mjd', scale='utc').tt.mjd


def _rate_to_arcsec_per_hr(deg_per_day: float) -> float:
    return deg_per_day * 3600.0 / 24.0


# Obliquity of the ecliptic at J2000.0 (radians)
_ECLIPTIC_OBL = 23.4392794 * _DEG2RAD


def _ecliptic_lat_rad(ra_deg: float, dec_deg: float) -> float:
    """Ecliptic latitude (radians) for J2000 equatorial (ra, dec) in degrees."""
    ra = ra_deg * _DEG2RAD
    dec = dec_deg * _DEG2RAD
    sin_beta = np.sin(dec) * np.cos(_ECLIPTIC_OBL) - np.cos(dec) * np.sin(_ECLIPTIC_OBL) * np.sin(ra)
    return float(np.arcsin(np.clip(sin_beta, -1.0, 1.0)))


# ---------------------------------------------------------------------------
# Geometry-aware H limit (Optimization 3)
# ---------------------------------------------------------------------------

# Giant planets: (name, approximate heliocentric distance AU)
_GIANT_PLANETS = [
    ('jupiter', 5.2),
    ('saturn',  9.5),
    ('uranus',  19.2),
    ('neptune', 30.1),
]
# Proximity threshold: if field is within this many degrees of a giant planet,
# use that planet's distance to compute a tighter H limit
_PLANET_CONE_DEG = 15.0


def _h_limit_from_vmag(
    vmag_limit: float,
    obs_helio: Optional[np.ndarray] = None,
    ra_deg: Optional[float] = None,
    dec_deg: Optional[float] = None,
    t_mjd: Optional[float] = None,
) -> float:
    """
    Estimate the maximum H magnitude that could produce vmag < vmag_limit.

    When obs_helio, ra_deg, dec_deg, and t_mjd are provided, uses the field's
    Sun elongation and proximity to outer planets for a tighter (faster) estimate.
    Conservative: uses typical distances rather than minimum possible distances
    so that real objects are not missed.
    """
    r_approx, delta_approx = 2.0, 1.5  # default: typical main-belt

    if obs_helio is not None and ra_deg is not None and dec_deg is not None:
        r_earth = float(np.linalg.norm(obs_helio))

        # Sun elongation of the field
        sun_dir = -obs_helio / r_earth
        obs_vec = np.array([
            np.cos(dec_deg * _DEG2RAD) * np.cos(ra_deg * _DEG2RAD),
            np.cos(dec_deg * _DEG2RAD) * np.sin(ra_deg * _DEG2RAD),
            np.sin(dec_deg * _DEG2RAD),
        ])
        elong_deg = float(np.arccos(np.clip(np.dot(sun_dir, obs_vec), -1, 1)) * _RAD2DEG)

        # Elongation-based estimate for outer solar system fields
        if elong_deg >= 150:
            r_approx, delta_approx = 2.8, 1.8
        elif elong_deg >= 90:
            r_approx, delta_approx = 2.5, 1.5
        # else: near Sun, keep defaults (conservative)

        # Check if field is near a giant planet — much tighter H limit
        if t_mjd is not None and elong_deg >= 60:
            for planet, r_planet in _GIANT_PLANETS:
                try:
                    ph = np.array(get_planet_helio(planet, round(t_mjd)))
                    # Planet's geocentric direction
                    planet_geo = ph - obs_helio
                    planet_delta = float(np.linalg.norm(planet_geo))
                    planet_dir = planet_geo / planet_delta
                    cos_sep = float(np.clip(np.dot(planet_dir, obs_vec), -1, 1))
                    sep_deg = float(np.arccos(cos_sep) * _RAD2DEG)
                    if sep_deg < _PLANET_CONE_DEG:
                        r_approx    = r_planet
                        delta_approx = planet_delta
                        log.debug('Field within %.1f° of %s (r=%.1f AU, delta=%.1f AU)',
                                  sep_deg, planet, r_approx, delta_approx)
                        break
                except Exception:
                    pass

    return vmag_limit - 5.0 * np.log10(r_approx * delta_approx) + 1.5


# ---------------------------------------------------------------------------
# Pre-filter helpers
# ---------------------------------------------------------------------------

def _asteroid_prefilter(
    catalog: np.ndarray,
    obs_helio: np.ndarray,
    t_tt: float,
    ra_obs: float,
    dec_obs: float,
    radius_deg: float,
) -> np.ndarray:
    """
    Vectorized Keplerian propagation over the entire asteroid catalog.
    Returns boolean mask of candidates within radius_deg.
    """
    i_rad     = catalog['i']     * _DEG2RAD
    Omega_rad = catalog['Omega'] * _DEG2RAD
    omega_rad = catalog['omega'] * _DEG2RAD
    M0_rad    = catalog['M']     * _DEG2RAD
    epoch_tt  = catalog['epoch']   # already TT

    ra_pred, dec_pred, _ = kep_to_radec(
        catalog['a'], catalog['e'],
        i_rad, Omega_rad, omega_rad, M0_rad,
        epoch_tt, t_tt,
        obs_helio,
    )

    sep = ang_sep_deg(ra_pred, dec_pred, ra_obs, dec_obs)
    return sep <= radius_deg


@dataclass
class AsteroidSOA:
    """Pre-extracted contiguous float64 column arrays from the asteroid catalog.
    Built once (e.g. at daemon startup) to eliminate strided copy overhead
    when running the full-catalog Keplerian pre-filter.
    """
    a:         np.ndarray
    e:         np.ndarray
    i_rad:     np.ndarray   # inclination (radians)
    Omega_rad: np.ndarray   # longitude of ascending node (radians)
    omega_rad: np.ndarray   # argument of periapsis (radians)
    M_rad:     np.ndarray   # mean anomaly (radians)
    epoch:     np.ndarray   # epoch MJD TT
    H:         np.ndarray   # absolute magnitude
    q:         np.ndarray   # perihelion distance (AU) = a*(1-e)


def build_asteroid_soa(asteroids: np.ndarray) -> AsteroidSOA:
    """Build pre-extracted contiguous float64 arrays from the structured asteroid catalog."""
    a = np.ascontiguousarray(asteroids['a'], dtype=np.float64)
    e = np.ascontiguousarray(asteroids['e'], dtype=np.float64)
    return AsteroidSOA(
        a=a,
        e=e,
        i_rad=np.ascontiguousarray(asteroids['i'],     dtype=np.float64) * _DEG2RAD,
        Omega_rad=np.ascontiguousarray(asteroids['Omega'], dtype=np.float64) * _DEG2RAD,
        omega_rad=np.ascontiguousarray(asteroids['omega'], dtype=np.float64) * _DEG2RAD,
        M_rad=np.ascontiguousarray(asteroids['M'],     dtype=np.float64) * _DEG2RAD,
        epoch=np.ascontiguousarray(asteroids['epoch'], dtype=np.float64),
        H=np.ascontiguousarray(asteroids['H'],     dtype=np.float64),
        q=a * (1.0 - e),
    )


def _asteroid_prefilter_soa(
    soa: AsteroidSOA,
    h_mask: np.ndarray,
    obs_helio: np.ndarray,
    t_tt: float,
    ra_obs: float,
    dec_obs: float,
    radius_deg: float,
) -> np.ndarray:
    """Fast prefilter for the full-catalog path using pre-extracted SOA arrays."""
    ra_pred, dec_pred, _ = kep_to_radec(
        soa.a[h_mask],         soa.e[h_mask],
        soa.i_rad[h_mask],     soa.Omega_rad[h_mask],
        soa.omega_rad[h_mask], soa.M_rad[h_mask],
        soa.epoch[h_mask],     t_tt,
        obs_helio,
    )
    sep = ang_sep_deg(ra_pred, dec_pred, ra_obs, dec_obs)
    return sep <= radius_deg


def _comet_prefilter(
    comets: np.ndarray,
    obs_helio: np.ndarray,
    t_tt: float,
    ra_obs: float,
    dec_obs: float,
    radius_deg: float,
) -> np.ndarray:
    """
    Keplerian pre-filter for comets.
    Converts COM (q, e, i, Ω, ω, Tp) → (a, e, i, Ω, ω, M) then calls kep_to_radec.
    For near-parabolic or hyperbolic comets this is only approximate,
    which is fine for a pre-filter.
    """
    e = comets['e']
    q = comets['q']
    Tp = comets['Tp']      # MJD TT
    i_rad     = comets['i']     * _DEG2RAD
    Omega_rad = comets['Omega'] * _DEG2RAD
    omega_rad = comets['omega'] * _DEG2RAD

    elliptic = e < 0.99
    a = np.where(elliptic, q / np.maximum(1 - e, 1e-6), 1e6)

    n = np.sqrt(0.01720209895**2 / np.maximum(a, 1e-6)**3)
    M = n * (t_tt - Tp)
    M = np.where(elliptic, M % (2*np.pi), M)

    ra_pred, dec_pred, _ = kep_to_radec(
        a, e, i_rad, Omega_rad, omega_rad, M,
        np.zeros_like(a), 0.0,
        obs_helio,
    )

    sep = ang_sep_deg(ra_pred, dec_pred, ra_obs, dec_obs)
    return sep <= radius_deg


# ---------------------------------------------------------------------------
# Phase 1 worker (extracted for multiprocessing)
# ---------------------------------------------------------------------------

def _phase1_one_obs(
    obs,
    asteroids:    np.ndarray,
    comets:       np.ndarray,
    obscodes:     dict,
    sky_index,
    mag_limit:    float,
    prefilter_deg: float,
    asteroid_soa: Optional[AsteroidSOA] = None,
) -> dict:
    """
    Run Phase 1 (Keplerian pre-filter) for a single observation.
    Returns a dict with t_tt, obs_helio, ast_cands, comet_cands.
    Safe to call in a forked worker process.
    """
    t_tt      = _utc_to_tt_mjd(obs.epoch_mjd)
    obs_helio = get_observer_helio(obs.epoch_mjd, obs.obscode, obscodes)
    h_cut     = _h_limit_from_vmag(mag_limit, obs_helio, obs.ra_deg, obs.dec_deg, obs.epoch_mjd)

    _H_arr        = asteroid_soa.H if asteroid_soa is not None else asteroids['H']
    n_h_filtered  = int((_H_arr <= h_cut).sum())
    ast_cands     = np.array([], dtype=np.intp)
    _INDEX_THRESHOLD = 400_000

    # Field candidate cache (P1-C): cache the main-scan result by coarse spatial
    # / temporal / magnitude bin.  Within a 0.5°/6-hr/0.5-mag bin, MBA motion
    # < 0.18° — well inside the bin margin.  The NEO safety passes below always
    # run regardless of cache state.
    _cache_key = (round(obs.ra_deg / 0.5) * 0.5,
                  round(obs.dec_deg / 0.5) * 0.5,
                  int(obs.epoch_mjd / 0.25),
                  round(h_cut / 0.5) * 0.5)
    if _cache_key in _FIELD_CACHE:
        ast_cands = _FIELD_CACHE[_cache_key].copy()
        log.debug('Field cache hit for %s (%d cands)', obs.designation, len(ast_cands))
    else:
        if (sky_index is not None and sky_index.is_fresh(obs.epoch_mjd)
                and n_h_filtered > _INDEX_THRESHOLD):
            idx_mask = sky_index.candidates(obs.ra_deg, obs.dec_deg, prefilter_deg,
                                            t_obs_mjd=obs.epoch_mjd)
            idx_mask &= (_H_arr <= h_cut)
            broad_idx = np.where(idx_mask)[0]
            log.debug('Index cone: %d candidates for %s', len(broad_idx), obs.designation)
            if len(broad_idx) > 0:
                kep_mask  = _asteroid_prefilter(
                    asteroids[broad_idx], obs_helio, t_tt,
                    obs.ra_deg, obs.dec_deg, prefilter_deg)
                ast_cands = broad_idx[kep_mask]
        else:
            h_mask = _H_arr <= h_cut
            # Element-space pre-filter: narrow h_mask before the Keplerian kernel.
            # (1) Inclination lower bound — objects with i < |ecliptic_lat| - buffer
            #     can never reach the observation's ecliptic latitude.  Safe: never
            #     produces false negatives.  Most effective for high-latitude fields.
            # (2) Perihelion brightness limit — objects whose brightness at closest
            #     approach (perihelion, opposition geometry) still exceeds mag_limit
            #     are never visible.  Handles faint outer-belt objects that slip past
            #     the H cut (which assumes a fixed r≈2.5 AU).
            # Both filters are only applied when the SOA is available (pre-extracted
            # arrays avoid the strided-access cost of the structured array).
            if asteroid_soa is not None and n_h_filtered > 0:
                r_obs_dist = float(np.linalg.norm(obs_helio))
                # (1) inclination lower bound
                ecl_lat = _ecliptic_lat_rad(obs.ra_deg, obs.dec_deg)
                i_min   = max(0.0, abs(ecl_lat) - 0.17)   # 0.17 rad ≈ 10° safety buffer
                h_mask &= asteroid_soa.i_rad >= i_min
                # (2) perihelion brightness: q*(q - r_obs) <= 10^((mag_limit-H)/5)
                q_arr   = asteroid_soa.q
                q_delta = np.maximum(q_arr - r_obs_dist, 0.001)
                h_mask &= q_arr * q_delta <= np.power(10.0, (mag_limit - asteroid_soa.H) / 5.0)
            log.debug('H≤%.1f + element filter: %d/%d asteroids for %s',
                      h_cut, int(h_mask.sum()), len(asteroids), obs.designation)
            if h_mask.any():
                if asteroid_soa is not None:
                    kep_mask = _asteroid_prefilter_soa(
                        asteroid_soa, h_mask, obs_helio, t_tt,
                        obs.ra_deg, obs.dec_deg, prefilter_deg)
                else:
                    kep_mask = _asteroid_prefilter(
                        asteroids[h_mask], obs_helio, t_tt,
                        obs.ra_deg, obs.dec_deg, prefilter_deg)
                bright_global_idx = np.where(h_mask)[0]
                ast_cands         = bright_global_idx[kep_mask]
        # Store main-scan result to cache (NEO passes run separately below)
        if len(_FIELD_CACHE) >= _FIELD_CACHE_MAX:
            _FIELD_CACHE.pop(next(iter(_FIELD_CACHE)))  # FIFO eviction
        _FIELD_CACHE[_cache_key] = ast_cands.copy()

    # Wide second pass for high-eccentricity objects (close-approach NEOs).
    # Keplerian propagation accumulates O(20') errors for NEOs at closest approach
    # due to planetary perturbations; a wider cone catches them.  We limit this
    # second pass to e > 0.5 (~30K objects out of 1.5M) to keep Phase 2 lean.
    _WIDE_PREFILTER_DEG = 0.5   # 30' absolute minimum for close-approach NEOs
    if prefilter_deg < _WIDE_PREFILTER_DEG and len(asteroids) > 0:
        high_e_h_mask = (asteroids['H'] <= h_cut) & (asteroids['e'] > 0.5)
        if np.any(high_e_h_mask):
            he_indices = np.where(high_e_h_mask)[0]
            kep_wide = _asteroid_prefilter(
                asteroids[high_e_h_mask], obs_helio, t_tt,
                obs.ra_deg, obs.dec_deg, _WIDE_PREFILTER_DEG)
            extra = he_indices[kep_wide]
            if len(extra):
                ast_cands = np.unique(np.concatenate([ast_cands, extra]))
                log.debug('%s: +%d high-e candidates from wide pre-filter',
                          obs.designation, len(extra))

    # Third pass: close-approach NEOs (Apollo/Aten/Amor, q < 1.3 AU) that are
    # fainter than h_cut in absolute magnitude but can be very bright at close
    # approach.  At delta=0.02 AU an H=24.5 object is mag ~17; the standard
    # H_cut (based on r≈2.5, delta≈1.5) entirely misses these.
    # ~22K catalog objects fall in this regime; Keplerian pre-filter on them
    # costs < 10 ms and false positives are handled by Phase 2.
    h_cut_neo = mag_limit - 5.0 * np.log10(1.0 * 0.02) + 1.5  # delta_min=0.02 AU
    if h_cut_neo > h_cut and len(asteroids) > 0:
        q_arr   = asteroids['a'] * (1.0 - asteroids['e'])
        neo_pass_mask = (q_arr < 1.3) & (asteroids['H'] > h_cut) & (asteroids['H'] <= h_cut_neo)
        if np.any(neo_pass_mask):
            neo_indices = np.where(neo_pass_mask)[0]
            kep_neo = _asteroid_prefilter(
                asteroids[neo_pass_mask], obs_helio, t_tt,
                obs.ra_deg, obs.dec_deg, max(prefilter_deg, _WIDE_PREFILTER_DEG))
            extra = neo_indices[kep_neo]
            if len(extra):
                ast_cands = np.unique(np.concatenate([ast_cands, extra]))
                log.debug('%s: +%d close-approach NEO candidates (q<1.3, H>%.1f)',
                          obs.designation, len(extra), h_cut)

    # Opt 9: MBA tightening pass.
    # After all three Phase 1 passes, re-screen MBA-class candidates at half
    # the prefilter radius to reduce Phase 2 (pyoorb) input by ~4×.
    #
    # Safety invariant: objects with q < 1.3 AU (Apollos, Atens, Amors + buffer)
    # or e > 0.5 (highly eccentric, large two-body errors) are NEVER tightened.
    # These objects must always reach pyoorb's N-body integrator.
    #
    # For MBA-class objects (q ≥ 1.3, e < 0.5), two-body position errors over
    # a 2-year catalog-epoch drift are < 30 arcsec, far below the half-prefilter
    # margin (~0.75° = 2700 arcsec for the default 30-arcmin search radius).
    # Running kep_to_radec on ~500 candidates costs < 1 ms (vs ~12 s full scan).
    if len(ast_cands) > 0:
        sub = asteroids[ast_cands]
        q_sub = sub['a'] * (1.0 - sub['e'])
        safe_mask = (q_sub >= 1.3) & (sub['e'] < 0.5)
        n_safe = int(safe_mask.sum())
        if n_safe > 0:
            tight_deg = prefilter_deg * 0.5
            safe_sub  = sub[safe_mask]
            kep_tight = _asteroid_prefilter(
                safe_sub, obs_helio, t_tt,
                obs.ra_deg, obs.dec_deg, tight_deg)
            # Rebuild ast_cands: always keep unsafe (NEO/high-e), keep only
            # tight-passing safe candidates.
            keep = np.ones(len(ast_cands), dtype=bool)
            safe_global = np.where(safe_mask)[0]   # indices into ast_cands
            keep[safe_global[~kep_tight]] = False
            ast_cands = ast_cands[keep]
            n_removed = n_safe - int(kep_tight.sum())
            if n_removed:
                log.debug('%s: MBA tightening removed %d/%d safe candidates '
                          '(tight_deg=%.3f°)',
                          obs.designation, n_removed, n_safe, tight_deg)

    comet_cands = np.array([], dtype=np.intp)
    if len(comets) > 0:
        comet_mask  = _comet_prefilter(
            comets, obs_helio, t_tt, obs.ra_deg, obs.dec_deg, prefilter_deg)
        comet_cands = np.where(comet_mask)[0]

    log.debug('%s: %d asteroid cands, %d comet cands',
              obs.designation, len(ast_cands), len(comet_cands))
    return {'t_tt': t_tt, 'obs_helio': obs_helio,
            'ast_cands': ast_cands, 'comet_cands': comet_cands}


def _phase1_worker(idx_obs):
    """
    Multiprocessing worker for Phase 1 pre-filter.
    Reads heavy arrays from _PHASE1_STATE (inherited via fork — no pickling).

    Numba is DISABLED in each worker. Although the threading layer is OpenMP,
    fork() only copies the calling thread; the OpenMP thread pool is not
    preserved, and attempting to spawn OpenMP worker threads in a forked child
    reliably deadlocks (barrier/mutex state from parent is inconsistent).
    Workers fall back to the NumPy kep_to_radec path, which is ~10× slower
    per-observation but scales linearly with n_workers — with 60-70 workers
    the aggregate throughput exceeds single-process Numba.
    """
    i, obs = idx_obs
    try:
        import mpchecker.propagator as _prop
        _prop._NUMBA_KEP_KERNEL = None   # force NumPy fallback in this process
    except Exception:
        pass
    st = _PHASE1_STATE
    return i, _phase1_one_obs(
        obs,
        st['asteroids'], st['comets'], st['obscodes'], st['sky_index'],
        st['mag_limit'], st['prefilter_deg'],
        asteroid_soa=st.get('asteroid_soa'),
    )


def _phase2_worker(group_item):
    """
    Multiprocessing worker for one obscode group in Phase 2.
    Reads shared arrays from _PHASE2_STATE (inherited via fork).

    Returns dict {obs_i: list[Match]} for the given obscode group.
    """
    import os
    # Prevent thread oversubscription: each worker runs single-threaded so
    # N concurrent workers don't saturate the NUMA node.
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    try:
        import numba as _nb
        _nb.set_num_threads(1)
    except Exception:
        pass

    import mpchecker.propagator as _prop
    _prop._init_oorb(force=True)   # clean pyoorb state in this forked worker

    obscode, obs_indices = group_item
    st            = _PHASE2_STATE
    asteroids     = st['asteroids']
    comets        = st['comets']
    observations  = st['observations']
    obs_meta      = st['obs_meta']
    dynmodel      = st['dynmodel']
    search_rad_deg = st['search_rad_deg']
    mag_limit     = st['mag_limit']
    obscodes      = st['obscodes']

    per_obs: dict = {i: [] for i in obs_indices}
    bright_table = st.get('bright_table')

    # ---- Asteroids ----
    all_ast = sorted(set().union(
        *[set(obs_meta[i]['ast_cands'].tolist()) for i in obs_indices]))
    if all_ast:
        all_ast_arr = np.array(all_ast, dtype=np.intp)
        sub = asteroids[all_ast_arr]
        n_all = len(sub)
        t_tts = [obs_meta[i]['t_tt'] for i in obs_indices]
        n_ep  = len(t_tts)

        eph_batch    = np.full((n_all, n_ep, 11), np.nan)
        eph_batch[:, :, 9] = 99.0
        pyoorb_mask  = np.ones(n_all, dtype=bool)

        if bright_table is not None:
            obs_helio_list = [obs_meta[i]['obs_helio'] for i in obs_indices]
            bt_valid, bt_eph = bright_table.get_eph(
                all_ast_arr, t_tts, obs_helio_list, obscode, obscodes)
            if bt_valid.any():
                eph_batch[bt_valid] = bt_eph[bt_valid]
                pyoorb_mask[bt_valid] = False
                log.debug('Phase 2 worker %s: %d/%d served from bright table',
                          obscode, int(bt_valid.sum()), n_all)

        pyoorb_idx = np.where(pyoorb_mask)[0]
        if len(pyoorb_idx) > 0:
            sub_p  = sub[pyoorb_idx]
            orbits = build_oorb_orbits_kep(
                sub_p['a'], sub_p['e'],
                sub_p['i']     * _DEG2RAD,
                sub_p['Omega'] * _DEG2RAD,
                sub_p['omega'] * _DEG2RAD,
                sub_p['M']     * _DEG2RAD,
                sub_p['epoch'], sub_p['H'], sub_p['G'],
            )
            eph_p = oorb_ephemeris_multi_epoch_split(
                orbits, t_tts, obscode,
                a_arr=sub_p['a'], e_arr=sub_p['e'],
                dynmodel_nea=dynmodel, obscodes=obscodes)
            eph_batch[pyoorb_idx] = eph_p

        for k, obs_i in enumerate(obs_indices):
            my_cands = obs_meta[obs_i]['ast_cands']
            if len(my_cands) == 0:
                continue
            pos = np.searchsorted(all_ast_arr, my_cands)
            eph_k = eph_batch[pos, k, :]
            obs = observations[obs_i]
            seps = ang_sep_deg(eph_k[:, 1], eph_k[:, 2], obs.ra_deg, obs.dec_deg)
            within = (seps <= search_rad_deg) & (eph_k[:, 9] <= mag_limit)
            for j in np.where(within)[0]:
                idx = my_cands[j]
                obj = asteroids[idx]
                per_obs[obs_i].append(Match(
                    name=obj['desig'],
                    packed=obj['packed'],
                    obj_type='minor_planet',
                    ra_deg=float(eph_k[j, 1]),
                    dec_deg=float(eph_k[j, 2]),
                    sep_arcsec=float(seps[j]) * 3600.0,
                    ra_rate=_rate_to_arcsec_per_hr(float(eph_k[j, 3])),
                    dec_rate=_rate_to_arcsec_per_hr(float(eph_k[j, 4])),
                    r_helio=float(eph_k[j, 7]),
                    delta=float(eph_k[j, 8]),
                    vmag=float(eph_k[j, 9]),
                    phase_deg=float(eph_k[j, 5]),
                    orbit_quality=str(obj['U']).strip(),
                ))

    # ---- Comets ----
    all_com = sorted(set().union(
        *[set(obs_meta[i]['comet_cands'].tolist()) for i in obs_indices]))
    if all_com:
        all_com_arr = np.array(all_com, dtype=np.intp)
        sub_c = comets[all_com_arr]
        orbits_c = build_oorb_orbits_com(
            sub_c['q'], sub_c['e'],
            sub_c['i']     * _DEG2RAD,
            sub_c['Omega'] * _DEG2RAD,
            sub_c['omega'] * _DEG2RAD,
            sub_c['Tp'], sub_c['H'], sub_c['G'],
        )
        t_tts = [obs_meta[i]['t_tt'] for i in obs_indices]
        eph_c_batch = oorb_ephemeris_multi_epoch(
            orbits_c, t_tts, obscode, dynmodel, obscodes=obscodes)

        for k, obs_i in enumerate(obs_indices):
            my_cands_c = obs_meta[obs_i]['comet_cands']
            if len(my_cands_c) == 0:
                continue
            pos = np.searchsorted(all_com_arr, my_cands_c)
            eph_ck = eph_c_batch[pos, k, :]
            obs = observations[obs_i]
            seps = ang_sep_deg(eph_ck[:, 1], eph_ck[:, 2], obs.ra_deg, obs.dec_deg)
            within = (seps <= search_rad_deg) & (eph_ck[:, 9] <= mag_limit)
            for j in np.where(within)[0]:
                idx = my_cands_c[j]
                obj = comets[idx]
                per_obs[obs_i].append(Match(
                    name=obj['desig'],
                    packed=obj['packed'],
                    obj_type='comet',
                    ra_deg=float(eph_ck[j, 1]),
                    dec_deg=float(eph_ck[j, 2]),
                    sep_arcsec=float(seps[j]) * 3600.0,
                    ra_rate=_rate_to_arcsec_per_hr(float(eph_ck[j, 3])),
                    dec_rate=_rate_to_arcsec_per_hr(float(eph_ck[j, 4])),
                    r_helio=float(eph_ck[j, 7]),
                    delta=float(eph_ck[j, 8]),
                    vmag=float(eph_ck[j, 9]),
                    phase_deg=float(eph_ck[j, 5]),
                    orbit_quality=str(obj['U']).strip(),
                ))

    return per_obs


# ---------------------------------------------------------------------------
# Main check function
# ---------------------------------------------------------------------------

def check_observations(
    observations: List[Observation],
    asteroids: np.ndarray,
    comets: np.ndarray,
    obscodes: dict,
    search_radius_arcmin: float = 30.0,
    mag_limit: float = 25.0,
    prefilter_factor: float = 3.0,
    dynmodel: str = 'N',
    check_sats: bool = True,
    sky_index=None,
    n_workers: int = 1,
    reepoch_threshold_days: float = 90.0,
    mpcat_index=None,
    fo_refit_q_threshold: float = 1.1,
    fo_epoch_window_days: float = 5.0,
    fo_progress: bool = False,
    asteroid_soa: Optional[AsteroidSOA] = None,
    bright_table=None,
) -> List[CheckResult]:
    """
    Check each observation against the asteroid/comet catalogs and satellites.

    Parameters
    ----------
    observations           : parsed input observations
    asteroids              : structured array from mpcorb.load_mpcorb()
    comets                 : structured array from mpcorb.load_comets()
    obscodes               : dict from mpcorb.load_obscodes()
    search_radius_arcmin   : maximum angular separation for a match
    mag_limit              : faint limit for reported matches (approximate V)
    prefilter_factor       : expand radius by this factor for Keplerian pre-filter
    dynmodel               : pyoorb dynamics ('2'=two-body fast, 'N'=N-body precise)
    check_sats             : whether to check planetary satellites
    sky_index              : optional MultiSkyIndex (or SkyIndex) for fast candidate
                             lookup; if None the full Keplerian prefilter is used
    n_workers              : number of parallel worker processes for Phase 1
                             (uses fork; >1 effective only when len(observations)>1)
    reepoch_threshold_days : if the catalog element epoch differs from the median
                             observation epoch by more than this many days, re-epoch
                             the high-e (e>0.5) asteroid subset via pyoorb N-body
                             before Phase 1.  Set to 0 to disable.  Default: 90.
    mpcat_index            : optional MPCATIndex; if provided, all objects with
                             perihelion q < fo_refit_q_threshold are orbit-refit
                             via fo before Phase 1, ensuring element epochs are
                             near the observation epoch regardless of close encounters.
    fo_refit_q_threshold   : perihelion distance threshold for fo refit (AU).
                             All objects with q < this value are refit.  Default: 1.1.
    fo_epoch_window_days   : cache-bin width in days for fo fits (default: 5).
    fo_progress            : if True, log fo batch progress every 100 objects.
    """
    search_rad_deg = search_radius_arcmin / 60.0
    prefilter_deg  = search_rad_deg * prefilter_factor

    results = [CheckResult(obs=obs) for obs in observations]

    # Load SPICE base kernels early so SPICE-based position lookups are fast
    # for get_earth_helio / get_planet_helio throughout the pipeline.
    try:
        from .satellites import _load_base_kernels
        _load_base_kernels()
    except Exception:
        pass

    # -----------------------------------------------------------------------
    # Pre-Phase 1a: Re-epoch high-e (e>0.5) asteroids via pyoorb N-body.
    # Fast vectorized batch: reduces Phase 1 position errors from O(20') to
    # < 1' for most close-approach objects by propagating to the observation
    # epoch.  Skipped when mpcat_index is provided (fo refit in Pre-Phase 1b
    # is more accurate and supercedes this for q < 1.1 AU objects).
    # -----------------------------------------------------------------------
    if (reepoch_threshold_days > 0 and len(asteroids) > 0
            and len(observations) > 0):
        t_tts_all = _utc_to_tt_mjd_vec(obs.epoch_mjd for obs in observations).tolist()
        obs_epoch_tt  = float(np.median(t_tts_all))
        catalog_epoch = float(np.median(asteroids['epoch']))
        epoch_gap     = abs(obs_epoch_tt - catalog_epoch)
        if epoch_gap > reepoch_threshold_days:
            log.info(
                'Catalog epoch (MJD %.1f) is %.0f days from observations '
                '(MJD %.1f); re-epoching high-e asteroids via %s',
                catalog_epoch, epoch_gap, obs_epoch_tt, dynmodel,
            )
            from .config import CACHE_DIR
            asteroids = reepoch_high_e_asteroids(
                asteroids, obs_epoch_tt, dynmodel=dynmodel,
                cache_dir=CACHE_DIR)

    # -----------------------------------------------------------------------
    # Pre-Phase 1b: fo orbit refit for all q < fo_refit_q_threshold objects.
    #
    # For any asteroid with perihelion inside ~1.1 AU, MPCORB elements may be
    # from a catalog epoch on the far side of a close Earth encounter.
    # Integrating through such an encounter accumulates errors of tens of
    # degrees.  By refitting from scratch using MPCAT observations up to the
    # observation date (max_date_mjd = obs epoch bin), fo produces elements at
    # an epoch just before the observation — pyoorb then only needs to
    # integrate a few days, eliminating close-encounter integration errors
    # entirely.  Fit results are cached per (packed, epoch_bin); cache hits
    # are ~1 ms.
    # -----------------------------------------------------------------------
    if (mpcat_index is not None
            and fo_refit_q_threshold > 0.0
            and len(asteroids) > 0
            and len(observations) > 0):
        from .orbitfit import select_neo_packed, refit_neo_batch, apply_fo_refits
        from .config import FO_CACHE_DIR

        obs_epoch_utc = float(np.median([obs.epoch_mjd for obs in observations]))
        neo_packed = select_neo_packed(asteroids, q_threshold=fo_refit_q_threshold)

        if neo_packed:
            log.info(
                'Pre-Phase 1 fo refit: %d objects with q < %.2f AU '
                'at obs epoch MJD %.1f (epoch window %.0f days)',
                len(neo_packed), fo_refit_q_threshold,
                obs_epoch_utc, fo_epoch_window_days,
            )
            refits = refit_neo_batch(
                neo_packed,
                mpcat_index,
                cache_dir=FO_CACHE_DIR,
                obs_epoch_mjd=obs_epoch_utc,
                epoch_window_days=fo_epoch_window_days,
                progress=fo_progress,
                n_workers=n_workers,
            )
            if refits:
                asteroids = apply_fo_refits(asteroids, refits)

    # -----------------------------------------------------------------------
    # Phase 1: Per-observation Keplerian pre-filter — parallelised over
    # observations when n_workers > 1 (fork-based, Linux only).
    # Each observation is independent; heavy arrays are inherited via fork
    # with copy-on-write so nothing is pickled.
    # -----------------------------------------------------------------------
    if n_workers > 1 and len(observations) > 1:
        import multiprocessing as mp
        global _PHASE1_STATE
        nw = min(n_workers, len(observations))
        _PHASE1_STATE = {
            'asteroids':    asteroids,
            'comets':       comets,
            'obscodes':     obscodes,
            'sky_index':    sky_index,
            'mag_limit':    mag_limit,
            'prefilter_deg': prefilter_deg,
            'asteroid_soa': asteroid_soa,
        }
        ctx = mp.get_context('fork')
        with ctx.Pool(nw) as pool:
            tagged = pool.map(_phase1_worker, list(enumerate(observations)))
        obs_meta = [meta for _, meta in sorted(tagged, key=lambda x: x[0])]
    else:
        obs_meta = [
            _phase1_one_obs(obs, asteroids, comets, obscodes, sky_index,
                            mag_limit, prefilter_deg, asteroid_soa=asteroid_soa)
            for obs in observations
        ]

    # -----------------------------------------------------------------------
    # Phase 2: Batched pyoorb — one call per obscode group.
    # For each group: take union of candidates, run pyoorb at all epochs,
    # then apply per-observation angular + magnitude cuts.
    # Groups are independent and run in parallel when n_workers > 1.
    # -----------------------------------------------------------------------
    by_obscode = defaultdict(list)
    for i, obs in enumerate(observations):
        by_obscode[obs.obscode].append(i)

    if n_workers > 1 and len(by_obscode) > 1:
        import multiprocessing as mp
        global _PHASE2_STATE
        _PHASE2_STATE = {
            'asteroids':     asteroids,
            'comets':        comets,
            'observations':  observations,
            'obs_meta':      obs_meta,
            'dynmodel':      dynmodel,
            'search_rad_deg': search_rad_deg,
            'mag_limit':     mag_limit,
            'obscodes':      obscodes,
            'bright_table':  bright_table,
        }
        nw2 = min(n_workers, len(by_obscode))
        ctx2 = mp.get_context('fork')
        with ctx2.Pool(nw2) as pool2:
            group_results = pool2.map(_phase2_worker, list(by_obscode.items()))
        for per_obs in group_results:
            for obs_i, matches in per_obs.items():
                results[obs_i].matches.extend(matches)
    else:
        for obscode, obs_indices in by_obscode.items():
            # --- Asteroids ---
            all_ast = sorted(set().union(
                *[set(obs_meta[i]['ast_cands'].tolist()) for i in obs_indices]))
            if all_ast:
                all_ast = np.array(all_ast, dtype=np.intp)
                sub = asteroids[all_ast]
                n_all = len(sub)
                t_tts = [obs_meta[i]['t_tt'] for i in obs_indices]
                n_ep  = len(t_tts)

                eph_batch   = np.full((n_all, n_ep, 11), np.nan)
                eph_batch[:, :, 9] = 99.0
                pyoorb_mask = np.ones(n_all, dtype=bool)

                if bright_table is not None:
                    obs_helio_list = [obs_meta[i]['obs_helio'] for i in obs_indices]
                    bt_valid, bt_eph = bright_table.get_eph(
                        all_ast, t_tts, obs_helio_list, obscode, obscodes)
                    if bt_valid.any():
                        eph_batch[bt_valid] = bt_eph[bt_valid]
                        pyoorb_mask[bt_valid] = False

                pyoorb_idx = np.where(pyoorb_mask)[0]
                if len(pyoorb_idx) > 0:
                    sub_p  = sub[pyoorb_idx]
                    orbits = build_oorb_orbits_kep(
                        sub_p['a'], sub_p['e'],
                        sub_p['i']     * _DEG2RAD,
                        sub_p['Omega'] * _DEG2RAD,
                        sub_p['omega'] * _DEG2RAD,
                        sub_p['M']     * _DEG2RAD,
                        sub_p['epoch'], sub_p['H'], sub_p['G'],
                    )
                    eph_p = oorb_ephemeris_multi_epoch_split(
                        orbits, t_tts, obscode,
                        a_arr=sub_p['a'], e_arr=sub_p['e'],
                        dynmodel_nea=dynmodel, obscodes=obscodes)
                    eph_batch[pyoorb_idx] = eph_p

                for k, obs_i in enumerate(obs_indices):
                    my_cands = obs_meta[obs_i]['ast_cands']
                    if len(my_cands) == 0:
                        continue
                    pos_in_union = np.searchsorted(all_ast, my_cands)
                    eph_k = eph_batch[pos_in_union, k, :]

                    obs = observations[obs_i]
                    seps = ang_sep_deg(eph_k[:, 1], eph_k[:, 2], obs.ra_deg, obs.dec_deg)
                    within = (seps <= search_rad_deg) & (eph_k[:, 9] <= mag_limit)

                    for j in np.where(within)[0]:
                        idx = my_cands[j]
                        obj = asteroids[idx]
                        results[obs_i].matches.append(Match(
                            name=obj['desig'],
                            packed=obj['packed'],
                            obj_type='minor_planet',
                            ra_deg=float(eph_k[j, 1]),
                            dec_deg=float(eph_k[j, 2]),
                            sep_arcsec=float(seps[j]) * 3600.0,
                            ra_rate=_rate_to_arcsec_per_hr(float(eph_k[j, 3])),
                            dec_rate=_rate_to_arcsec_per_hr(float(eph_k[j, 4])),
                            r_helio=float(eph_k[j, 7]),
                            delta=float(eph_k[j, 8]),
                            vmag=float(eph_k[j, 9]),
                            phase_deg=float(eph_k[j, 5]),
                            orbit_quality=str(obj['U']).strip(),
                        ))

            # --- Comets ---
            all_com = sorted(set().union(
                *[set(obs_meta[i]['comet_cands'].tolist()) for i in obs_indices]))
            if all_com:
                all_com = np.array(all_com, dtype=np.intp)
                sub_c = comets[all_com]
                orbits_c = build_oorb_orbits_com(
                    sub_c['q'], sub_c['e'],
                    sub_c['i']     * _DEG2RAD,
                    sub_c['Omega'] * _DEG2RAD,
                    sub_c['omega'] * _DEG2RAD,
                    sub_c['Tp'], sub_c['H'], sub_c['G'],
                )
                t_tts = [obs_meta[i]['t_tt'] for i in obs_indices]
                eph_c_batch = oorb_ephemeris_multi_epoch(orbits_c, t_tts, obscode, dynmodel,
                                                         obscodes=obscodes)

                for k, obs_i in enumerate(obs_indices):
                    my_cands_c = obs_meta[obs_i]['comet_cands']
                    if len(my_cands_c) == 0:
                        continue
                    pos_in_union = np.searchsorted(all_com, my_cands_c)
                    eph_ck = eph_c_batch[pos_in_union, k, :]

                    obs = observations[obs_i]
                    seps = ang_sep_deg(eph_ck[:, 1], eph_ck[:, 2], obs.ra_deg, obs.dec_deg)
                    within = (seps <= search_rad_deg) & (eph_ck[:, 9] <= mag_limit)

                    for j in np.where(within)[0]:
                        idx = my_cands_c[j]
                        obj = comets[idx]
                        results[obs_i].matches.append(Match(
                            name=obj['desig'],
                            packed=obj['packed'],
                            obj_type='comet',
                            ra_deg=float(eph_ck[j, 1]),
                            dec_deg=float(eph_ck[j, 2]),
                            sep_arcsec=float(seps[j]) * 3600.0,
                            ra_rate=_rate_to_arcsec_per_hr(float(eph_ck[j, 3])),
                            dec_rate=_rate_to_arcsec_per_hr(float(eph_ck[j, 4])),
                            r_helio=float(eph_ck[j, 7]),
                            delta=float(eph_ck[j, 8]),
                            vmag=float(eph_ck[j, 9]),
                            phase_deg=float(eph_ck[j, 5]),
                            orbit_quality=str(obj['U']).strip(),
                        ))

    # -----------------------------------------------------------------------
    # Phase 3: Planetary satellite check (per-observation, no batching needed)
    # -----------------------------------------------------------------------
    if check_sats:
        # Build packed→index lookup for dwarf planet primaries once (avoids
        # O(N_catalog) string scan per satellite per observation).
        from .config import DWARF_PLANET_SATELLITES as _DP_SATS
        dp_primary_idx: dict = {}
        for _sat in _DP_SATS:
            _pk = _sat['primary_packed']
            if _pk not in dp_primary_idx:
                _hits = np.where(asteroids['packed'] == _pk)[0]
                if len(_hits):
                    dp_primary_idx[_pk] = int(_hits[0])

        for obs_i, obs in enumerate(observations):
            sat_matches = check_satellites(
                obs.ra_deg, obs.dec_deg, obs.epoch_mjd, obs.obscode,
                obscodes, search_rad_deg, mag_limit)
            for sm in sat_matches:
                results[obs_i].matches.append(Match(
                    name=sm['name'],
                    packed=str(sm['naif_id']),
                    obj_type='satellite',
                    ra_deg=sm['ra_deg'],
                    dec_deg=sm['dec_deg'],
                    sep_arcsec=sm['sep_deg'] * 3600.0,
                    ra_rate=0.0,
                    dec_rate=0.0,
                    r_helio=0.0,
                    delta=sm['delta_au'],
                    vmag=sm['vmag'],
                    phase_deg=0.0,
                ))

            dp_matches = check_dwarf_planet_satellites(
                obs.ra_deg, obs.dec_deg, obs.epoch_mjd, obs.obscode,
                obscodes, asteroids, dp_primary_idx, search_rad_deg, mag_limit)
            for sm in dp_matches:
                results[obs_i].matches.append(Match(
                    name=sm['name'],
                    packed=sm.get('primary_packed', ''),
                    obj_type='satellite',
                    ra_deg=sm['ra_deg'],
                    dec_deg=sm['dec_deg'],
                    sep_arcsec=sm['sep_deg'] * 3600.0,
                    ra_rate=0.0,
                    dec_rate=0.0,
                    r_helio=0.0,
                    delta=sm['delta_au'],
                    vmag=sm['vmag'],
                    phase_deg=0.0,
                ))

    # Sort each result by separation
    for res in results:
        res.matches.sort(key=lambda m: m.sep_arcsec)

    return results


# ---------------------------------------------------------------------------
# Tracklet identification
# ---------------------------------------------------------------------------

def _find_catalog_match_by_elements(
    fo_arr: np.ndarray,
    asteroids: np.ndarray,
    threshold: float = 0.05,
) -> tuple:
    """
    Find the best-matching catalog object for a fo-fitted orbit using (a, e, i)
    orbital element similarity.

    The similarity score is:
        score = |Δa| + |Δe| + |Δi°| / 10

    This is insensitive to epoch differences because a, e, i change very
    little between observations and catalog epochs (even across close
    approaches at 0.056 AU, Δa < 0.01 AU).

    Parameters
    ----------
    fo_arr    : 1-element ASTEROID_DTYPE array from refit_from_obs()
    asteroids : MPCORB structured array (may be H-limited)
    threshold : maximum score to accept a match (default: 0.05)

    Returns
    -------
    (name, packed, score) if a match is found; (None, None, None) otherwise.
    """
    a_fo = float(fo_arr['a'][0])
    e_fo = float(fo_arr['e'][0])
    i_fo = float(fo_arr['i'][0])

    da = np.abs(asteroids['a'] - a_fo)
    de = np.abs(asteroids['e'] - e_fo)
    di = np.abs(asteroids['i'] - i_fo)
    score = da + de + di / 10.0

    best_idx   = int(np.argmin(score))
    best_score = float(score[best_idx])

    if best_score < threshold:
        name   = str(asteroids['desig'][best_idx]).strip()
        packed = str(asteroids['packed'][best_idx]).strip()
        return name, packed, best_score

    # Not found in the (possibly H-limited) catalog.  If the fitted orbit has
    # q < 1.3 AU (potential Apollo/Aten/Amor), fall back to searching the full
    # MPCORB catalog which includes faint close-approach NEOs (H > H_cut).
    # These are excluded from the runtime catalog for Phase 1/2 efficiency, but
    # must be included in element-similarity identification.
    q_fo = a_fo * (1.0 - e_fo)
    if q_fo < 1.3:
        try:
            from .mpcorb import load_mpcorb
            full_ast = load_mpcorb()   # uses cache — fast after first load
            if len(full_ast) > len(asteroids):
                da2 = np.abs(full_ast['a'] - a_fo)
                de2 = np.abs(full_ast['e'] - e_fo)
                di2 = np.abs(full_ast['i'] - i_fo)
                score2 = da2 + de2 + di2 / 10.0
                best2  = int(np.argmin(score2))
                s2     = float(score2[best2])
                if s2 < threshold:
                    return (str(full_ast['desig'][best2]).strip(),
                            str(full_ast['packed'][best2]).strip(),
                            s2)
        except Exception:
            pass

    return None, None, None


def identify_tracklet(
    observations: List[Observation],
    results: List[CheckResult],
    asteroids: np.ndarray,
    comets: np.ndarray,
    obscodes: dict,
    dynmodel: str = 'N',
    residual_threshold_arcsec: float = 2.0,
    min_obs: Optional[int] = None,
    fo_fit: bool = False,
    fo_timeout_sec: int = 60,
    satellite_threshold_arcsec: float = 120.0,
) -> List[Identification]:
    """
    Identify which catalog candidates explain all input observations as a tracklet.

    For each candidate that appeared in Phase 2 results for >= min_obs of the
    input observations, this function computes precise pyoorb predicted positions
    at EVERY input observation epoch and reports per-observation O-C residuals.
    Any candidate with all residuals <= residual_threshold_arcsec is returned as
    an Identification, sorted by RMS residual (best first).

    When fo_fit=True and 'fo' (Project Pluto's find_orb) is available, an
    independent orbit is fitted to the input observations themselves.  The
    fitted orbit's per-observation residuals (O-C from the fo solution) are
    added as a separate method='orbit_fit' Identification entry alongside the
    best ephemeris match.  This brute-force orbit check independently confirms
    that the observations are self-consistent with a single physical orbit and
    quantifies the fit quality.

    Parameters
    ----------
    observations              : input observations (same list as check_observations)
    results                   : Phase 2 output from check_observations()
    asteroids                 : structured array from load_mpcorb()
    comets                    : structured array from load_comets()
    obscodes                  : dict from load_obscodes()
    dynmodel                  : pyoorb dynamics ('N' = N-body, '2' = two-body)
    residual_threshold_arcsec : max O-C (arcsec) for any single observation.
                                Candidates with any residual above this are
                                excluded from the identification list. Default: 2".
    min_obs                   : minimum number of observations a candidate must
                                appear in to be tested.  Default (None): require
                                the candidate to appear in ALL input observations.
    fo_fit                    : if True, additionally run find_orb on the input
                                observations as an independent orbit determination.
                                Requires 'fo' to be on PATH.
    fo_timeout_sec            : find_orb timeout in seconds (default: 60)

    Returns
    -------
    List of Identification objects sorted by rms_arcsec ascending (best first).
    Returns an empty list if len(observations) < 2, or no candidates satisfy
    the threshold.
    """
    if len(observations) < 2:
        return []

    n_obs     = len(observations)
    threshold = n_obs if min_obs is None else max(1, min_obs)

    # ---- Count how many input observations each Phase 2 candidate appears in ----
    count:      Counter = Counter()
    best_match: dict    = {}
    # Also count satellite (Phase 3 / SPICE) matches by name across observations
    sat_count:   Counter = Counter()
    sat_obs_map: dict    = {}   # sat_name -> {obs_i: Match}
    for obs_i, res in enumerate(results):
        for m in res.matches:
            if m.obj_type == 'satellite':
                sat_count[m.name] += 1
                if m.name not in sat_obs_map:
                    sat_obs_map[m.name] = {}
                prior = sat_obs_map[m.name].get(obs_i)
                if prior is None or m.sep_arcsec < prior.sep_arcsec:
                    sat_obs_map[m.name][obs_i] = m
                continue
            key = (m.packed, m.obj_type)
            count[key] += 1
            if key not in best_match or m.sep_arcsec < best_match[key].sep_arcsec:
                best_match[key] = m

    qualifying = [(pk, ot) for (pk, ot), cnt in count.items() if cnt >= threshold]
    if not qualifying:
        log.debug('identify_tracklet: no candidates appeared in >= %d/%d observations',
                  threshold, n_obs)
        if not fo_fit:
            return []
        # fo_fit=True: still try a blind orbit fit even without Phase 2 candidates

    # ---- Pre-compute TT epochs; group by obscode for batched pyoorb calls ----
    t_tts      = _utc_to_tt_mjd_vec(obs.epoch_mjd for obs in observations).tolist()
    by_obscode: dict = defaultdict(list)
    for i, obs in enumerate(observations):
        by_obscode[obs.obscode].append(i)

    identifications: List[Identification] = []

    # ---- Ephemeris method: batch pyoorb across ALL qualifying candidates ----
    #
    # Phase 2 calls pyoorb once per obscode group with a BATCH of orbits; calling
    # it one-by-one for each candidate causes single-orbit N-body failures for
    # close-approach objects (they work fine in a batch but fail alone).
    # Replicate the Phase 2 batching pattern here.

    # 1. Build orbit arrays for all qualifying candidates
    cand_orbits:   List[np.ndarray] = []   # parallel to valid_qualifying
    valid_qualifying: List[tuple]   = []   # (packed, obj_type)
    for packed, obj_type in qualifying:
        try:
            if obj_type == 'minor_planet':
                idx = np.where(asteroids['packed'] == packed)[0]
                if not len(idx):
                    continue
                sub = asteroids[idx[:1]]
                orb = build_oorb_orbits_kep(
                    sub['a'], sub['e'],
                    sub['i']     * _DEG2RAD, sub['Omega'] * _DEG2RAD,
                    sub['omega'] * _DEG2RAD, sub['M']     * _DEG2RAD,
                    sub['epoch'], sub['H'], sub['G'],
                )
            elif obj_type == 'comet':
                idx = np.where(comets['packed'] == packed)[0]
                if not len(idx):
                    continue
                sub = comets[idx[:1]]
                orb = build_oorb_orbits_com(
                    sub['q'], sub['e'],
                    sub['i']     * _DEG2RAD, sub['Omega'] * _DEG2RAD,
                    sub['omega'] * _DEG2RAD, sub['Tp'],
                    sub['H'], sub['G'],
                )
            else:
                continue
            cand_orbits.append(orb)
            valid_qualifying.append((packed, obj_type))
        except Exception as exc:
            log.debug('identify_tracklet: orbit build failed for %s: %s', packed, exc)

    # Force pyoorb re-init unconditionally: after heavy Phase 2 use the N-body
    # integrator accumulates internal state (step-size caches) that causes NaN
    # for close-approach orbits.  Must happen before ANY pyoorb call below,
    # including the fo_fit single-orbit call when valid_qualifying is empty.
    _init_oorb(force=True)

    if not valid_qualifying:
        pass   # fall through to fo_fit block
    else:
        # 2. Stack all orbits with consecutive IDs (required by pyoorb)
        n_cands = len(valid_qualifying)
        batch_orbits = np.vstack(cand_orbits)
        batch_orbits[:, 0] = np.arange(1, n_cands + 1)

        # 3. Per-candidate residual arrays
        all_residuals = [[np.inf] * n_obs for _ in range(n_cands)]

        # 4. One pyoorb batch call per obscode group
        for obscode, obs_indices in by_obscode.items():
            epoch_tts = [t_tts[i] for i in obs_indices]
            try:
                eph = oorb_ephemeris_multi_epoch(
                    batch_orbits, epoch_tts, obscode, dynmodel, obscodes=obscodes)
                # eph shape: [n_cands, n_epochs, 11]
                for ci in range(n_cands):
                    for k, obs_i in enumerate(obs_indices):
                        obs = observations[obs_i]
                        ra_pred  = float(eph[ci, k, 1])
                        dec_pred = float(eph[ci, k, 2])
                        if np.isfinite(ra_pred) and np.isfinite(dec_pred):
                            sep = ang_sep_deg(ra_pred, dec_pred,
                                              obs.ra_deg, obs.dec_deg) * 3600.0
                            all_residuals[ci][obs_i] = sep
            except Exception as exc:
                log.debug('identify_tracklet: batch ephemeris failed for obscode %s: %s',
                          obscode, exc)

        # 5. Evaluate each candidate
        for ci, (packed, obj_type) in enumerate(valid_qualifying):
            obs_residuals = all_residuals[ci]
            finite = [r for r in obs_residuals if np.isfinite(r)]
            if not finite:
                continue
            rms = float(np.sqrt(np.mean(np.array(finite) ** 2)))
            # Use RMS rather than per-observation max: catalog-orbit errors
            # for close-approach NEOs grow systematically across a multi-night
            # arc, so individual far-end observations often exceed the threshold
            # even for genuine identifications.  Guard against extreme single-obs
            # outliers with a 4× ceiling.
            max_resid = float(max(finite))
            if rms <= residual_threshold_arcsec and max_resid <= residual_threshold_arcsec * 4:
                identifications.append(Identification(
                    match=best_match[(packed, obj_type)],
                    residuals=obs_residuals,
                    rms_arcsec=rms,
                    n_obs=n_obs,
                    method='ephemeris',
                ))
            else:
                log.debug(
                    'identify_tracklet: %s (%s) — RMS=%.1f" max=%.1f" '
                    '(threshold=%.1f"); not reported',
                    packed, obj_type, rms, max_resid, residual_threshold_arcsec,
                )

    identifications.sort(key=lambda x: x.rms_arcsec)

    # ---- Satellite identification via SPICE positional offsets ----
    #
    # Unlike minor planets/comets, SPICE positions for planetary satellites are
    # computed directly (no pyoorb needed).  The per-observation separations
    # stored in Match.sep_arcsec ARE the residuals.  For irregular satellites
    # (e.g. Carme group) SPICE element uncertainty can be 10–100", so we use a
    # separate, more generous threshold.
    #
    # A satellite is identified when:
    #   1. It appears in >= threshold observations (within the search radius)
    #   2. Its RMS separation is below satellite_threshold_arcsec
    #   3. It is uniquely the closest satellite by a significant margin
    #
    sat_identifications: List[Identification] = []
    if satellite_threshold_arcsec > 0 and sat_count:
        # Find the minimum RMS among qualifying satellites to check uniqueness
        qualifying_sats = [
            name for name, cnt in sat_count.items() if cnt >= threshold
        ]
        sat_rms_map = {}
        for sat_name in qualifying_sats:
            obs_by_i = sat_obs_map[sat_name]
            residuals = [obs_by_i[i].sep_arcsec if i in obs_by_i else np.inf
                         for i in range(n_obs)]
            finite = [r for r in residuals if np.isfinite(r)]
            if not finite:
                continue
            rms = float(np.sqrt(np.mean(np.array(finite) ** 2)))
            sat_rms_map[sat_name] = (rms, residuals)

        if sat_rms_map:
            sorted_sats = sorted(sat_rms_map.items(), key=lambda kv: kv[1][0])
            best_sat_name, (best_rms, best_residuals) = sorted_sats[0]
            second_rms = sorted_sats[1][1][0] if len(sorted_sats) > 1 else np.inf

            # Accept if within threshold AND at least 3× closer than next satellite
            if best_rms <= satellite_threshold_arcsec and best_rms * 3 < second_rms:
                obs_by_i = sat_obs_map[best_sat_name]
                best_m = min(obs_by_i.values(), key=lambda m: m.sep_arcsec)
                sat_identifications.append(Identification(
                    match=best_m,
                    residuals=best_residuals,
                    rms_arcsec=best_rms,
                    n_obs=n_obs,
                    method='satellite',
                ))
                log.info(
                    'identify_tracklet: satellite match %s  RMS=%.1f"  '
                    '(next closest=%.1f")',
                    best_sat_name, best_rms, second_rms,
                )

    # Satellite identifications take priority: prepend before orbit-based results
    identifications = sat_identifications + identifications

    # ---- fo orbit fit (optional): brute-force orbit from input observations ----
    if fo_fit:
        try:
            from .orbitfit import refit_from_obs, _fo_available
            if _fo_available():
                log.info('identify_tracklet: running fo orbit fit on %d observations', n_obs)
                result = refit_from_obs(observations, timeout_sec=fo_timeout_sec)
                if result is not None:
                    fo_orbit, fo_quality = result
                    # fo's own internal RMS is reliable even for close-approach objects
                    # where pyoorb re-evaluation diverges.
                    fo_rms_internal = fo_quality.get('rms_arcsec', np.nan)
                    fo_n_obs_fit    = fo_quality.get('n_obs', None)
                    fo_earth_moid   = fo_quality.get('earth_moid_au', None)

                    # Build pyoorb orbit from fo-fitted elements for per-obs O-C display
                    fo_pyoorb = build_oorb_orbits_kep(
                        fo_orbit['a'], fo_orbit['e'],
                        fo_orbit['i']     * _DEG2RAD, fo_orbit['Omega'] * _DEG2RAD,
                        fo_orbit['omega'] * _DEG2RAD, fo_orbit['M']     * _DEG2RAD,
                        fo_orbit['epoch'], fo_orbit['H'], fo_orbit['G'],
                    )
                    # Re-init pyoorb before fo single-orbit call (may have been
                    # corrupted by ephemeris-method batch above)
                    _init_oorb(force=True)
                    # Compute per-observation O-C from fo orbit via pyoorb
                    fo_residuals: List[float] = [np.inf] * n_obs
                    try:
                        for obscode, obs_indices in by_obscode.items():
                            epoch_tts = [t_tts[i] for i in obs_indices]
                            eph = oorb_ephemeris_multi_epoch(
                                fo_pyoorb, epoch_tts, obscode, dynmodel,
                                obscodes=obscodes)
                            for k, obs_i in enumerate(obs_indices):
                                obs = observations[obs_i]
                                ra_p  = float(eph[0, k, 1])
                                dec_p = float(eph[0, k, 2])
                                if np.isfinite(ra_p) and np.isfinite(dec_p):
                                    sep = ang_sep_deg(ra_p, dec_p,
                                                      obs.ra_deg, obs.dec_deg) * 3600.0
                                    fo_residuals[obs_i] = sep
                    except Exception as exc:
                        log.debug('identify_tracklet: fo residual computation failed: %s', exc)

                    # Use fo's own RMS as the primary quality metric.
                    # pyoorb re-evaluation can be unreliable for close-approach objects
                    # (e.g. 119" pyoorb vs 0.4" fo-internal for a genuine new NEO).
                    # Fall back to pyoorb RMS only if fo's value is not available.
                    finite_resids = [r for r in fo_residuals if np.isfinite(r)]
                    fo_rms_pyoorb = (float(np.sqrt(np.mean(np.array(finite_resids) ** 2)))
                                     if finite_resids else np.nan)
                    primary_rms = (fo_rms_internal
                                   if np.isfinite(fo_rms_internal) else fo_rms_pyoorb)

                    if np.isfinite(primary_rms):
                        # Anchor: prefer best ephemeris match; fall back to
                        # Phase 2 candidate; or None when fo-only (no Phase 2).
                        anchor = (identifications[0].match if identifications else
                                  (best_match[qualifying[0]] if qualifying else None))

                        # Catalog match by orbital element similarity
                        cat_name, cat_packed, cat_score = _find_catalog_match_by_elements(
                            fo_orbit, asteroids)

                        # If fo embedded a real designation in desig[] (not temp),
                        # prefer it over the element-similarity result — but only
                        # when the designation's catalog elements are consistent
                        # with the fitted orbit (score < threshold). fo sometimes
                        # mis-identifies close-approach objects, embedding names
                        # that belong to completely different orbits.
                        fo_desig = str(fo_orbit['desig'][0]).strip()
                        if fo_desig and fo_desig not in ('ZMPCK', ''):
                            idx = np.where(
                                (asteroids['desig'] == fo_desig) |
                                (asteroids['packed'] == fo_desig)
                            )[0]
                            if len(idx):
                                fo_desig_score = float(
                                    abs(asteroids['a'][idx[0]] - float(fo_orbit['a'][0]))
                                    + abs(asteroids['e'][idx[0]] - float(fo_orbit['e'][0]))
                                    + abs(asteroids['i'][idx[0]] - float(fo_orbit['i'][0])) / 10.0
                                )
                                if fo_desig_score < 0.05:
                                    cat_name   = str(asteroids['desig'][idx[0]]).strip()
                                    cat_packed = str(asteroids['packed'][idx[0]]).strip()
                                    cat_score  = fo_desig_score
                                # else: fo's designation doesn't match fitted elements;
                                # fall through and use element-similarity result
                            else:
                                cat_name  = fo_desig
                                cat_score = 0.0

                        # Validate catalog match: confirm the catalog orbit actually
                        # predicts the observed positions.  Element similarity alone
                        # is insufficient — a short-arc fit to a planetary satellite's
                        # apparent motion can yield heliocentric elements close to a
                        # real asteroid at the same sky distance.  The satellite's
                        # SPICE ephemeris is precise, but the heliocentric orbit fit
                        # is an artefact of projecting a bound-satellite trajectory
                        # onto a heliocentric conic.  Running the catalog orbit through
                        # pyoorb exposes the true ~100" positional mismatch.
                        if cat_name is not None and cat_packed:
                            _cat_idx = np.where(
                                (asteroids['packed'] == cat_packed) |
                                (asteroids['desig']  == cat_packed)
                            )[0]
                            if len(_cat_idx):
                                try:
                                    _sub = asteroids[_cat_idx[:1]]
                                    _cat_orb = build_oorb_orbits_kep(
                                        _sub['a'], _sub['e'],
                                        _sub['i']     * _DEG2RAD,
                                        _sub['Omega'] * _DEG2RAD,
                                        _sub['omega'] * _DEG2RAD,
                                        _sub['M']     * _DEG2RAD,
                                        _sub['epoch'], _sub['H'], _sub['G'],
                                    )
                                    _val_seps = []
                                    for _obc, _oidxs in by_obscode.items():
                                        _ettvs = [t_tts[_ii] for _ii in _oidxs]
                                        try:
                                            _eph_v = oorb_ephemeris_multi_epoch(
                                                _cat_orb, _ettvs, _obc,
                                                dynmodel, obscodes=obscodes)
                                            for _k, _oi in enumerate(_oidxs):
                                                _o = observations[_oi]
                                                _ra  = float(_eph_v[0, _k, 1])
                                                _dec = float(_eph_v[0, _k, 2])
                                                if np.isfinite(_ra) and np.isfinite(_dec):
                                                    _val_seps.append(
                                                        ang_sep_deg(_ra, _dec,
                                                                    _o.ra_deg,
                                                                    _o.dec_deg) * 3600.0
                                                    )
                                        except Exception:
                                            pass
                                    if _val_seps:
                                        _val_rms = float(np.sqrt(
                                            np.mean(np.array(_val_seps) ** 2)))
                                        _val_thr = max(10.0,
                                                       5.0 * residual_threshold_arcsec)
                                        if _val_rms > _val_thr:
                                            log.info(
                                                'identify_tracklet: catalog match %s '
                                                'REJECTED — catalog orbit RMS=%.1f" '
                                                'vs threshold=%.1f"; element similarity '
                                                'was a false positive (degenerate short-arc fit)',
                                                cat_name, _val_rms, _val_thr,
                                            )
                                            cat_name   = None
                                            cat_packed = None
                                            cat_score  = None
                                except Exception as _exc:
                                    log.debug(
                                        'identify_tracklet: catalog orbit validation '
                                        'failed for %s: %s', cat_name, _exc)

                        log.info(
                            'identify_tracklet: fo fit RMS=%.2f" (fo-internal)'
                            '  catalog match: %s (score=%.4f)',
                            primary_rms,
                            cat_name or 'none',
                            cat_score if cat_score is not None else float('nan'),
                        )

                        identifications.append(Identification(
                            match=anchor,
                            residuals=list(fo_residuals),
                            rms_arcsec=primary_rms,
                            n_obs=n_obs,
                            method='orbit_fit',
                            fo_elements=fo_orbit,
                            fo_catalog_name=cat_name,
                            fo_catalog_score=cat_score,
                            fo_rms_internal=fo_rms_internal,
                            fo_n_obs=fo_n_obs_fit,
                            fo_earth_moid_au=fo_earth_moid,
                        ))
        except Exception as exc:
            log.debug('identify_tracklet: fo fit failed: %s', exc)

    # Deduplicate: if a satellite identification and an orbit_fit identification
    # refer to the same object (same match name), keep only the satellite entry.
    # This avoids showing both [1] Carme [SPICE satellite] and [2] Carme [fo orbit fit]
    # when the fo fit's anchor was set to the satellite match.
    sat_names = {
        ident.match.name
        for ident in identifications
        if ident.method == 'satellite' and ident.match is not None
    }
    if sat_names:
        identifications = [
            ident for ident in identifications
            if not (
                ident.method == 'orbit_fit'
                and ident.match is not None
                and ident.match.name in sat_names
            )
        ]

    return identifications
