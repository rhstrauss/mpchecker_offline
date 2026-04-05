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
    oorb_ephemeris_multi_epoch, reepoch_high_e_asteroids,
    _init_oorb,
    _DEG2RAD, _RAD2DEG,
)
from .satellites import check_satellites, check_dwarf_planet_satellites
from .index import SkyIndex, MultiSkyIndex

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level state for fork-based multiprocessing workers (Optimization 4)
# Set by check_observations before Pool creation; inherited copy-on-write.
# ---------------------------------------------------------------------------
_PHASE1_STATE: dict = {}


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


def _rate_to_arcsec_per_hr(deg_per_day: float) -> float:
    return deg_per_day * 3600.0 / 24.0


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
) -> dict:
    """
    Run Phase 1 (Keplerian pre-filter) for a single observation.
    Returns a dict with t_tt, obs_helio, ast_cands, comet_cands.
    Safe to call in a forked worker process.
    """
    t_tt      = _utc_to_tt_mjd(obs.epoch_mjd)
    obs_helio = get_observer_helio(obs.epoch_mjd, obs.obscode, obscodes)
    h_cut     = _h_limit_from_vmag(mag_limit, obs_helio, obs.ra_deg, obs.dec_deg, obs.epoch_mjd)

    n_h_filtered  = int((asteroids['H'] <= h_cut).sum())
    ast_cands     = np.array([], dtype=np.intp)
    _INDEX_THRESHOLD = 400_000

    if (sky_index is not None and sky_index.is_fresh(obs.epoch_mjd)
            and n_h_filtered > _INDEX_THRESHOLD):
        idx_mask = sky_index.candidates(obs.ra_deg, obs.dec_deg, prefilter_deg,
                                        t_obs_mjd=obs.epoch_mjd)
        idx_mask &= (asteroids['H'] <= h_cut)
        broad_idx = np.where(idx_mask)[0]
        log.debug('Index cone: %d candidates for %s', len(broad_idx), obs.designation)
        if len(broad_idx) > 0:
            kep_mask  = _asteroid_prefilter(
                asteroids[broad_idx], obs_helio, t_tt,
                obs.ra_deg, obs.dec_deg, prefilter_deg)
            ast_cands = broad_idx[kep_mask]
    else:
        h_mask = asteroids['H'] <= h_cut
        log.debug('H≤%.1f cut: %d/%d asteroids for %s',
                  h_cut, n_h_filtered, len(asteroids), obs.designation)
        if n_h_filtered > 0:
            kep_mask          = _asteroid_prefilter(
                asteroids[h_mask], obs_helio, t_tt,
                obs.ra_deg, obs.dec_deg, prefilter_deg)
            bright_global_idx = np.where(h_mask)[0]
            ast_cands         = bright_global_idx[kep_mask]

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

    Disables the Numba parallel kernel in each worker to avoid TBB thread-pool
    fork-safety deadlocks (libgomp/TBB pools from the parent are not fork-safe).
    The vectorized NumPy fallback is used instead; it is still fast because each
    worker processes only 1/N of the observations.
    """
    i, obs = idx_obs
    # Patch out the Numba kernel in this worker process (fork-local, doesn't
    # affect the parent).  kep_to_radec falls back to vectorized NumPy.
    try:
        import mpchecker.propagator as _prop
        _prop._NUMBA_KEP_KERNEL = None
    except Exception:
        pass
    st = _PHASE1_STATE
    return i, _phase1_one_obs(
        obs,
        st['asteroids'], st['comets'], st['obscodes'], st['sky_index'],
        st['mag_limit'], st['prefilter_deg'],
    )


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
        t_tts_all = [_utc_to_tt_mjd(obs.epoch_mjd) for obs in observations]
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
        _PHASE1_STATE = {
            'asteroids':    asteroids,
            'comets':       comets,
            'obscodes':     obscodes,
            'sky_index':    sky_index,
            'mag_limit':    mag_limit,
            'prefilter_deg': prefilter_deg,
        }
        nw = min(n_workers, len(observations))
        ctx = mp.get_context('fork')
        with ctx.Pool(nw) as pool:
            tagged = pool.map(_phase1_worker, list(enumerate(observations)))
        obs_meta = [meta for _, meta in sorted(tagged, key=lambda x: x[0])]
    else:
        obs_meta = [
            _phase1_one_obs(obs, asteroids, comets, obscodes, sky_index,
                            mag_limit, prefilter_deg)
            for obs in observations
        ]

    # -----------------------------------------------------------------------
    # Phase 2: Batched pyoorb — one call per obscode group (Optimization 2).
    # For each group: take union of candidates, run pyoorb at all epochs,
    # then apply per-observation angular + magnitude cuts.
    # -----------------------------------------------------------------------
    # Group observation indices by obscode
    by_obscode = defaultdict(list)
    for i, obs in enumerate(observations):
        by_obscode[obs.obscode].append(i)

    for obscode, obs_indices in by_obscode.items():
        # --- Asteroids ---
        all_ast = sorted(set().union(
            *[set(obs_meta[i]['ast_cands'].tolist()) for i in obs_indices]))
        if all_ast:
            all_ast = np.array(all_ast, dtype=np.intp)
            sub = asteroids[all_ast]
            orbits = build_oorb_orbits_kep(
                sub['a'], sub['e'],
                sub['i']     * _DEG2RAD,
                sub['Omega'] * _DEG2RAD,
                sub['omega'] * _DEG2RAD,
                sub['M']     * _DEG2RAD,
                sub['epoch'], sub['H'], sub['G'],
            )
            t_tts = [obs_meta[i]['t_tt'] for i in obs_indices]
            # Single pyoorb call for all epochs in this obscode group
            eph_batch = oorb_ephemeris_multi_epoch(orbits, t_tts, obscode, dynmodel,
                                                   obscodes=obscodes)
            # eph_batch: [n_union_cands, n_epochs, 11]

            for k, obs_i in enumerate(obs_indices):
                my_cands = obs_meta[obs_i]['ast_cands']
                if len(my_cands) == 0:
                    continue
                # Positions of my_cands within all_ast (all_ast is sorted)
                pos_in_union = np.searchsorted(all_ast, my_cands)
                eph_k = eph_batch[pos_in_union, k, :]   # [n_my_cands, 11]

                obs = observations[obs_i]
                seps = ang_sep_deg(eph_k[:, 1], eph_k[:, 2], obs.ra_deg, obs.dec_deg)
                vmags = eph_k[:, 9]
                within = (seps <= search_rad_deg) & (vmags <= mag_limit)

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
                vmags = eph_ck[:, 9]
                within = (seps <= search_rad_deg) & (vmags <= mag_limit)

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
    asteroids : full MPCORB structured array
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
    for res in results:
        for m in res.matches:
            if m.obj_type == 'satellite':
                continue   # satellites have no catalog orbit to test
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
    t_tts      = [_utc_to_tt_mjd(obs.epoch_mjd) for obs in observations]
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
                        # prefer that over element-similarity result
                        fo_desig = str(fo_orbit['desig'][0]).strip()
                        if fo_desig and fo_desig not in ('ZMPCK', ''):
                            idx = np.where(
                                (asteroids['desig'] == fo_desig) |
                                (asteroids['packed'] == fo_desig)
                            )[0]
                            if len(idx):
                                cat_name   = str(asteroids['desig'][idx[0]]).strip()
                                cat_packed = str(asteroids['packed'][idx[0]]).strip()
                                cat_score  = 0.0
                            else:
                                cat_name  = fo_desig
                                cat_score = 0.0

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

    return identifications
