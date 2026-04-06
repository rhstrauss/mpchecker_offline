"""
Spatial sky index for fast candidate pre-selection.

Pre-computes heliocentric sky positions for the entire asteroid catalog at
multiple reference epochs and stores them in scipy KD-trees on unit (x,y,z)
vectors. At query time a cone search retrieves candidates in O(log N + k)
rather than the O(N) Keplerian propagation over the full 1.5M-object catalog.

Multi-snapshot design
---------------------
Four snapshots are built at 2-day intervals (t-3, t-1, t+1, t+3 days around
the center epoch).  At query time the nearest snapshot is selected so the
observation is at most ~1 day from a reference epoch.  The required motion
buffer shrinks from 5° (single-snapshot, 7-day validity) to ~0.7° (nearest
snapshot ≤1 day away), reducing the cone area by ~8× and the candidate count
proportionally.

Objects with semi-major axis < MIN_A_AU move faster than the index can bound;
they are always propagated directly (they are a tiny fraction of the catalog).

Typical workflow:
    multi = get_or_build_index(asteroids, obscodes, CACHE_DIR, t_now_mjd)
    multi.save(cache_dir)                          # auto-done by get_or_build_index

    multi = get_or_build_index(asteroids, obscodes, CACHE_DIR, t_now_mjd)  # loads from cache
    mask  = multi.candidates(ra_deg, dec_deg, prefilter_deg, t_obs_mjd)
"""

import logging
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

from .propagator import kep_to_radec, get_observer_helio, _DEG2RAD, _RAD2DEG

log = logging.getLogger(__name__)

# Objects with a < MIN_A_AU are fast movers; always propagated directly.
_MIN_A_AU = 1.5

# Maximum apparent angular rate (deg/day) for objects with a ≥ MIN_A_AU.
# Derived from the single-snapshot defaults: 5° buffer / 7-day validity.
_MOTION_RATE_DEG_PER_DAY = 5.0 / 7.0          # ≈ 0.714 deg/day

# Multi-snapshot parameters
_SNAPSHOT_INTERVAL_DAYS = 2.0                  # spacing between snapshots
_N_SNAPSHOTS            = 4                    # total snapshots (covers ±(N/2 * interval) days)

# Kept for backward-compat / single-index builds
_DEFAULT_VALIDITY_DAYS    = 7.0
_DEFAULT_MOTION_BUFFER_DEG = 5.0


# ---------------------------------------------------------------------------
# Single-epoch index
# ---------------------------------------------------------------------------

class SkyIndex:
    """
    KD-tree spatial index over asteroid sky positions at one reference epoch.
    """

    def __init__(
        self,
        tree,
        ref_ra_deg:              np.ndarray,
        ref_dec_deg:             np.ndarray,
        indexed_mask:            np.ndarray,
        fast_mask:               np.ndarray,
        t_ref_mjd:               float,
        validity_days:           float,
        motion_buffer_deg:       float,
        motion_rate_deg_per_day: float = _MOTION_RATE_DEG_PER_DAY,
    ):
        self._tree             = tree
        self._ref_ra           = ref_ra_deg
        self._ref_dec          = ref_dec_deg
        self._indexed_mask     = indexed_mask
        self._fast_mask        = fast_mask
        self._indexed_idx      = np.where(indexed_mask)[0]
        self._fast_idx         = np.where(fast_mask)[0]
        self.t_ref_mjd               = t_ref_mjd
        self.validity_days           = validity_days
        self.motion_buffer_deg       = motion_buffer_deg
        self.motion_rate_deg_per_day = motion_rate_deg_per_day

    # ------------------------------------------------------------------
    # Factory: build from catalog
    # ------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        asteroids:               np.ndarray,
        t_ref_mjd:               float,
        obscodes:                dict,
        obscode:                 str   = '500',
        validity_days:           float = _DEFAULT_VALIDITY_DAYS,
        motion_buffer_deg:       float = _DEFAULT_MOTION_BUFFER_DEG,
        motion_rate_deg_per_day: float = _MOTION_RATE_DEG_PER_DAY,
    ) -> 'SkyIndex':
        from scipy.spatial import cKDTree

        t0 = time.time()
        log.info('Building SkyIndex: %d objects at MJD=%.2f …', len(asteroids), t_ref_mjd)

        indexed_mask = asteroids['a'] >= _MIN_A_AU
        fast_mask    = ~indexed_mask

        log.info('  %d slow-mover (a≥%.1f AU) + %d fast-mover objects',
                 indexed_mask.sum(), _MIN_A_AU, fast_mask.sum())

        obs_helio = get_observer_helio(t_ref_mjd, obscode, obscodes)

        sub = asteroids[indexed_mask]
        ra_deg, dec_deg, _ = kep_to_radec(
            sub['a'], sub['e'],
            sub['i']     * _DEG2RAD,
            sub['Omega'] * _DEG2RAD,
            sub['omega'] * _DEG2RAD,
            sub['M']     * _DEG2RAD,
            sub['epoch'],
            t_ref_mjd,
            obs_helio,
        )

        ra_r  = ra_deg  * _DEG2RAD
        dec_r = dec_deg * _DEG2RAD
        cos_d = np.cos(dec_r)
        xyz = np.column_stack([
            cos_d * np.cos(ra_r),
            cos_d * np.sin(ra_r),
            np.sin(dec_r),
        ])

        tree = cKDTree(xyz, leafsize=32)
        log.info('  SkyIndex built in %.1fs', time.time() - t0)

        return cls(
            tree=tree,
            ref_ra_deg=ra_deg,
            ref_dec_deg=dec_deg,
            indexed_mask=indexed_mask,
            fast_mask=fast_mask,
            t_ref_mjd=t_ref_mjd,
            validity_days=validity_days,
            motion_buffer_deg=motion_buffer_deg,
            motion_rate_deg_per_day=motion_rate_deg_per_day,
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        import pickle
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            ref_ra_deg=self._ref_ra,
            ref_dec_deg=self._ref_dec,
            indexed_mask=self._indexed_mask,
            fast_mask=self._fast_mask,
            t_ref_mjd=np.array([self.t_ref_mjd]),
            validity_days=np.array([self.validity_days]),
            motion_buffer_deg=np.array([self.motion_buffer_deg]),
            motion_rate_deg_per_day=np.array([self.motion_rate_deg_per_day]),
        )
        pkl_path = path.with_suffix('.pkl')
        try:
            with open(pkl_path, 'wb') as f:
                pickle.dump(self._tree, f, protocol=4)
        except Exception as exc:
            log.warning('Failed to save KDTree pickle %s: %s', pkl_path, exc)
        log.info('SkyIndex saved to %s', path)

    @classmethod
    def load(cls, path: Path) -> 'SkyIndex':
        import pickle
        from scipy.spatial import cKDTree

        t0 = time.time()
        path = Path(path)
        d = np.load(path)
        ra_deg  = d['ref_ra_deg']
        dec_deg = d['ref_dec_deg']

        tree = None
        pkl_path = path.with_suffix('.pkl')
        if (pkl_path.exists()
                and pkl_path.stat().st_mtime >= path.stat().st_mtime):
            try:
                with open(pkl_path, 'rb') as f:
                    tree = pickle.load(f)
                log.info('SkyIndex loaded from %s (KDTree from pickle in %.2fs)',
                         path, time.time() - t0)
            except Exception as exc:
                log.warning('KDTree pickle load failed (%s); rebuilding', exc)
                tree = None

        if tree is None:
            ra_r  = ra_deg  * _DEG2RAD
            dec_r = dec_deg * _DEG2RAD
            cos_d = np.cos(dec_r)
            xyz = np.column_stack([
                cos_d * np.cos(ra_r),
                cos_d * np.sin(ra_r),
                np.sin(dec_r),
            ])
            tree = cKDTree(xyz, leafsize=32)
            log.info('SkyIndex loaded from %s (cKDTree rebuilt in %.1fs)',
                     path, time.time() - t0)

        mrate = (float(d['motion_rate_deg_per_day'][0])
                 if 'motion_rate_deg_per_day' in d else _MOTION_RATE_DEG_PER_DAY)

        return cls(
            tree=tree,
            ref_ra_deg=ra_deg,
            ref_dec_deg=dec_deg,
            indexed_mask=d['indexed_mask'],
            fast_mask=d['fast_mask'],
            t_ref_mjd=float(d['t_ref_mjd'][0]),
            validity_days=float(d['validity_days'][0]),
            motion_buffer_deg=float(d['motion_buffer_deg'][0]),
            motion_rate_deg_per_day=mrate,
        )

    # ------------------------------------------------------------------
    # Freshness
    # ------------------------------------------------------------------

    def is_fresh(self, t_mjd: float) -> bool:
        return abs(t_mjd - self.t_ref_mjd) <= self.validity_days

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query_cone(self, ra_deg: float, dec_deg: float, radius_deg: float) -> np.ndarray:
        """
        Return indices into the *indexed* sub-array of objects within radius_deg
        of (ra_deg, dec_deg) at the reference epoch.
        """
        ra_r  = ra_deg  * _DEG2RAD
        dec_r = dec_deg * _DEG2RAD
        cos_d = np.cos(dec_r)
        qvec  = np.array([cos_d*np.cos(ra_r), cos_d*np.sin(ra_r), np.sin(dec_r)])

        chord = 2.0 * np.sin(radius_deg * _DEG2RAD / 2.0)
        hits  = self._tree.query_ball_point(qvec, chord)
        return np.array(hits, dtype=np.intp)

    def candidates(
        self,
        ra_deg:       float,
        dec_deg:      float,
        prefilter_deg: float,
        t_obs_mjd:    Optional[float] = None,
    ) -> np.ndarray:
        """
        Return a boolean mask over the *full* asteroids array of candidates.

        When t_obs_mjd is given, the motion buffer is computed dynamically as
        motion_rate * |t_obs_mjd - t_ref_mjd|, giving a tighter cone when the
        observation is close to the reference epoch (as with multi-snapshot).
        Fast movers (a < MIN_A_AU) are always included unconditionally.
        """
        if t_obs_mjd is not None:
            dt  = abs(t_obs_mjd - self.t_ref_mjd)
            buf = self.motion_rate_deg_per_day * dt
        else:
            buf = self.motion_buffer_deg

        cone_deg    = prefilter_deg + buf
        hits_in_sub = self.query_cone(ra_deg, dec_deg, cone_deg)

        global_hits = self._indexed_idx[hits_in_sub]

        n_total = len(self._indexed_mask)
        mask = np.zeros(n_total, dtype=bool)
        if len(global_hits):
            mask[global_hits] = True
        mask[self._fast_idx] = True
        return mask


# ---------------------------------------------------------------------------
# Multi-snapshot wrapper
# ---------------------------------------------------------------------------

class MultiSkyIndex:
    """
    Wraps several SkyIndex snapshots at different reference epochs.

    At query time the nearest snapshot is chosen and queried with a
    dynamically-computed motion buffer proportional to the time offset.
    The validity window spans the full set of snapshots.
    """

    def __init__(
        self,
        snapshots:         List[SkyIndex],
        snapshot_interval: float = _SNAPSHOT_INTERVAL_DAYS,
    ):
        self._snaps          = sorted(snapshots, key=lambda s: s.t_ref_mjd)
        self._half_interval  = snapshot_interval / 2.0
        # Expose t_ref_mjd / validity_days for code that treats this like a SkyIndex
        if self._snaps:
            mid = len(self._snaps) // 2
            self.t_ref_mjd    = self._snaps[mid].t_ref_mjd
            self.validity_days = (self._snaps[-1].t_ref_mjd
                                  - self._snaps[0].t_ref_mjd
                                  + snapshot_interval)
        else:
            self.t_ref_mjd    = 0.0
            self.validity_days = 0.0

    def is_fresh(self, t_mjd: float) -> bool:
        """True if at least one snapshot is within half_interval of t_mjd."""
        if not self._snaps:
            return False
        return min(abs(s.t_ref_mjd - t_mjd) for s in self._snaps) <= self._half_interval

    def covers(self, t_min: float, t_max: float) -> bool:
        """True if the index covers the entire range [t_min, t_max].

        Requires that every point in [t_min, t_max] is within half_interval of
        at least one snapshot — i.e. the first snapshot covers t_min and the
        last covers t_max.
        """
        if not self._snaps:
            return False
        t_first = self._snaps[0].t_ref_mjd - self._half_interval
        t_last  = self._snaps[-1].t_ref_mjd + self._half_interval
        return t_min >= t_first and t_max <= t_last

    def candidates(
        self,
        ra_deg:        float,
        dec_deg:       float,
        prefilter_deg: float,
        t_obs_mjd:     Optional[float] = None,
    ) -> np.ndarray:
        """Delegate to the nearest snapshot, passing t_obs_mjd for tight buffer."""
        if not self._snaps:
            return np.zeros(0, dtype=bool)
        if t_obs_mjd is not None:
            best = min(self._snaps, key=lambda s: abs(s.t_ref_mjd - t_obs_mjd))
        else:
            best = self._snaps[len(self._snaps) // 2]
        return best.candidates(ra_deg, dec_deg, prefilter_deg, t_obs_mjd)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _index_path(cache_dir: Path, t_ref_mjd: float) -> Path:
    return cache_dir / f'sky_index_{int(t_ref_mjd)}.npz'


def get_or_build_index(
    asteroids:         np.ndarray,
    obscodes:          dict,
    cache_dir:         Path,
    t_now_mjd:         Optional[float] = None,
    t_min_mjd:         Optional[float] = None,
    t_max_mjd:         Optional[float] = None,
    n_snapshots:       int   = _N_SNAPSHOTS,
    snapshot_interval: float = _SNAPSHOT_INTERVAL_DAYS,
    validity_days:     float = _DEFAULT_VALIDITY_DAYS,
    motion_buffer_deg: float = _DEFAULT_MOTION_BUFFER_DEG,
    force_rebuild:     bool  = False,
) -> MultiSkyIndex:
    """
    Load a fresh multi-snapshot index from cache_dir, or build and save a new one.

    When t_min_mjd and t_max_mjd are provided (recommended for batch processing),
    the index is extended to cover the full epoch range: n_snapshots is derived
    from the span and t_now_mjd is set to the midpoint.  This eliminates the
    full-catalog Keplerian scan for historical or future batch files.

    Without t_min/t_max: four snapshots at ±1 and ±3 days around t_now_mjd
    (n_snapshots=4, interval=2) cover ±4 days.

    Old index files are cleaned up automatically when a rebuild is needed.

    Parameters
    ----------
    asteroids         : full asteroid catalog (structured array)
    obscodes          : observatory code dict
    cache_dir         : directory to store snapshot files
    t_now_mjd         : center epoch (MJD UTC). Defaults to midpoint of
                        [t_min_mjd, t_max_mjd] if provided, else now.
    t_min_mjd         : earliest observation epoch in the batch (MJD UTC)
    t_max_mjd         : latest observation epoch in the batch (MJD UTC)
    n_snapshots       : number of snapshots (overridden when t_min/t_max given)
    snapshot_interval : days between consecutive snapshots (default 2)
    validity_days     : validity stored in each individual SkyIndex
    force_rebuild     : always rebuild, ignoring any cached files
    """
    from astropy.time import Time

    # Derive span-aware parameters when batch epoch range is provided
    if t_min_mjd is not None and t_max_mjd is not None:
        span = t_max_mjd - t_min_mjd
        n_snapshots = max(_N_SNAPSHOTS, int(np.ceil(span / snapshot_interval)) + 2)
        if t_now_mjd is None:
            t_now_mjd = (t_min_mjd + t_max_mjd) / 2.0
        log.info('Batch epoch range %.1f days → %d snapshots', span, n_snapshots)

    if t_now_mjd is None:
        t_now_mjd = Time.now().mjd

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if not force_rebuild:
        # Try to load whatever snapshot files are present
        snaps = []
        for npz in sorted(cache_dir.glob('sky_index_*.npz')):
            try:
                snaps.append(SkyIndex.load(npz))
            except Exception as exc:
                log.warning('Dropping corrupt index file %s: %s', npz.name, exc)
                npz.unlink(missing_ok=True)
                npz.with_suffix('.pkl').unlink(missing_ok=True)

        if snaps:
            multi = MultiSkyIndex(snaps, snapshot_interval)
            # Reject index if it was built from a different-sized catalog
            index_n = len(snaps[0]._indexed_mask)
            if index_n != len(asteroids):
                log.info(
                    'Cached index size %d != catalog size %d; rebuilding …',
                    index_n, len(asteroids),
                )
                snaps = []
            else:
                # Use covers() when a batch range is known, is_fresh() otherwise
                if t_min_mjd is not None and t_max_mjd is not None:
                    cache_ok = multi.covers(t_min_mjd, t_max_mjd)
                else:
                    cache_ok = multi.is_fresh(t_now_mjd)
                if cache_ok:
                    log.info('Using %d cached index snapshot(s) (nearest %.1f days away)',
                             len(snaps),
                             min(abs(s.t_ref_mjd - t_now_mjd) for s in snaps))
                    return multi
                log.info('Cached index does not cover required range, rebuilding …')

        # Remove stale files before rebuild
        for npz in cache_dir.glob('sky_index_*.npz'):
            npz.unlink(missing_ok=True)
            npz.with_suffix('.pkl').unlink(missing_ok=True)

    # Build fresh snapshots centered on t_now_mjd
    half   = (n_snapshots - 1) / 2.0
    t_snaps = [t_now_mjd + (i - half) * snapshot_interval
               for i in range(n_snapshots)]

    log.info('Building %d index snapshots around MJD=%.1f (interval=%.0fd) …',
             n_snapshots, t_now_mjd, snapshot_interval)

    snaps = []
    for t_snap in t_snaps:
        snap = SkyIndex.build(
            asteroids, t_snap, obscodes,
            validity_days=validity_days,
            motion_buffer_deg=motion_buffer_deg,
        )
        snap.save(_index_path(cache_dir, t_snap))
        snaps.append(snap)

    return MultiSkyIndex(snaps, snapshot_interval)
