"""
Benchmark for mpchecker performance optimizations 1–6.
Run from the repo root: python bench.py
"""
import time
import numpy as np
import os, sys

sys.path.insert(0, os.path.dirname(__file__))

# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def _hdr(title):
    print(f'\n{"="*60}')
    print(f'  {title}')
    print(f'{"="*60}')

def _row(label, t_ms, note=''):
    note_str = f'  ({note})' if note else ''
    print(f'  {label:<40s}  {t_ms:8.1f} ms{note_str}')

def _timeit(fn, n=3):
    """Return median wall time in ms over n calls."""
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times))

# ──────────────────────────────────────────────────────────────
# Load catalog once (shared by all benchmarks)
# ──────────────────────────────────────────────────────────────
print('Loading catalog …', flush=True)
from mpchecker.mpcorb import load_mpcorb, load_obscodes
asteroids = load_mpcorb()
obscodes  = load_obscodes()
print(f'  {len(asteroids):,} asteroids loaded', flush=True)

# Representative observation parameters
OBS_RA, OBS_DEC, OBS_T_MJD = 90.0, -20.0, 60700.0
OBS_OBSCODE = 'T08'

# ──────────────────────────────────────────────────────────────
# Benchmark 1 & 2: cKDTree + pickle persistence
# ──────────────────────────────────────────────────────────────
_hdr('Opt 1+2: cKDTree + pickle persistence')

from pathlib import Path
from mpchecker.config import CACHE_DIR
from mpchecker.propagator import get_observer_helio, kep_to_radec, _DEG2RAD
import numpy as np

# 1a: Time to REBUILD tree from .npz (no pickle) — simulates old behavior with KDTree
npz_files = sorted(Path(CACHE_DIR).glob('sky_index_*.npz'))
pkl_files  = sorted(Path(CACHE_DIR).glob('sky_index_*.pkl'))
print(f'  Found {len(npz_files)} .npz files, {len(pkl_files)} .pkl files')

# Time rebuild from raw xyz (as old KDTree load did)
sample_npz = npz_files[0]
d = np.load(sample_npz)
ra_deg  = d['ref_ra_deg']
dec_deg = d['ref_dec_deg']
ra_r  = ra_deg  * _DEG2RAD
dec_r = dec_deg * _DEG2RAD
cos_d = np.cos(dec_r)
xyz = np.column_stack([cos_d * np.cos(ra_r), cos_d * np.sin(ra_r), np.sin(dec_r)])

# Old: KDTree
from scipy.spatial import KDTree, cKDTree

def _build_kdtree():
    KDTree(xyz, leafsize=32)

def _build_ckdtree():
    cKDTree(xyz, leafsize=32)

t_kd  = _timeit(_build_kdtree, n=3)
t_ckd = _timeit(_build_ckdtree, n=3)
_row('KDTree  build (1 snapshot)', t_kd,  'old')
_row('cKDTree build (1 snapshot)', t_ckd, 'new')
print(f'  → tree build speedup: {t_kd/t_ckd:.1f}×')

# 1b: Query time
tree_kd  = KDTree(xyz, leafsize=32)
tree_ckd = cKDTree(xyz, leafsize=32)

ra_q, dec_q = OBS_RA * _DEG2RAD, OBS_DEC * _DEG2RAD
cos_q = np.cos(dec_q)
qvec = np.array([cos_q*np.cos(ra_q), cos_q*np.sin(ra_q), np.sin(dec_q)])
chord = 2.0 * np.sin(0.6 * _DEG2RAD / 2.0)

def _query_kd():  tree_kd.query_ball_point(qvec, chord)
def _query_ckd(): tree_ckd.query_ball_point(qvec, chord)

t_qkd  = _timeit(_query_kd,  n=20)
t_qckd = _timeit(_query_ckd, n=20)
_row('KDTree  query_ball_point',   t_qkd,  'old')
_row('cKDTree query_ball_point',   t_qckd, 'new')
print(f'  → query speedup: {t_qkd/t_qckd:.1f}×')

# 2: Pickle save/load vs rebuild
import pickle
pkl_path = Path('/tmp/_bench_tree.pkl')
with open(pkl_path, 'wb') as f:
    pickle.dump(tree_ckd, f, protocol=4)

def _load_rebuild():
    d2 = np.load(sample_npz)
    ra2, dec2 = d2['ref_ra_deg'], d2['ref_dec_deg']
    r2, d2_ = ra2 * _DEG2RAD, dec2 * _DEG2RAD
    c2 = np.cos(d2_)
    xyz2 = np.column_stack([c2*np.cos(r2), c2*np.sin(r2), np.sin(d2_)])
    cKDTree(xyz2, leafsize=32)

def _load_pickle():
    with open(pkl_path, 'rb') as f:
        pickle.load(f)

t_rebuild = _timeit(_load_rebuild, n=3)
t_pickle  = _timeit(_load_pickle,  n=10)
_row('cKDTree from rebuild (1 snap)', t_rebuild)
_row('cKDTree from pickle  (1 snap)', t_pickle)
print(f'  → pickle speedup: {t_rebuild/t_pickle:.1f}×')
print(f'  → 4-snapshot cold start: rebuild={4*t_rebuild:.0f} ms  pickle={4*t_pickle:.0f} ms')
pkl_path.unlink(missing_ok=True)

# ──────────────────────────────────────────────────────────────
# Benchmark 3: Vectorized UTC→TT
# ──────────────────────────────────────────────────────────────
_hdr('Opt 3: Vectorized UTC→TT conversion')

from mpchecker.checker import _utc_to_tt_mjd, _utc_to_tt_mjd_vec

for n_obs in (10, 100, 500):
    epochs = [OBS_T_MJD + i * 0.01 for i in range(n_obs)]

    def _per_obs():
        return [_utc_to_tt_mjd(e) for e in epochs]

    def _vec():
        return _utc_to_tt_mjd_vec(iter(epochs)).tolist()

    t_per = _timeit(_per_obs, n=5)
    t_vec = _timeit(_vec,     n=5)
    _row(f'per-obs  [{n_obs:4d} obs]', t_per, 'old')
    _row(f'vectorized [{n_obs:4d} obs]', t_vec, 'new')
    print(f'  → speedup: {t_per/t_vec:.0f}×')

# ──────────────────────────────────────────────────────────────
# Benchmark 4: Numba thread count
# ──────────────────────────────────────────────────────────────
_hdr('Opt 4: Numba thread count')

try:
    import numba
    from mpchecker.propagator import kep_to_radec, _NUMBA_KEP_KERNEL

    if _NUMBA_KEP_KERNEL is None:
        print('  Numba kernel not available — skipping')
    else:
        # Use a large subset for a meaningful test
        N = min(len(asteroids), 300_000)
        sub = asteroids[:N]
        obs_helio = get_observer_helio(OBS_T_MJD, OBS_OBSCODE, obscodes)
        t_tt = OBS_T_MJD + 69.184 / 86400.0

        def _run_kep():
            kep_to_radec(
                sub['a'], sub['e'],
                sub['i'] * _DEG2RAD, sub['Omega'] * _DEG2RAD,
                sub['omega'] * _DEG2RAD, sub['M'] * _DEG2RAD,
                sub['epoch'], t_tt, obs_helio,
            )

        for n_threads in (16, 32, 64, 128):
            if n_threads > numba.config.NUMBA_NUM_THREADS:
                continue
            numba.set_num_threads(n_threads)
            t = _timeit(_run_kep, n=5)
            marker = ' ← current cap' if n_threads == 64 else ''
            _row(f'{n_threads:3d} threads  [{N//1000}K objects]', t, marker.strip())

        # Restore to cap
        numba.set_num_threads(min(64, numba.config.NUMBA_NUM_THREADS))
except Exception as exc:
    print(f'  Numba benchmark failed: {exc}')

# ──────────────────────────────────────────────────────────────
# Benchmark 5: Satellite planet pre-filter
# ──────────────────────────────────────────────────────────────
_hdr('Opt 5: Satellite planet proximity pre-filter')

try:
    from mpchecker.satellites import check_satellites, get_satellite_positions, SATELLITE_NAIF_IDS
    all_ids = list(SATELLITE_NAIF_IDS.keys())

    # Field far from all planets (typical survey field)
    ra_far, dec_far = 90.0, -20.0

    # Field near Jupiter (stress test)
    # Jupiter is currently around ~55°, but let's compute a rough position
    # We'll just test both "far" and "near" by manipulating the filter
    from mpchecker.satellites import _PLANET_SAT_GROUPS, _planet_radec_cached

    t_mjd_r = round(OBS_T_MJD, 2)
    skipped = 0
    for planet_naif, sat_ids, guard_deg in _PLANET_SAT_GROUPS:
        p_ra, p_dec = _planet_radec_cached(planet_naif, t_mjd_r)
        if p_ra is not None:
            from mpchecker.satellites import ang_sep_scalar
            sep = ang_sep_scalar(p_ra, p_dec, ra_far, dec_far)
            search_r = 0.5  # 30 arcmin
            if sep > search_r + guard_deg:
                skipped += len(sat_ids & set(all_ids))

    queried = len(all_ids) - skipped
    print(f'  Field ({ra_far:.0f}°, {dec_far:.0f}°): {len(all_ids)} total IDs, '
          f'{skipped} skipped, {queried} queried')

    def _check_no_filter():
        get_satellite_positions(all_ids, OBS_T_MJD, OBS_OBSCODE, obscodes)

    def _check_with_filter():
        check_satellites(ra_far, dec_far, OBS_T_MJD, OBS_OBSCODE, obscodes,
                         search_radius_deg=0.5)

    t_no   = _timeit(_check_no_filter,   n=5)
    t_with = _timeit(_check_with_filter, n=5)
    _row('get_satellite_positions (all IDs)', t_no,   'no filter')
    _row('check_satellites (with pre-filter)', t_with, 'new')
    print(f'  → speedup: {t_no/t_with:.1f}×')
except Exception as exc:
    print(f'  Satellite benchmark failed: {exc}')

# ──────────────────────────────────────────────────────────────
# Benchmark 6: AsteroidSOA vs structured array prefilter
# ──────────────────────────────────────────────────────────────
_hdr('Opt 6: AsteroidSOA vs structured array prefilter (full-catalog path)')

from mpchecker.checker import (
    _asteroid_prefilter, _asteroid_prefilter_soa,
    build_asteroid_soa, AsteroidSOA, _h_limit_from_vmag,
)

obs_helio = get_observer_helio(OBS_T_MJD, OBS_OBSCODE, obscodes)
h_cut = _h_limit_from_vmag(25.0, obs_helio, OBS_RA, OBS_DEC, OBS_T_MJD)
h_mask = asteroids['H'] <= h_cut
n_h = int(h_mask.sum())
print(f'  H≤{h_cut:.1f} cut: {n_h:,}/{len(asteroids):,} objects pass')

# Time SOA build (daemon startup cost)
t_soa_build = _timeit(lambda: build_asteroid_soa(asteroids), n=3)
_row('build_asteroid_soa (once at startup)', t_soa_build)

asteroid_soa = build_asteroid_soa(asteroids)
t_tt = OBS_T_MJD + 69.184 / 86400.0

def _old_path():
    _asteroid_prefilter(asteroids[h_mask], obs_helio, t_tt, OBS_RA, OBS_DEC, 0.6)

def _soa_path():
    _asteroid_prefilter_soa(asteroid_soa, h_mask, obs_helio, t_tt, OBS_RA, OBS_DEC, 0.6)

t_old = _timeit(_old_path, n=5)
t_soa = _timeit(_soa_path, n=5)
_row(f'asteroids[h_mask] prefilter [{n_h//1000}K]', t_old, 'old structured')
_row(f'SOA prefilter [{n_h//1000}K]',                t_soa, 'new contiguous')
print(f'  → speedup: {t_old/t_soa:.1f}×')

# Also measure structured array column extraction cost alone
sub = asteroids[h_mask]
def _extract_cols():
    _ = sub['a'].copy()
    _ = sub['e'].copy()
    _ = (sub['i'] * _DEG2RAD)
    _ = (sub['Omega'] * _DEG2RAD)
    _ = (sub['omega'] * _DEG2RAD)
    _ = (sub['M'] * _DEG2RAD)
    _ = sub['epoch'].copy()

def _extract_soa():
    _ = asteroid_soa.a[h_mask]
    _ = asteroid_soa.e[h_mask]
    _ = asteroid_soa.i_rad[h_mask]
    _ = asteroid_soa.Omega_rad[h_mask]
    _ = asteroid_soa.omega_rad[h_mask]
    _ = asteroid_soa.M_rad[h_mask]
    _ = asteroid_soa.epoch[h_mask]

t_cols_old = _timeit(_extract_cols, n=10)
t_cols_soa = _timeit(_extract_soa,  n=10)
_row(f'column extraction only (structured)', t_cols_old)
_row(f'column extraction only (SOA)',         t_cols_soa)
print(f'  → extraction speedup: {t_cols_old/t_cols_soa:.1f}×')

# ──────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────
_hdr('Summary')
print('  See per-section speedup lines above.')
print('  Next: benchmark full end-to-end mpchecker run on real obs file.')
print()
