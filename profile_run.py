"""
Profile a real mpchecker run to break down Phase 1 / Phase 2 / Phase 3 timing.
Usage: python profile_run.py <obs_file>
"""
import sys, time, os
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))

OBS_FILE = (sys.argv[1] if len(sys.argv) > 1
            else '/astro/store/shire/rstrau/ATLAS/cneo_parseout/fo_combined_test/combined.mpc80')

# ── Load infrastructure ─────────────────────────────────────────────────────
print('Loading catalog …', flush=True)
from mpchecker.mpcorb import load_mpcorb, load_comets, load_obscodes
asteroids = load_mpcorb()
comets    = load_comets()
obscodes  = load_obscodes()
print(f'  {len(asteroids):,} asteroids, {len(comets):,} comets', flush=True)

from mpchecker.checker import build_asteroid_soa
asteroid_soa = build_asteroid_soa(asteroids)

from mpchecker.config import CACHE_DIR
from mpchecker.index import get_or_build_index
from mpchecker.obs_parser import parse_file_auto as _pfa
import numpy as np
_probe_obs = _pfa(OBS_FILE)
_t_center  = float(np.median([o.epoch_mjd for o in _probe_obs]))
print(f'Loading sky index (centered on MJD={_t_center:.1f}) …', flush=True)
_index_cache = CACHE_DIR.parent / 'profile_index_cache'
sky_index = get_or_build_index(asteroids, obscodes, _index_cache,
                               t_now_mjd=_t_center, force_rebuild=False)

from mpchecker.obs_parser import parse_file_auto
print(f'Parsing {OBS_FILE} …', flush=True)
observations = parse_file_auto(OBS_FILE)
print(f'  {len(observations)} observations', flush=True)

# ── Patch in timing around each phase ──────────────────────────────────────
import mpchecker.checker as _ck
import mpchecker.propagator as _prop
import mpchecker.satellites as _sat

_timings = {
    'phase1_total':   0.0,
    'phase1_index':   0,   # count of index-path obs
    'phase1_noindex': 0,   # count of full-catalog obs
    'phase2_total':   0.0,
    'phase2_calls':   0,
    'phase2_orbits':  0,   # total orbit-epochs
    'phase3_total':   0.0,
    'phase3_calls':   0,
    'reepoch_total':  0.0,
    'check_obs_total': 0.0,
}

# Patch _phase1_one_obs
_orig_phase1 = _ck._phase1_one_obs
def _timed_phase1(obs, asteroids, comets, obscodes, sky_index, mag_limit,
                  prefilter_deg, asteroid_soa=None):
    t0 = time.perf_counter()
    result = _orig_phase1(obs, asteroids, comets, obscodes, sky_index,
                          mag_limit, prefilter_deg, asteroid_soa=asteroid_soa)
    _timings['phase1_total'] += time.perf_counter() - t0
    # Detect which path was taken: index path has small candidate count
    n_cands = len(result['ast_cands'])
    if sky_index is not None and sky_index.is_fresh(obs.epoch_mjd):
        _timings['phase1_index'] += 1
    else:
        _timings['phase1_noindex'] += 1
    return result
_ck._phase1_one_obs = _timed_phase1

# Patch oorb_ephemeris_multi_epoch
_orig_oorb = _prop.oorb_ephemeris_multi_epoch
def _timed_oorb(orbits, t_tts, obscode, dynmodel='N', obscodes=None):
    t0 = time.perf_counter()
    result = _orig_oorb(orbits, t_tts, obscode, dynmodel, obscodes=obscodes)
    dt = time.perf_counter() - t0
    _timings['phase2_total']  += dt
    _timings['phase2_calls']  += 1
    _timings['phase2_orbits'] += len(orbits) * len(t_tts)
    return result
_prop.oorb_ephemeris_multi_epoch = _timed_oorb
# Also patch the import reference in checker
_ck.oorb_ephemeris_multi_epoch = _timed_oorb

# Patch check_satellites
_orig_sat = _sat.check_satellites
def _timed_sat(ra_deg, dec_deg, t_mjd, obscode, obscodes, search_radius_deg,
               mag_limit=25.0, naif_ids=None):
    t0 = time.perf_counter()
    result = _orig_sat(ra_deg, dec_deg, t_mjd, obscode, obscodes,
                       search_radius_deg, mag_limit, naif_ids)
    _timings['phase3_total'] += time.perf_counter() - t0
    _timings['phase3_calls'] += 1
    return result
_sat.check_satellites = _timed_sat
_ck.check_satellites = _timed_sat

# Patch reepoch_high_e_asteroids
_orig_reepoch = _prop.reepoch_high_e_asteroids
def _timed_reepoch(*args, **kwargs):
    t0 = time.perf_counter()
    result = _orig_reepoch(*args, **kwargs)
    _timings['reepoch_total'] += time.perf_counter() - t0
    return result
_prop.reepoch_high_e_asteroids = _timed_reepoch
_ck.reepoch_high_e_asteroids = _timed_reepoch

# ── Run check_observations ──────────────────────────────────────────────────
from mpchecker.checker import check_observations

import sys
n_workers = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 1
print(f'\nRunning check_observations (n_workers={n_workers}) …', flush=True)
t_wall0 = time.perf_counter()
results = check_observations(
    observations, asteroids, comets, obscodes,
    sky_index=sky_index,
    asteroid_soa=asteroid_soa,
    search_radius_arcmin=30.0,
    mag_limit=25.0,
    check_sats=True,
    n_workers=n_workers,
)
t_wall = time.perf_counter() - t_wall0

n_obs   = len(observations)
n_match = sum(1 for r in results if r.matches)

# ── Report ──────────────────────────────────────────────────────────────────
print(f'\n{"="*60}')
print(f'  Pipeline timing  ({n_obs} obs, {n_match} with matches)')
print(f'{"="*60}')

def _ms(t): return t * 1000

p1  = _ms(_timings['phase1_total'])
p2  = _ms(_timings['phase2_total'])
p3  = _ms(_timings['phase3_total'])
pre = _ms(_timings['reepoch_total'])
total_accounted = p1 + p2 + p3 + pre
total_wall = _ms(t_wall)
other = total_wall - total_accounted

fmt = '  {:<38s}  {:8.1f} ms  {:5.1f}%'
print(fmt.format('Pre-Phase 1 (reepoch high-e)',    pre, 100*pre/total_wall))
print(fmt.format(f'Phase 1 total  ({n_obs} obs)',   p1,  100*p1/total_wall))
idx_n  = _timings['phase1_index']
noi_n  = _timings['phase1_noindex']
print(f'    index path: {idx_n} obs   full-catalog: {noi_n} obs')
if n_obs > 0:
    print(f'    per-obs avg: {p1/n_obs:.1f} ms')

print(fmt.format(f'Phase 2 (pyoorb, {_timings["phase2_calls"]} calls)', p2, 100*p2/total_wall))
if _timings['phase2_calls'] > 0:
    orbit_epochs = _timings['phase2_orbits']
    print(f'    {orbit_epochs:,} orbit-epochs  →  '
          f'{_ms(_timings["phase2_total"])/orbit_epochs*1000:.2f} µs/orbit-epoch')
    print(f'    per-call avg: {p2/_timings["phase2_calls"]:.1f} ms')

print(fmt.format(f'Phase 3 (satellites, {_timings["phase3_calls"]} calls)', p3, 100*p3/total_wall))
if _timings['phase3_calls'] > 0:
    print(f'    per-call avg: {p3/_timings["phase3_calls"]:.2f} ms')

print(fmt.format('Other (overhead, format, etc.)',  other, 100*other/total_wall))
print(f'  {"─"*52}')
print(fmt.format('Total wall time',                 total_wall, 100.0))
print(f'  per-observation avg: {total_wall/n_obs:.1f} ms')

# Phase 1 candidate counts
cand_counts = [len(r.matches) for r in results]
print(f'\n  Phase 2 input candidates per obs:')
phase1_cands = []
# Re-run phase1 quickly just to get candidate counts (without timing overhead)
from mpchecker.checker import _phase1_one_obs as _p1
for obs in observations[:5]:  # sample
    meta = _p1(obs, asteroids, comets, obscodes, sky_index, 25.0, 0.6,
               asteroid_soa=asteroid_soa)
    phase1_cands.append(len(meta['ast_cands']))
if phase1_cands:
    print(f'    sample (5 obs): {phase1_cands}')
    print(f'    (union of all obs fed to pyoorb per obscode batch)')
