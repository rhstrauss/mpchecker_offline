"""
Orbit refitting via find_orb (fo) for close-approach NEOs.

When a candidate NEO has a large epoch gap between the MPCORB element epoch
and the observation date — and has a close Earth encounter in between — the
pyoorb N-body integration accumulates significant positional errors.  This
module retrieves the object's full observational history from MPCAT and runs
a fresh orbit fit using `fo` (Project Pluto's Find_Orb batch tool).

The resulting elements, epoched to the date of the most recent observation
used, are returned in a format compatible with the asteroid structured array
used throughout mpchecker, and can be cached to avoid re-fitting on subsequent
calls.

Usage
-----
    from mpchecker.orbitfit import refit_neo
    new_elements = refit_neo('j8122', mpcat_index, cache_dir=CACHE_DIR)
    # Returns a 1-element structured array with updated a, e, i, Omega, omega, M, epoch
    # or None if fo is unavailable / fit fails.

Notes on accuracy
-----------------
For objects with very close Earth encounters (δ < 0.05 AU) where the
encounter epoch falls BETWEEN the fo fit epoch and the target observation
epoch, the fit will still require N-body integration through the encounter.
In those cases accuracy is limited to ~1–2' regardless of orbit fit quality.
Objects without such encounters typically improve from several arcminutes to
sub-arcsecond accuracy after a fresh fit.
"""

import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

log = logging.getLogger(__name__)

# MPC80 line length for validation
_MPC80_MIN_LEN = 77


def _fo_available() -> bool:
    return shutil.which('fo') is not None


def _parse_fo_json(json_path: Path) -> Optional[dict]:
    """
    Read a fo output JSON file and return the element dict for the best solution.

    When fo produces multi-arc solutions (e.g. 'ObjW' designations for weaker arcs),
    prefer the solution with the most residuals (largest observation arc).
    """
    try:
        with open(json_path) as f:
            d = json.load(f)
        objects = d.get('objects', {})
        if not objects:
            return None
        # Return solution with most n_resids (primary arc)
        best = max(objects.values(),
                   key=lambda v: v.get('elements', {}).get('n_resids', 0))
        return best['elements']
    except Exception as exc:
        log.debug('Failed to parse fo JSON %s: %s', json_path, exc)
        return None


def _obs_date_mjd(line: str) -> Optional[float]:
    """
    Parse the date from an MPC 80-column observation line and return MJD UTC.
    Date is in cols 15–31: 'YYYY MM DD.dddddd' (1-indexed cols 16-32).
    Returns None on parse failure.
    """
    try:
        y = int(line[15:19])
        m = int(line[20:22])
        d = float(line[23:32])
        return _gregorian_to_mjd(y, m, d)
    except Exception:
        return None


def _gregorian_to_mjd(y: int, m: int, d: float) -> float:
    """
    Convert a proleptic Gregorian calendar date to MJD (UTC).

    Uses the standard algorithm from Meeus "Astronomical Algorithms".
    Works for any date from antiquity to far future without astropy,
    avoiding ERFA 'dubious year' warnings for pre-1972 observations.
    """
    if m <= 2:
        y -= 1
        m += 12
    A = int(y / 100)
    B = 2 - A + int(A / 4)
    jd = int(365.25 * (y + 4716)) + int(30.6001 * (m + 1)) + d + B - 1524.5
    return jd - 2400000.5


def refit_neo(
    packed: str,
    mpcat_index,
    cache_dir: Optional[Path] = None,
    max_obs: int = 2000,
    timeout_sec: int = 60,
    max_date_mjd: Optional[float] = None,
    fo_home_dir: Optional[Path] = None,
) -> Optional[np.ndarray]:
    """
    Refit the orbit for a NEO using MPCAT observations.

    Parameters
    ----------
    packed       : MPC packed designation (e.g. 'j8122', 'K10E45W')
    mpcat_index  : MPCATIndex instance
    cache_dir    : if provided, cache fit results here; reload on subsequent calls
    max_obs      : cap on observations passed to fo (evenly sampled if exceeded)
    timeout_sec  : maximum wall time for fo
    max_date_mjd : if given, only use observations with date <= this MJD (UTC).
                   Useful to avoid crossing a close encounter when integrating
                   backward from a post-encounter epoch.
    fo_home_dir  : if given, set HOME=fo_home_dir for the fo subprocess and read
                   output from fo_home_dir/.find_orb/.  Allows concurrent fo runs
                   without output collisions.  Caller is responsible for cleanup.

    Returns
    -------
    numpy structured array with same dtype as mpcorb.ASTEROID_DTYPE, length 1,
    with updated elements and epoch; or None if fit fails or fo is unavailable.
    """
    if not _fo_available():
        log.debug('fo not found; skipping orbit refit for %s', packed)
        return None

    # ---- Build cache key (includes date filter if used) ----
    cache_path = None
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        if max_date_mjd is not None:
            # Round to nearest day for cache granularity
            cache_key = f'fo_fit_{packed}_pre{int(max_date_mjd)}.json'
        else:
            cache_key = f'fo_fit_{packed}.json'
        cache_path = cache_dir / cache_key
        if cache_path.exists():
            try:
                el = _parse_fo_json(cache_path)
                if el is not None:
                    return _elements_to_array(el, packed)
            except Exception:
                pass

    # ---- Fetch observations ----
    lines = mpcat_index.get_obs(packed, max_obs=0)   # fetch all, filter below
    if len(lines) < 3:
        log.debug('Too few observations for %s (%d); skipping refit', packed, len(lines))
        return None

    # Filter to valid-length lines
    lines = [l for l in lines if len(l.rstrip('\n')) >= _MPC80_MIN_LEN]

    # Apply date filter if requested
    if max_date_mjd is not None:
        filtered = [l for l in lines
                    if (d := _obs_date_mjd(l)) is not None and d <= max_date_mjd]
        if len(filtered) < 3:
            log.debug('%s: only %d obs before MJD %.1f; skipping refit',
                      packed, len(filtered), max_date_mjd)
            return None
        log.debug('%s: using %d/%d obs (date <= MJD %.1f)',
                  packed, len(filtered), len(lines), max_date_mjd)
        lines = filtered

    # Apply max_obs cap: evenly sample if exceeded
    if max_obs and len(lines) > max_obs:
        stride = len(lines) // max_obs
        lines = lines[::stride][:max_obs]

    if len(lines) < 3:
        return None

    log.info('Refitting orbit for %s using %d observations', packed, len(lines))

    # ---- Write temp obs file ----
    with tempfile.NamedTemporaryFile(
            mode='w', suffix='.mpc80', delete=False, prefix='mpchecker_fo_') as tf:
        tf.writelines(lines)
        obs_file = tf.name

    # fo writes output to $HOME/.find_orb/; override HOME for parallel isolation
    fo_output_dir = (fo_home_dir / '.find_orb') if fo_home_dir else (Path.home() / '.find_orb')
    fo_env = ({**os.environ, 'HOME': str(fo_home_dir)} if fo_home_dir else None)

    try:
        t0 = time.time()
        result = subprocess.run(
            ['fo', obs_file],
            capture_output=True, text=True, timeout=timeout_sec,
            env=fo_env,
        )
        elapsed = time.time() - t0
        log.debug('fo completed in %.1fs (returncode=%d)', elapsed, result.returncode)

        if result.returncode != 0:
            log.warning('fo returned %d for %s: %s',
                        result.returncode, packed, result.stderr[:200])
            return None

        # Prefer total.json (has all solutions including multi-arc) over
        # elem_short.json (may have only the secondary/weaker arc).
        json_path = fo_output_dir / 'total.json'
        if not json_path.exists():
            json_path = fo_output_dir / 'elem_short.json'
        if not json_path.exists():
            log.warning('fo produced no output JSON for %s', packed)
            return None

        el = _parse_fo_json(json_path)
        if el is None:
            return None

        # ---- Cache result ----
        if cache_path is not None:
            try:
                cache_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(json_path, cache_path)
                log.debug('Cached fo fit: %s', cache_path.name)
            except Exception as exc:
                log.debug('Cache write failed: %s', exc)

        return _elements_to_array(el, packed)

    except subprocess.TimeoutExpired:
        log.warning('fo timed out (%ds) for %s', timeout_sec, packed)
        return None
    except Exception as exc:
        log.warning('fo fit failed for %s: %s', packed, exc)
        return None
    finally:
        try:
            os.unlink(obs_file)
        except OSError:
            pass


def _elements_to_array(el: dict, packed: str) -> Optional[np.ndarray]:
    """
    Convert a fo elem_short.json element dict to a 1-element numpy structured
    array compatible with mpcorb.ASTEROID_DTYPE.
    """
    try:
        from .mpcorb import ASTEROID_DTYPE

        # fo uses JD; convert to MJD TT
        epoch_mjd = float(el['epoch']) - 2400000.5

        # fo elements are in J2000 ecliptic; angles in degrees
        # 'M' in fo output is mean anomaly in degrees
        arr = np.zeros(1, dtype=ASTEROID_DTYPE)
        arr['packed'][0] = packed
        arr['desig'][0]  = packed          # will be overridden by caller if needed
        arr['a'][0]      = float(el['a'])
        arr['e'][0]      = float(el['e'])
        arr['i'][0]      = float(el['i'])
        arr['Omega'][0]  = float(el['asc_node'])
        arr['omega'][0]  = float(el['arg_per'])
        arr['M'][0]      = float(el['M'])
        arr['epoch'][0]  = epoch_mjd
        arr['H'][0]      = float(el.get('H', 20.0))
        arr['G'][0]      = float(el.get('G', 0.15))
        return arr

    except Exception as exc:
        log.warning('Failed to convert fo elements for %s: %s', packed, exc)
        return None


def refit_cache_key(packed: str, last_obs_mjd: float) -> str:
    """Cache key string: packed designation + rounded last-obs epoch."""
    epoch_rounded = round(last_obs_mjd / 10) * 10   # 10-day bins
    return f'fo_fit_{packed}_{epoch_rounded:.0f}'


# ---------------------------------------------------------------------------
# Batch helpers for pre-Phase 1 NEO orbit refitting
# ---------------------------------------------------------------------------

def select_neo_packed(
    asteroids: np.ndarray,
    q_threshold: float = 1.1,
) -> List[str]:
    """
    Return packed designations for all asteroids with perihelion q < q_threshold.

    Parameters
    ----------
    asteroids    : structured array from mpcorb.load_mpcorb()
    q_threshold  : perihelion distance threshold in AU (default: 1.1)

    Returns
    -------
    List of packed designation strings.
    """
    q = asteroids['a'] * (1.0 - asteroids['e'])
    mask = q < q_threshold
    return list(asteroids['packed'][mask])


def _refit_worker(args: tuple) -> tuple:
    """
    Process-pool worker for parallel fo orbit refits.

    Each worker constructs its own MPCATIndex (private file handles) and runs fo
    in a private HOME directory so output files do not collide across workers.

    Parameters
    ----------
    args : (packed, mpcat_dir, cache_dir, max_obs, timeout_sec, max_date_mjd)

    Returns
    -------
    (packed, arr) — arr is None on failure or cache miss with no fo available
    """
    packed, mpcat_dir, cache_dir, max_obs, timeout_sec, max_date_mjd = args
    import tempfile
    from .mpcat import MPCATIndex

    idx = MPCATIndex(mpcat_dir)
    fo_home = Path(tempfile.mkdtemp(prefix='mpchecker_fo_home_'))
    try:
        _seed_fo_home(fo_home)
        arr = refit_neo(
            packed, idx,
            cache_dir=cache_dir,
            max_obs=max_obs,
            timeout_sec=timeout_sec,
            max_date_mjd=max_date_mjd,
            fo_home_dir=fo_home,
        )
    finally:
        shutil.rmtree(fo_home, ignore_errors=True)
    return packed, arr


# fo output/cache files that should NOT be copied into a worker's private HOME
# (sof.txt is fo's internal orbit cache — copying it would return stale fits
# for objects that happen to share a designation with a prior cached run)
_FO_SKIP_FILES = frozenset({
    'sof.txt', 'total.json', 'elem_short.json', 'debug.txt',
    'covar.txt', 'monte.txt', 'elements.txt', 'guide.txt',
    'mpc_fmt.txt', 'neocp.txt', 'state.txt', 'vectors.txt', 'observe.txt',
})


def _seed_fo_home(fo_home: Path) -> None:
    """
    Seed a fresh fo HOME directory with config files from the real HOME.

    fo asserts that $HOME/.find_orb/environ.dat is readable at startup.
    Copy all non-output config files from the real ~/.find_orb/, then
    append OUTPUT_DIR=<tmp_fo_dir>/ to the copied environ.dat so fo
    writes its JSON output to the worker's private directory.

    Without OUTPUT_DIR, fo ignores the HOME override for output and
    always writes total.json to the original ~/.find_orb/ — causing
    parallel workers to clobber each other.
    """
    real_fo_dir = Path.home() / '.find_orb'
    tmp_fo_dir = fo_home / '.find_orb'
    tmp_fo_dir.mkdir(parents=True, exist_ok=True)
    if real_fo_dir.is_dir():
        for src in real_fo_dir.iterdir():
            if src.is_file() and src.name not in _FO_SKIP_FILES:
                shutil.copy2(src, tmp_fo_dir / src.name)
    # Tell fo to write output here, not to the real ~/.find_orb/
    with open(tmp_fo_dir / 'environ.dat', 'a') as f:
        f.write(f'\nOUTPUT_DIR={tmp_fo_dir}/\n')


def refit_neo_batch(
    packed_list: List[str],
    mpcat_index,
    cache_dir: Optional[Path],
    obs_epoch_mjd: float,
    epoch_window_days: float = 5.0,
    max_obs: int = 2000,
    timeout_sec: int = 60,
    progress: bool = False,
    n_workers: int = 1,
) -> Dict[str, np.ndarray]:
    """
    Run fo orbit refits for a list of packed designations.

    Each object is refit using only MPCAT observations up to the epoch bin
    boundary, so the resulting element epoch is close to (and never beyond)
    the observation epoch.  Results are cached per (packed, epoch_bin) pair;
    subsequent calls at the same epoch bin return instantly.

    Parameters
    ----------
    packed_list       : list of MPC packed designations to refit
    mpcat_index       : MPCATIndex instance
    cache_dir         : directory for fo JSON cache files (FO_CACHE_DIR)
    obs_epoch_mjd     : representative observation epoch (UTC MJD); used to
                        define the epoch bin boundary (max_date_mjd for fo)
    epoch_window_days : cache bin width in days; fits are shared across all
                        observations within the same bin (default: 5)
    max_obs           : cap on observations passed to fo per object
    timeout_sec       : per-object fo timeout
    progress          : if True, show a tqdm progress bar (falls back to
                        log.info every 100 objects if tqdm is unavailable)
    n_workers         : parallel fo processes (default: 1 = serial).  Each
                        worker gets a private MPCATIndex and HOME directory so
                        fo output files do not collide.

    Returns
    -------
    Dict mapping packed designation → 1-element ASTEROID_DTYPE array for
    successful refits.  Objects with no MPCAT observations or fo failures
    are absent from the dict.
    """
    if not _fo_available():
        log.debug('fo not found; skipping batch refit')
        return {}

    # Round obs epoch to nearest epoch_window_days boundary
    epoch_bin = round(obs_epoch_mjd / epoch_window_days) * epoch_window_days

    refits: Dict[str, np.ndarray] = {}
    n_fit = 0
    n_fail = 0

    if n_workers > 1:
        mpcat_dir = getattr(mpcat_index, 'mpcat_dir', None)
        if mpcat_dir is None:
            log.warning('mpcat_index has no mpcat_dir attribute; '
                        'falling back to serial fo refit')
            n_workers = 1

    def _progress_wrap(iterable, total, desc):
        """Wrap iterable with tqdm if available and progress=True."""
        if not progress:
            return iterable
        try:
            from tqdm import tqdm
            return tqdm(iterable, total=total, desc=desc, unit='obj',
                        dynamic_ncols=True)
        except ImportError:
            return iterable

    if n_workers > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        worker_args = [
            (packed, mpcat_dir, cache_dir, max_obs, timeout_sec, epoch_bin)
            for packed in packed_list
        ]
        log.info('fo refit batch: %d objects, %d workers', len(packed_list), n_workers)

        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_refit_worker, arg): arg[0]
                       for arg in worker_args}
            completed = 0
            for future in _progress_wrap(as_completed(futures),
                                         total=len(futures),
                                         desc='fo refit'):
                pk = futures[future]
                try:
                    pk, arr = future.result()
                    if arr is not None:
                        refits[pk] = arr
                        n_fit += 1
                    else:
                        n_fail += 1
                except Exception as exc:
                    log.warning('fo worker failed for %s: %s', pk, exc)
                    n_fail += 1
                completed += 1
                if progress and completed % 100 == 0:
                    log.info('fo refit: %d/%d done (%d ok, %d fail)',
                             completed, len(packed_list), n_fit, n_fail)

    else:
        for packed in _progress_wrap(packed_list,
                                     total=len(packed_list),
                                     desc='fo refit'):
            arr = refit_neo(
                packed, mpcat_index,
                cache_dir=cache_dir,
                max_obs=max_obs,
                timeout_sec=timeout_sec,
                max_date_mjd=epoch_bin,
            )
            if arr is not None:
                refits[packed] = arr
                n_fit += 1
            else:
                n_fail += 1

    log.info('fo refit batch done: %d/%d succeeded, %d failed',
             len(refits), len(packed_list), n_fail)
    return refits


def apply_fo_refits(
    asteroids: np.ndarray,
    refits: Dict[str, np.ndarray],
) -> np.ndarray:
    """
    Replace orbital elements in the asteroid array with fo-refitted values.

    Parameters
    ----------
    asteroids : structured array from mpcorb.load_mpcorb()
    refits    : dict from refit_neo_batch() — packed → 1-element array

    Returns
    -------
    Copy of asteroids with updated elements for all keys in refits.
    Objects not in refits are unchanged.  If refits is empty, returns
    asteroids unchanged (no copy).
    """
    if not refits:
        return asteroids

    result = asteroids.copy()

    # Build packed→row-index map once
    packed_to_idx: Dict[str, int] = {}
    for i, pk in enumerate(asteroids['packed']):
        packed_to_idx[pk] = i

    updated = 0
    for packed, fo_arr in refits.items():
        idx = packed_to_idx.get(packed)
        if idx is None:
            continue
        for col in ('a', 'e', 'i', 'Omega', 'omega', 'M', 'epoch', 'H', 'G'):
            result[col][idx] = fo_arr[col][0]
        updated += 1

    log.debug('apply_fo_refits: updated %d rows', updated)
    return result
