"""
Command-line interface for local mpchecker.

Usage examples:

  # Check an 80-column obs file
  mpchecker obs.txt

  # Check with custom search radius and magnitude limit
  mpchecker obs.txt --radius 15 --maglim 22

  # Check a single RA/Dec/epoch/obscode (no input file)
  mpchecker --ra 123.456 --dec -12.345 --epoch 2024-03-15 --obscode 568

  # Download / update orbit data before running
  mpchecker --download-data
  mpchecker --download-data --satellites-only

  # Pre-compute position cache for today's date (speeds up future runs)
  mpchecker --build-cache

  # Use two-body dynamics (faster, less accurate for NEOs)
  mpchecker obs.txt --dynmodel 2
"""

import argparse
import logging
import sys
from pathlib import Path

from .config import DATA_DIR, ORBS_DIR, CACHE_DIR


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def _ra_to_hms(ra_deg: float) -> str:
    ra_h = ra_deg / 15.0
    h = int(ra_h)
    m = int((ra_h - h) * 60)
    s = ((ra_h - h) * 60 - m) * 60
    return f'{h:02d} {m:02d} {s:06.3f}'


def _dec_to_dms(dec_deg: float) -> str:
    sign = '+' if dec_deg >= 0 else '-'
    dec_abs = abs(dec_deg)
    d = int(dec_abs)
    m = int((dec_abs - d) * 60)
    s = ((dec_abs - d) * 60 - m) * 60
    return f'{sign}{d:02d} {m:02d} {s:05.2f}'


def format_results(results, search_radius_arcmin: float) -> str:
    lines = []
    for res in results:
        obs = res.obs
        hdr = (
            f'\n--- Observation: {obs.designation or "unknown"} '
            f'| RA {_ra_to_hms(obs.ra_deg)}  '
            f'Dec {_dec_to_dms(obs.dec_deg)}  '
            f'Epoch {obs.epoch_mjd:.4f} MJD  '
            f'Obs {obs.obscode} ---'
        )
        lines.append(hdr)

        if not res.matches:
            lines.append('  No known objects found within '
                         f'{search_radius_arcmin:.0f} arcmin.')
            continue

        # Header row
        lines.append(
            f'  {"Name":<30} {"Type":<12} '
            f'{"RA":>15} {"Dec":>15} '
            f'{"Sep":>8} {"dRA":>8} {"dDec":>8} '
            f'{"r(AU)":>7} {"Δ(AU)":>7} {"V":>5} {"Ph°":>5}'
        )
        lines.append('  ' + '-'*120)

        for m in res.matches:
            lines.append(
                f'  {m.name:<30} {m.obj_type:<12} '
                f'{_ra_to_hms(m.ra_deg):>15} {_dec_to_dms(m.dec_deg):>15} '
                f'{m.sep_arcsec:>7.1f}" '
                f'{m.ra_rate:>+8.2f} {m.dec_rate:>+8.2f} '
                f'{m.r_helio:>7.3f} {m.delta:>7.3f} '
                f'{m.vmag:>5.1f} {m.phase_deg:>5.1f}'
            )

    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Data download
# ---------------------------------------------------------------------------

def download_data(satellites_only: bool = False,
                  asteroids_only: bool = False,
                  show_progress: bool = True) -> None:
    """Download orbit data files and SPICE kernels."""
    import requests
    from tqdm import tqdm

    ORBS_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def _dl(url: str, dest: Path, label: str = '') -> None:
        if dest.exists():
            print(f'  Already exists: {dest.name}')
            return
        print(f'  Downloading {label or dest.name} …')
        r = requests.get(url, stream=True, timeout=120)
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
        print(f'  Saved: {dest}')

    from .config import (URL_MPCORB, URL_COMETS, URL_OBSCODES,
                         MPCORB_GZ, COMET_FILE, OBSCODE_FILE,
                         SPICE_DIR, SPICE_URLS, SPICE_LSK, SPICE_DE,
                         SPICE_PCK, SPICE_MAR, SPICE_JUP, SPICE_JUP2, SPICE_JUP3,
                         SPICE_SAT, SPICE_URA, SPICE_NEP, SPICE_PLU)

    if not asteroids_only:
        # SPICE kernels
        SPICE_DIR.mkdir(parents=True, exist_ok=True)
        print('\n[SPICE kernels]')
        _dl(SPICE_URLS['lsk'], SPICE_LSK)
        _dl(SPICE_URLS['de'],  SPICE_DE,  'Planetary ephemeris DE440s (~100 MB)')
        _dl(SPICE_URLS['pck'], SPICE_PCK)
        if not satellites_only:
            pass  # satellite kernels downloaded on demand by satellites.py

        print('\n[Satellite SPK kernels]')
        sat_map = {'mar': SPICE_MAR,
                   'jup': SPICE_JUP, 'jup2': SPICE_JUP2, 'jup3': SPICE_JUP3,
                   'sat': SPICE_SAT, 'ura':  SPICE_URA,
                   'nep': SPICE_NEP, 'plu':  SPICE_PLU}
        for key, path in sat_map.items():
            _dl(SPICE_URLS[key], path, f'{key.upper()} satellites')

    if not satellites_only:
        print('\n[MPC orbit catalogs]')
        _dl(URL_MPCORB, MPCORB_GZ,   'MPCORB.DAT.gz (~200 MB)')
        _dl(URL_COMETS, COMET_FILE,   'AllCometEls.txt')
        _dl(URL_OBSCODES, OBSCODE_FILE, 'ObsCodes')

    print('\nDownload complete.')


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        prog='mpchecker',
        description=(
            'Local minor planet / planetary satellite checker.\n'
            'Replicates MPC MPChecker with local orbit data and pyoorb propagation.'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Input
    parser.add_argument('obsfile', nargs='?',
                        help='MPC 80-column observation file (or - for stdin)')
    parser.add_argument('--ra',      type=float, metavar='DEG',
                        help='RA of target (degrees J2000)')
    parser.add_argument('--dec',     type=float, metavar='DEG',
                        help='Dec of target (degrees J2000)')
    parser.add_argument('--epoch',   metavar='ISO_OR_MJD',
                        help='Epoch of target (ISO date or MJD UTC)')
    parser.add_argument('--obscode', default='500',
                        help='MPC observatory code (default: 500 = geocenter)')

    # Search parameters
    parser.add_argument('--radius', type=float, default=30.0, metavar='ARCMIN',
                        help='Search radius in arcmin (default: 30)')
    parser.add_argument('--maglim', type=float, default=25.0, metavar='MAG',
                        help='Faint magnitude limit (default: 25)')

    # Dynamics
    parser.add_argument('--dynmodel', choices=['N', '2'], default='2',
                        help='pyoorb dynamics: 2=two-body (default, fast), N=N-body (precise)')
    parser.add_argument('--workers', type=int, default=1, metavar='N',
                        help='Parallel workers for Keplerian pre-filter phase (default: 1; '
                             'try 8 on HPC nodes for large obs batches)')

    # Object types
    parser.add_argument('--no-satellites', action='store_true',
                        help='Skip planetary satellite check')
    parser.add_argument('--no-comets', action='store_true',
                        help='Skip comet check')
    parser.add_argument('--no-asteroids', action='store_true',
                        help='Skip asteroid check')

    # Daemon management
    parser.add_argument('--serve', action='store_true',
                        help='Run as a foreground daemon server (keeps catalog in memory)')
    parser.add_argument('--start-daemon', action='store_true',
                        help='Start daemon server in the background')
    parser.add_argument('--stop-daemon', action='store_true',
                        help='Stop a running daemon server')
    parser.add_argument('--no-daemon', action='store_true',
                        help='Skip daemon, run standalone even if daemon is available')

    # Data management
    parser.add_argument('--download-data', action='store_true',
                        help='Download/update orbit data and SPICE kernels')
    parser.add_argument('--build-index', action='store_true',
                        help='Build (or rebuild) the sky position index for fast lookup')
    parser.add_argument('--satellites-only', action='store_true',
                        help='(with --download-data) only download satellite data')
    parser.add_argument('--asteroids-only', action='store_true',
                        help='(with --download-data) only download asteroid data')
    parser.add_argument('--data-dir', type=Path, metavar='DIR',
                        help='Override data directory (default: $MPCHECKER_DATA)')

    # Misc
    parser.add_argument('--dedup', action='store_true',
                        help='Only check the first observation per object designation '
                             '(useful when a file contains many epochs of the same object)')
    parser.add_argument('--output', '-o', type=Path,
                        help='Write results to file instead of stdout')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose logging')
    parser.add_argument('--debug', action='store_true',
                        help='Debug logging')

    args = parser.parse_args(argv)

    # Logging
    level = logging.DEBUG if args.debug else (
            logging.INFO  if args.verbose else logging.WARNING)
    logging.basicConfig(level=level, format='%(levelname)s: %(message)s')

    # --serve / --start-daemon / --stop-daemon modes
    if args.serve:
        from .daemon import serve
        serve(mag_limit=args.maglim, n_workers=args.workers)
        return 0

    if args.start_daemon:
        from .daemon import start_daemon_background, is_daemon_running
        if is_daemon_running():
            print('Daemon is already running.')
            return 0
        print('Starting daemon …')
        pid = start_daemon_background(mag_limit=args.maglim, n_workers=args.workers)
        from .daemon import _log_path
        import time
        for _ in range(40):          # wait up to 20 s for ready message
            time.sleep(0.5)
            if is_daemon_running():
                print(f'Daemon ready (pid={pid}). Log: {_log_path()}')
                return 0
        print(f'WARNING: daemon did not respond within 20 s. '
              f'Check {_log_path()} for errors.', file=sys.stderr)
        return 1

    if args.stop_daemon:
        from .daemon import stop_daemon
        if stop_daemon():
            print('Daemon stopped.')
        else:
            print('No daemon found (or already stopped).')
        return 0

    # Override data dir if requested
    if args.data_dir:
        import mpchecker.config as cfg
        cfg.DATA_DIR    = args.data_dir
        cfg.ORBS_DIR    = args.data_dir / 'orbits'
        cfg.SPICE_DIR   = args.data_dir / 'spice'
        cfg.CACHE_DIR   = args.data_dir / 'cache'
        cfg.MPCORB_FILE = cfg.ORBS_DIR / 'MPCORB.DAT'
        cfg.MPCORB_GZ   = cfg.ORBS_DIR / 'MPCORB.DAT.gz'
        cfg.COMET_FILE  = cfg.ORBS_DIR / 'AllCometEls.txt'
        cfg.OBSCODE_FILE = cfg.ORBS_DIR / 'ObsCodes.txt'

    # --build-index mode
    if args.build_index:
        from .mpcorb import load_mpcorb, load_obscodes
        from .index import get_or_build_index
        from .config import CACHE_DIR
        import numpy as np
        print('Loading asteroid catalog …', file=sys.stderr)
        asts = load_mpcorb(H_limit=_h_limit(args.maglim))
        obscodes_bi = load_obscodes()
        print(f'Building sky index for {len(asts)} objects …', file=sys.stderr)
        idx = get_or_build_index(asts, obscodes_bi, CACHE_DIR, force_rebuild=True)
        print(f'Index built (ref MJD {idx.t_ref_mjd:.1f}, valid {idx.validity_days:.0f} days).',
              file=sys.stderr)
        return 0

    # --download-data mode
    if args.download_data:
        download_data(
            satellites_only=args.satellites_only,
            asteroids_only=args.asteroids_only,
        )
        return 0

    # Build observations
    from .obs_parser import parse_observations, parse_file, Observation

    observations = []

    if args.obsfile:
        if args.obsfile == '-':
            text = sys.stdin.read()
            observations = parse_observations(text)
        else:
            observations = parse_file(args.obsfile)
    elif args.ra is not None and args.dec is not None and args.epoch is not None:
        # Single RA/Dec/epoch check — synthesize a dummy observation
        from .obs_parser import Observation as Obs
        epoch_mjd = _parse_epoch(args.epoch)
        obs = Obs(
            line='',
            designation='',
            packed_desig='',
            discovery=False,
            note1='',
            note2='',
            epoch_mjd=epoch_mjd,
            ra_deg=args.ra,
            dec_deg=args.dec,
            mag=None,
            band=None,
            obscode=args.obscode,
            obj_type='minor_planet',
        )
        observations = [obs]
    else:
        parser.print_help()
        return 1

    if not observations:
        print('No valid observations found in input.', file=sys.stderr)
        return 1

    # --dedup: keep only the first observation per packed designation
    if args.dedup:
        seen = set()
        deduped = []
        for obs in observations:
            if obs.designation not in seen:
                seen.add(obs.designation)
                deduped.append(obs)
        if len(deduped) < len(observations):
            print(f'Dedup: {len(observations)} → {len(deduped)} observations '
                  f'({len(observations)-len(deduped)} duplicates removed)',
                  file=sys.stderr)
        observations = deduped

    print(f'Checking {len(observations)} observation(s) …', file=sys.stderr)

    # Try daemon first (skip if --no-daemon or daemon not running)
    if not args.no_daemon:
        from .daemon import query_daemon, is_daemon_running
        if is_daemon_running():
            daemon_params = {
                'search_radius_arcmin': args.radius,
                'mag_limit':            args.maglim,
                'dynmodel':             args.dynmodel,
                'check_sats':           not args.no_satellites,
                'n_workers':            args.workers,
            }
            results = query_daemon(observations, daemon_params)
            if results is not None:
                output = format_results(results, args.radius)
                if args.output:
                    with open(args.output, 'w') as f:
                        f.write(output + '\n')
                    print(f'Results written to {args.output}', file=sys.stderr)
                else:
                    print(output)
                return 0
            # Daemon unreachable — fall through to standalone
            print('Daemon unreachable, running standalone.', file=sys.stderr)

    # Load catalogs
    import numpy as np
    from .mpcorb import (load_mpcorb, load_comets, load_obscodes,
                         ASTEROID_DTYPE, COMET_DTYPE)

    obscodes = load_obscodes()

    asteroids = np.zeros(0, dtype=ASTEROID_DTYPE)
    if not args.no_asteroids:
        try:
            asteroids = load_mpcorb(H_limit=_h_limit(args.maglim))
        except FileNotFoundError as e:
            print(f'WARNING: {e}', file=sys.stderr)
            print('Run "mpchecker --download-data" to fetch orbit catalogs.',
                  file=sys.stderr)

    comets = np.zeros(0, dtype=COMET_DTYPE)
    if not args.no_comets:
        try:
            comets = load_comets()
        except Exception:
            pass

    # Try to load a cached sky index for fast candidate lookup
    sky_index = None
    if not args.no_asteroids and len(asteroids) > 0:
        try:
            from .index import get_or_build_index
            from .config import CACHE_DIR
            t_rep = observations[0].epoch_mjd if observations else None
            sky_index = get_or_build_index(
                asteroids, obscodes, CACHE_DIR, t_now_mjd=t_rep)
        except Exception as exc:
            logging.getLogger(__name__).debug('Sky index unavailable: %s', exc)

    # Run checks
    from .checker import check_observations
    results = check_observations(
        observations,
        asteroids,
        comets,
        obscodes,
        search_radius_arcmin=args.radius,
        mag_limit=args.maglim,
        dynmodel=args.dynmodel,
        check_sats=(not args.no_satellites),
        sky_index=sky_index,
        n_workers=args.workers,
    )

    # Format and output
    output = format_results(results, args.radius)

    if args.output:
        with open(args.output, 'w') as f:
            f.write(output + '\n')
        print(f'Results written to {args.output}', file=sys.stderr)
    else:
        print(output)

    return 0


def _parse_epoch(s: str) -> float:
    """Parse ISO date or MJD number to MJD UTC."""
    try:
        return float(s)
    except ValueError:
        from astropy.time import Time
        return Time(s, scale='utc').mjd


def _h_limit(maglim: float) -> float:
    """Rough H limit from V magnitude limit (assume r~2, delta~1.5 AU)."""
    import numpy as np
    return maglim - 5*np.log10(2.0 * 1.5) + 1.5


if __name__ == '__main__':
    sys.exit(main())
