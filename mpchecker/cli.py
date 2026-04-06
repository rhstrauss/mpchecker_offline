"""
Command-line interface for local mpchecker.

Usage examples:

  # Check an 80-column obs file (format auto-detected)
  mpchecker obs.txt

  # Check an ADES PSV file
  mpchecker observations.psv --format ades

  # Check a HelioLinC detection CSV (parseout.csv)
  mpchecker parseout.csv --format hldet

  # Check multiple files at once (globs supported)
  mpchecker night1.txt night2.txt night3.txt
  mpchecker obs_*.txt
  mpchecker data/**/*.csv --format hldet

  # Check with custom search radius and magnitude limit
  mpchecker obs.txt --radius 15 --maglim 22

  # Check a single RA/Dec/epoch/obscode (no input file)
  mpchecker --ra 123.456 --dec -12.345 --epoch 2024-03-15 --obscode 568

  # Machine-readable JSON output
  mpchecker obs.txt --json

  # Download / update orbit data before running
  mpchecker --download-data
  mpchecker --download-data --satellites-only

  # Pre-compute position cache for today's date (speeds up future runs)
  mpchecker --build-cache

  # Use two-body dynamics (faster, less accurate for NEOs)
  mpchecker obs.txt --dynmodel 2
"""

import argparse
import glob
import json as _json_mod
import logging
import sys
from pathlib import Path

import numpy as np

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


def _ident_display_name(ident) -> str:
    """Return the best human-readable name for an Identification."""
    if ident.method == 'orbit_fit':
        # fo's element-derived catalog match is more reliable than the Phase 2
        # anchor, which can be a coincidental ephemeris hit near a close-approach
        # tracklet.  Prefer fo_catalog_name when available.
        if ident.fo_catalog_name:
            return ident.fo_catalog_name
        if ident.match is not None:
            return ident.match.name
        return 'unknown'
    # ephemeris method: Phase 2 match is definitive
    m = ident.match
    if m is not None:
        return m.name
    if ident.fo_catalog_name:
        return ident.fo_catalog_name
    return 'unknown'


def _ident_status(ident, obs_list) -> str:
    """Classify an Identification for summary display."""
    if ident.method == 'ephemeris':
        return 'IDENTIFIED'
    # orbit_fit method
    if ident.fo_catalog_name or (ident.match is not None):
        return 'IDENTIFIED'
    # No catalog match — evaluate if likely a real new object
    n_nights = len(set(int(o.epoch_mjd) for o in obs_list))
    n_fit = ident.fo_n_obs or 0
    rms = ident.rms_arcsec
    if n_fit >= 4 and n_nights >= 2 and np.isfinite(rms) and rms < 5.0:
        return 'CANDIDATE_NEW'
    return 'UNRESOLVED'


def format_identifications(identifications, observations,
                           designation: str = '') -> str:
    """
    Format tracklet identification results for console output.

    Parameters
    ----------
    identifications : list of Identification objects from identify_tracklet()
    observations    : the input observation list for this tracklet
    designation     : cluster/tracklet label to show in the header (optional)
    """
    if not identifications:
        return ''
    label = f' [{designation}]' if designation else ''
    lines = [f'\n{"="*60}',
             f'Tracklet identification{label} ({len(observations)} obs):',
             f'{"="*60}']
    fo_rms_note = False
    for rank, ident in enumerate(identifications, 1):
        m = ident.match
        method_label = {
            'ephemeris':  'catalog orbit (pyoorb)',
            'orbit_fit':  'fo orbit fit',
        }.get(ident.method, ident.method)

        display_name = _ident_display_name(ident)

        # For orbit_fit: rms_arcsec is fo's own internal RMS (reliable).
        # Flag with † to distinguish from pyoorb O-C residuals.
        if ident.method == 'orbit_fit' and ident.fo_rms_internal is not None:
            rms_str = f'{ident.rms_arcsec:.2f}"†'
            fo_rms_note = True
        else:
            rms_str = f'{ident.rms_arcsec:.2f}"'

        lines.append(
            f'\n [{rank}] {display_name}  [{method_label}]'
            f'  RMS = {rms_str}'
        )
        # Show individual residuals (truncate if very many observations)
        resids = ident.residuals
        finite_resids = [r for r in resids if np.isfinite(r)]
        if finite_resids:
            if len(resids) <= 12:
                resid_str = '  '.join(
                    f'{r:.2f}"' if np.isfinite(r) else '  —  ' for r in resids)
            else:
                first6 = '  '.join(
                    f'{r:.2f}"' if np.isfinite(r) else '—' for r in resids[:6])
                last2 = '  '.join(
                    f'{r:.2f}"' if np.isfinite(r) else '—' for r in resids[-2:])
                resid_str = f'{first6}  …  {last2}  (n={len(resids)})'
            lines.append(f'     O-C per obs (pyoorb): {resid_str}')
        if ident.method == 'ephemeris' and m is not None:
            uq = f'  U={m.orbit_quality}' if m.orbit_quality else ''
            lines.append(
                f'     Catalog: V={m.vmag:.1f}  sep={m.sep_arcsec:.1f}"  '
                f'r={m.r_helio:.3f} AU  Δ={m.delta:.3f} AU  phase={m.phase_deg:.1f}°'
                f'{uq}'
            )
        if ident.method == 'orbit_fit' and ident.fo_elements is not None:
            el = ident.fo_elements
            moid_str = (f'  Earth MOID={ident.fo_earth_moid_au:.4f} AU'
                        if ident.fo_earth_moid_au is not None
                           and np.isfinite(ident.fo_earth_moid_au) else '')
            lines.append(
                f'     fo elements: a={float(el["a"][0]):.4f} AU  '
                f'e={float(el["e"][0]):.4f}  '
                f'i={float(el["i"][0]):.2f}°  '
                f'epoch MJD {float(el["epoch"][0]):.1f}'
                f'{moid_str}'
            )
            if ident.fo_catalog_name and ident.fo_catalog_score is not None:
                lines.append(
                    f'     Catalog match (Δ-elements): {ident.fo_catalog_name}'
                    f'  (score={ident.fo_catalog_score:.4f})'
                )
    if fo_rms_note:
        lines.append('     † fo internal residual (reliable for close-approach objects)')
    lines.append('')
    return '\n'.join(lines)


def format_identification_summary(
    all_idents_by_desig: dict,
    obs_by_desig: dict,
) -> str:
    """
    One-line-per-cluster summary table for batch --identify runs.

    Parameters
    ----------
    all_idents_by_desig : dict mapping designation → list[Identification]
    obs_by_desig        : dict mapping designation → list[Observation]
    """
    if not all_idents_by_desig and not obs_by_desig:
        return ''

    n_total = len(obs_by_desig)
    lines = [
        f'\n{"="*70}',
        f'Identification Summary ({n_total} cluster{"s" if n_total != 1 else ""}):',
        f'{"="*70}',
        f'  {"Cluster":<12} {"Obs":>4}  {"Matched":<28} {"Method":<10} {"RMS":>8}',
        f'  {"-"*12} {"-"*4}  {"-"*28} {"-"*10} {"-"*8}',
    ]
    fo_rms_any = False
    for desig in sorted(obs_by_desig.keys()):
        obs_list  = obs_by_desig[desig]
        n_obs_d   = len(obs_list)
        idents    = all_idents_by_desig.get(desig, [])

        if not idents:
            lines.append(f'  {desig:<12} {n_obs_d:>4}  {"—":<28} {"—":<10} {"—":>8}')
            continue

        best = idents[0]
        status = _ident_status(best, obs_list)
        name   = _ident_display_name(best)

        if status == 'CANDIDATE_NEW':
            el = best.fo_elements
            if el is not None:
                a_str = f'a={float(el["a"][0]):.3f}'
                name  = f'CANDIDATE NEW ({a_str})'
            else:
                name = 'CANDIDATE NEW'
        elif status == 'UNRESOLVED':
            name = 'UNRESOLVED'

        method_str = {
            'ephemeris': 'ephemeris',
            'orbit_fit': 'fo_fit',
        }.get(best.method, best.method)

        if best.method == 'orbit_fit' and best.fo_rms_internal is not None:
            rms_str = f'{best.rms_arcsec:.1f}"†'
            fo_rms_any = True
        else:
            rms_str = f'{best.rms_arcsec:.1f}"'

        lines.append(
            f'  {desig:<12} {n_obs_d:>4}  {name:<28} {method_str:<10} {rms_str:>8}'
        )

    if fo_rms_any:
        lines.append('  † fo internal residual (not pyoorb re-evaluation)')
    lines.append(f'{"="*70}')
    return '\n'.join(lines)


def format_results_csv(results, search_radius_arcmin: float) -> str:
    """
    Output matches as CSV (one row per observation × match).

    Columns: designation, epoch_mjd, obs_ra_deg, obs_dec_deg, obscode,
             name, packed, obj_type, sep_arcsec,
             ra_rate_arcsec_hr, dec_rate_arcsec_hr,
             match_ra_deg, match_dec_deg, r_au, delta_au, vmag,
             phase_deg, orbit_quality
    """
    import csv, io
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow([
        'designation', 'epoch_mjd', 'obs_ra_deg', 'obs_dec_deg', 'obscode',
        'name', 'packed', 'obj_type',
        'sep_arcsec', 'ra_rate_arcsec_hr', 'dec_rate_arcsec_hr',
        'match_ra_deg', 'match_dec_deg',
        'r_au', 'delta_au', 'vmag', 'phase_deg', 'orbit_quality',
    ])
    for res in results:
        obs = res.obs
        for m in res.matches:
            w.writerow([
                obs.designation or '',
                round(obs.epoch_mjd, 6),
                round(obs.ra_deg, 6),
                round(obs.dec_deg, 6),
                obs.obscode,
                m.name, m.packed, m.obj_type,
                round(m.sep_arcsec, 2),
                round(m.ra_rate, 3),
                round(m.dec_rate, 3),
                round(m.ra_deg, 6),
                round(m.dec_deg, 6),
                round(m.r_helio, 5),
                round(m.delta, 5),
                round(m.vmag, 2),
                round(m.phase_deg, 2),
                m.orbit_quality,
            ])
    return buf.getvalue()


def format_identifications_csv(
    idents_by_desig: dict,
    obs_by_desig: dict,
) -> str:
    """
    Output identification results as CSV (one row per cluster).

    Columns: designation, n_obs, n_nights, status, method, match_name,
             rms_arcsec, fo_rms_arcsec, fo_n_obs, fo_earth_moid_au,
             fo_a, fo_e, fo_i, fo_epoch_mjd,
             catalog_match, catalog_match_score
    """
    import csv, io
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow([
        'designation', 'n_obs', 'n_nights', 'status', 'method',
        'match_name', 'rms_arcsec', 'fo_rms_arcsec', 'fo_n_obs',
        'fo_earth_moid_au', 'fo_a', 'fo_e', 'fo_i', 'fo_epoch_mjd',
        'catalog_match', 'catalog_match_score',
    ])
    for desig in sorted(obs_by_desig.keys()):
        obs_list = obs_by_desig[desig]
        n_obs_d  = len(obs_list)
        n_nights = len(set(int(o.epoch_mjd) for o in obs_list))
        idents   = idents_by_desig.get(desig, [])

        if not idents:
            w.writerow([desig, n_obs_d, n_nights,
                        'NO_CANDIDATES', '', '', '', '', '', '',
                        '', '', '', '', '', ''])
            continue

        best   = idents[0]
        status = _ident_status(best, obs_list)
        name   = _ident_display_name(best)
        if name == 'unknown':
            name = ''

        rms    = (round(best.rms_arcsec, 3)
                  if np.isfinite(best.rms_arcsec) else '')
        fo_rms = (round(best.fo_rms_internal, 3)
                  if best.fo_rms_internal is not None
                  and np.isfinite(best.fo_rms_internal) else '')
        fo_n   = best.fo_n_obs or ''
        fo_moid = (round(best.fo_earth_moid_au, 5)
                   if best.fo_earth_moid_au is not None
                   and np.isfinite(best.fo_earth_moid_au) else '')

        fo_a = fo_e = fo_i = fo_ep = ''
        if best.fo_elements is not None:
            el   = best.fo_elements
            fo_a = round(float(el['a'][0]),     6)
            fo_e = round(float(el['e'][0]),     6)
            fo_i = round(float(el['i'][0]),     4)
            fo_ep = round(float(el['epoch'][0]), 2)

        cat_match = best.fo_catalog_name or ''
        cat_score = (round(best.fo_catalog_score, 5)
                     if best.fo_catalog_score is not None else '')

        w.writerow([
            desig, n_obs_d, n_nights, status, best.method,
            name, rms, fo_rms, fo_n, fo_moid,
            fo_a, fo_e, fo_i, fo_ep,
            cat_match, cat_score,
        ])
    return buf.getvalue()


def format_results_json(
    results,
    search_radius_arcmin: float,
    identifications_by_desig: dict = None,
    obs_by_desig: dict = None,
) -> dict:
    """
    Produce a JSON-serialisable dict of all results.

    Suitable for machine-readable output via --json.  Structure:
      meta          — run metadata and summary counts
      observations  — list of per-observation dicts, each with a 'matches' list
      identifications — per-cluster identification results (when --identify used)
    """
    import time as _time

    n_obs          = len(results)
    n_with_matches = sum(1 for r in results if r.matches)
    epoch_mjds     = [r.obs.epoch_mjd for r in results]
    median_epoch   = float(np.median(epoch_mjds)) if epoch_mjds else None

    out: dict = {
        'meta': {
            'generated_mjd':      median_epoch,
            'search_radius_arcmin': search_radius_arcmin,
            'n_observations':      n_obs,
            'n_with_matches':      n_with_matches,
        },
        'observations': [],
    }

    for res in results:
        obs = res.obs
        obs_dict = {
            'designation': obs.designation or '',
            'epoch_mjd':   round(obs.epoch_mjd, 6),
            'ra_deg':      round(obs.ra_deg, 6),
            'dec_deg':     round(obs.dec_deg, 6),
            'obscode':     obs.obscode,
            'matches': [],
        }
        if obs.mag is not None:
            obs_dict['obs_mag']  = round(float(obs.mag), 2)
            obs_dict['obs_band'] = obs.band or ''
        for m in res.matches:
            mdict = {
                'name':               m.name,
                'packed':             m.packed,
                'obj_type':           m.obj_type,
                'sep_arcsec':         round(m.sep_arcsec, 2),
                'ra_deg':             round(m.ra_deg, 6),
                'dec_deg':            round(m.dec_deg, 6),
                'ra_rate_arcsec_hr':  round(m.ra_rate, 3),
                'dec_rate_arcsec_hr': round(m.dec_rate, 3),
                'r_au':               round(m.r_helio, 5),
                'delta_au':           round(m.delta, 5),
                'vmag':               round(m.vmag, 2),
                'phase_deg':          round(m.phase_deg, 2),
            }
            if m.orbit_quality:
                mdict['orbit_quality'] = m.orbit_quality
            obs_dict['matches'].append(mdict)
        out['observations'].append(obs_dict)

    # Identifications section
    if identifications_by_desig:
        idents_list = []
        for desig, idents in identifications_by_desig.items():
            if not idents:
                continue
            grp_obs = (obs_by_desig or {}).get(desig, [])
            for ident in idents:
                status = _ident_status(ident, grp_obs)
                idict: dict = {
                    'designation': desig,
                    'n_obs':       ident.n_obs,
                    'method':      ident.method,
                    'status':      status,
                    'rms_arcsec':  round(ident.rms_arcsec, 3)
                                   if np.isfinite(ident.rms_arcsec) else None,
                }
                # Match name
                name = _ident_display_name(ident)
                idict['match_name'] = name if name != 'unknown' else None

                if ident.method == 'ephemeris':
                    # Per-obs residuals
                    idict['residuals_arcsec'] = [
                        round(r, 2) if np.isfinite(r) else None
                        for r in ident.residuals]
                    if ident.match:
                        idict['catalog_sep_arcsec'] = round(ident.match.sep_arcsec, 2)
                        idict['vmag']               = round(ident.match.vmag, 2)
                        idict['delta_au']           = round(ident.match.delta, 5)
                        if ident.match.orbit_quality:
                            idict['orbit_quality']  = ident.match.orbit_quality

                if ident.method == 'orbit_fit':
                    if ident.fo_rms_internal is not None and np.isfinite(ident.fo_rms_internal):
                        idict['fo_rms_arcsec'] = round(ident.fo_rms_internal, 3)
                    if ident.fo_n_obs is not None:
                        idict['fo_n_obs']      = ident.fo_n_obs
                    if (ident.fo_earth_moid_au is not None
                            and np.isfinite(ident.fo_earth_moid_au)):
                        idict['fo_earth_moid_au'] = round(ident.fo_earth_moid_au, 5)
                    if ident.fo_catalog_name:
                        idict['catalog_match']       = ident.fo_catalog_name
                        idict['catalog_match_score'] = (
                            round(ident.fo_catalog_score, 5)
                            if ident.fo_catalog_score is not None else None)
                    if ident.fo_elements is not None:
                        el = ident.fo_elements
                        idict['fo_elements'] = {
                            'a':        round(float(el['a'][0]),     6),
                            'e':        round(float(el['e'][0]),     6),
                            'i':        round(float(el['i'][0]),     4),
                            'Omega':    round(float(el['Omega'][0]), 4),
                            'omega':    round(float(el['omega'][0]), 4),
                            'M':        round(float(el['M'][0]),     4),
                            'epoch_mjd': round(float(el['epoch'][0]), 2),
                            'H':        round(float(el['H'][0]),     2),
                        }
                    # pyoorb re-evaluation residuals (may be unreliable for close-approach)
                    finite_resids = [r for r in ident.residuals if np.isfinite(r)]
                    if finite_resids:
                        idict['pyoorb_residuals_arcsec'] = [
                            round(r, 2) if np.isfinite(r) else None
                            for r in ident.residuals]

                idents_list.append(idict)

        out['identifications'] = idents_list

    return out


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

        # Header row — U column shows MPC orbit quality code
        lines.append(
            f'  {"Name":<30} {"U":1}  {"Type":<12} '
            f'{"RA":>15} {"Dec":>15} '
            f'{"Sep":>8} {"dRA":>8} {"dDec":>8} '
            f'{"r(AU)":>7} {"Δ(AU)":>7} {"V":>5} {"Ph°":>5}'
        )
        lines.append('  ' + '-'*125)

        for m in res.matches:
            uq = m.orbit_quality if m.orbit_quality else '-'
            lines.append(
                f'  {m.name:<30} {uq:1}  {m.obj_type:<12} '
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
    parser.add_argument('obsfile', nargs='*',
                        help='Observation file(s) to check.  Accepts multiple files, '
                             'shell globs (e.g. "obs_*.txt"), and - for stdin.')
    parser.add_argument('--format', '-f', dest='fmt',
                        choices=['auto', 'mpc80', 'ades', 'hldet'],
                        default='auto',
                        help='Input format: auto (default), mpc80 (MPC 80-column), '
                             'ades (ADES PSV), hldet (HelioLinC detection CSV)')
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
    parser.add_argument('--dynmodel', choices=['N', '2'], default='N',
                        help='pyoorb dynamics: N=N-body (default, precise), 2=two-body (faster)')
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

    # fo orbit refinement
    parser.add_argument('--mpcat-dir', type=Path, metavar='DIR',
                        help='Directory containing NumObs.txt / UnnObs.txt and '
                             'their .idx.npy index files.  When provided, all objects '
                             'with perihelion q < --fo-refit-q are refit via fo before '
                             'Phase 1, ensuring accurate positions near close approaches.')
    parser.add_argument('--fo-refit-q', type=float, default=1.1, metavar='AU',
                        help='Perihelion threshold for pre-Phase 1 fo orbit refit '
                             '(default: 1.1 AU; covers all Atiras/Atens/Apollos/inner Amors)')
    parser.add_argument('--fo-epoch-window', type=float, default=5.0, metavar='DAYS',
                        help='Cache bin width in days for fo refits (default: 5; '
                             'fits are shared across observations within the same bin)')

    # Tracklet identification
    parser.add_argument('--identify', action='store_true',
                        help='After finding matches, test which candidate fits ALL '
                             'input observations as a single physical orbit (O-C residuals '
                             'via pyoorb for each candidate that appears in every observation)')
    parser.add_argument('--id-threshold', type=float, default=2.0, metavar='ARCSEC',
                        help='Maximum O-C residual (arcsec) for any single observation '
                             'to count as an identification (default: 2.0)')
    parser.add_argument('--id-min-obs', type=int, default=None, metavar='N',
                        help='Minimum number of input observations a candidate must appear '
                             'in to be tested for identification (default: all observations)')
    parser.add_argument('--fo-fit', action='store_true',
                        help='(with --identify) additionally run find_orb on the input '
                             'observations to independently fit a brute-force orbit and '
                             'confirm physical self-consistency')

    # Misc
    parser.add_argument('--dedup', action='store_true',
                        help='Only check the first observation per object designation '
                             '(useful when a file contains many epochs of the same object)')
    parser.add_argument('--json', action='store_true',
                        help='Output results as JSON (machine-readable) instead of text table. '
                             'When combined with --identify the JSON includes per-cluster '
                             'identification results and orbital elements.')
    parser.add_argument('--csv', action='store_true',
                        help='Output matches as CSV instead of text table '
                             '(one row per observation × match). '
                             'When combined with --identify, identification results are '
                             'appended as a second CSV section after a blank line.')
    parser.add_argument('--summary-only', action='store_true',
                        help='Suppress per-observation match table and (with --identify) '
                             'per-cluster identification detail. '
                             'Prints only a compact count summary, or (with --identify) '
                             'only the identification summary table. '
                             'Compatible with --json (drops observations array) '
                             'and --csv (drops matches rows).')
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
        serve(mag_limit=args.maglim, n_workers=args.workers,
              mpcat_dir=args.mpcat_dir)
        return 0

    if args.start_daemon:
        from .daemon import start_daemon_background, is_daemon_running
        if is_daemon_running():
            print('Daemon is already running.')
            return 0
        print('Starting daemon …')
        pid = start_daemon_background(mag_limit=args.maglim, n_workers=args.workers,
                                      mpcat_dir=args.mpcat_dir)
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
        cfg.FO_CACHE_DIR = cfg.CACHE_DIR / 'fo_fits'
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
    from .obs_parser import (parse_observations, parse_file, Observation,
                             parse_ades_psv, parse_ades_file,
                             parse_hldet, parse_hldet_file,
                             parse_auto, parse_file_auto)

    # Select parser based on --format
    _parsers = {
        'mpc80': (parse_observations, parse_file),
        'ades':  (parse_ades_psv,     parse_ades_file),
        'hldet': (parse_hldet,        parse_hldet_file),
        'auto':  (parse_auto,         parse_file_auto),
    }
    parse_text_fn, parse_file_fn = _parsers[args.fmt]

    observations = []

    if args.obsfile:
        # Expand globs and collect all file paths
        resolved_paths = []
        read_stdin = False
        for pattern in args.obsfile:
            if pattern == '-':
                read_stdin = True
            else:
                expanded = sorted(glob.glob(pattern))
                if not expanded:
                    # No glob match — treat as literal path (will error naturally)
                    resolved_paths.append(pattern)
                else:
                    resolved_paths.extend(expanded)

        if read_stdin:
            text = sys.stdin.read()
            observations.extend(parse_text_fn(text))

        for fpath in resolved_paths:
            try:
                observations.extend(parse_file_fn(fpath))
            except FileNotFoundError:
                print(f'WARNING: file not found: {fpath}', file=sys.stderr)
            except Exception as exc:
                print(f'WARNING: error reading {fpath}: {exc}', file=sys.stderr)

        if resolved_paths and not read_stdin:
            n_files = len(resolved_paths)
            if n_files > 1:
                print(f'Read {n_files} files, {len(observations)} observation(s) total.',
                      file=sys.stderr)
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

    # --dedup + --identify are incompatible: dedup collapses every tracklet to
    # 1 observation, making orbit identification impossible (need >= 2 obs).
    if args.dedup and args.identify:
        print(
            'WARNING: --identify and --dedup are incompatible. '
            '--dedup reduces each tracklet to 1 observation; '
            'identify_tracklet requires at least 2 observations per cluster.\n'
            'Remove --dedup to enable identification.',
            file=sys.stderr,
        )

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

    # Try daemon first (skip if --no-daemon, --identify, or daemon not running)
    # --identify requires catalog data loaded locally (can't delegate to daemon)
    if not args.no_daemon and not args.identify:
        from .daemon import query_daemon, is_daemon_running
        if is_daemon_running():
            daemon_params = {
                'search_radius_arcmin': args.radius,
                'mag_limit':            args.maglim,
                'dynmodel':             args.dynmodel,
                'check_sats':           not args.no_satellites,
                'n_workers':            args.workers,
                'fo_refit_q_threshold': args.fo_refit_q,
                'fo_epoch_window_days': args.fo_epoch_window,
            }
            results = query_daemon(observations, daemon_params)
            if results is not None:
                n_with_m = sum(1 for r in results if r.matches)
                tot_m    = sum(len(r.matches) for r in results)
                if args.json:
                    output = _json_mod.dumps(
                        format_results_json(results, args.radius), indent=2)
                    if args.summary_only:
                        d = _json_mod.loads(output)
                        d.pop('observations', None)
                        output = _json_mod.dumps(d, indent=2)
                elif args.csv:
                    output = ('' if args.summary_only
                              else format_results_csv(results, args.radius))
                elif args.summary_only:
                    output = (f'{len(results)} observation(s), '
                              f'{n_with_m} with matches, '
                              f'{tot_m} total match(es).')
                else:
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

    print(f'Loaded {len(asteroids)} asteroids, {len(comets)} comets.',
          file=sys.stderr)
    if len(asteroids) == 0 and not args.no_asteroids:
        print('WARNING: No asteroids loaded — run "mpchecker --download-data" '
              'or check --maglim setting.', file=sys.stderr)

    # Try to load a cached sky index for fast candidate lookup
    sky_index = None
    if not args.no_asteroids and len(asteroids) > 0:
        try:
            from .index import get_or_build_index
            from .config import CACHE_DIR
            epochs = [o.epoch_mjd for o in observations] if observations else []
            t_min = min(epochs) if epochs else None
            t_max = max(epochs) if epochs else None
            sky_index = get_or_build_index(
                asteroids, obscodes, CACHE_DIR,
                t_min_mjd=t_min, t_max_mjd=t_max)
        except Exception as exc:
            logging.getLogger(__name__).debug('Sky index unavailable: %s', exc)

    # Optionally load MPCAT index for fo orbit refinement
    mpcat_index = None
    if args.mpcat_dir:
        try:
            from .mpcat import MPCATIndex
            mpcat_index = MPCATIndex(args.mpcat_dir)
            print(f'MPCAT index loaded for fo orbit refinement.', file=sys.stderr)
        except Exception as exc:
            print(f'WARNING: could not load MPCAT index: {exc}', file=sys.stderr)

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
        mpcat_index=mpcat_index,
        fo_refit_q_threshold=args.fo_refit_q,
        fo_epoch_window_days=args.fo_epoch_window,
        fo_progress=(args.verbose or args.debug),
    )

    # Optional: tracklet identification (O-C residuals across all observations)
    identification_output = ''
    idents_by_desig:    dict = {}   # designation → list[Identification] (for JSON / summary)
    obs_grp_by_desig:   dict = {}   # designation → list[Observation]

    if args.identify:
        from .checker import identify_tracklet
        from collections import defaultdict as _dd

        # Group by designation so each cluster is identified independently.
        # If there's only one designation (or all are blank), treat all obs as
        # a single tracklet (backwards-compatible with the --ra/--dec path).
        desig_keys = [obs.packed_desig or obs.designation or '' for obs in observations]
        unique_desigs = list(dict.fromkeys(desig_keys))   # preserve order, dedup

        id_parts = []
        if len(unique_desigs) <= 1:
            # Single tracklet — original behaviour
            desig0 = unique_desigs[0] if unique_desigs else ''
            idents = identify_tracklet(
                observations, results,
                asteroids, comets, obscodes,
                dynmodel=args.dynmodel,
                residual_threshold_arcsec=args.id_threshold,
                min_obs=args.id_min_obs,
                fo_fit=args.fo_fit,
                fo_timeout_sec=60,
            )
            idents_by_desig[desig0]  = idents
            obs_grp_by_desig[desig0] = observations
            if idents:
                id_parts.append(format_identifications(
                    idents, observations, designation=desig0))
            else:
                id_parts.append(
                    f'\n{"="*60}\n'
                    f'Tracklet identification: no candidates satisfy all '
                    f'{len(observations)} observations with O-C ≤ '
                    f'{args.id_threshold:.1f}" threshold.\n'
                )
        else:
            # Multi-cluster file: run identify_tracklet per designation group.
            idx_by_desig: dict = _dd(list)
            for i, key in enumerate(desig_keys):
                idx_by_desig[key].append(i)

            n_identified = 0
            for desig in unique_desigs:
                grp_indices = idx_by_desig[desig]
                grp_obs     = [observations[i] for i in grp_indices]
                obs_grp_by_desig[desig] = grp_obs
                if len(grp_indices) < 2:
                    idents_by_desig[desig] = []
                    continue   # single observation — can't fit an orbit
                grp_results = [results[i] for i in grp_indices]

                idents = identify_tracklet(
                    grp_obs, grp_results,
                    asteroids, comets, obscodes,
                    dynmodel=args.dynmodel,
                    residual_threshold_arcsec=args.id_threshold,
                    min_obs=args.id_min_obs,
                    fo_fit=args.fo_fit,
                    fo_timeout_sec=60,
                )
                idents_by_desig[desig] = idents
                if idents:
                    n_identified += 1
                    id_parts.append(format_identifications(
                        idents, grp_obs, designation=desig))
                # (silently skip unidentified clusters in per-cluster block)

            # Summary table always shown for multi-cluster runs
            summary_str = format_identification_summary(
                idents_by_desig, obs_grp_by_desig)
            if id_parts:
                id_parts.insert(0,
                    f'\nIdentified {n_identified}/{len(unique_desigs)} clusters:\n')
            id_parts.append(summary_str)

        identification_output = ''.join(id_parts)

    # -----------------------------------------------------------------------
    # Format and output
    # -----------------------------------------------------------------------
    total_matches = sum(len(r.matches) for r in results)
    n_with_matches = sum(1 for r in results if r.matches)

    # Validate mutually-exclusive format flags
    if args.json and args.csv:
        print('ERROR: --json and --csv are mutually exclusive.', file=sys.stderr)
        return 1

    # Build the primary output string
    if args.json:
        json_dict = format_results_json(
            results, args.radius,
            identifications_by_desig=idents_by_desig if args.identify else None,
            obs_by_desig=obs_grp_by_desig if args.identify else None,
        )
        if args.summary_only:
            # Drop per-observation data; keep only meta + identifications
            json_dict.pop('observations', None)
        output = _json_mod.dumps(json_dict, indent=2)

    elif args.csv:
        if args.summary_only:
            # Matches suppressed — only write identifications section (if any)
            output = ''
        else:
            output = format_results_csv(results, args.radius)
        if args.identify and idents_by_desig:
            idents_csv = format_identifications_csv(
                idents_by_desig, obs_grp_by_desig)
            if output:
                output = output.rstrip('\n') + '\n\n# identifications\n' + idents_csv
            else:
                output = '# identifications\n' + idents_csv

    else:
        # Plain text
        if args.summary_only:
            # Compact count line instead of full match table
            output = (f'{len(results)} observation(s), '
                      f'{n_with_matches} with matches, '
                      f'{total_matches} total match(es).')
        else:
            output = format_results(results, args.radius)

    # identification_output holds per-cluster detail + summary (text only)
    # --summary-only suppresses per-cluster detail but keeps the summary table
    if args.summary_only and not args.json and not args.csv:
        # Re-build identification output to show only the summary table
        if args.identify and obs_grp_by_desig:
            if len(obs_grp_by_desig) > 1:
                identification_output = format_identification_summary(
                    idents_by_desig, obs_grp_by_desig)
            else:
                # Single tracklet: show status line instead of full detail
                desig0    = next(iter(obs_grp_by_desig))
                obs_list0 = obs_grp_by_desig[desig0]
                idents0   = idents_by_desig.get(desig0, [])
                if idents0:
                    best   = idents0[0]
                    status = _ident_status(best, obs_list0)
                    name   = _ident_display_name(best)
                    rms    = (f'{best.rms_arcsec:.2f}"'
                              if np.isfinite(best.rms_arcsec) else '—')
                    identification_output = (
                        f'\n{status}: {name}  '
                        f'[{best.method}]  RMS={rms}  '
                        f'({len(obs_list0)} obs)')
                else:
                    identification_output = '\nNo identification found.'
        else:
            identification_output = ''

    # Write output
    if args.output:
        with open(args.output, 'w') as f:
            f.write(output + '\n')
            if not args.json and not args.csv and identification_output:
                f.write(identification_output + '\n')
        print(f'Results written to {args.output}', file=sys.stderr)
    else:
        print(output)
        if not args.json and not args.csv and identification_output:
            print(identification_output)

    if total_matches == 0 and len(asteroids) > 0 and not args.summary_only:
        print('Hint: 0 matches found. Run with --debug for pipeline diagnostics.',
              file=sys.stderr)

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
