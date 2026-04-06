"""
Pre-built geocentric ephemeris table for bright asteroids (H ≤ h_limit).

Stores RA, Dec, V, r_helio, delta at 1-day intervals using two-body
propagation.  Linear interpolation replaces pyoorb calls for non-NEO bright
candidates in Phase 2, trading < 1 arcsec accuracy for a large speed gain.

Build cost:  ~1–2 s two-body run for ~20K H≤16 objects × 360 days.
Run once after each MPCORB update (e.g. weekly via cron or '--build-bright-table').

Usage
-----
    from mpchecker.bright_table import BrightEphemTable
    from mpchecker.config import CACHE_DIR

    # Build and cache
    table = BrightEphemTable.build(asteroids, obscodes,
                                    t0_mjd=Time.now().mjd,
                                    cache_path=CACHE_DIR / 'bright_table.npz')

    # Load from cache
    table = BrightEphemTable.load(CACHE_DIR / 'bright_table.npz')

    # Use in Phase 2 (see checker.py)
    valid, eph_interp = table.get_eph(global_idx_arr, t_mjd_tt_list,
                                       obs_helio_list, obscode, obscodes)

Accuracy
--------
Interpolation error for MBAs at 1-day steps: < 1 arcsec.
NEOs (q < 1.3 AU or e ≥ 0.5) are flagged — callers must use pyoorb for these.
Two-body propagation error (relative to N-body) for MBAs over catalog-epoch
drift: < 30 arcsec, well below the 30-arcmin Phase 2 search radius.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

# Objects this close in inclination/perihelion classification to the NEA boundary
# are conservatively routed to pyoorb to avoid classification edge-cases.
_Q_NEA_THRESHOLD = 1.3    # AU; q < this → NEA
_E_NEA_THRESHOLD = 0.5    # e ≥ this → NEA (high-eccentricity)


class BrightEphemTable:
    """
    Pre-built two-body ephemeris table for bright (H ≤ h_limit) non-NEA asteroids.

    Attributes
    ----------
    data       : float32 [n_bright, n_days, 5]  — geocentric RA, Dec, V, r, delta
    global_idx : int32   [n_bright]              — global indices into asteroid catalog
    t0_mjd     : float                           — MJD of day 0 in the table
    n_days     : int                             — number of columns
    is_nea     : bool    [n_bright]              — True → skip interp, use pyoorb
    """

    # Columns in data array
    _RA    = 0
    _DEC   = 1
    _VMAG  = 2
    _R     = 3
    _DELTA = 4

    def __init__(
        self,
        data:       np.ndarray,   # float32 [n, n_days, 5]
        global_idx: np.ndarray,   # int32   [n]
        t0_mjd:     float,
        is_nea:     np.ndarray,   # bool    [n]
    ):
        self.data       = data
        self.global_idx = global_idx
        self.t0_mjd     = float(t0_mjd)
        self.n_days     = int(data.shape[1])
        self.is_nea     = is_nea
        # dict: catalog_index → local row (for O(1) lookup)
        self._inv_idx: dict = {int(g): i for i, g in enumerate(global_idx)}

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        asteroids:  np.ndarray,
        obscodes:   dict,
        t0_mjd:     float,
        n_days:     int   = 360,
        h_limit:    float = 16.0,
        dynmodel:   str   = '2',
        cache_path: Optional[Path] = None,
    ) -> 'BrightEphemTable':
        """
        Build a BrightEphemTable for asteroids with H ≤ h_limit.

        Parameters
        ----------
        asteroids  : full asteroid catalog structured array
        obscodes   : observatory code dict (from load_obscodes)
        t0_mjd     : first epoch to tabulate (MJD TT)
        n_days     : number of daily steps (default 360 = ±180 days)
        h_limit    : absolute magnitude limit (default 16.0, ~20K objects)
        dynmodel   : pyoorb dynmodel for build ('2' = two-body, 'N' = N-body)
        cache_path : if given, save to this .npz file
        """
        import time
        from .propagator import (
            build_oorb_orbits_kep, oorb_ephemeris_multi_epoch,
            _init_oorb, _DEG2RAD,
        )

        bright_mask = asteroids['H'] <= h_limit
        bright = asteroids[bright_mask]
        gidx = np.where(bright_mask)[0].astype(np.int32)
        n_bright = len(bright)

        q_arr  = bright['a'] * (1.0 - bright['e'])
        is_nea = (q_arr < _Q_NEA_THRESHOLD) | (bright['e'] >= _E_NEA_THRESHOLD)

        log.info('BrightEphemTable: building for %d H≤%.1f objects (%d NEA, %d MBA) …',
                 n_bright, h_limit, is_nea.sum(), (~is_nea).sum())
        t0 = time.time()

        orbits = build_oorb_orbits_kep(
            bright['a'], bright['e'],
            bright['i']     * _DEG2RAD,
            bright['Omega'] * _DEG2RAD,
            bright['omega'] * _DEG2RAD,
            bright['M']     * _DEG2RAD,
            bright['epoch'], bright['H'], bright['G'],
        )

        t_list = [t0_mjd + i for i in range(n_days)]
        _init_oorb(force=True)
        eph = oorb_ephemeris_multi_epoch(orbits, t_list, '500', dynmodel=dynmodel)
        # eph: [n_bright, n_days, 11]; extract RA, Dec, V, r, delta
        data = eph[:, :, [1, 2, 9, 7, 8]].astype(np.float32)

        log.info('BrightEphemTable: built in %.1f s', time.time() - t0)

        table = cls(data, gidx, t0_mjd, is_nea)
        if cache_path is not None:
            table.save(cache_path)
        return table

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            data=self.data,
            global_idx=self.global_idx,
            t0_mjd=np.array([self.t0_mjd]),
            is_nea=self.is_nea,
        )
        log.info('BrightEphemTable saved to %s', path)

    @classmethod
    def load(cls, path: Path) -> 'BrightEphemTable':
        path = Path(path)
        d = np.load(path)
        table = cls(
            data=d['data'],
            global_idx=d['global_idx'].astype(np.int32),
            t0_mjd=float(d['t0_mjd'][0]),
            is_nea=d['is_nea'],
        )
        log.info('BrightEphemTable loaded from %s (%d objects, %d days from MJD=%.1f)',
                 path, len(table.global_idx), table.n_days, table.t0_mjd)
        return table

    # ------------------------------------------------------------------
    # Freshness
    # ------------------------------------------------------------------

    def covers(self, t_mjd: float) -> bool:
        """True if the table has data for this epoch."""
        return self.t0_mjd <= t_mjd < self.t0_mjd + self.n_days - 1

    def covers_range(self, t_min: float, t_max: float) -> bool:
        """True if the table covers the full epoch range."""
        return self.covers(t_min) and self.covers(t_max)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get_eph(
        self,
        global_idx_arr:  np.ndarray,    # [n_cands] int — catalog indices
        t_mjd_tt_list:   List[float],   # [n_epochs] TT MJD
        obs_helio_list:  List[np.ndarray],  # [n_epochs] [3] observer helio pos (AU)
        obscode:         str,
        obscodes:        dict,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Interpolate ephemeris for Phase 2 candidates.

        Returns
        -------
        valid      : bool [n_cands]
            True if served from the table (non-NEA, epoch in range).
            False → caller must use pyoorb.
        eph_interp : float64 [n_cands, n_epochs, 11]
            Topocentric ephemeris in pyoorb column format for valid entries.
            Invalid entries are filled with NaN (vmag column = 99.0).
            Columns: 0=MJD, 1=RA, 2=Dec, 3=dRA/dt, 4=dDec/dt, 5=phase,
                     7=r_helio, 8=delta, 9=vmag.  (6=elongation: 0.0)
        """
        from .propagator import _apply_topocentric_correction, _RAD2DEG

        n_cands = len(global_idx_arr)
        n_ep    = len(t_mjd_tt_list)
        eph = np.full((n_cands, n_ep, 11), np.nan)
        eph[:, :, 9] = 99.0
        valid = np.zeros(n_cands, dtype=bool)

        for i, gidx in enumerate(global_idx_arr):
            li = self._inv_idx.get(int(gidx), -1)
            if li < 0 or self.is_nea[li]:
                continue

            all_epochs_ok = True
            for k, t_tt in enumerate(t_mjd_tt_list):
                offset = t_tt - self.t0_mjd
                if offset < 0.0 or offset >= self.n_days - 1:
                    all_epochs_ok = False
                    break

                lo   = int(offset)
                frac = offset - lo

                d0 = self.data[li, lo].astype(np.float64)
                d1 = self.data[li, lo + 1].astype(np.float64)
                vals = d0 + frac * (d1 - d0)

                ra_geo  = float(vals[self._RA])
                dec_geo = float(vals[self._DEC])
                vmag    = float(vals[self._VMAG])
                r_helio = float(vals[self._R])
                delta   = float(vals[self._DELTA])

                # Sky motion rates (deg/day) via central difference
                if 0 < lo < self.n_days - 2:
                    ra_rate  = (float(self.data[li, lo+1, self._RA])
                                - float(self.data[li, lo-1, self._RA])) / 2.0
                    dec_rate = (float(self.data[li, lo+1, self._DEC])
                                - float(self.data[li, lo-1, self._DEC])) / 2.0
                else:
                    ra_rate  = float(d1[self._RA]  - d0[self._RA])
                    dec_rate = float(d1[self._DEC] - d0[self._DEC])

                # Phase angle from r, delta, r_obs (law of cosines)
                r_obs = float(np.linalg.norm(obs_helio_list[k]))
                denom = max(2.0 * r_helio * delta, 1e-10)
                cos_ph = (r_helio**2 + delta**2 - r_obs**2) / denom
                phase_deg = float(np.arccos(np.clip(cos_ph, -1.0, 1.0))) * _RAD2DEG

                eph[i, k, 0] = t_tt
                eph[i, k, 1] = ra_geo
                eph[i, k, 2] = dec_geo
                eph[i, k, 3] = ra_rate
                eph[i, k, 4] = dec_rate
                eph[i, k, 5] = phase_deg
                eph[i, k, 6] = 0.0       # elongation: not computed
                eph[i, k, 7] = r_helio
                eph[i, k, 8] = delta      # geocentric for now; corrected below
                eph[i, k, 9] = vmag

            if all_epochs_ok:
                valid[i] = True

        if valid.any() and obscode != '500' and obscodes is not None:
            eph[valid] = _apply_topocentric_correction(
                eph[valid], t_mjd_tt_list, obscode, obscodes)

        return valid, eph
