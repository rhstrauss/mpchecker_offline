"""
MPCAT observation archive index and lookup.

Provides fast random access to MPC observation files (NumObs.txt, UnnObs.txt)
via a pre-built byte-offset index.  The index maps each packed designation to
its start byte and line count in the flat text file, enabling microsecond
lookup for any object by a single seek + read.

Index build
-----------
Run the index builder once after downloading fresh MPCAT files:

    python /astro/store/shire/rstrau/mpcat/build_index.py \\
        /astro/store/shire/rstrau/mpcat/NumObs.txt \\
        /astro/store/shire/rstrau/mpcat/UnnObs.txt

This creates NumObs.idx.npy and UnnObs.idx.npy alongside the text files.

Usage
-----
    from mpchecker.mpcat import MPCATIndex
    idx = MPCATIndex()
    lines = idx.get_obs('j8122')          # all 2010 EW45 observations
    lines = idx.get_obs('K10E45W')        # provisional packed form
    lines = idx.get_obs('j8122', max_obs=200)  # cap for speed

MPC packed designation packing reference
-----------------------------------------
Numbers 1–99999: 5-digit zero-padded string ('00001', '00433', ...).
Numbers 100000–359999: uppercase A–Z as leading digit, then 4 digits.
    A=10 (100000–109999), B=11, ..., Z=35 (350000–359999)
Numbers 360000–619999: lowercase a–z as leading digit, then 4 digits.
    a=36 (360000–369999), b=37, ..., z=61 (610000–619999)
Numbers ≥620000: extended encoding starting with '~'.

This is OPPOSITE to the common documentation — uppercase comes before
lowercase in the MPC scheme for historical/sorting reasons.
"""

import logging
import os
from pathlib import Path
from typing import List, Optional

import numpy as np

log = logging.getLogger(__name__)

# Default locations
_DEFAULT_MPCAT_DIR = Path('/astro/store/shire/rstrau/mpcat')
_FALLBACK_NUMOBS   = Path('/astro/store/shire/rstrau/NumObs.txt')
_FALLBACK_UNNOBS   = Path('/astro/store/shire/rstrau/UnnObs.txt')


def pack_mpc_number(n: int) -> str:
    """
    Convert an integer minor planet number to the MPC packed 5-char form.

    Uppercase A-Z for 100000-359999; lowercase a-z for 360000-619999.
    """
    if n < 100000:
        return f'{n:05d}'
    lead = n // 10000
    rest = n % 10000
    if lead < 36:                          # 10-35 → uppercase A-Z
        c = chr(ord('A') + lead - 10)
    else:                                  # 36-61 → lowercase a-z
        c = chr(ord('a') + lead - 36)
    return f'{c}{rest:04d}'


def unpack_mpc_number(s: str) -> Optional[int]:
    """Decode an MPC packed permanent designation to an integer, or None."""
    s = s.strip()
    if not s:
        return None
    c = s[0]
    if c.isdigit():
        try:
            return int(s)
        except ValueError:
            return None
    if c.isupper():
        lead = ord(c) - ord('A') + 10     # A=10 … Z=35
    else:
        lead = ord(c) - ord('a') + 36     # a=36 … z=61
    try:
        return lead * 10000 + int(s[1:])
    except ValueError:
        return None


class MPCATIndex:
    """
    Fast lookup of MPC80 observation lines by packed designation.

    Wraps a pre-built numpy byte-offset index and the corresponding flat
    observation text file.  Lookup is O(log n) over the index + a single
    fseek/read for the observations.

    Parameters
    ----------
    mpcat_dir : directory containing NumObs.txt, UnnObs.txt, and their
                .idx.npy index files.  Defaults to the shire MPCAT location.
    """

    def __init__(self, mpcat_dir: Optional[Path] = None):
        if mpcat_dir is None:
            mpcat_dir = _DEFAULT_MPCAT_DIR

        mpcat_dir = Path(mpcat_dir)
        self.mpcat_dir = mpcat_dir   # stored so parallel workers can reconstruct index
        self._num_fh  = None
        self._unn_fh  = None
        self._num_idx = None
        self._unn_idx = None

        # Prefer mpcat/ subdirectory, fall back to legacy location
        for name, fallback in [('NumObs.txt', _FALLBACK_NUMOBS),
                                ('UnnObs.txt', _FALLBACK_UNNOBS)]:
            txt  = mpcat_dir / name
            idx  = (mpcat_dir / name).with_suffix('.idx.npy')
            if not txt.exists():
                txt = fallback
                idx = fallback.with_suffix('.idx.npy')
            if txt.exists() and idx.exists():
                fh  = open(txt, 'rb')
                arr = np.load(idx, allow_pickle=False)
                if 'Num' in name:
                    self._num_fh  = fh
                    self._num_idx = arr
                    log.info('Loaded NumObs index: %d objects from %s', len(arr), txt)
                else:
                    self._unn_fh  = fh
                    self._unn_idx = arr
                    log.info('Loaded UnnObs index: %d objects from %s', len(arr), txt)
            elif txt.exists():
                log.warning('%s index not found at %s — run build_index.py first',
                            name, idx)

        if self._num_idx is None and self._unn_idx is None:
            raise FileNotFoundError(
                f'No MPCAT files found in {mpcat_dir} or fallback locations. '
                f'Run build_index.py to create the index.')

    def _lookup(self, idx_arr: np.ndarray, fh, packed: str,
                max_obs: int) -> List[str]:
        """
        Collect all observation lines whose index key starts with `packed`.

        NumObs records sometimes have both the permanent number AND the packed
        provisional designation concatenated in cols 0–11 (e.g. 'j8122K10E45W').
        A prefix scan catches these alongside the plain 'j8122' entries.
        Multiple non-contiguous blocks for the same key are also merged.
        """
        pos = np.searchsorted(idx_arr['packed'], packed)
        lines: List[str] = []
        while pos < len(idx_arr) and idx_arr['packed'][pos].startswith(packed):
            offset = int(idx_arr['offset'][pos])
            n      = int(idx_arr['n_obs'][pos])
            fh.seek(offset)
            lines.extend(fh.readline().decode('ascii', errors='replace')
                         for _ in range(n))
            pos += 1
        if max_obs and len(lines) > max_obs:
            stride = max(1, len(lines) // max_obs)
            lines = lines[::stride][:max_obs]
        return lines

    def get_obs(self, packed: str, max_obs: int = 0) -> List[str]:
        """
        Return all MPC80 observation lines for the given packed designation.

        Searches both NumObs (numbered) and UnnObs (unnumbered) indexes.
        Uses a prefix scan so that records with both number and provisional
        packed in cols 0–11 (e.g. 'j8122K10E45W') are included alongside
        plain numbered records ('j8122').
        If max_obs > 0, returns an evenly-spaced sample for speed.

        Parameters
        ----------
        packed  : MPC packed designation, e.g. 'j8122', 'K10E45W', '00001'
        max_obs : cap on returned observations (0 = unlimited)
        """
        packed = packed.strip()
        lines = []
        if self._num_idx is not None:
            lines = self._lookup(self._num_idx, self._num_fh, packed, max_obs)
        if not lines and self._unn_idx is not None:
            lines = self._lookup(self._unn_idx, self._unn_fh, packed, max_obs)
        return lines

    def get_obs_for_number(self, number: int, max_obs: int = 0) -> List[str]:
        """Convenience wrapper: look up by integer permanent number."""
        return self.get_obs(pack_mpc_number(number), max_obs=max_obs)

    def close(self):
        for fh in (self._num_fh, self._unn_fh):
            if fh is not None:
                fh.close()

    def __del__(self):
        self.close()
