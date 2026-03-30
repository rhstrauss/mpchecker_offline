# mpchecker

A local replication of the [MPC MPChecker service](https://www.minorplanetcenter.net/cgi-bin/checkmp.cgi) that runs entirely offline on your own machine. Given MPC 80-column astrometric observation records (or a bare RA/Dec/epoch), it searches the full MPC orbit catalog for known solar system objects at each observed position.

**Extends the MPC service with:**
- Planetary satellites (~100 bodies via JPL SPICE kernels)
- Dwarf planet satellites (Dysnomia, Hi'iaka, Namaka, Weywot, Vanth, Xiangliu, Actaea, MK2 — Keplerian orbit model, no SPICE coverage available)
- Persistent daemon mode for interactive use (eliminates ~10–15 s cold start)
- Multi-snapshot KD-tree index for fast catalog pre-filtering (~5 ms/obs)
- Parallel Phase 1 pre-filter for large observation batches

---

## Requirements

- **Python ≥ 3.11** (tested with 3.11)
- [pyoorb](https://github.com/oorb/oorb) (for precise N-body ephemeris) — easiest via conda
- [SpiceyPy](https://github.com/AndrewAnnex/SpiceyPy) (SPICE Python bindings)
- ~800 MB disk space for data files (orbit catalogs + SPICE kernels)

Optional but recommended:
- [numba](https://numba.pydata.org/) — JIT-compiled parallel Keplerian propagation (~10× speedup for the pre-filter)

---

## Installation

### 1. Create the conda environment

```bash
conda create -n mpchecker -c conda-forge python=3.11 openorb spiceypy numba numpy astropy astroquery requests tqdm
conda activate mpchecker
```

If `openorb` is not available on your platform via conda-forge, see the [pyoorb build instructions](https://github.com/oorb/oorb).

### 2. Install the package

```bash
git clone https://github.com/rhstrauss/mpchecker.git
cd mpchecker
pip install -e .
```

### 3. Set the data directory (optional)

By default, data files are stored in `/gscratch/astro/rstrau/mpchecker_data/`. To use a different location:

```bash
export MPCHECKER_DATA=/path/to/your/data
```

Add this to your shell profile or conda activation script to make it permanent.

### 4. Download data files (~800 MB total)

```bash
# All-in-one:
mpchecker --download-data

# Or with a custom data directory:
bash scripts/download_data.sh --data-dir /path/to/data

# SPICE kernels only (if you already have MPC catalogs):
bash scripts/download_data.sh --satellites-only
```

This downloads:
| File | Size | Source |
|------|------|--------|
| `MPCORB.DAT.gz` | ~200 MB | Minor Planet Center |
| `AllCometEls.txt` | ~5 MB | Minor Planet Center |
| `ObsCodes.html` | ~1 MB | Minor Planet Center |
| `de440s.bsp` | ~100 MB | NAIF/JPL |
| `naif0012.tls`, `pck00011.tpc` | <1 MB | NAIF/JPL |
| Satellite SPK files | ~500 MB | NAIF/JPL |

### 5. Pre-parse the asteroid catalog (recommended)

Parsing 1.5 M MPCORB entries on first run takes ~30 s. Cache it ahead of time:

```bash
conda run -n mpchecker python -c "from mpchecker.mpcorb import load_mpcorb; load_mpcorb()"
```

Subsequent runs load the pre-parsed numpy binary in ~1 s.

---

## Usage

### Check an 80-column MPC observation file

```bash
mpchecker obs.txt
```

The 80-column format is the standard MPC astrometry format (see [MPC format docs](https://www.minorplanetcenter.net/iau/info/OpticalObs.html)):

```
     K04Q02F  C2004 08 21.41941 00 32 29.55 +07 55 34.9          21.0 R      568
```

### Check a single RA/Dec/epoch

```bash
mpchecker --ra 26.09 --dec -0.97 --epoch 2024-01-01 --obscode 568
```

### Key options

```bash
mpchecker obs.txt --radius 30         # search radius in arcsec (default: 30)
mpchecker obs.txt --maglim 22         # faint V-mag limit (default: 25)
mpchecker obs.txt --dynmodel N        # N-body integration (default: two-body)
mpchecker obs.txt --workers 8         # parallel Phase 1 over observations
mpchecker obs.txt --dedup             # only check first obs per designation
mpchecker obs.txt --no-asteroids      # skip MPCORB (satellites only)
mpchecker obs.txt --no-comets         # skip comet catalog
mpchecker obs.txt --no-satellites     # skip satellite checks
```

### Persistent daemon (recommended for interactive use)

The first `mpchecker` run loads ~1.5 M orbits and builds SPICE/pyoorb state — roughly 10–15 s. The daemon keeps everything in memory:

```bash
mpchecker --start-daemon     # fork daemon into background
mpchecker obs.txt            # instantly uses running daemon
mpchecker --stop-daemon      # stop daemon
mpchecker --serve            # foreground server (debugging)
mpchecker obs.txt --no-daemon  # bypass daemon, run standalone
```

The daemon auto-rebuilds the KD-tree index when it goes stale (~every 7 days).

### Rebuild the sky index

```bash
mpchecker --build-index
```

The multi-snapshot KD-tree index (4 snapshots, 2-day intervals) is built automatically on first run and cached. Rebuild manually after updating the orbit catalog.

---

## Pipeline overview

For each input observation the pipeline runs three stages:

1. **Keplerian pre-filter** (`propagator.py`, `index.py`): An H-magnitude cut reduces the 1.5 M object catalog to a manageable subset. A multi-snapshot KD-tree index (4 snapshots, 2 days apart) provides a 7× tighter cone search than a single-epoch index (~50× fewer pyoorb candidates). Typical time: **5 ms/obs**.

2. **Precise ephemeris** (`propagator.py`): pyoorb two-body or N-body integration for pre-filter candidates. Observations are grouped by observatory code and processed in one batched call. Typical time: **15–50 ms/obs**.

3. **Satellite check** (`satellites.py`): ~100 planetary satellites via SpiceyPy + JPL SPICE kernels; 8 dwarf planet satellites via Keplerian orbit models.

---

## Data updates

The MPC releases updated orbit catalogs frequently. To refresh:

```bash
# Re-download MPCORB and comets
bash scripts/download_data.sh --asteroids-only

# Clear the cached parsed catalog so it's rebuilt on next run
rm "$MPCHECKER_DATA/cache/mpcorb_H35.npy" 2>/dev/null

# Rebuild the KD-tree index
mpchecker --build-index
```

---

## Notes on HPC environments

On nodes with many CPUs (e.g. Hyak/Klone login nodes with 384 cores), Numba's default thread count can exceed the OS limit:

```bash
export NUMBA_NUM_THREADS=4
```

The `--workers N` flag uses `fork`-based multiprocessing, which is incompatible with Numba's TBB thread pool. Worker processes automatically fall back to the NumPy propagation path.

---

## Covered solar system populations

| Population | Source | Notes |
|-----------|--------|-------|
| Minor planets | MPCORB (~1.5 M objects) | Includes NEOs, MBAs, TNOs, dwarf planets |
| Comets | MPC AllCometEls | ~1 000 objects |
| Planetary satellites | JPL SPICE SPK files | ~100 objects (Mars through Pluto) |
| Dwarf planet satellites | Keplerian orbit models | Dysnomia, Hi'iaka, Namaka, Weywot, Vanth, Xiangliu, Actaea, MK2 |

**Not currently covered:** small irregular moons beyond the cataloged SPICE kernels, interstellar objects, newly discovered bodies not yet in MPCORB.

---

## License

MIT
