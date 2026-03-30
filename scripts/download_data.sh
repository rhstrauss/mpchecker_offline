#!/usr/bin/env bash
# Download all data required by mpchecker:
#   - MPCORB.DAT.gz    (MPC asteroid orbit catalog, ~200 MB)
#   - AllCometEls.txt  (MPC comet elements)
#   - ObsCodes.txt     (MPC observatory codes)
#   - SPICE kernels:
#       de440s.bsp     (planetary ephemeris, ~100 MB)
#       naif0012.tls   (leapseconds)
#       pck00011.tpc   (planet constants)
#       mar097.bsp, jup365.bsp, sat441.bsp, ura116.bsp, nep097.bsp, plu060.bsp
#         (planetary satellite ephemerides)
#
# Usage:
#   bash scripts/download_data.sh [--data-dir /path/to/data]
#   bash scripts/download_data.sh --satellites-only
#   bash scripts/download_data.sh --asteroids-only

set -euo pipefail

DATA_DIR="${MPCHECKER_DATA:-/gscratch/astro/rstrau/mpchecker_data}"
SATELLITES_ONLY=0
ASTEROIDS_ONLY=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --data-dir)    DATA_DIR="$2"; shift 2 ;;
        --satellites-only) SATELLITES_ONLY=1; shift ;;
        --asteroids-only)  ASTEROIDS_ONLY=1; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

ORBS_DIR="$DATA_DIR/orbits"
SPICE_DIR="$DATA_DIR/spice"
mkdir -p "$ORBS_DIR" "$SPICE_DIR"

NAIF_BASE="https://naif.jpl.nasa.gov/pub/naif/generic_kernels"
MPC_BASE="https://www.minorplanetcenter.net"

dl() {
    local url="$1" dest="$2"
    if [[ -f "$dest" ]]; then
        echo "  [skip] $(basename $dest) already exists"
        return
    fi
    echo "  Downloading $(basename $dest) ..."
    curl -L --retry 3 --retry-delay 5 -o "$dest" "$url"
    echo "  Saved: $dest"
}

# ----------------------------------------------------------------
# SPICE kernels (always downloaded unless --asteroids-only)
# ----------------------------------------------------------------
if [[ $ASTEROIDS_ONLY -eq 0 ]]; then
    echo ""
    echo "=== SPICE base kernels ==="
    dl "$NAIF_BASE/lsk/naif0012.tls"          "$SPICE_DIR/naif0012.tls"
    dl "$NAIF_BASE/spk/planets/de440s.bsp"    "$SPICE_DIR/de440s.bsp"
    dl "$NAIF_BASE/pck/pck00011.tpc"          "$SPICE_DIR/pck00011.tpc"

    echo ""
    echo "=== Satellite SPK kernels ==="
    dl "$NAIF_BASE/spk/satellites/mar099.bsp"   "$SPICE_DIR/mar099.bsp"
    dl "$NAIF_BASE/spk/satellites/jup365.bsp"   "$SPICE_DIR/jup365.bsp"
    dl "$NAIF_BASE/spk/satellites/sat441.bsp"   "$SPICE_DIR/sat441.bsp"
    dl "$NAIF_BASE/spk/satellites/ura116xl.bsp" "$SPICE_DIR/ura116xl.bsp"
    dl "$NAIF_BASE/spk/satellites/nep097.bsp"   "$SPICE_DIR/nep097.bsp"
    dl "$NAIF_BASE/spk/satellites/plu060.bsp"   "$SPICE_DIR/plu060.bsp"
fi

# ----------------------------------------------------------------
# MPC orbit catalogs (skipped if --satellites-only)
# ----------------------------------------------------------------
if [[ $SATELLITES_ONLY -eq 0 ]]; then
    echo ""
    echo "=== MPC orbit catalogs ==="
    dl "$MPC_BASE/iau/MPCORB/MPCORB.DAT.gz"        "$ORBS_DIR/MPCORB.DAT.gz"
    dl "$MPC_BASE/iau/MPCORB/AllCometEls.txt"       "$ORBS_DIR/AllCometEls.txt"
    dl "$MPC_BASE/iau/lists/ObsCodes.html"           "$ORBS_DIR/ObsCodes.txt"
fi

echo ""
echo "Data download complete."
echo "Data directory: $DATA_DIR"
echo ""
echo "To parse and cache the asteroid catalog (recommended before first use):"
echo "  conda run -n mpchecker python -c \\"
echo "    'from mpchecker.mpcorb import load_mpcorb; load_mpcorb()'"
