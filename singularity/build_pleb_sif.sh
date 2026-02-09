#!/usr/bin/env bash
set -euo pipefail

BASE_IMAGE="${BASE_IMAGE:-${1:-}}"
OUT_SIF="${OUT_SIF:-${2:-pleb.sif}}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PLEB_SRC="${PLEB_SRC:-$REPO_ROOT/pleb}"
PQC_SRC="${PQC_SRC:-$REPO_ROOT/pqc}"
DEF_IN="$SCRIPT_DIR/pleb.def.in"
DEF_OUT="$SCRIPT_DIR/pleb.def"

if [[ -z "$BASE_IMAGE" ]]; then
  echo "BASE_IMAGE is required (path to psrpta.sif)." >&2
  echo "Usage: BASE_IMAGE=/path/to/psrpta.sif $0 [out.sif]" >&2
  echo "   or: $0 /path/to/psrpta.sif [out.sif]" >&2
  exit 1
fi

sed \
  -e "s|{{BASE_IMAGE}}|$BASE_IMAGE|g" \
  -e "s|{{PLEB_SRC}}|$PLEB_SRC|g" \
  -e "s|{{PQC_SRC}}|$PQC_SRC|g" \
  "$DEF_IN" > "$DEF_OUT"

echo "Building $OUT_SIF from $BASE_IMAGE"
echo "Using PLEB_SRC=$PLEB_SRC"
echo "Using PQC_SRC=$PQC_SRC"
singularity build --fakeroot "$OUT_SIF" "$DEF_OUT"
