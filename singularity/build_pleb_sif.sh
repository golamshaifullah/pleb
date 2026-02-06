#!/usr/bin/env bash
set -euo pipefail

BASE_IMAGE="${BASE_IMAGE:-}"
OUT_SIF="${OUT_SIF:-pleb.sif}"
DEF_IN="$(dirname "$0")/pleb.def.in"
DEF_OUT="$(dirname "$0")/pleb.def"

if [[ -z "$BASE_IMAGE" ]]; then
  echo "BASE_IMAGE is required (path to psrpta.sif)." >&2
  exit 1
fi

sed "s|{{BASE_IMAGE}}|$BASE_IMAGE|g" "$DEF_IN" > "$DEF_OUT"

echo "Building $OUT_SIF from $BASE_IMAGE"
sudo singularity build "$OUT_SIF" "$DEF_OUT"
