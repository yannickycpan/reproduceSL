#!/usr/bin/env bash
set -euo pipefail
export FULLTD_METR_LA_SOURCE_FRACTION="${FULLTD_METR_LA_SOURCE_FRACTION:-0.10}"
tag="$(python3 - "$FULLTD_METR_LA_SOURCE_FRACTION" <<'PY'
import math, sys
f = float(sys.argv[1])
if not math.isfinite(f) or not 0 < f <= 1:
    raise SystemExit("FULLTD_METR_LA_SOURCE_FRACTION must be in (0,1]")
print("f" + format(f * 100, ".8g").replace(".", "p") + "pct")
PY
)"
base="${FULLTD_RESULTS_ROOT:-$HOME/scratch/fulltd}"
dest="$base/metr_la_sampling/$tag/metr_la_results"
if [[ -n "${FULLTD_METR_LA_ROOT:-}" && "${FULLTD_METR_LA_ROOT}" != "$dest" ]]; then
  echo "Refusing conflicting METR-LA root: $FULLTD_METR_LA_ROOT" >&2
  echo "Required isolated root: $dest" >&2
  exit 1
fi
export FULLTD_METR_LA_ROOT="$dest"
