#!/usr/bin/env bash
# Locked validation selection + parallel selected endpoint inference + archive.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/dscpuscripts/env.sh"
cd "$ROOT"
DATASET="${1:-}"
WORKERS="${2:-8}"
if [[ $# -lt 1 || $# -gt 2 || ( "$DATASET" != "electricity" && "$DATASET" != "metr_la" ) ||
      ! "$WORKERS" =~ ^[0-9]+$ ]] || (( WORKERS < 1 || WORKERS > 32 )); then
  echo "Usage: bash dscpuscripts/analyze_pairwise.sh electricity|metr_la [EVAL_WORKERS=8]" >&2
  exit 2
fi
mapfile -t ROOTS < <("$FULLTD_ENV/bin/python" - "$DATASET" <<'PY'
import sys
from dscpuscripts.pairwise_runner import paths
print(paths(sys.argv[1])[1])
PY
)
if [[ "${#ROOTS[@]}" -ne 1 ]]; then
  echo "Pairwise result root resolution failed" >&2
  exit 1
fi
NEW_ROOT="${ROOTS[0]}"
mkdir -p "$NEW_ROOT"
exec 9>"$NEW_ROOT/.pairwise_analysis.lock"
if ! flock -n 9; then
  echo "Another Pairwise analysis is active under $NEW_ROOT" >&2
  exit 1
fi
"$FULLTD_ENV/bin/python" -m dscpuscripts.pairwise_analysis "$DATASET" --all --workers "$WORKERS"

if [[ "$DATASET" == "electricity" ]]; then
  ARCHIVE="$HOME/ELECTRICITY_pairwise_final.tar.gz"
else
  ARCHIVE="$HOME/METR_LA_pairwise_final.tar.gz"
fi
TMP="$(mktemp "$HOME/.pairwise_analysis.XXXXXX.tar.gz")"
trap 'rm -f "$TMP"' EXIT
tar -czf "$TMP" -C "$NEW_ROOT" pairwise_outputs
if [[ ! -s "$TMP" ]]; then
  echo "Incomplete Pairwise archive" >&2
  exit 1
fi
mv -f "$TMP" "$ARCHIVE"
trap - EXIT
echo "COMPLETE: $ARCHIVE"
echo "Summary: $NEW_ROOT/pairwise_outputs/summary.csv"
