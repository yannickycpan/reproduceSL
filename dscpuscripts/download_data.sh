#!/usr/bin/env bash
# Idempotently download and prepare original FD001 (or other original datasets).
# Does not reprepare existing processed data while experiments may be running.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/dscpuscripts/env.sh"
cd "$ROOT"
PYTHON="$FULLTD_ENV/bin/python"
[[ -x "$PYTHON" ]] || { echo "Run bash dscpuscripts/setup_server.sh first." >&2; exit 1; }
DATASET="${1:-fd001}"
[[ $# -le 1 ]] || { echo "Usage: bash dscpuscripts/download_data.sh [fd001|electricity|metr_la|all]" >&2; exit 2; }
case "$DATASET" in
  fd001|electricity|metr_la) DATASETS=("$DATASET") ;;
  all) DATASETS=(fd001 electricity metr_la) ;;
  *) echo "Unknown dataset: $DATASET" >&2; exit 2 ;;
esac
mkdir -p "$FULLTD_DATA_DIR"
for name in "${DATASETS[@]}"; do
  echo "Acquiring original $name data with recorded source and SHA-256..."
  "$PYTHON" -m fulltd.download --dataset "$name" --data-dir "$FULLTD_DATA_DIR"
  if [[ -f "$FULLTD_DATA_DIR/processed/$name/metadata.json" ]]; then
    echo "Existing processed $name data found; verifying, not overwriting."
  else
    "$PYTHON" -m fulltd.data prepare --dataset "$name" --data-dir "$FULLTD_DATA_DIR"
  fi
  "$PYTHON" -m fulltd.data verify --dataset "$name" --data-dir "$FULLTD_DATA_DIR"
done
echo "Dataset preparation verified: $FULLTD_DATA_DIR"
