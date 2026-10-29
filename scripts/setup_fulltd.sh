#!/usr/bin/env bash
# Download public raw data and install the tested Python/PyTorch environment.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
: "${FULLTD_DATA_DIR:?Set FULLTD_DATA_DIR to your approved project storage directory}"
PYTHON="${FULLTD_PYTHON:-python3.12}"
VENV="${FULLTD_ENV:-$ROOT/.venv-fulltd}"
TORCH_INDEX="${FULLTD_TORCH_INDEX:-https://download.pytorch.org/whl/cu118}"
"$PYTHON" -c 'import sys; assert sys.version_info[:2] == (3,12), "Use Python 3.12 (tested 3.12.14)"'
"$PYTHON" -m venv "$VENV"
"$VENV/bin/python" -m pip install 'torch==2.5.1' --index-url "$TORCH_INDEX"
"$VENV/bin/python" -m pip install -r requirements-fulltd.txt
mkdir -p "$FULLTD_DATA_DIR"
"$VENV/bin/python" -m pip freeze > "$FULLTD_DATA_DIR/environment-freeze.txt"
"$VENV/bin/python" -m fulltd.download --dataset "${FULLTD_DATASET:-all}" --data-dir "$FULLTD_DATA_DIR"
printf 'Environment and raw data ready. Preprocessing and GPU checks run in the submission preflight job.\n'
