#!/usr/bin/env bash
# Install the exact pinned CPU dependencies without HPC modules or Slurm.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/dscpuscripts/env.sh"
PYTHON="${FULLTD_PYTHON:-python3.12}"
TORCH_INDEX="${FULLTD_TORCH_INDEX:-https://download.pytorch.org/whl/cpu}"
command -v "$PYTHON" >/dev/null 2>&1 || {
  echo "Python 3.12 not found. Set FULLTD_PYTHON to its path, or install Python 3.12." >&2
  exit 1
}
"$PYTHON" -c 'import sys; assert sys.version_info[:2] == (3, 12), "Python 3.12 is required for the pinned environment"'
mkdir -p "$(dirname "$FULLTD_ENV")" "$FULLTD_DATA_DIR"
if [[ ! -x "$FULLTD_ENV/bin/python" ]]; then
  "$PYTHON" -m venv "$FULLTD_ENV"
fi
"$FULLTD_ENV/bin/python" -c 'import sys; assert sys.version_info[:2] == (3, 12), "Existing venv uses the wrong Python version"'
"$FULLTD_ENV/bin/python" -m pip install 'torch==2.5.1' --index-url "$TORCH_INDEX"
"$FULLTD_ENV/bin/python" -m pip install -r "$ROOT/requirements-fulltd.txt"
"$FULLTD_ENV/bin/python" -m pip check
"$FULLTD_ENV/bin/python" -m pip freeze > "$FULLTD_DATA_DIR/environment-freeze.txt"
echo "CPU environment ready: $FULLTD_ENV"
echo "Next: bash dscpuscripts/download_data.sh fd001"
