#!/usr/bin/env bash
# Install the exact pinned CPU dependencies without HPC modules or Slurm.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/dscpuscripts/env.sh"
PYTHON="${FULLTD_PYTHON:-}"
TORCH_INDEX="${FULLTD_TORCH_INDEX:-https://download.pytorch.org/whl/cpu}"
if [[ -z "$PYTHON" ]]; then
  for candidate in python3.12 python3.11 python3.10 python3; do
    if command -v "$candidate" >/dev/null 2>&1 && \
       "$candidate" -c 'import sys; assert (3,10) <= sys.version_info[:2] <= (3,12)' 2>/dev/null; then
      PYTHON="$candidate"
      break
    fi
  done
fi
[[ -n "$PYTHON" ]] && command -v "$PYTHON" >/dev/null 2>&1 || {
  echo "Python 3.10, 3.11, or 3.12 is needed. Set FULLTD_PYTHON to an available interpreter." >&2
  exit 1
}
"$PYTHON" -c 'import sys; assert (3,10) <= sys.version_info[:2] <= (3,12), "Use Python 3.10-3.12"'
mkdir -p "$(dirname "$FULLTD_ENV")" "$FULLTD_DATA_DIR"
if [[ ! -x "$FULLTD_ENV/bin/python" ]]; then
  "$PYTHON" -m venv "$FULLTD_ENV"
fi
"$FULLTD_ENV/bin/python" -c 'import sys; assert (3,10) <= sys.version_info[:2] <= (3,12), "Existing venv needs Python 3.10-3.12"'
"$FULLTD_ENV/bin/python" -m pip install 'torch==2.5.1' --index-url "$TORCH_INDEX"
"$FULLTD_ENV/bin/python" -m pip install -r "$ROOT/requirements-fulltd.txt"
"$FULLTD_ENV/bin/python" -m pip check
"$FULLTD_ENV/bin/python" -m pip freeze > "$FULLTD_DATA_DIR/environment-freeze.txt"
echo "CPU environment ready: $FULLTD_ENV"
echo "Next: bash dscpuscripts/download_data.sh fd001"
