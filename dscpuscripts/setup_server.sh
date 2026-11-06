#!/usr/bin/env bash
# CPU environment setup WITHOUT requiring pip, ensurepip, sudo, or HPC modules.
# Uses Astral uv as a standalone installer; preserves existing environments.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/dscpuscripts/env.sh"
PYTHON="${FULLTD_PYTHON:-}"
TORCH_INDEX="${FULLTD_TORCH_INDEX:-https://download.pytorch.org/whl/cpu}"

if [[ -z "$PYTHON" ]]; then
  for candidate in python3.12 python3.11 python3.10 python3; do
    if command -v "$candidate" >/dev/null 2>&1 &&
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

# uv is a standalone binary. Unlike Python's pip/ensurepip, it works when
# those modules are missing, and can create a venv without python3.x-venv.
UV="${FULLTD_UV:-}"
if [[ -z "$UV" ]]; then
  for candidate in "$HOME/scratch/fulltd/.uv-bin/uv" "$HOME/.local/bin/uv" uv; do
    if command -v "$candidate" >/dev/null 2>&1; then
      UV="$candidate"
      break
    fi
  done
fi
if [[ -z "$UV" ]]; then
  UV_DIR="${FULLTD_UV_DIR:-$HOME/scratch/fulltd/.uv-bin}"
  mkdir -p "$UV_DIR"
  installer="$(mktemp)"
  trap 'rm -f "$installer"' EXIT
  echo "Standalone uv not found. Downloading official Astral installer (no pip/sudo needed)."
  if command -v curl >/dev/null 2>&1; then
    curl --proto '=https' --tlsv1.2 -fsSL "https://astral.sh/uv/install.sh" -o "$installer" || {
      echo "Cannot download uv. Check network access to https://astral.sh/uv/install.sh." >&2
      exit 1
    }
  elif command -v wget >/dev/null 2>&1; then
    wget -qO "$installer" "https://astral.sh/uv/install.sh" || {
      echo "Cannot download uv. Check network access to https://astral.sh/uv/install.sh." >&2
      exit 1
    }
  else
    echo "Neither curl nor wget is available. Copy a standalone uv binary and set FULLTD_UV=/path/to/uv." >&2
    exit 1
  fi
  # Official installer, installed only under the user's home directory;
  # do not edit shell profiles or the system Python installation.
  UV_UNMANAGED_INSTALL="$UV_DIR" sh "$installer"
  UV="$UV_DIR/uv"
fi
command -v "$UV" >/dev/null 2>&1 || {
  echo "uv is unavailable at '$UV'. Set FULLTD_UV=/absolute/path/to/uv." >&2
  exit 1
}
"$UV" --version

mkdir -p "$(dirname "$FULLTD_ENV")" "$FULLTD_DATA_DIR"
if [[ ! -x "$FULLTD_ENV/bin/python" ]]; then
  echo "Creating Python environment with uv (no pip or ensurepip)..."
  "$UV" venv --python "$PYTHON" "$FULLTD_ENV"
fi
"$FULLTD_ENV/bin/python" -c 'import sys; assert (3,10) <= sys.version_info[:2] <= (3,12), "Existing venv needs Python 3.10-3.12"'

# uv pip is a standalone command, NOT 'python -m pip'. It directly installs
# wheels into the specified existing venv even when pip/ensurepip are absent.
echo "Installing pinned CPU PyTorch and experiment dependencies with uv..."
"$UV" pip install --python "$FULLTD_ENV/bin/python" --index-url "$TORCH_INDEX" 'torch==2.5.1'
"$UV" pip install --python "$FULLTD_ENV/bin/python" -r "$ROOT/requirements-fulltd.txt"
"$UV" pip check --python "$FULLTD_ENV/bin/python"
"$UV" pip freeze --python "$FULLTD_ENV/bin/python" > "$FULLTD_DATA_DIR/environment-freeze.txt"
"$FULLTD_ENV/bin/python" -c 'import torch, numpy, pandas, scipy, matplotlib; print("Verified:", torch.__version__, "CPU PyTorch", numpy.__version__)'
echo "CPU environment ready: $FULLTD_ENV"
echo "Next: bash dscpuscripts/download_data.sh fd001"
