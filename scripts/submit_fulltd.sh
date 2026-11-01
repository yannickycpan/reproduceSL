#!/usr/bin/env bash
# Default: array 0-6%7, one node and 90 independent single-thread tasks per element.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
VENV="${FULLTD_ENV:-$HOME/scratch/fulltd/.venv-fulltd}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
exec "$VENV/bin/python" -m fulltd.hpc "$@"
