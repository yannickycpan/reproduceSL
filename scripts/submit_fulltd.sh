#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
VENV="${FULLTD_ENV:-$ROOT/.venv-fulltd}"
exec "$VENV/bin/python" -m fulltd.hpc "$@"
