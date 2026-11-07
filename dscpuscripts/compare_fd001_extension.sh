#!/usr/bin/env bash
# Backward-compatible alias; reporting logic lives in compare_fd001.sh.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$ROOT/dscpuscripts/compare_fd001.sh" "$@"
