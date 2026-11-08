#!/usr/bin/env bash
# AMD CPU launcher: pass --dry-run, --status, or --run (then worker options).
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$ROOT/dscpuscripts/run_standard.sh" electricity "$@"
