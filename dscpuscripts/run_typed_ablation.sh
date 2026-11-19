#!/usr/bin/env bash
# Launch ONLY the new typed TD ablations, using the same environment as existing runs.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/dscpuscripts/env.sh"
cd "$ROOT"
if [[ ! -x "$FULLTD_ENV/bin/python" ]]; then
  printf 'Missing Python environment: %s\n' "$FULLTD_ENV/bin/python" >&2
  exit 1
fi
exec "$FULLTD_ENV/bin/python" -m dscpuscripts.typed_ablation_runner "$@"
