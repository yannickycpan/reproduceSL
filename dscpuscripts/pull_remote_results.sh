#!/usr/bin/env bash
# On the primary server: copy disjoint finished FD001 Gamma runs and import safely.
# Usage: bash dscpuscripts/pull_remote_results.sh USER@SECONDARY_HOST [remote-results-path]
set -euo pipefail
[[ $# -ge 1 && $# -le 2 ]] || {
  echo "Usage: bash dscpuscripts/pull_remote_results.sh USER@HOST [remote-results-path]" >&2
  exit 2
}
REMOTE="$1"
REMOTE_DIR="${2:-~/scratch/fulltd/fd001_gamma_v1/results}"
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/dscpuscripts/env.sh"
cd "$ROOT"
command -v rsync >/dev/null 2>&1 || { echo "rsync is required to transfer server results." >&2; exit 1; }
PYTHON="$FULLTD_ENV/bin/python"
[[ -x "$PYTHON" ]] || { echo "Run bash dscpuscripts/setup_server.sh first." >&2; exit 1; }
STAGING="$(dirname "$FULLTD_FD001_GAMMA_RESULTS_DIR")/from_secondary_server"
mkdir -p "$STAGING"
# Include ONLY canonical run directories; exclude remote logs and shared files.
rsync -av --prune-empty-dirs \
  --include='fd001_gamma_v1_*/' --include='fd001_gamma_v1_*/***' --exclude='*' \
  "$REMOTE:$REMOTE_DIR/" "$STAGING/"
"$PYTHON" -m dscpuscripts.import_results --source "$STAGING" --dry-run
"$PYTHON" -m dscpuscripts.import_results --source "$STAGING"
echo "All imported results: $FULLTD_FD001_GAMMA_RESULTS_DIR"
