#!/usr/bin/env bash
# Submit analysis of the NEW Gamma sweep only, after every candidate is terminal.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/fulltd_environment.sh"
cd "$ROOT"
"$FULLTD_ENV/bin/python" -m fd001_gamma.analysis check
export FULLTD_FD001_GAMMA_RESULTS_DIR="${FULLTD_FD001_GAMMA_RESULTS_DIR:-$HOME/scratch/fulltd/fd001_gamma_v1/results}"
export FULLTD_FD001_GAMMA_OUTPUT_DIR="${FULLTD_FD001_GAMMA_OUTPUT_DIR:-$HOME/scratch/fulltd/fd001_gamma_v1/outputs}"
LOGS="$FULLTD_FD001_GAMMA_RESULTS_DIR/logs"
mkdir -p -- "$LOGS"
QUEUE="$(squeue --noheader --user "$USER" --format '%100j')" || {
    echo 'ERROR: cannot inspect scheduler; no analysis submitted.' >&2
    exit 1
}
if printf '%s\n' "$QUEUE" | grep -Eq '^[[:space:]]*fulltd_fd001_gamma_analysis[[:space:]]*$'; then
    echo 'ERROR: Gamma analysis already queued/running.' >&2
    exit 1
fi
printf -v COMMAND 'set -euo pipefail; source %q; exec %q -m fd001_gamma.analysis run' "$ROOT/scripts/fulltd_environment.sh" "$FULLTD_ENV/bin/python"
printf -v WRAP 'exec bash -c %q' "$COMMAND"
sbatch --parsable --partition=tiny --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=32G --time=01:00:00 \
    --job-name=fulltd_fd001_gamma_analysis --chdir="$ROOT" --export=ALL \
    --output="$LOGS/gamma_analysis_%j.out" --error="$LOGS/gamma_analysis_%j.err" --wrap="$WRAP"
echo "Analysis outputs: $FULLTD_FD001_GAMMA_OUTPUT_DIR"
