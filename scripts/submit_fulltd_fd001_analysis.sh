#!/usr/bin/env bash
# Verify FD001 completion, then submit the existing FD001-only analysis pipeline.
# Usage: bash scripts/submit_fulltd_fd001_analysis.sh
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source "$ROOT/scripts/fulltd_environment.sh"

# Never evaluate held-out FD001 data while some candidates are unfinished.
bash "$ROOT/scripts/check_fulltd_fd001.sh"

export FULLTD_DATASET=fd001
export FULLTD_OUTPUT_DIR="${FULLTD_FD001_OUTPUT_DIR:-$HOME/scratch/fulltd/outputs_fd001}"
mkdir -p -- "$FULLTD_OUTPUT_DIR" "$FULLTD_RESULTS_DIR/logs"

# Avoid accidentally scheduling duplicate endpoint analyses.
if squeue --noheader --user "$USER" --format '%j' | grep -Fxq 'fulltd_fd001_analysis'; then
    echo 'ERROR: an FD001 analysis job is already queued/running; not submitting another.' >&2
    exit 1
fi

PARTITION="${FULLTD_FD001_ANALYSIS_PARTITION:-tiny}"
TIME="${FULLTD_FD001_ANALYSIS_TIME:-01:00:00}"
MEMORY="${FULLTD_FD001_ANALYSIS_MEMORY:-32G}"
LOGS="$FULLTD_RESULTS_DIR/logs"

# --wrap consumes a single Bash command; never pass a separate script argument.
job_id="$(sbatch --parsable \
    --partition="$PARTITION" \
    --nodes=1 --ntasks=1 --cpus-per-task=1 \
    --mem="$MEMORY" --time="$TIME" \
    --job-name=fulltd_fd001_analysis \
    --chdir="$ROOT" \
    --output="$LOGS/fd001_analysis_%j.out" \
    --error="$LOGS/fd001_analysis_%j.err" \
    --export=ALL \
    --wrap='bash scripts/analyze_fulltd.sh')"
job_id="${job_id%%;*}"
echo "Submitted FD001-only analysis job: $job_id"
echo "Outputs: $FULLTD_OUTPUT_DIR"
echo "Logs: $LOGS/fd001_analysis_${job_id}.out and .err"
echo 'The existing script selects, evaluates and plots FD001; no training is submitted.'
