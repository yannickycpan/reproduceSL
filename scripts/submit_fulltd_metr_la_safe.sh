#!/usr/bin/env bash
# OOM-safe restart of unfinished METR-LA Graph WaveNet candidates only.
# FD001/Electricity are never submitted or inspected for candidate selection.
# Run: bash scripts/submit_fulltd_metr_la_safe.sh --dry-run
#      bash scripts/submit_fulltd_metr_la_safe.sh --submit
set -euo pipefail

if [[ $# -ne 1 || ( "$1" != --dry-run && "$1" != --submit ) ]]; then
    printf 'Usage: bash scripts/submit_fulltd_metr_la_safe.sh --dry-run|--submit\n' >&2
    exit 2
fi
MODE="$1"
ROOT="${FULLTD_REPO:-$HOME/reproduceSL}"
ROOT="$(cd -- "$ROOT" && pwd -P)"
SCRIPT="$ROOT/scripts/submit_fulltd_metr_la_safe.sh"
if [[ ! -f "$ROOT/configs/fulltd/manifest.json" || ! -f "$SCRIPT" ]]; then
    echo 'ERROR: FULLTD_REPO must be the checked-out fullTD repository.' >&2
    exit 1
fi
source "$ROOT/scripts/fulltd_environment.sh"
cd -- "$ROOT"
RESULTS="${FULLTD_RESULTS_DIR:-$HOME/scratch/fulltd/results}"
LOGS="$RESULTS/logs"

# Do not overlap workers from an old mixed METR-LA/FD001 array. A deferred
# coordinator waits until those arrays terminate before selecting pending runs.
command -v squeue >/dev/null || { echo 'ERROR: squeue unavailable.' >&2; exit 1; }
QUEUE="$(squeue --array --noheader --user "$USER" --format '%F|%50j|%T')" || {
    echo 'ERROR: cannot inspect active Slurm jobs; refusing submission.' >&2; exit 1;
}
LIVE=()
COORD=()
while IFS='|' read -r job_id name state; do
    job_id="${job_id//[[:space:]]/}"
    name="${name#"${name%%[![:space:]]*}"}"
    name="${name%"${name##*[![:space:]]}"}"
    [[ "$job_id" =~ ^[0-9]+$ ]] || continue
    case "$name" in
      fulltd_cpu_metr_la|fulltd_cpu_metr_la_fd001|fulltd_cpu_all)
          LIVE+=("$job_id") ;;
      fulltd_submit_remaining|fulltd_gw_safe_submit)
          if [[ "$job_id" != "${SLURM_JOB_ID:-}" ]]; then COORD+=("$job_id"); fi ;;
    esac
done <<< "$QUEUE"

if [[ "$MODE" == --submit && ${#COORD[@]} -gt 0 ]]; then
    printf 'ERROR: existing fullTD submission coordinator(s): %s. Nothing submitted.\n' "${COORD[*]}" >&2
    exit 1
fi
if [[ ${#LIVE[@]} -gt 0 ]]; then
    mapfile -t LIVE < <(printf '%s\n' "${LIVE[@]}" | sort -u)
    if [[ "$MODE" == --dry-run ]]; then
        printf 'NOTE: active METR-LA-containing arrays: %s. --submit will defer until they terminate.\n' "${LIVE[*]}"
    else
        if [[ -n "${SLURM_JOB_ID:-}" ]]; then
            echo 'ERROR: a mixed METR-LA array is still active inside the deferred coordinator; not overlapping.' >&2
            exit 1
        fi
        mkdir -p -- "$LOGS"
        dependency="$(IFS=:; echo "${LIVE[*]}")"
        printf 'Active METR-LA-containing arrays: %s\n' "${LIVE[*]}"
        printf 'Deferring restart until those arrays end; no training overlaps.\n'
        sbatch --parsable --partition=tiny --job-name=fulltd_gw_safe_submit \
            --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=32G --time=00:15:00 \
            --dependency="afterany:$dependency" --export=ALL --chdir="$ROOT" \
            --output="$LOGS/fulltd_gw_safe_submit_%j.out" \
            --error="$LOGS/fulltd_gw_safe_submit_%j.err" \
            --wrap="$(printf 'exec bash %q --submit' "$SCRIPT")"
        exit 0
    fi
fi

# The canonical manifest is immutable. Select only unfinished METR-LA runs;
# normal fulltd.hpc validation/checkpoint/requeue remains authoritative.
"$FULLTD_ENV/bin/python" - "$ROOT" "$RESULTS" "$MODE" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

root, results, mode = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
from fulltd.protocol import digest, load_manifest
from fulltd.train import code_hash

configs = [config for config in load_manifest(root / 'configs/fulltd/manifest.json')
           if config['dataset'] == 'metr_la' and config['backbone'] == 'graph_wavenet']
if len(configs) != 210:
    raise SystemExit(f'Expected 210 canonical METR-LA runs; found {len(configs)}')
expected_code = code_hash()
pending = []
counts = dict(complete=0, numerical_failure=0, start_epoch_1=0, resume_checkpoint=0, repair_final=0)
for config in configs:
    directory = results / config['run_id']
    status_file = directory / 'status.json'
    status = json.loads(status_file.read_text()) if status_file.exists() else None
    final, recovery = directory / 'final.pt', directory / 'recovery.pt'
    if status is not None:
        identity = status.get('identity', {})
        if (status.get('config') != config or identity.get('code_sha256') != expected_code
                or identity.get('config_sha256') != digest(config)):
            raise SystemExit(f'ABORT: incompatible existing result {status_file}; nothing submitted')
        if status.get('state') == 'complete' and final.is_file():
            counts['complete'] += 1
            continue
        if status.get('state') == 'numerical_failure':
            counts['numerical_failure'] += 1
            continue
    category = 'repair_final' if final.is_file() else ('resume_checkpoint' if recovery.is_file() else 'start_epoch_1')
    counts[category] += 1
    pending.append(config)

print('METR-LA Graph WaveNet: 210 canonical candidates')
print('Counts:', ', '.join(f'{key}={value}' for key, value in counts.items()))
print(f'Unfinished: {len(pending)}')
print('Resources: 32 CPU workers/node, physical microbatch 32, node memory 400G, 18h, concurrency 2')
print('Logical batches (128/256), updates, epochs, and scientific code are unchanged.')
if not pending:
    print('Nothing to submit.')
    raise SystemExit(0)
if mode == '--dry-run':
    print('DRY RUN: no files or jobs modified.')
    raise SystemExit(0)

# Freeze a uniquely identified subset in the results area. This never edits
# tracked manifests, existing per-run data, or checkpoints.
key = hashlib.sha256(json.dumps(pending, sort_keys=True, separators=(',', ':')).encode()).hexdigest()[:20]
folder = results / 'resume_manifests'
folder.mkdir(parents=True, exist_ok=True)
subset = folder / f'metr_la_safe_{key}.json'
if subset.exists():
    if json.loads(subset.read_text()) != pending:
        raise SystemExit(f'ABORT: immutable manifest conflict at {subset}')
else:
    temporary = folder / f'.{subset.name}.{uuid.uuid4().hex}.tmp'
    temporary.write_text(json.dumps(pending, indent=2) + '\n')
    os.replace(temporary, subset)
print('Submitting pending METR-LA runs with the existing fulltd.hpc runner:', subset, flush=True)
subprocess.run([sys.executable, '-m', 'fulltd.hpc', '--manifest', str(subset),
                '--dataset', 'metr_la', '--partition', 'batch', '--time', '18:00:00',
                '--concurrency', '2', '--tasks-per-node', '32', '--microbatch', '32',
                '--memory', '400G', '--data-dir', os.environ['FULLTD_DATA_DIR'],
                '--results-dir', str(results), '--log-dir', str(results / 'logs')], check=True)
PY
