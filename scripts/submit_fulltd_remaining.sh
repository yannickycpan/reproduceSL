#!/usr/bin/env bash
# Safe, protocol-preserving restart of only unfinished fullTD candidates.
# Default: METR-LA + FD001; set FULLTD_SCOPE=electricity for the other campaign.
# Usage: bash submit_fulltd_remaining.sh --dry-run | --submit
# --submit automatically defers submission until existing fullTD training arrays end.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash submit_fulltd_remaining.sh --dry-run
       bash submit_fulltd_remaining.sh --submit

Environment overrides (optional):
  FULLTD_REPO                 default: $HOME/reproduceSL (checked-out fullTD branch)
  FULLTD_SCOPE                default: metr_la_fd001 (or electricity, metr_la, fd001, all)
  FULLTD_REMAINING_TIME       default: 12:00:00 (e.g. 24:00:00)
  FULLTD_CONCURRENCY         default: 4
  FULLTD_TASKS_PER_NODE      default: 90
  FULLTD_MEMORY              default: 400G
  FULLTD_COORD_PARTITION     default: tiny (for deferred 15-min coordinator)
  FULLTD_COORD_MEMORY        default: 32G
  FULLTD_PARTITION           default: batch
  FULLTD_RESULTS_DIR         default: $HOME/scratch/fulltd/results
  FULLTD_DATA_DIR            default: $HOME/scratch/fulltd/data

The original fulltd.hpc worker and epoch-boundary checkpoint/resume logic are used.
EOF
}

if [[ $# != 1 ]] || [[ "$1" != "--dry-run" && "$1" != "--submit" ]]; then
    usage >&2
    exit 2
fi
MODE="$1"
ROOT="${FULLTD_REPO:-$HOME/reproduceSL}"
if [[ ! -d "$ROOT" ]]; then
    echo "ERROR: repository directory missing: $ROOT" >&2; exit 1
fi
ROOT="$(cd -- "$ROOT" && pwd -P)"
SCRIPT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)/$(basename -- "${BASH_SOURCE[0]}")"
if [[ ! -f "$ROOT/fulltd/hpc.py" || ! -f "$ROOT/configs/fulltd/manifest.json" ]]; then
    echo "ERROR: FULLTD_REPO must point to the checked-out fullTD branch: $ROOT" >&2; exit 1
fi

# Use the repository's own CityU modules, Python and result/data locations.
source "$ROOT/scripts/fulltd_environment.sh"
cd -- "$ROOT"
RESULTS="${FULLTD_RESULTS_DIR:-$HOME/scratch/fulltd/results}"
LOGS="$RESULTS/logs"
SCOPE="${FULLTD_SCOPE:-metr_la_fd001}"
TIME="${FULLTD_REMAINING_TIME:-12:00:00}"
CONCURRENCY="${FULLTD_CONCURRENCY:-4}"
TASKS="${FULLTD_TASKS_PER_NODE:-90}"
MEMORY="${FULLTD_MEMORY:-400G}"
PARTITION="${FULLTD_PARTITION:-batch}"

for pair in "CONCURRENCY:$CONCURRENCY" "TASKS_PER_NODE:$TASKS"; do
    value="${pair#*:}"
    if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: invalid positive integer $pair" >&2; exit 1
    fi
done

# Check the scheduler *before* planning. Do not assume a stale status means idle.
if ! command -v squeue >/dev/null 2>&1; then
    echo "ERROR: squeue is not available; refusing an unsafe submission." >&2; exit 1
fi
QUEUE="$(squeue --array --noheader --user "$USER" --format '%F|%50j|%T')" || {
    echo "ERROR: unable to inspect live Slurm jobs; refusing submission." >&2; exit 1;
}
mapfile -t LIVE_IDS < <(printf '%s\n' "$QUEUE" | awk -F'|' '
    {n=$2;gsub(/^[ \t]+|[ \t]+$/, "", n); if (n ~ /^fulltd_cpu_$/ || n ~ /^fulltd_cpu_/) {
        id=$1;gsub(/^[ \t]+|[ \t]+$/, "", id);if(id ~ /^[0-9]+$/) print id
    }}' | sort -u)
mapfile -t COORD_IDS < <(printf '%s\n' "$QUEUE" | awk -F'|' '
    {n=$2;gsub(/^[ \t]+|[ \t]+$/, "", n); if(n == "fulltd_submit_remaining") {
        id=$1;gsub(/^[ \t]+|[ \t]+$/, "", id);if(id ~ /^[0-9]+$/) print id
    }}' | sort -u)

# A previously queued coordinator is an independent concurrent submitter.
# Never make a second submission while it is still queued or running.
if [[ "$MODE" == "--submit" && ${#COORD_IDS[@]} -gt 0 ]]; then
    for coordinator_id in "${COORD_IDS[@]}"; do
        if [[ "$coordinator_id" != "${SLURM_JOB_ID:-}" ]]; then
            echo "ERROR: deferred fullTD coordinator $coordinator_id already exists; no duplicate submission." >&2
            exit 1
        fi
    done
fi

if [[ "$MODE" == "--submit" && ${#LIVE_IDS[@]} -gt 0 ]]; then
    # The deferred coordinator will recompute the unfinished set AFTER these
    # whole arrays finish. No currently running or queued configuration collides.
    if [[ -n "${SLURM_JOB_ID:-}" ]]; then
        echo "ERROR: another fullTD array is still present from the coordinator's view." >&2
        echo "Refusing to schedule another overlapping array. Retry manually later." >&2
        exit 1
    fi
    if [[ ${#COORD_IDS[@]} -gt 0 ]]; then
        printf 'ERROR: a deferred fullTD submission already exists: %s\n' "${COORD_IDS[*]}" >&2
        echo "Do not queue duplicates. Check its logs/state first." >&2
        exit 1
    fi
    mkdir -p -- "$LOGS"
    dep="$(IFS=:; echo "${LIVE_IDS[*]}")"
    echo "Live fullTD job-array IDs: ${LIVE_IDS[*]}"
    echo "Deferring re-selection and submission until ALL existing arrays terminate."
    echo "Scheduling small coordinator with dependency afterany:$dep (no training yet)."
    sbatch --parsable --partition="${FULLTD_COORD_PARTITION:-tiny}" \
        --job-name=fulltd_submit_remaining \
        --nodes=1 --ntasks=1 --cpus-per-task=1 \
        --mem="${FULLTD_COORD_MEMORY:-32G}" --time=00:15:00 \
        --dependency="afterany:$dep" --export=ALL \
        --chdir="$ROOT" \
        --output="$LOGS/fulltd_submit_remaining_%j.out" \
        --error="$LOGS/fulltd_submit_remaining_%j.err" \
        "$SCRIPT" --submit
    echo "After existing arrays terminate, check the coordinator's .out/.err log."
    exit 0
fi

if [[ "$MODE" == "--dry-run" && ${#LIVE_IDS[@]} -gt 0 ]]; then
    echo "NOTE: existing fullTD arrays are live: ${LIVE_IDS[*]}"
    echo "      --submit will defer its selection until those arrays finish."
fi

# Select only outstanding canonical configurations (not just directories).
# A subset manifest is valid under protocol.load_manifest; the per-run configs
# and scientific code hashes are unchanged. No training files are modified here.
"$FULLTD_ENV/bin/python" - "$ROOT" "$RESULTS" "$SCOPE" "$MODE" "$TIME" "$CONCURRENCY" "$TASKS" "$MEMORY" "$PARTITION" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

root = Path(sys.argv[1])
results = Path(sys.argv[2])
scope, mode, timelimit, concurrency, tasks, memory, partition = sys.argv[3:]

from fulltd.protocol import datasets_for_scope, digest, load_manifest
from fulltd.train import code_hash

manifest = load_manifest(root / "configs/fulltd/manifest.json")
scoped = [c for c in manifest if c["dataset"] in datasets_for_scope(scope)]
if not scoped:
    raise SystemExit(f"No candidate configurations for scope={scope!r}")

scientific_hash = code_hash()
pending = []
counts = dict(complete=0, numerical_failure=0, restart_epoch_1=0,
              resume_checkpoint=0, repair_final=0)
examples = []

for config in scoped:
    run_dir = results / config["run_id"]
    path = run_dir / "status.json"
    final = run_dir / "final.pt"
    recovery = run_dir / "recovery.pt"
    status = json.loads(path.read_text()) if path.exists() else None
    if status is not None:
        if status.get("config") != config:
            raise SystemExit(f"ABORT: incompatible configuration in {path} -- preserving old results")
        ident = status.get("identity", {})
        if (ident.get("code_sha256") != scientific_hash or
                ident.get("config_sha256") != digest(config)):
            raise SystemExit(f"ABORT: incompatible code/config identity in {path}. "
                             "Do not relabel or overwrite the previous runs.")
    if status is not None and status.get("state") == "complete" and final.is_file():
        counts["complete"] += 1
        continue
    if status is not None and status.get("state") == "numerical_failure":
        counts["numerical_failure"] += 1
        continue
    if final.is_file():
        # train_candidate validates, loads and repairs sidecars from final.pt.
        category = "repair_final"
    elif recovery.is_file():
        category = "resume_checkpoint"
    else:
        category = "restart_epoch_1"
    counts[category] += 1
    pending.append(config)
    if len(examples) < 15:
        epoch = status.get("next_epoch") if status is not None else None
        examples.append((config["run_id"], category, epoch))

print(f"Scope: {scope} | original candidates: {len(scoped)}")
print("Counts:", ", ".join(f"{k}={v}" for k, v in counts.items()))
print("  Pending to submit:", len(pending))
print("  Scientific code SHA-256:", scientific_hash)
for rid, category, epoch in examples:
    print(f"  {rid}: {category}; status.next_epoch={epoch}")

if not pending:
    print("No remaining non-terminal candidates. Nothing submitted.")
    raise SystemExit(0)
if mode == "--dry-run":
    print("DRY RUN: no jobs submitted, no manifests/results changed.")
    raise SystemExit(0)

# Store one small, immutable subset manifest outside tracked scientific sources.
# Immutable mapping/receipt files are then managed by the repository itself.
encoded = json.dumps(pending, sort_keys=True, separators=(",", ":")).encode()
key = hashlib.sha256(encoded).hexdigest()[:20]
folder = results / "resume_manifests"
folder.mkdir(parents=True, exist_ok=True)
subset = folder / f"remaining_{scope}_{key}.json"
if subset.exists():
    if json.loads(subset.read_text()) != pending:
        raise SystemExit(f"ABORT: existing manifest differs: {subset}")
else:
    tmp = folder / f".{subset.name}.{uuid.uuid4().hex}.tmp"
    tmp.write_text(json.dumps(pending, indent=2) + "\n")
    os.replace(tmp, subset)
print("Immutable subset manifest:", subset)
print("Submitting through original fulltd.hpc; no scientific code is patched.", flush=True)
subprocess.run([sys.executable, "-m", "fulltd.hpc",
                "--manifest", str(subset), "--dataset", scope,
                "--time", timelimit, "--concurrency", concurrency,
                "--tasks-per-node", tasks, "--memory", memory,
                "--partition", partition,
                "--data-dir", os.environ.get("FULLTD_DATA_DIR", str(Path.home() / "scratch/fulltd/data")),
                "--results-dir", str(results),
                "--log-dir", str(results / "logs")], check=True)
PY
