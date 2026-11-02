#!/usr/bin/env bash
#SBATCH --job-name=fulltd_gw_smoke
#SBATCH --output=fulltd_gw_smoke_%j.out
#SBATCH --error=fulltd_gw_smoke_%j.err
#SBATCH --partition=tiny
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --export=ALL
# Run with: sbatch fulltd_graphwavenet_smoke.sbatch
# Uses 1 CPU on a compute node; does not modify existing fullTD results.
# To select another METR-LA Graph WaveNet candidate:
#   FULLTD_SMOKE_RUN_ID=metr_la_graph_wavenet_td_full_b256_h0_s0 sbatch --export=ALL fulltd_graphwavenet_smoke.sbatch
set -euo pipefail

REPO="${FULLTD_REPO:-$HOME/reproduceSL}"
source "$REPO/scripts/fulltd_environment.sh"
cd "$REPO"

BASE="${FULLTD_SMOKE_BASE:-$HOME/scratch/fulltd}"
mkdir -p -- "$BASE"
SMOKE_DIR="$(mktemp -d "$BASE/gw-smoke.XXXXXXXX")"
cleanup() {
  rc=$?
  if [[ $rc -eq 0 ]]; then
    rm -rf -- "$SMOKE_DIR"
    echo "Smoke-test temporary files cleaned. Main results were never touched."
  else
    echo "Smoke test exited with code $rc. Temporary files preserved at: $SMOKE_DIR" >&2
  fi
}
trap cleanup EXIT

"$FULLTD_ENV/bin/python" - "$SMOKE_DIR" <<'PY'
import json
import os
import sys
import time
from pathlib import Path

import torch
from fulltd.data import load_dataset
from fulltd.protocol import load_manifest
from fulltd.train import train_candidate

scratch = Path(sys.argv[1])
run_id = os.environ.get("FULLTD_SMOKE_RUN_ID", "metr_la_graph_wavenet_td_full_b128_h0_s1")
manifest = load_manifest("configs/fulltd/manifest.json")
matching = [c for c in manifest if c["run_id"] == run_id]
if len(matching) != 1:
    raise ValueError(f"Unknown fullTD run ID: {run_id}")
config = matching[0]
if config["dataset"] != "metr_la" or config["backbone"] != "graph_wavenet":
    raise ValueError("Smoke test must select a METR-LA Graph WaveNet configuration")

# Match the original single-thread CPU training path. For batch128, the
# original default microbatch256 is effectively physical microbatch128.
torch.set_num_threads(1)
if torch.get_num_interop_threads() != 1:
    torch.set_num_interop_threads(1)
data_dir = os.environ.get("FULLTD_DATA_DIR", str(Path.home() / "scratch/fulltd/data"))
train = load_dataset("metr_la", data_dir, "train")
validation = load_dataset("metr_la", data_dir, "validation")

print("SMOKE TEST ONLY — no production checkpoint or result changes", flush=True)
print("Run ID:", run_id, flush=True)
print("Device: CPU; training batches: 2; microbatch:", config["batch_size"], flush=True)
t0 = time.perf_counter()
result = train_candidate(config, train, validation, scratch,
                         device="cpu", microbatch=config["batch_size"],
                         stop_after_batches=2)
elapsed = time.perf_counter() - t0

# The test interruption intentionally discards uncheckpointed model steps.
# Read the *discarded_counters*, not the zeroed durable status counters.
run_dir = scratch / run_id
interruptions = json.loads((run_dir / "interruptions.json").read_text())
last = interruptions[-1]
counts = last["discarded_counters"]
print("Elapsed including setup (seconds):", round(elapsed, 2), flush=True)
print("Actual optimizer steps:", counts["optimizer_steps"], flush=True)
print("Actual logical batches:", counts["update_opportunities"], flush=True)
print("Valid TD pairs:", counts["pair_valid_count"], flush=True)
print("Result state (expected):", result["state"], flush=True)
if result["state"] != "stopped_no_checkpoint":
    raise RuntimeError(f"Unexpected smoke-test stop state: {result['state']}")
if counts["update_opportunities"] != 2 or counts["optimizer_steps"] != 2:
    raise RuntimeError("Two expected parameter updates did not complete")
if counts["pair_valid_count"] <= 0:
    raise RuntimeError("No valid TD training pairs encountered")
print("PASS: 2 Graph WaveNet TD optimization steps completed with finite loss/gradients/parameters.", flush=True)
print("NOTE: This does not test epoch-boundary checkpointing or scheduled validation.", flush=True)
PY
