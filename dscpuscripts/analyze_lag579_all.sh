#!/usr/bin/env bash
# One-command validation selection, endpoint evaluation, reconciliation, and packaging.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/dscpuscripts/env.sh"
cd "$ROOT"
DATASET="${1:-fd001}"
EVAL_WORKERS="${2:-8}"
if [[ $# -gt 2 || ( "$DATASET" != "fd001" && "$DATASET" != "electricity" ) ||
      ! "$EVAL_WORKERS" =~ ^[0-9]+$ ]] || (( EVAL_WORKERS < 1 || EVAL_WORKERS > 32 )); then
  echo "Usage: bash dscpuscripts/analyze_lag579_all.sh [fd001|electricity] [EVAL_WORKERS=8 (1..32)]" >&2
  exit 2
fi
if [[ ! -x "$FULLTD_ENV/bin/python" ]]; then
  echo "Missing Python environment: $FULLTD_ENV/bin/python" >&2
  exit 1
fi

# Resolve the same configured roots that the training and analysis modules use.
mapfile -t ROOTS < <("$FULLTD_ENV/bin/python" - "$DATASET" <<'PY'
import sys
from dscpuscripts.lag579_runner import paths as new_paths
from dscpuscripts.typed_ablation_runner import paths as old_paths
print(new_paths(sys.argv[1])[1])
print(old_paths(sys.argv[1])[1])
PY
)
if [[ "${#ROOTS[@]}" -ne 2 ]]; then
  echo "Cannot determine analysis result roots" >&2
  exit 1
fi
NEW_ROOT="${ROOTS[0]}"
ABL_ROOT="${ROOTS[1]}"
NEW_OUTPUT="$NEW_ROOT/lag579_outputs"
ABL_OUTPUT="$ABL_ROOT/typed_ablation_outputs"
mkdir -p "$NEW_ROOT"
exec 9>"$NEW_ROOT/.lag579_analysis_all.lock"
if ! flock -n 9; then
  echo "A complete lag analysis is already running in: $NEW_ROOT" >&2
  exit 1
fi

echo "Dataset: $DATASET"
if [[ "$DATASET" == "fd001" ]]; then
  echo "Reusing original five-lag training and selected outputs from: $ABL_ROOT"
else
  echo "Electricity: evaluating the new five-lag-only TD sweep with the original MSE reference"
fi
echo "Reading new TD results from: $NEW_ROOT"

# The full Electricity report requires the baseline explicitly requested by
# the experiment protocol. Never silently discard MSE from the final output.
if [[ "$DATASET" == "electricity" ]]; then
  "$FULLTD_ENV/bin/python" - <<'PY'
import sys
from collections import Counter
from dscpuscripts.application_protocol import make_manifest
from dscpuscripts.application_train import code_hash
from dscpuscripts.result_paths import paths
from dscpuscripts.standard_runner import classify

configs = [c for c in make_manifest()
           if c["dataset"] == "electricity" and c["method"] == "mse"
           and c["batch_size"] == 256]
if len(configs) != 50:
    raise RuntimeError("Expected exactly 50 Electricity MSE training candidates")
root = paths("electricity")[1]
expected = code_hash(configs[0])
states = Counter(classify(c, root, expected) for c in configs)
print(f"Fresh MSE baseline status: {dict(states)} (50 expected).", flush=True)
if any(s not in ("complete", "numerical_failure") for s in states):
    print("MSE baseline not finished. Run:\n"
          "  bash dscpuscripts/run_electricity.sh --run --method mse --workers 64",
          file=sys.stderr)
    raise SystemExit(1)
PY
fi

# Reconstruct old FD001 five-lag artifacts only where the current plan reuses them.
if [[ "$DATASET" == "fd001" ]]; then
if [[ ! -f "$ABL_OUTPUT/selection.json" ]]; then
  echo "Restoring missing five-lag validation selection..."
  bash dscpuscripts/analyze_typed_ablation.sh "$DATASET" --select
fi
if [[ ! -f "$ABL_OUTPUT/four_method_selection.json" ]]; then
  echo "Restoring missing four-method validation selection..."
  bash dscpuscripts/analyze_typed_ablation.sh "$DATASET" --compare-select
fi
if [[ ! -f "$ABL_OUTPUT/test_per_seed_metrics.csv" ]]; then
  echo "Restoring missing five-lag test metrics from saved models..."
  bash dscpuscripts/analyze_typed_ablation.sh "$DATASET" --evaluate
fi
if [[ ! -f "$ABL_OUTPUT/four_method_per_seed.csv" ]]; then
  echo "Restoring missing four-method report from saved models..."
  bash dscpuscripts/analyze_typed_ablation.sh "$DATASET" --compare-evaluate
fi
fi

echo "Auditing and selecting the prescribed TD lag candidates..."
bash dscpuscripts/analyze_lag579.sh "$DATASET" --select
echo "Evaluating selected TD endpoints and the validation-selected MSE reference..."
bash dscpuscripts/analyze_lag579.sh "$DATASET" --evaluate --workers "$EVAL_WORKERS"

# Validate the actual finished reports before producing any distributable archive.
"$FULLTD_ENV/bin/python" - "$DATASET" "$NEW_OUTPUT" <<'PY'
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from dscpuscripts.lag579_protocol import LAG_SETS, METHODS
import json

dataset, output = sys.argv[1], Path(sys.argv[2])
with (output/"selection.json").open(encoding="utf-8") as f:
    selection = json.load(f)
has_mse = dataset=="fd001" or selection.get("mse_reference") is not None
metrics = {("cycles","rmse"), ("cycles","mae"), ("score","rul_score_sum")} if dataset=="fd001" else {("kW","rmse"), ("kW","mae")}
lag_keys = ["-".join(str(n) for n in lags) for lags in LAG_SETS[dataset]]
groups_expected = {(m,lag) for lag in lag_keys for m in METHODS}
if has_mse:
    groups_expected.add(("mse","none"))
else:
    print("NOTICE: Historical MSE is missing; verifying the three TD methods only.")
keys_expected = {(m,lag,u,k) for m,lag in groups_expected for u,k in metrics}

def read(name):
    path = output/name
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"Missing completed analysis file: {path}")
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

per = read("per_seed.csv")
groups = defaultdict(list)
for row in per:
    if row["dataset"] != dataset or int(row["batch_size"]) != 256:
        raise ValueError("Inconsistent dataset/batch in output")
    value = float(row["value"])
    if not math.isfinite(value):
        raise ValueError("Nonfinite output metric")
    groups[(row["method"],row["lag_set"],row["units"],row["metric"])].append((int(row["seed"]),value))
if set(groups) != keys_expected:
    raise ValueError("Final comparison has missing or additional method/lag/metric groups")
if any(sorted(seed for seed,_ in values) != [0,1,2,3,4] for values in groups.values()):
    raise ValueError("Final comparison requires five distinct seeds per group")

summary = read("summary.csv")
if len(summary) != len(keys_expected):
    raise ValueError("Incomplete summary row count")
seen = set()
for row in summary:
    key = (row["method"],row["lag_set"],row["units"],row["metric"])
    if key in seen or key not in groups or int(row["n"]) != 5:
        raise ValueError("Invalid summary identity")
    seen.add(key)
    values = [v for _,v in sorted(groups[key])]
    if not math.isclose(float(row["mean"]), mean(values), rel_tol=1e-11, abs_tol=1e-10):
        raise ValueError("Summary mean does not match the five saved seeds")
    if not math.isclose(float(row["standard_error"]), stdev(values)/(5**.5), rel_tol=1e-11, abs_tol=1e-10):
        raise ValueError("Summary standard error does not match the five saved seeds")

paired = read("paired.csv")
if len(paired) != len(lag_keys)*len(metrics)*(3 if has_mse else 2):
    raise ValueError("Paired-comparison report incomplete")
if len(read("candidate_validation_scores.csv")) != (150 if dataset=="fd001" else 75):
    raise ValueError("Seven/nine-lag validation search is incomplete")
print(f"Validated {len(per)} per-seed records, {len(summary)} summaries, {len(paired)} comparisons.")
PY

ARCHIVE="$HOME/${DATASET^^}_lag579_final.tar.gz"
TMP="$(mktemp "$HOME/.${DATASET}_lag579_final.XXXXXX.tar.gz")"
trap 'rm -f "$TMP"' EXIT
if [[ "$DATASET" == "fd001" ]]; then
  tar -czf "$TMP" -C "$NEW_ROOT" lag579_outputs -C "$ABL_ROOT" typed_ablation_outputs
else
  tar -czf "$TMP" -C "$NEW_ROOT" lag579_outputs
fi
if [[ ! -s "$TMP" ]]; then
  echo "Failed to create archive" >&2
  exit 1
fi
mv -f "$TMP" "$ARCHIVE"
trap - EXIT
echo "COMPLETE: $ARCHIVE"
echo "Main result: $NEW_OUTPUT/summary.csv"
echo "Paired comparisons: $NEW_OUTPUT/paired.csv"
