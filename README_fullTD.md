# TD v7 neural applications — CityU HPC

This branch supplies the complete **630-candidate** campaign code: 450 primary
and 180 TD batch-sensitivity runs. The campaign has **not** been trained here.
All original datasets were downloaded and preprocessing was checked locally;
CPU tests check mathematics and execution. GPU execution is checked by a Slurm
preflight job on your allocated compute node before candidate jobs start.

This entry point is independent of the repository's legacy TensorFlow `main.py`
and its split/model-selection rules. Algorithms are separate `agents/fulltd_*.py`
files; data, models, fitting and analysis are in `fulltd/`.

## Frozen protocol

- Electricity: **100 stochastic rounds**, each with exactly `ceil(N/24)` iid
  uniform draws **with replacement** from all eligible customer/origin windows.
  Partners use the full training universe with the unchanged 1/24-hour law.
- METR-LA and FD001: 100 complete, shuffled source passes, including short batches.
- Six configurations × seeds 0–4 for every prescribed method/batch combination.
  TD batches are 256 pairs (primary) and 128 pairs (sensitivity); ordinary batches
  are 256 sources. Rates are constant Adam rates; full TD gradients and exactly
  zero signed operators. No clipping, scheduling, penalties or early stopping.
- Complete validation at 10,20,...,100; one common configuration selected using
  ordinary MSE averaged over **80,90,100 and all five seeds**, separately per batch.
  Report the actual epoch/round-100 model. Test evaluation is a separate command.

See [protocol details](docs/fulltd_protocol.md), [architectures and attribution](docs/fulltd_architectures.md),
and [measured data checks](docs/fulltd_data_verification.json).

## 1. Pull, install and download

```bash
git fetch origin
git switch --track origin/fullTD  # first checkout; otherwise git switch fullTD && git pull
export FULLTD_DATA_DIR=/path/to/approved/project/storage/fulltd-data
export FULLTD_RESULTS_DIR=/path/to/approved/project/storage/fulltd-results
export FULLTD_OUTPUT_DIR=/path/to/approved/project/storage/fulltd-analysis
export FULLTD_PYTHON=python3.12
# Optional: FULLTD_ENV=/path/to/project/storage/fulltd-env
bash scripts/setup_fulltd.sh
```

Python 3.12.14 and CPU PyTorch 2.5.1 were tested. Setup defaults to the same
PyTorch release's CUDA 11.8 wheel; verify your assigned GPU driver, or set
`FULLTD_TORCH_INDEX=https://download.pytorch.org/whl/cu121` (or `cpu` for checks).
Use your site's existing Python module or installation; module names are not
assumed. Setup records the installed package versions and raw-file source URLs,
retrieval dates and SHA-256 hashes. Original UCI 370-customer data are required;
the 321-customer benchmark is not a replacement. METR-LA uses the authors' Drive
file and pinned graph. Downloads are reusable and hash checked.

## 2. Submit the complete campaign

CityU Burgundy uses Slurm. Run production computation on allocated nodes.
Verify your actual partition, GRES, limits and storage allocation with `sinfo`,
`scontrol show partition`, `scontrol show config` and your account documentation.
The detailed CityU guide was access-restricted; no queue name is guessed here.
Use project storage with enough quota for raw data, processed arrays and
candidate/recovery checkpoints, not the small default home allocation.

```bash
bash scripts/submit_fulltd.sh --partition YOUR_GPU_PARTITION --gres gpu:1 \
  --data-dir "$FULLTD_DATA_DIR" --results-dir "$FULLTD_RESULTS_DIR" \
  --concurrency 4 --array-chunk 100 --memory 32G --time 24:00:00
```

Add `--account`, `--qos` or `--constraint` if your allocation requires them.
Use `--dry-run` to print commands without submission. A GPU preflight preprocesses
missing datasets, verifies hashes, checks exact source/update counts and measures
three throwaway updates per backbone. Candidate arrays depend on its success.
Array chunks run sequentially, at most `--concurrency` simultaneous candidates
per invocation. Choose chunk and time limits permitted by your account. Use
`--dataset electricity`, `metr_la` or `fd001` to submit a dataset separately;
their candidate counts are 300, 210 and 120. A `submission_*.json` receipt records
job IDs. These are real submitted jobs only when run on HPC; local dry runs
do not submit or continue work in the background.

Re-run the same command after time limits or operational failures. Completed
compatible candidates and recorded numerical failures are preserved; interrupted
runs restore the rolling checkpoint. Do not resubmit while old jobs are active.
`--microbatch 32` changes physical memory use while preserving logical batches
and one globally normalized Adam update. CUDA forward/backward OOM retries with
smaller physical chunks; other resource failures remain operational failures.

For an individual manifest entry on a compute node:
```bash
.venv-fulltd/bin/python -m fulltd.train --manifest configs/fulltd/manifest.json \
  --index 0 --data-dir "$FULLTD_DATA_DIR" --results-dir "$FULLTD_RESULTS_DIR" --device cuda
```

## 3. Select, evaluate and regenerate figures

After all jobs for a dataset complete, run on a compute node (endpoint evaluation
and the full-data frozen-readout fits can be substantial):
```bash
export FULLTD_DATASET=all  # or electricity / metr_la / fd001; use a separate output directory per scope
bash scripts/analyze_fulltd.sh
```

The script audits every candidate before selecting, evaluates final train/val/test
predictions, computes the two prescribed frozen-readout diagnostics, and writes
per-seed metrics, pooled scores, failure records, paired comparisons, sensitivity
tables, and caption-free PDF/SVG figures with CSV sources. It retains per-engine
FD001 endpoint predictions. No LaTeX/paper report is generated at this stage.
Rebuild tables/figures from saved metrics without model inference or retraining:
```bash
.venv-fulltd/bin/python -m fulltd.analysis summarize \
  --metrics "$FULLTD_OUTPUT_DIR/metrics.csv" --table-dir "$FULLTD_OUTPUT_DIR" \
  --figures-dir "$FULLTD_OUTPUT_DIR/figures"
```

## Verification and stored records

```bash
.venv-fulltd/bin/python -m pytest tests -q
.venv-fulltd/bin/python -m fulltd.data verify --dataset all --data-dir "$FULLTD_DATA_DIR"
```

CPU synthetic checks are correctness evidence, not application results. Each run
stores its exact config, ten validation values, source/partner sequence hashes,
counts, runtime, initial-state identity, final checkpoint and one rolling recovery
checkpoint. Shared initial mean states are saved once per backbone/seed. All
candidate final checkpoints remain available; this code does not delete them.

No previously completed FD001 checkpoint was present in the supplied repository.
Keep the four earlier runs in their original location. Do not relabel them with
the new protocol identifier. Reuse requires evidence of matching data, objective,
mean initialization, pair/source streams, rates, final weights and validation
records at 80/90/100; absent that evidence, run the corresponding manifest entries
in a fresh results directory. Existing incompatible checkpoints are rejected
without being overwritten. See [verification record](docs/fulltd_verification.md).
