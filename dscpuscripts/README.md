# One AMD EPYC server: FD001 Gamma sweep

**New FD001 pairwise-Gamma and alpha-extension campaign:** see
[`STANDARD_CAMPAIGNS.md`](STANDARD_CAMPAIGNS.md#fd001-gamma-extension-two-representation-variants-140-new-runs)
for the 140 new runs, original-representation HP extension, baseline reruns,
and six-method plots with optimizer updates on the x-axis. This original
180-run campaign is frozen and its results are reused unchanged.

All operational scripts are inside `dscpuscripts/`. The scientific model, the
three Gamma variants, the 3-by-4 learning-rate grid, five seeds, original
sampler, training objective, Adam, and validation/test protocol are unchanged.
**All 180 configurations run on one machine.** No Slurm, cluster job time
limit, second server, sharding, copying or result merging is involved.

## First-time setup

On the dual AMD EPYC 9754 Ubuntu server:

```bash
git clone --branch fullTD https://github.com/yannickycpan/reproduceSL.git
cd reproduceSL
bash dscpuscripts/inspect_server.sh
bash dscpuscripts/setup_server.sh
bash dscpuscripts/download_data.sh fd001
```

If you already cloned the repository, instead run:

```bash
cd ~/reproduceSL
git switch fullTD
git pull --ff-only origin fullTD
```

`setup_server.sh` installs the existing version-pinned CPU requirements
with a standalone `uv` executable, **without Python pip, ensurepip, or sudo**.
It reuses an installed `uv` or downloads the official Astral installer into
`~/scratch/fulltd/.uv-bin/`; it does not modify shell profiles or system Python.
It also reuses an existing pip-less virtual environment without deleting it.
The official installer and Python package indexes require internet access.
If downloading `uv` is blocked, obtain its standalone Linux binary on another
machine, copy it to the server, and run `FULLTD_UV=/absolute/path/to/uv bash
dscpuscripts/setup_server.sh`.

The script accepts Python 3.10, 3.11 or 3.12 (Ubuntu 22.04 commonly uses 3.10).
If needed, specify an interpreter with `FULLTD_PYTHON=/path/to/python`.

`download_data.sh fd001` reuses the original NASA C-MAPSS downloader,
retains SHA-256/source sidecars, performs the original train/val/test split
and training-only standardization, and verifies every processed file. It
does not overwrite already prepared data. It can optionally download
`electricity`, `metr_la` or `all`, but this direct training campaign is
**FD001-only**.

## CPU concurrency: choose from the actual server state

AMD EPYC 9754 provides 128 physical cores and 256 SMT threads per socket:
dual socket means 256 physical cores and up to 512 logical CPUs.

The old 16-process suggestion was a conservative check, **not the capacity**.
An idle machine may support 96, 128 or more independent single-thread runs,
but the practical optimum depends on memory bandwidth, process memory and
other active users.

The login report showed load 246.91. On a shared 256-core machine, that is
a reason to measure current utilization rather than blindly start 128 jobs.
Linux load is not identical to instantaneous CPU utilization; it can include
I/O-waiting tasks. The script prints both a short CPU-idle measurement and
the load average. **It warns rather than blocking on a high load**.

By default, `--workers auto` measures CPU idleness **once at startup** and
chooses up to 128 processes. Fixed `--workers 48` or `--workers 128`
starts that many immediately and is **not load-aware**.

For ongoing load-aware dispatch, explicitly enable `--adaptive`. The
`--workers` value becomes the **simultaneous-process ceiling**. The runner
samples available physical-core capacity **at launch and every 600 seconds
(10 minutes)**. At each sample it admits up to **80% of currently idle
physical-core capacity**, subject to the ceiling and the number of pending
runs. For example, 100 estimated idle physical cores can admit up to 80 new
jobs; if only 30 pending runs remain, it admits 30. The previous arbitrary
8-jobs-per-check limit has been removed.

The cap is on **running** jobs: if 80 jobs are already active and the next
sample finds 50 idle cores, at most 40 more may be admitted (and never more
than `max_workers - 80`). Jobs that finish are immediately replaced up to
the latest allowed concurrency. If other users occupy additional cores,
no running jobs are terminated; the runner instead stops increasing its
concurrency until capacity recovers. If estimated idle capacity is zero,
no *additional* concurrent workers are launched.

This is an advisory 0.75-second CPU-idle measurement, **not a CPU reservation**
or guarantee of spare physical cores. The 10-minute interval means that
sudden changes in others' workloads are not immediately detected.
All candidates remain queued until completed, failed or interrupted.
The logical training batch remains 128 regardless of worker count.

The logical training batch remains 128 regardless of CPU worker count.

Run the preflight/dry-run first:

```bash
bash dscpuscripts/inspect_server.sh
bash dscpuscripts/run_fd001_gamma.sh --dry-run
```

For an initial **full 100-epoch single-run smoke test**, use:

```bash
bash dscpuscripts/run_fd001_gamma.sh --run --workers 1 --max-runs 1
```

It trains one canonical configuration and leaves its checkpoint/result
available for the next invocation. If the server is not busy, run the
remaining candidates with the default auto worker selection:

```bash
nohup bash dscpuscripts/run_fd001_gamma.sh --run \
  > "$HOME/fd001_gamma_direct.log" 2>&1 < /dev/null &
echo "Dispatcher PID: $!"
```

Or explicitly choose a higher concurrency on an otherwise idle machine:

```bash
nohup bash dscpuscripts/run_fd001_gamma.sh --run --workers 96 \
  > "$HOME/fd001_gamma_direct.log" 2>&1 < /dev/null &
echo "Dispatcher PID: $!"
```

For a shared server where more cores may become available **while the
campaign is running**, use an adaptive ceiling of 128:

```bash
nohup bash dscpuscripts/run_fd001_gamma.sh --run --workers 128 --adaptive \
  > "$HOME/fd001_gamma_direct.log" 2>&1 < /dev/null &
echo "Dispatcher PID: $!"
```

Before launching, inspect the pending configurations and starting capacity with
`bash dscpuscripts/run_fd001_gamma.sh --dry-run --workers 128 --adaptive`.
The default is `--check-seconds 600`, and the admission fraction is fixed
at 80%. You can explicitly change the check interval with `--check-seconds`.
There is no longer a `--ramp-up` option.
A new launcher version is used only after stopping and restarting the old
dispatcher; pulling Git updates does not change a process already running.

Do not launch two dispatchers simultaneously. A filesystem lock enforces
this. All workers run as independent one-thread Python CPU processes.
One completed run gets its own log in
`~/scratch/fulltd/fd001_gamma_v1/results/direct_logs/`.
Progress and restart receipts are in `results/direct_launches/`.
No hidden job time limit, requeue or Slurm command is involved.
On Ctrl-C/SIGTERM, in-progress runs are asked to stop gracefully and
restart later from their most recent **completed epoch**. Repeating
`--run` skips completed or terminal numerical-failure records.

Monitor:

```bash
tail -f "$HOME/fd001_gamma_direct.log"
vmstat 1 5
free -h
```

The `vmstat` columns `r`, `b`, `si`, `so`, `us`, `sy`,
`id`, `wa` help distinguish CPU contention, I/O wait, and swapping.
Pay particular attention to sustained `si/so` activity and low
`MemAvailable`. `--workers auto` without `--adaptive` measures availability
only once; `--adaptive` checks every ten minutes and changes only the
*admission of additional runs*, not the CPU allocation of workers already
running. Do not launch a second dispatcher while an old one is active;
restart the old one gracefully if you change scheduling modes.

## Check status and analyze after the 180 Gamma runs

Check the campaign without loading datasets or opening checkpoints:

```bash
cd ~/reproduceSL
bash dscpuscripts/status_fd001_gamma.sh
```

This reports counts for `complete`, `numerical_failure`, `running`,
`checkpointed`, `missing`, and incompatible/incomplete records, plus the
most recent saved epoch. For scripting, add `--strict` to return a nonzero
status unless **all 180 runs completed successfully**. A numerical failure
is terminal but **not** a successful training result.

```bash
bash dscpuscripts/status_fd001_gamma.sh --strict
```

For a full integrity audit of all 180 candidate configurations, validation
records, source draws, and compatible scientific code, run:

```bash
bash dscpuscripts/analyze_fd001_gamma.sh check
```

To run the complete Gamma analysis on the **same AMD server**, including
validation-only hyperparameter selection, evaluation of the existing
epoch-100 trained model on train/validation/test, descriptive five-seed
statistics, paired comparisons and publication-ready PDF/SVG figures:

```bash
bash dscpuscripts/analyze_fd001_gamma.sh
```

This is **direct CPU execution with no Slurm submission, no new training, no
readout refit, no OLS/TD refit and no test-based tuning**. Do not use the old
`scripts/analyze_fulltd.sh` or `fulltd.evaluate` for this Gamma campaign:
those are a different multi-dataset experiment and may invoke refit
diagnostics. The Gamma-specific `fd001_gamma.analysis` is the correct
analysis/plotting engine.

Once the tables exist, regenerate *only figures*, without opening training
checkpoints or rerunning test inference:

```bash
bash dscpuscripts/analyze_fd001_gamma.sh plots
```

Results and outputs default to:

```text
~/scratch/fulltd/fd001_gamma_v1/results/
~/scratch/fulltd/fd001_gamma_v1/outputs/
```

Key output files:

- `outputs/selections.json` and `outputs/scores.csv`: validation-selected
  Gamma/rate combinations; selection only uses epochs 80, 90 and 100
- `outputs/candidate_accounting.csv`, `outputs/campaign_audit.json`:
  accounting and scientific-identity audit for all 180 runs
- `outputs/metrics.csv`, `outputs/summary_metrics.csv`,
  `outputs/paired_comparisons.csv`: selected five-seed trained-model metrics
- `outputs/gamma_records.csv`: realized Gamma values from training probes
- `outputs/figures/`: vector PDF and SVG plots, each with its matching CSV
- `outputs/predictions/<run_id>/test_engine_predictions.csv`: test-only
  engine predictions for each selected seed
- `outputs/endpoint_cache/`: resumable checkpoint-identified analysis cache,
  without modifying completed training-result directories

The full audit is more stringent than the fast status checker. It can reject
a campaign that appears terminal if record identities, checkpoints,
training-source sequences or validation evidence do not match. If the
analysis reports an error, inspect it rather than rename, alter or delete
any training results.

No original FD001 or ongoing METR-LA results are changed or deleted. Do not
delete `~/scratch/fulltd/results/`.

## Consolidate all FD001 methods (previous baselines + new Gamma)

The old FD001 v7 experiment already evaluated two trained TCN baselines:
**MSE (B=256)** and **Gaussian NLL (B=256)**, each with five seeds.
The new experiment evaluated **TD typed**, **TD raw26** and
**TD representation** (all B=128), five seeds each.

The old B=128/B=256 TD-full runs and all refit_OLS/refit_TD outputs are
**excluded completely**. There is no training or readout refit.
The baseline source is the previously generated \`outputs_fd001\` directory
(\`selections.json\`, \`scores.csv\`, \`candidate_accounting.csv\`,
\`validation_records.csv\`, \`metrics.csv\`, \`evaluation_manifest.json\`)
or the provided small \`fd001_legacy_baselines_only.zip\` archive filtered
from the exact earlier FD001 analysis.

If the older \`~/scratch/fulltd/outputs_fd001\` directory is available on
the AMD server, no data transfer is needed. Otherwise download the
**fd001_legacy_baselines_only.zip** from this ChatGPT conversation to your
own computer, then copy it to the AMD server, for example:

\`\`\`bash
scp fd001_legacy_baselines_only.zip yangcpan@144.214.209.203:~/scratch/fulltd/
\`\`\`

On the AMD server run:

\`\`\`bash
cd ~/reproduceSL
git pull --ff-only origin fullTD
~/scratch/fulltd/.venv-fulltd/bin/python -m pytest -q dscpuscripts/test_combined_fd001.py
bash dscpuscripts/analyze_fd001_all.sh
\`\`\`

The command automatically locates the already imported baseline-only records,
the earlier \`outputs_fd001\` directory, or the uploaded ZIP. On first run
it creates one **baseline-only** local copy under
\`~/scratch/fulltd/fd001_gamma_v1/results/legacy_baselines/\`.
The 180 Gamma run directories remain unchanged. Original results on the old
server are not copied, changed, or deleted.

Combined reports go to:

\`\`\`text
~/scratch/fulltd/fd001_gamma_v1/outputs/combined/
    metrics.csv
    summary_metrics.csv
    paired_comparisons.csv
    selections.json
    provenance.json
    validation_records.csv
    validation_summary.csv
    figures/
\`\`\`

Figures include all five methods, with explicit B128/B256 labels, for test
RMSE/MAE (cycles), asymmetric RUL score, and validation MSE/MAE curves.
Comparisons are descriptive five-seed matched-index differences. Batch size
differs between Gamma and the old baselines; the combined chart must not be
presented as a controlled same-batch method comparison. No test results enter
hyperparameter selection.

For **plots only**, after the unified tables already exist:

\`\`\`bash
bash dscpuscripts/analyze_fd001_all.sh plots
\`\`\`

RUL means *Remaining Useful Life*: the number of engine operating cycles
remaining until failure. Errors are reported in **cycles**. The NASA
asymmetric RUL score penalizes underestimating and overestimating remaining
life differently; smaller scores indicate better predictions.

The old TD data are not imported into the consolidated FD001 results, so
they are no longer required for these comparisons. Removing original legacy
TD training directories on another server is a separate irreversible
operation; do not delete any baseline or Gamma checkpoint directories.

## Verification

```bash
"$HOME/scratch/fulltd/.venv-fulltd/bin/python" -m pytest -q \
  dscpuscripts/test_direct_servers.py \
  dscpuscripts/test_gamma_reporting.py
```

These tests cover one-server manifest coverage, checkpoint identity and status
counting, as well as replotting existing Gamma tables without test inference
or readout refitting. They do not replace the on-server audit and actual
endpoint evaluation of your completed 180-run campaign.

If the setup or performance is unclear, send the output of:

```bash
bash dscpuscripts/inspect_server.sh
```

The default paths are exported by `dscpuscripts/env.sh` and may be
overridden through `FULLTD_ENV`, `FULLTD_DATA_DIR`,
`FULLTD_FD001_GAMMA_RESULTS_DIR` and
`FULLTD_FD001_GAMMA_OUTPUT_DIR`. Use fast local storage if available.
