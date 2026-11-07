# Dataset results and direct CPU experiments

This is the **shared** results, analysis and plotting workflow for the AMD
server or a supported HPC host. Machines differ only in environment setup
and how processes are launched, **not** in scientific checkpoint formats,
hyperparameter selection, report definitions or figures.

## Directory convention (same on every server)

All defaults are relative to `$HOME/scratch/fulltd`, or override the common
`FULLTD_RESULTS_ROOT` environment variable:

```text
~/scratch/fulltd/
  data/                            # prepared datasets shared by campaigns
  .venv-fulltd/                   # direct CPU Python environment
  fd001_results/
    fd001_gamma_v1_typed_b128_a0_b0_s0/    # run IDs live HERE, not in results/
    fd001_gamma_v1_raw26_b128_a0_b0_s0/
    fd001_gamma_v1_representation_b128_a0_b0_s0/
    ...                         # 180 original + 140 extension + 60 baseline IDs
    direct_logs/                # worker logs
    direct_launches/            # dispatcher receipts
    initial_states/             # initial checkpoint states
    outputs/                    # analysis output files are separate from run folders
      rmse_report/                # original three-Gamma RMSE report
      extension/                  # four-Gamma selection, tables and plots
      baselines/                  # direct MSE and Gaussian NLL analysis
      comparison/                 # single six-method test table and common RMSE figure
  electricity_results/
    <canonical-electricity-run-id>/
    outputs/
  metr_la_results/
    <canonical-metr_la-run-id>/
    outputs/
```

FD001 run folders live directly in `fd001_results/` because their frozen run IDs are disjoint.
The run directories, statuses, checkpoints and original selection files must
**not** be renamed. Different analysis outputs are separated to prevent
overwriting each other's selection evidence.

The canonical paths are defined in `dscpuscripts/result_paths.py`. Both
`dscpuscripts/env.sh` and direct Python reporting commands use them. Environment
overrides take precedence, including `FULLTD_FD001_ROOT`,
`FULLTD_ELECTRICITY_ROOT`, `FULLTD_METR_LA_ROOT` and `FULLTD_DATA_DIR`.

**Your moved original FD001 results are expected at**
`/home/yangcpan/scratch/fulltd/fd001_results/`;
the original Gamma analysis outputs are expected at
`/home/yangcpan/scratch/fulltd/fd001_results/outputs/`.
No script will move, overwrite, or delete those trained results.

## Why there are fd001_gamma and fd001_gamma_ext source packages

- `fd001_gamma/` is the **frozen** original 180-candidate protocol, its
  objective/training implementation, and its validated checkpoint identities.
- `fd001_gamma_ext/` is a **separate scientific protocol** for 140 additional
  candidates: 40 extra old-representation rates and 100 new pairwise-head
  candidates. It imports the original epoch trainer instead of copying it.
- The common CPU dispatcher is `dscpuscripts/direct_runner.py`; baseline,
  Electricity and METR-LA training share `dscpuscripts/standard_runner.py`.
  The new code-head dimension is inferred from `model.feature_dim`, not fixed
  at 64.

Merging these scientific modules would change recorded code hashes and risk
invalidating existing original/extension checkpoints. Keeping the source
protocols separate is intentional; **they do not require separate results
folders**.

## One-time migration of earlier baseline or extension runs, if needed

The original 180 run directories are **already directly inside**
`fd001_results/`, as shown by your listing. **Do not move them again.**
Only older baseline/extension runs saved in other directories may require copying.

If the earlier separate paths still contain baseline or extension
checkpoints, stop any associated dispatchers, then copy their run directories
into the common FD001 results folder **without overwriting files**:

```bash
dest="$HOME/scratch/fulltd/fd001_results"
mkdir -p "$dest" "$dest/outputs/baselines" "$dest/outputs/extension"
# ONLY run the following commands if the respective source exists.
# Some runs may have been started using the mistaken nested layout:
test ! -d "$dest/results" || rsync -av --ignore-existing "$dest/results/" "$dest/"
test ! -d "$HOME/scratch/fulltd/direct/fd001_baselines/results" || rsync -av --ignore-existing "$HOME/scratch/fulltd/direct/fd001_baselines/results/" "$dest/"
test ! -d "$HOME/scratch/fulltd/direct/fd001_baselines/outputs" || rsync -av --ignore-existing "$HOME/scratch/fulltd/direct/fd001_baselines/outputs/" "$dest/outputs/baselines/"
test ! -d "$HOME/scratch/fulltd/fd001_gamma_extension_v1/results" || rsync -av --ignore-existing "$HOME/scratch/fulltd/fd001_gamma_extension_v1/results/" "$dest/"
test ! -d "$HOME/scratch/fulltd/fd001_gamma_extension_v1/outputs" || rsync -av --ignore-existing "$HOME/scratch/fulltd/fd001_gamma_extension_v1/outputs/" "$dest/outputs/extension/"
```

Use `test -d SOURCE_DIRECTORY` first: `rsync` errors on nonexistent sources.
The runners detect existing old/nested checkpoints that were not
migrated and refuse to silently rerun those configurations. **Never delete
the old directories until the new status and analysis checks pass.**

## Run (single AMD server, no Slurm)

```bash
cd ~/reproduceSL
git switch fullTD
git pull --ff-only origin fullTD
bash dscpuscripts/setup_server.sh
bash dscpuscripts/download_data.sh fd001

# Inspect the 180 original Gamma runs.
bash dscpuscripts/status_fd001_gamma.sh --terminal  # allows recorded numerical failures
# Use --strict instead to require every single training run to have completed

# The 60 previously defined MSE/NLL baselines (B256).
bash dscpuscripts/run_standard.sh fd001_baselines --dry-run
nohup bash dscpuscripts/run_standard.sh fd001_baselines --run \
  --workers 48 --adaptive > "$HOME/fd001_baselines.log" 2>&1 < /dev/null &

# The 140 Gamma-extension configurations (B128).
# Launch after the baseline dispatcher finishes to avoid resource contention.
bash dscpuscripts/run_fd001_gamma_extension.sh --dry-run
nohup bash dscpuscripts/run_fd001_gamma_extension.sh --run \
  --workers 128 --adaptive > "$HOME/fd001_gamma_extension.log" 2>&1 < /dev/null &
```

The source files have no maximum wall time. `--adaptive` is a 10-minute
CPU-idleness sampling heuristic, not a scheduler reservation. It never kills
healthy workers when other users become busy. A repeated `--run` resumes
compatible checkpoints and skips completed configurations.

For **Electricity** or **METR-LA** use the same scripts and dataset naming:

```bash
bash dscpuscripts/download_data.sh electricity
bash dscpuscripts/run_standard.sh electricity --dry-run
# Then --run --workers 8 --adaptive

bash dscpuscripts/download_data.sh metr_la
bash dscpuscripts/run_standard.sh metr_la --dry-run
# Then --run --workers 8 --microbatch 32 --adaptive
```

## Numerical failures versus missing records

The original Gamma selector assigns `+Infinity` to every hyperparameter
combination for which **even one of the five seeds** reports a genuine
`numerical_failure`. It cannot be chosen ahead of a finite five-seed
configuration. The 320-run joint Gamma selector does the same. A missing
`status.json`, incomplete checkpoint, code/data identity mismatch, or absent
run folder is **not** a numerical divergence and must be repaired/audited;
these must never silently be scored as failed HPs.

`status_fd001_gamma.sh --strict` requires 180 actual successful completions.
`status_fd001_gamma.sh --terminal` succeeds when all 180 have finished or
documented numerical failure statuses; it does **not** pretend the failed
runs trained successfully.

## Analyze and plot (exact same commands on either machine)

```bash
# No retraining or readout refit; validation selects each method separately.
bash dscpuscripts/analyze_fd001_gamma.sh check
bash dscpuscripts/analyze_standard.sh fd001_baselines
bash dscpuscripts/analyze_fd001_gamma_extension.sh check
bash dscpuscripts/analyze_fd001_gamma_extension.sh run
bash dscpuscripts/compare_fd001.sh
```

The unified six-method figure is:
`fd001_results/outputs/comparison/figures/fd001_six_methods_validation_rmse_updates.pdf`
(and SVG, PNG), with **actual cumulative Adam optimizer updates** on the
horizontal axis. Tables include test RMSE, MAE and summed RUL score. Each
method chooses HPs from five-seed validation MSE at epochs 80, 90 and 100;
test evaluation uses the actual epoch-100 trained model, **never a refit**.

For plots only:

```bash
bash dscpuscripts/compare_fd001.sh plots
```

The original `fd001_gamma/analysis.py` retains its three-Gamma audit and
operator diagnostics, `fd001_gamma_ext/analysis.py` audits all 320 Gamma
candidates and jointly selects representation HPs, and
`dscpuscripts/standard_analysis.py` selects/evaluates baselines plus
Electricity and METR-LA. The **single common FD001 RMSE curve renderer**
lives in `dscpuscripts/fd001_report.py`; the unified six-method comparison
uses `dscpuscripts/compare_fd001_extension.py`. Previous ZIP-only,
five-method reporting duplicates have been deleted.

## Verification

```bash
~/scratch/fulltd/.venv-fulltd/bin/python -m pytest -q dscpuscripts/test_result_paths.py \
  dscpuscripts/test_direct_servers.py dscpuscripts/test_gamma_extension.py \
  dscpuscripts/test_gamma_extension_selection.py dscpuscripts/test_gamma_reporting.py \
  dscpuscripts/test_standard_workflow.py
```

A passing synthetic suite is not a replacement for a real 60/140-run
completion check, checkpoint/data-identity audit, or successful report.
