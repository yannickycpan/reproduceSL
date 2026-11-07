# Direct-server experiments: FD001 fresh four-method sweep

This is the recommended workflow for the new, **270-run** FD001 comparison.
The earlier 180-run Gamma, 140-run extension, and 60-run baseline experiments
remain intact with their original checkpoint identities and run IDs.
Only one launcher and one analyzer are used for the fresh sweep.

## Files and directories

```text
~/scratch/fulltd/
    data/                       # previously verified processed datasets
    fd001_results/              # ALL training runs; flat run-ID directories
        fd001_v2_mse_.../
        fd001_v2_gaussian_nll_.../
        fd001_v2_td_typed_.../
        fd001_v2_td_pairwise_.../
        fd001_gamma_v1_.../     # historical data remain unchanged
        fd001_gamma_ext_v1_.../
    fd001_outputs/              # NEW shared FD001 analysis, SIBLING of results/
        selections.json
        hp_scores.csv
        selected_hyperparameters.csv
        per_seed_metrics.csv
        performance_summary.csv
        validation_learning_curves.csv
        validation_curve_per_seed.csv
        figures/
        td_diagnostics/
        .cache/
```

No new nested result directory is created. Results and analysis are shared by
the direct CPU server and HPC: host-specific differences are only environment
setup and dispatch. `dscpuscripts/result_paths.py` resolves the root;
`FULLTD_RESULTS_ROOT` and `FULLTD_FD001_ROOT` allow explicit overrides.
Existing historical `fd001_results/outputs/` are *not deleted*.

## New FD001 scientific protocol

One frozen TCN architecture, the same FD001 train/validation/test split, same
train-data standardization, same deterministic source/partner sampling, 100
full epochs, Adam with constant rates and no clipping, weight decay, learning
schedule, refit, auxiliary supervised term, or early stopping.

| Method | Predictor alpha sweep | Operator beta sweep | Total runs |
| --- | --- | --- | ---: |
| MSE, B256 | .0002, .0003, .0005, .0007, .001, .003, .01 | — | 35 |
| Gaussian NLL, B256 | same seven alpha values | — | 35 |
| TD typed, B128 | .0003, .001, .003 | .00003, .0001, .0003, .001, .003 | 75 |
| TD pairwise, B128 | .00003, .0001, .0003, .001, .003 | same five beta values | 125 |
| **Total** | | | **270** |

All methods use five seeds. Raw26, source-only representation TD and MAE-trained
FD001 algorithms are **absent**. The TD operators and baseline losses are reused
from the frozen implementations.

**Selection is unchanged:** mean standardized validation MSE over epochs
80, 90 and 100, pooled across five seeds. Any genuine nonfinite numerical
failure invalidates the *entire five-seed HP setting* with infinite score.
Missing/incomplete checkpoints block analysis; they are not misclassified as
divergence. Test data are loaded only after selection has been locked.

**Learning-curve observations are independent of selection:** initial
evaluation at update **0**, then at **500, 1000, 1500, ...** actual Adam
optimizer updates. Each observation evaluates validation RMSE, MAE and summed
RUL score, and each scheduled step retains a full checkpoint (model, Adam,
RNG state, source cursor and records). There are also separate epoch-end
selection evaluations and rolling recovery checkpoints. Interrupted training
resumes from the last durable source cursor within an epoch without changing
the source/partner sampling stream. No interpolation of historical learning
curves or evaluation on test during training is used.

The default 500-update cadence is configurable for **future, distinct**
experiments via `FULLTD_FD001_EVAL_EVERY`; this value is saved in every run's
config and its checkpoint identity. **Do not change it after the sweep starts.**

Original Graph WaveNet's reference training uses masked MAE. Its *historical*
MAE baseline remains available solely for that cited-reference exception;
the fresh FD001 sweep and all four current methods do not train using MAE.
All four are nevertheless **evaluated using both MAE and RMSE**, plus FD001
RUL score.

## Launch on the AMD server

```bash
cd ~/reproduceSL
git switch fullTD
git pull --ff-only origin fullTD
~/scratch/fulltd/.venv-fulltd/bin/python -m pytest -q \
  dscpuscripts/test_fd001_v2.py \
  dscpuscripts/test_standard_workflow.py \
  dscpuscripts/test_result_paths.py

# No jobs launched by --dry-run:
bash dscpuscripts/run_standard.sh fd001_v2 --dry-run

# Start the entire 270-run campaign with one direct scheduler:
nohup bash dscpuscripts/run_standard.sh fd001_v2 \
  --run --workers 48 --adaptive \
  > "$HOME/fd001_v2.log" 2>&1 < /dev/null &
echo $!
```

Check progress without launching additional jobs:

```bash
bash dscpuscripts/run_standard.sh fd001_v2 --status
tail -n 40 "$HOME/fd001_v2.log"
```

The shared server is heavily utilized. Start at most one launcher and use
`--adaptive` rather than assuming all 256 physical cores are idle. Logs,
snapshots, and training outputs are under the flat `fd001_results` tree.
The launcher skips completed/numerically failed run IDs and resumes any
compatible checkpoints. It refuses to overwrite a record from another
protocol/code/data identity.

## Unified results analysis (one command)

Only after all 270 runs have finished successfully or have documented
numerical failures:

```bash
bash dscpuscripts/analyze_standard.sh fd001_v2 check
bash dscpuscripts/analyze_standard.sh fd001_v2 run
```

`check` audits all candidates and determines the four winners without
opening test labels. `run` evaluates the **actual epoch-100 models**
and produces one common FD001 report in `fd001_outputs/`: a validation/test
table with RMSE, MAE and RUL score, one HP selection table, measured
0/500/... learning curves on the real optimizer-update axis, and TD operator
diagnostics. There is **no baseline-only analysis, extension-only analysis,
or rmse_report** in the fresh workflow.

Regenerate figures only without checkpoint evaluation:

```bash
bash dscpuscripts/analyze_standard.sh fd001_v2 plots
```

## Legacy experiments

Older commands `run_fd001_gamma.sh`,
`run_fd001_gamma_extension.sh`, and `analyze_fd001_gamma.sh` are retained
for *historical checkpoint reproducibility*, not part of this fresh FD001
comparison. The older Electricity/METR-LA runner and their published
experimental protocols remain unchanged; they can use the direct-server
`run_standard.sh electricity` or `run_standard.sh metr_la` commands.
