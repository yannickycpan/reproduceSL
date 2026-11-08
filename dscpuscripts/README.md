# Direct CPU experiments — FD001, Electricity and METR-LA

**There is one FD001 scientific protocol:** `dscpuscripts/fd001_protocol.py`.
Its training and analysis implementations are `dscpuscripts/fd001_train.py`
and `dscpuscripts/fd001_analysis.py`. The two TD objectives (typed and
pairwise) are in `dscpuscripts/fd001_objectives.py`. There are no historical
FD001 source protocols, no extra FD001-specific launchers, and no baseline-only
or extension-only reporting pipelines.

Electricity and METR-LA now use `application_protocol.py`,
`application_train.py`, `application_objectives.py`, and `application_analysis.py`
in this directory. They share the frozen data/model/sampling utilities with
FD001, and match its optimizer, grids, seeds, TD definitions and selection rule.
See [the application protocol](../docs/fulltd_protocol.md) for the explicit
Electricity computation exceptions. FD001's scientific files and hash are
unchanged.

## Experiment definition

Dataset: NASA C-MAPSS FD001. Predictors: the existing four-block TCN.
Optimization: Adam, constant learning rates, 100 full epochs, no weight
decay, clipping, auxiliary penalty, learning-rate schedule, refit, or early
stopping. Same five initialization/source seeds across methods. For TD, all
gradients flow through both predictions and the operator.

| Method | Predictor α values | Operator β values | Batch | Runs |
|---|---|---|---|---:|
| MSE | 0.0001, 0.0003, 0.0005, 0.0007, 0.0009, 0.001, 0.003, 0.005, 0.007, 0.01 | — | 256 | 50 |
| Gaussian NLL | same 10 | — | 256 | 50 |
| TD typed | 0.00003, 0.0001, 0.0003, 0.001, 0.003 | 0.00003, 0.0001, 0.0003, 0.001, 0.003 | 128, 256 | 250 |
| TD pairwise | same 5 | same 5 | 128, 256 | 250 |
| **Total** | | | | **600** |

The method set includes no raw26, source-only representation, MAE-trained
FD001, or other TD parameterization. All methods are evaluated using the
**same** validation/test RMSE, MAE, and RUL score.

## Measurements and hyperparameter selection

Evaluation is performed before training at update **0**, then every **500
actual Adam updates** by default. Measurements are on FD001 validation only.
A corresponding full checkpoint (weights, Adam state, RNG, current source
cursor, and measurements) is saved at each of those updates. The rolling
epoch-end recovery checkpoint remains as operational fault tolerance.
A stopped run resumes exactly from its saved cursor; this does not change
the loss, optimizer, source order, batch, or sampling protocol.

For future clean runs, `FULLTD_FD001_EVAL_EVERY` overrides the update period.
Do not change it after starting; the interval belongs to the per-run
scientific identity.

**Analysis-only selection:** for each candidate, ignore update-zero and
take the final `ceil(0.25 × number_of_positive_update_evaluations)`
measured validation-MSE values. Average them per seed, then across all five
seeds, and choose the best α/β/batch setting separately for each method.
For the default 500-update interval and FD001 training sizes, these are
updates 4500/5000/5500 for B256 and
8500/9000/9500/10000/10500/11000 for B128. No test labels
are opened until the four method-level selections are locked.
A documented numerical failure of any seed gives that entire HP
setting infinite selection score.

Test results come from the five actual 100-epoch endpoints, with no
retraining or refit. The output includes validation/test RMSE, MAE, and
RUL score, matched-seed differences, update-based learning curves, and
typed/pairwise Gamma diagnostics.

## Results and outputs

```text
~/scratch/fulltd/
  data/
  .venv-fulltd/
  fd001_results/              # Flat run directories; fd001_<method>_b..._h..._s...
    fd001_mse_b256_h0_s0/
    fd001_gaussian_nll_b256_h0_s0/
    fd001_td_typed_b128_h0_s0/
    fd001_td_pairwise_b256_h0_s0/
  fd001_outputs/              # Sibling analysis, NEVER inside fd001_results/
    selections.json
    hp_scores.csv
    selected_hyperparameters.csv
    per_seed_metrics.csv
    performance_summary.csv
    paired_comparisons.csv
    validation_learning_curves.csv
    figures/
    td_diagnostics/
```

No script deletes or renames existing server-side results. If there are
old run directories on your server, review and remove them separately after
verifying backups; the current protocol has unversioned run IDs.

## AMD server: run and analyze

```bash
cd ~/reproduceSL
git switch fullTD
git pull --ff-only origin fullTD

$HOME/scratch/fulltd/.venv-fulltd/bin/python -m pytest -q \
  dscpuscripts/test_fd001.py \
  dscpuscripts/test_standard_workflow.py \
  dscpuscripts/test_result_paths.py

bash dscpuscripts/run_standard.sh fd001 --dry-run
nohup bash dscpuscripts/run_standard.sh fd001 \
  --run --workers 48 --adaptive \
  > $HOME/fd001.log 2>&1 < /dev/null &

bash dscpuscripts/run_standard.sh fd001 --status
tail -n 40 $HOME/fd001.log

# Only after all candidates complete or document true numerical failures:
bash dscpuscripts/analyze_standard.sh fd001 check
bash dscpuscripts/analyze_standard.sh fd001 run
# Figures only, from saved tables:
bash dscpuscripts/analyze_standard.sh fd001 plots
```

The direct dispatcher is shared with Electricity and METR-LA, and admits
workers based on real-time idle CPU capacity when `--adaptive` is given.
No Slurm is required on the AMD host. Use `--workers auto` or a
conservative maximum on a heavily shared server; do not run two dispatchers
against the same result root.

## Electricity and METR-LA on the AMD CPU server

The current application sweeps replace the older TD-full/TD-scalar campaign.
Each Electricity backbone (TCN and PatchTST) has the same four principal
methods as FD001. DLinear is a separate architecture and is not in the sweep.
METR-LA uses Graph WaveNet for all methods, including additional masked-MAE
and pairwise-TD-MAE comparators. The totals are 1,200 and 900 candidates.

Use the existing pinned CPU environment; only run setup if it is not ready.
The download script verifies existing prepared data without overwriting it.

```bash
cd ~/reproduceSL
git switch fullTD
git pull --ff-only origin fullTD
# Only if the environment has not yet been installed:
# bash dscpuscripts/setup_server.sh
bash dscpuscripts/download_data.sh electricity
bash dscpuscripts/download_data.sh metr_la

bash dscpuscripts/run_electricity.sh --dry-run
bash dscpuscripts/run_metr_la.sh --dry-run

nohup bash dscpuscripts/run_electricity.sh --run --workers 8 --adaptive \
  > "$HOME/electricity.log" 2>&1 < /dev/null &
nohup bash dscpuscripts/run_metr_la.sh --run --workers 8 --adaptive \
  > "$HOME/metr_la.log" 2>&1 < /dev/null &

bash dscpuscripts/run_electricity.sh --status
bash dscpuscripts/run_metr_la.sh --status
```

Eight single-threaded workers per campaign are a conservative starting point,
not a measured throughput optimum. Electricity's default physical microbatch
is 128 and METR-LA's is 32; gradients accumulate to the prescribed logical
128/256 batch using the global valid-target count. `--microbatch` controls
memory only. Both launchers resume compatible incomplete runs and skip
completed runs and recorded numerical failures. They stop on operational
errors. A new launch never silently adopts incompatible old checkpoints.
If an earlier application sweep occupies these result paths, preserve it and
choose a fresh path with `FULLTD_ELECTRICITY_ROOT` or `FULLTD_METR_LA_ROOT`.
Do not mix the old campaign with this one or change settings halfway through.

After the corresponding sweep is complete:

```bash
bash dscpuscripts/analyze_standard.sh electricity check
bash dscpuscripts/analyze_standard.sh electricity run
bash dscpuscripts/analyze_standard.sh metr_la check
bash dscpuscripts/analyze_standard.sh metr_la run
# Regenerate the requested tables/plots from cached measurements:
bash dscpuscripts/analyze_standard.sh electricity plots
bash dscpuscripts/analyze_standard.sh metr_la plots
```

Training states live in `~/scratch/fulltd/electricity_results/` and
`~/scratch/fulltd/metr_la_results/`; reports are in the sibling
`electricity_outputs/` and `metr_la_outputs/` directories. Each backbone
gets an RMSE table and an MAE table with algorithms as rows and Test /
Validation columns (five-seed mean ± standard error). TD B128 and B256 have
separate validation-selected HPs, with additional B128 matched-update rows.
Learning curves use the same selected settings, with test and validation
panels, complete update axes and standard-error bands. Electricity metrics
are standardized and equally weighted by customer; its validation column
and curves use the fixed 20% customer-stratified view, while test uses the
full held-out split. METR-LA uses masked all-horizon RMSE/MAE in mph.
`selected_hyperparameters.csv` records the chosen rates and budgets.

No test data are opened until validation-only selections are locked. No
additional model fitting, persistence baselines, diagnostic plots, or refits
are part of this report. Generating full test learning curves is more costly
than endpoint evaluation; measurements are cached so rerunning the command
does not repeat completed evaluations.


## FD001 fixed-selection batch and matched-update report

This secondary **reporting-only** analysis does not change the canonical
four-method selection, training configuration, optimizer, or trained weights.
For full-budget performance it selects within each fixed TD batch size using
the *same* five-seed final-quarter validation-MSE rule and evaluates the
100-epoch final checkpoints. It also compares TD B128 at the exact B256
100-epoch optimizer-update budget, reapplying the **same final-quarter rule**
to the validation observations within that shorter budget and evaluating
the retained B128 checkpoint at that matched update. In the current sweep
B256 has 5,500 updates and B128 has 11,000; the matched selection uses
validation observations at 4,500, 5,000 and 5,500 updates, then evaluates
step 5,500 rather than the B128 epoch-100 model.

This is **not** best-seed selection, test-tuned checkpoint selection,
additional training, or replacement of the original primary report.

On the server containing all original FD001 checkpoints, run:

```bash
cd ~/reproduceSL
git switch fullTD
git pull --ff-only origin fullTD
bash dscpuscripts/analyze_fd001_batches.sh check
bash dscpuscripts/analyze_fd001_batches.sh run
cat ~/scratch/fulltd/fd001_outputs/batch_comparison/comparison_table.md
```

The check locks validation-only choices without opening test labels. The
run reads the original final/step checkpoints and creates separate cached
evaluations, raw per-seed values, mean ± standard error tables and a Markdown
report under `~/scratch/fulltd/fd001_outputs/batch_comparison/`.
It never writes in the training-results directories. If a required snapshot
is missing, the command stops rather than replacing it.
