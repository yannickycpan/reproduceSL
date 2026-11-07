# FD001 — single current experiment

**There is one FD001 scientific protocol:** `dscpuscripts/fd001_protocol.py`.
Its training and analysis implementations are `dscpuscripts/fd001_train.py`
and `dscpuscripts/fd001_analysis.py`. The two TD objectives (typed and
pairwise) are in `dscpuscripts/fd001_objectives.py`. There are no historical
FD001 source protocols, no extra FD001-specific launchers, and no baseline-only
or extension-only reporting pipelines.

Electricity and METR-LA use the existing `fulltd` implementations; their
data preparation, networks, losses, sampling, and ordinary training procedures
are separate from the FD001 experiment.

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

## Other datasets

The unchanged processing and model implementations for Electricity and
METR-LA are still available:

```bash
bash dscpuscripts/run_standard.sh electricity --dry-run
bash dscpuscripts/run_standard.sh metr_la --dry-run
```

The non-FD001 protocol and corresponding manifests under `fulltd/` and
`configs/fulltd/` now contain 510 Electricity/METR-LA runs only. The
historical MAE objective is retained only where required by the reference
Graph WaveNet training procedure; it is not part of FD001.
