# Direct CPU campaigns and consistent FD001 metrics

All additions are operational/reporting code in `dscpuscripts/`. Scientific
training files and hashes are unchanged; no completed run needs retraining to
use the reporting changes. No scheduler or time limit is required.

## Metric conventions

The earlier pipeline used standardized **MSE** for validation selection and its
validation MSE graph, standardized **MAE** for its other validation graph, and
**RMSE/MAE in cycles plus summed RUL score** for FD001 test figures. The Gamma
pipeline already used those same quantities. MSE and RMSE are not interchangeable
labels. None of the old values or the selection criterion has been changed.

`analyze_fd001_gamma.sh plots` now also writes an explicit validation RMSE curve
in `outputs/rmse_report/validation_rmse_fd001_tcn.{png,pdf,svg}`. The curve takes
the square root of each seed's validation MSE, multiplies by the verified target
scale, then averages the five RMSEs. It is NOT sqrt(mean MSE), and uncertainty is
computed after transformation. Legacy MSE/MAE plots remain intact.

The selected HP is still the minimum five-seed average validation MSE at epochs
80/90/100. Test performance uses each seed's actual epoch-100 checkpoint, not
an average of the last checkpoints and not a test-selected best epoch.

## Rerun only the two original FD001 baselines

```bash
cd ~/reproduceSL
git pull --ff-only origin fullTD
bash dscpuscripts/run_standard.sh fd001_baselines --dry-run
nohup bash dscpuscripts/run_standard.sh fd001_baselines --run --workers 48 --adaptive > "$HOME/fd001_baselines.log" 2>&1 < /dev/null &
```

This is exactly **60 runs**: MSE and Gaussian NLL x six original predictor rates
x five seeds. Logical and default physical batch are both **256**, preserving
the old baseline configuration. It does NOT change the baseline grid, add a
batch-128 control, or rerun any TD candidates. A new batch-128 controlled study
would require a separately declared experiment, not relabeling these results.

Status and no-refit analysis:

```bash
bash dscpuscripts/run_standard.sh fd001_baselines --status
bash dscpuscripts/analyze_standard.sh fd001_baselines
bash dscpuscripts/compare_fd001.sh
```

Fresh baselines live in `~/scratch/fulltd/direct/fd001_baselines/results/` and
`.../outputs/`. The final comparison automatically discovers these outputs,
combines them with the existing Gamma outputs on this server, and writes
`~/scratch/fulltd/fd001_gamma_v1/outputs/comparison_report/`. No remote download
or old-checkpoint transfer is needed. The report contains one five-method RMSE
learning curve, the three test measures with SD/SE, and selected HPs.

## Electricity and METR-LA on the same server

```bash
bash dscpuscripts/download_data.sh electricity
bash dscpuscripts/run_standard.sh electricity --dry-run
nohup bash dscpuscripts/run_standard.sh electricity --run --workers 8 --adaptive > "$HOME/electricity_direct.log" 2>&1 < /dev/null &
```

```bash
bash dscpuscripts/download_data.sh metr_la
bash dscpuscripts/run_standard.sh metr_la --dry-run
nohup bash dscpuscripts/run_standard.sh metr_la --run --workers 8 --microbatch 32 --adaptive > "$HOME/metr_la_direct.log" 2>&1 < /dev/null &
```

These reproduce the **existing canonical experiments**, not an unrequested
extension of the FD001-specific raw26/representation heads to graph or
customer-window inputs. Electricity has **300** runs; METR-LA has **210**.
Their architectures, HPs, seeds, source sampling, validation sampling and
training budgets are imported unchanged from `fulltd.protocol` and
`fulltd.train`. In particular, Electricity retains its original stochastic-round
budget and sampled validation policy, rather than being silently converted to
full passes. Physical microbatches affect memory and floating-point reduction
order, not logical batch size or number of Adam updates.

Each campaign has independent output paths under
`~/scratch/fulltd/direct/<campaign>/`. There are no conflicts with Gamma or old
HPC result paths. `FULLTD_DIRECT_ROOT` overrides the root. Checkpoints resume at
completed epoch/round boundaries. Numerical failures are retained, not retried
as fresh seeds. A repeated launcher cannot race another dispatcher in the same
campaign. Scientific-code or data mismatches stop rather than overwrite runs.

The default worker counts are operational starting points (16 for FD001, 8 for
the other datasets), not benchmarked memory guarantees. Eight Graph WaveNet
workers with microbatch 32 is intentionally different from launching 90 large
unmicrobatched graph models. Raise concurrency only after observing RAM and
throughput on this shared server. `--adaptive` treats the supplied count as a
ceiling and checks 80% of estimated idle physical-core capacity every 600 seconds.
There is no allocation wall-time. Do not concurrently consume the same free
capacity with several large campaigns without accounting for their total load.

After completion:

```bash
bash dscpuscripts/analyze_standard.sh electricity
bash dscpuscripts/analyze_standard.sh metr_la
```

`check` audits candidate records; `plots` redraws existing tables. All endpoint
analysis evaluates only completed mean predictors, with no neural retraining or
OLS/TD readout refits. Baseline failure groups remain failures. Persistence is
included for the two time-series datasets using the original validation view.

## Tests

```bash
~/scratch/fulltd/.venv-fulltd/bin/python -m pytest -q dscpuscripts/test_standard_workflow.py dscpuscripts/test_gamma_objective_audit.py
```

The objective tests check the actual Gamma implementation by finite differences,
zero-Gamma MSE equivalence, parameter-group partitioning, the source feature
path, raw26 time/sign conventions, and bitwise typed/old-TD equivalence on the
same synthetic CPU problem. Launcher tests mock dataset access and training;
they do not constitute full-data runs on the AMD hardware.


## FD001 Gamma extension: two representation variants, 140 new runs

**Scientific protocols remain unchanged for all existing results.**
The original 180 FD001 Gamma runs live at
`~/scratch/fulltd/fd001_gamma_v1/results/`. The legacy
`fd001_gamma/train.py`, `protocol.py`, `objectives.py`, and their scientific
hash remain untouched. New runs use the exact existing epoch trainer and
fixed seed, sampler, TCN, Adam, logical batch 128, full-gradient loss, and
validation rule.

The extension is a fresh protocol with two components:

| Gamma head | Predictor alpha grid | Operator beta grid | New runs |
| --- | --- | --- | ---: |
| Source representation `Gamma(h_i)` (unchanged old head) | 0.00003, 0.0001 only | 0.00003, 0.0001, 0.0003, 0.001 | 40 |
| Pairwise representation `Gamma(h_i,h_j)` | 0.00003, 0.0001, 0.0003, 0.001, 0.003 | same four values | 100 |

Total: **140 new runs**, with five seeds per rate pair. The old
representation's original alpha values 0.0003/0.001/0.003 remain in the
previous 180 runs. The extension's final selector compares **all 20 rate
pairs for representation**, not just the eight new ones, and independently
selects from **all 20 pairwise rate pairs**. The old typed and raw26 winners
are retained after auditing all 180 previous records.

The new pairwise operator builds
`[dot(h_i,h_j)/d, abs(h_i-h_j), norm(h_i-h_j)/sqrt(d)]`,
a `d+2` feature vector, where `d=model.feature_dim`, then applies a
zero-initialized **type-specific signed affine head** with
`2(d+2)+2` trainable parameters for two lag types. Both source and
neighbor representations participate in the full gradient; neither
representation nor the sampled residual is detached. The `1/d` and
`1/sqrt(d)` factors are simple, fixed scaling conventions, not
feature whitening or trainable normalization. No clipping, weight decay,
anchors, schedules, or other new training interventions are introduced.

### Run the two original baselines and the extension on the same AMD server

```bash
cd ~/reproduceSL
git switch fullTD
git pull --ff-only origin fullTD

# Before running: verify mathematical gradients, campaign identities, and basic startup
~/scratch/fulltd/.venv-fulltd/bin/python -m pytest -q \
  dscpuscripts/test_gamma_extension.py \
  dscpuscripts/test_gamma_reporting.py \
  dscpuscripts/test_standard_workflow.py

# Earlier original baselines; 60 runs, only MSE/Gaussian NLL, B256
bash dscpuscripts/run_standard.sh fd001_baselines --dry-run
nohup bash dscpuscripts/run_standard.sh fd001_baselines \
  --run --workers 48 --adaptive \
  > "$HOME/fd001_baselines.log" 2>&1 < /dev/null &

# Check the baseline progress anytime
bash dscpuscripts/run_standard.sh fd001_baselines --status
```

Run the 140 new Gamma configurations when the baseline dispatcher has
finished (or choose a conservative concurrency and monitor shared CPU use):

```bash
bash dscpuscripts/run_fd001_gamma_extension.sh --dry-run
nohup bash dscpuscripts/run_fd001_gamma_extension.sh \
  --run --workers 128 --adaptive \
  > "$HOME/fd001_gamma_extension.log" 2>&1 < /dev/null &

tail -f "$HOME/fd001_gamma_extension.log"
```

Extension outputs and checkpoints are isolated under:

```text
~/scratch/fulltd/fd001_gamma_extension_v1/results/
~/scratch/fulltd/fd001_gamma_extension_v1/outputs/
```

Once the baselines and extension complete, analyze from **existing epoch-100
models only** with no readout refit or new training:

```bash
bash dscpuscripts/analyze_standard.sh fd001_baselines
bash dscpuscripts/analyze_fd001_gamma_extension.sh check
bash dscpuscripts/analyze_fd001_gamma_extension.sh
bash dscpuscripts/compare_fd001_extension.sh
```

This produces a six-method comparison (MSE, Gaussian NLL, typed, raw26,
original source representation with expanded alpha grid, new pairwise
representation), including test RMSE, MAE, RUL score, and one
validation-RMSE learning curve over **cumulative Adam optimizer updates**.
The mean and standard error are computed over five fixed seeds.
Hyperparameter selection remains validation MSE across epochs 80,90,100.
The test checkpoint is always epoch 100. Importantly, B256 baselines
receive fewer updates per epoch than B128 Gamma models; the plots show the
real update counts rather than disguising that distinction.

Outputs:

```text
~/scratch/fulltd/fd001_gamma_extension_v1/outputs/
  selections.json
  scores.csv
  summary_metrics.csv
  validation_rmse.csv
  figures/fd001_gamma_four_learning_curves.pdf
  comparison/test_performance.csv
  comparison/selected_hyperparameters.csv
  comparison/figures/fd001_six_methods_validation_rmse_updates.pdf
```

Recreate figures **without rerunning inference**:

```bash
bash dscpuscripts/analyze_fd001_gamma_extension.sh plots
bash dscpuscripts/compare_fd001_extension.sh plots
```

Other datasets, unchanged: `bash dscpuscripts/run_standard.sh electricity --run`
and `bash dscpuscripts/run_standard.sh metr_la --run` (with appropriate
worker ceilings). They continue using the original standard runner.

