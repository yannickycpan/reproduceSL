# FD001 Gamma sweep v1

This is a separate, additive experiment on branch `fullTD`. It does not edit the
legacy `fulltd` package, `agents/fulltd_*.py`, original manifests, or prior results.
Those files participate in legacy checkpoint hashes and may still be in use by
METR-LA jobs. The new campaign uses new IDs and separate output directories.

## Exactly three choices of one categorical hyperparameter

All models use the original FD001 TCN: 24 standardized input channels, 30-cycle
history, four causal residual blocks, 64-dimensional final representation,
and scalar linear prediction head (92,801 mean-network parameters).
`gamma_variant` has these three values:

- `typed`: the original two unrestricted scalar coefficients, selected by lag
  type: type 1 is distance one cycle, type 2 is distance five cycles. Two Gamma
  parameters; 92,803 total parameters.
- `raw26`: `phi_ij = concat(one_hot(type_ij, 2), x_i[:, -1] - x_j[:, -1])` and
  `Gamma_ij = dot(theta, phi_ij)`. This uses the latest observations, not all
  30 history positions. The raw observable inputs retain the existing
  training-only per-channel standardization. Differences are signed, not absolute.
  There is no extra bias, because the two exhaustive lag indicators already
  supply intercepts. 26 features/parameters; 92,827 total parameters.
- `representation`: `gamma_i = Linear(64, 2)(h_i)` and
  `Gamma_ij = gamma_i[type_ij]`. The source representation is shared with the
  mean predictor. This is the requested flexible two-output head, with 128
  weights and two biases (130 Gamma parameters; 92,931 total parameters).

Every Gamma parameter starts at exactly zero. All outputs are unrestricted and
signed. Both residual prediction paths and the representation-to-Gamma path stay
in autograd. Predictor parameters receive learning rate alpha exactly once;
only the operator/head parameters receive beta. The representation head does
not detach the source representation. Gamma is not used for point prediction.

The sole training objective remains half the mean squared sampled-pair residual:
`0.5 * mean((e_i - Gamma_ij * e_j)**2)` over the global valid-mask intersection.
There is no MSE anchor, clipping, output constraint, regularization, additional
Gamma architecture, or variance/correlation model in this campaign.

## Grid and preserved protocol

```
gamma_variant = typed, raw26, representation
alpha = 0.0003, 0.001, 0.003
beta  = 0.00003, 0.0001, 0.0003, 0.001
seed  = 0, 1, 2, 3, 4
logical_batch = 128
```

This is **3 x 12 x 5 = 180 runs**, not 36 runs and not an adaptive search.
The only experimental changes are the requested Gamma parameterization/rate
sweep and fixing logical batch 128 as agreed. Baseline experiments are neither
changed nor resubmitted. The original split, target definition, preprocessing,
sampling distribution, within-engine lag-1/lag-5 neighbors, 100 full epochs,
Adam settings, and constant learning rates are retained. Seeds share the same
initial predictor and source/partner sequences across all Gamma variants and HPs.
No test labels are accessed during training or hyperparameter selection.

The physical microbatch defaults to 128 (one backward pass per logical update).
A smaller physical microbatch accumulates losses divided by the logical batch's
global valid count and still performs exactly one Adam step. There are no
schedules or gradient clipping. Final short batches are retained.

Validation uses all prescribed validation prefixes at epochs 10,20,...,100.
Each configuration's score averages validation MSE over all five seeds and
epochs 80,90,100. A numerical failure in any seed assigns infinite score to
that configuration. All candidates must be terminal before selection succeeds.
The primary selection is joint over all 36 `(gamma_variant, alpha, beta)`
choices; ties use smaller alpha, then smaller beta, then declared variant order.
Per-variant validation winners are retained only for comparisons/diagnostics;
test scores cannot change the primary choice. Evaluation uses actual epoch-100
mean networks, never a best-epoch or test-selected model.

The old refit scope was primary batch-256 TD TCN models. This new batch-128
campaign does not add refit variants or redefine that legacy diagnostic.

## Run on the HPC login node

```bash
cd ~/reproduceSL
git switch fullTD
git pull --ff-only origin fullTD
bash scripts/submit_fulltd_fd001_gamma.sh --dry-run
bash scripts/submit_fulltd_fd001_gamma.sh --submit
```

The launcher uses the existing `scripts/fulltd_environment.sh` and virtualenv.
Default allocation: **batch partition, 90 single-thread CPU workers per node,
400G node memory, two concurrent nodes, two hours per allocation**. The original
FD001 TD-128 accounting supplied in this investigation recorded about 28-31
minutes per 100-epoch run. That motivates headroom in the two-hour request; it is
not a timing guarantee for the new heads or current cluster load.

Only unfinished new-campaign runs are submitted. Repeating the command resumes
compatible epoch checkpoints after the prior new array has ended. An existing
new-campaign array is refused rather than duplicated. Old METR-LA arrays need
not be cancelled or waited for, since no legacy scientific files or run paths
are modified. Scheduler lookup errors fail closed.

Epoch-boundary checkpoints and Slurm warning/requeue preserve continuation over
multiple allocations. Partial epochs are discarded and replayed. Requeue is
refused on operational failures or if no new complete epoch was saved. A longer
allocation can be requested without editing scientific settings:

```bash
bash scripts/submit_fulltd_fd001_gamma.sh --submit --time 04:00:00
```

Optional operational flags: `--workers`, `--concurrency`, `--memory`,
`--microbatch`, `--partition`, `--signal-seconds`. Changing physical microbatch
can change floating-point reduction order, not the mathematical objective or
logical update count. Resource defaults are estimates, not cluster measurements.

## Result locations and analysis

Defaults (intentionally ignore `FULLTD_RESULTS_DIR` and `FULLTD_OUTPUT_DIR`):

```
~/scratch/fulltd/fd001_gamma_v1/results/
~/scratch/fulltd/fd001_gamma_v1/outputs/
```

Overrides: `FULLTD_FD001_GAMMA_RESULTS_DIR`, `FULLTD_FD001_GAMMA_OUTPUT_DIR`.
`FULLTD_DATA_DIR` continues to point to the existing prepared data.

After training is terminal:

```bash
cd ~/reproduceSL
bash scripts/submit_fulltd_fd001_gamma_analysis.sh
```

This verifies all 180 records before submitting a one-CPU, 32G, one-hour `tiny`
analysis job. The job independently repeats the audit before evaluating test
labels. It produces `scores.csv`, `selections.json`, `selected_runs.csv`,
`candidate_accounting.csv`, `validation_records.csv`, `per_seed_metrics.csv`,
`summary_metrics.csv`, `paired_comparisons.csv`, `gamma_records.csv`, and
PDF/SVG figures with exact numerical CSV sidecars. Selected runs also retain
`test_engine_predictions.csv` and checkpoint/code/data/selection-bound endpoint
caches. All five seeds, including nonfinite endpoint results, remain explicit.
No old result or baseline cache is overwritten.

The Gamma trajectory plots display actual coefficient-function values, not
neural-head weights mislabeled as coefficients. Every scheduled validation also
records a fixed label-free training-source probe, with exact local pair weights,
per-type mean/SD/min/max, negative fraction, and fraction within 0.05 of one.
This probe consumes no training RNG state and does not affect updates. Its
statistics are descriptive, not a new tuning or stopping criterion.

## Verification before commit

The two new test modules have **42 passing tests**, run in two invocations:

```bash
python -m pytest -q tests/test_fd001_gamma.py
python -m pytest -q tests/test_fd001_gamma_pipeline.py
```

Checks include exact Cartesian grid/IDs, parameter counts, zero initialization,
matched predictor/RNG state, raw feature order/sign/time index, all operator
finite differences and sampled predictor finite differences, the additional
Gamma-to-representation derivative, both prediction paths, original typed-TD
bitwise Adam equivalence, masks, microbatch gradients/updates, 100-epoch actual
TCN fixture runs for all three variants, exact checkpoint/Adam replay, validation
interruption, final-sidecar recovery, failure retention, frozen-code identity,
all-180 validation-only joint selection, five-seed aggregation, cached endpoint
evaluation, per-engine exports, seven rendered PDF/SVG plots, shell quoting,
submission limits, duplicate/scheduler-error protection, worker mapping, warning
forwarding and requeue decisions. Local tests used PyTorch 2.10.0 CPU. The
full historical test suite and a live Slurm allocation were not executed here.
No full-data scientific training result is claimed by these synthetic tests.

Remaining empirical uncertainty: richer Gamma functions can still reduce pair
loss without improving ordinary prediction. No safeguard altering the requested
objective was added. This is an empirical extension, not a claim that the linear
theory automatically proves gains for jointly learned neural representations.
