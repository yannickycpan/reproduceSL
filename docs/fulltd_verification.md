# Verification before release

Checked on 1 November 2026 with Python 3.12.14, PyTorch 2.5.1 CPU and the core
versions in `requirements-fulltd.txt`. **176 focused tests passed** in a clean
source export containing no legacy files or pre-existing Python caches. The five
shell scripts pass `sh -n`; Python compilation and Git whitespace checks pass.
Slurm submission was exercised with mocked receipts/dry runs; no HPC jobs were
submitted from this workstation. The actual CityU CPU resource settings must
pass the supplied compute-node preflight. No cluster speedup is claimed from
local correctness tests.

The branch cleanup removed 75 unused legacy source/data/cache files and the
obsolete generic submission entry. All active imports, six objective factories,
630 manifest records, source licenses and correctness tests remain. Training,
evaluation and manifest hashes are identical before and after cleanup, preserving
compatibility with checkpoints from the preceding release. Temporary validation
fixtures, shell mocks, logs and generated test outputs are excluded from Git.

## Mathematical and protocol checks

- Nonlinear TD predictor and both operator gradients agree with finite
  differences; the partner derivative is present. At exactly zero theta,
  complete-target predictor gradients equal half-MSE, while theta gradients can
  be nonzero. Missing-mask reductions and Gaussian variance initialization pass.
- Every prescribed backbone has correct input/output shapes and matching mean
  initialization across loss variants. Transformer layers are independently
  initialized. Full/microbatch gradients and Adam updates agree, including the
  adaptive graph parameters and unequal valid-mask counts.
- Source sampling, last short batches, type/direction probabilities, entity and
  temporal boundaries, uniform weighting and deterministic source/partner streams
  pass. Exactly 630 candidate records are required, with 450 primary/180 secondary.
- Training-only preprocessing, graph sensor ordering, fixed engine split, causal
  input filling, target-disjoint chronological boundaries and no test access in
  tuning pass. Metrics verify unscaling, horizon indexing, equal-entity aggregation,
  square root after MSE aggregation, and the asymmetric RUL score signs.
- Interrupted/restarted fitting gives identical final model parameters and Adam
  state on a deterministic fixture. The code retains all ten validation records,
  exactly 100 training units and the actual last model. A crash after final.pt but
  before sidecars is repaired. Original execution hardware/runtime provenance is
  retained, and failed logical batches/Adam steps are counted.
- Checkpoint tests verify exactly 99 epoch-end recovery writes and one final
  write for all three datasets, with no model writes inside training/validation
  or on a warning. Interrupted work replays from the previous complete epoch;
  weights, Adam state, counters and validation scores equal uninterrupted runs
  on a deterministic fixture. Completing the final validation batch permits its
  completed epoch checkpoint. A warning before epoch1 completes creates no state.
- CPU submission tests verify array `0-6%7`, 90 single-thread/single-CPU ranks on
  each node, all 630 indices exactly once, and a single coordinator per element.
  Only warning-interrupted elements with durable progress requeue. Cancellation,
  operational errors and lack of progress cannot cause retry loops. A real tiny
  worker/trainer integration completes 100 epochs and ten validations. Live CityU
  requeue permission, signal delivery, memory use and throughput remain untested.
- Electricity validation tests verify per-customer ceil20% counts, sampling
  without replacement, independent reproducible run/round streams, no mutation
  of training RNG state, and correct equal-customer MSE. The original model and
  readouts use the same round100 subset. Full train/test and METR-LA/FD001
  validation sets are preserved. Forged subset records are rejected before test
  access. The selector remains five seeds pooled over 80/90/100.
- The all-horizon validation-only accumulator returns both MSE and MAE with
  bitwise agreement against the previous metric path on tested complete/masked
  tensors, unequal entity counts and multiple batch sizes. It performs no
  per-horizon/RMSE reductions. Training records both values at all ten scheduled
  evaluations, and final endpoint metrics retain their full reporting scope.
- Independent Electricity and METR-LA/FD001 submissions cover disjoint sets of
  300 and 330 canonical indices (90/90/90/30 and 90/90/90/60 ranks). Scoped
  preprocessing/selection/evaluation avoid unrequested datasets. Combined
  METR-LA/FD001 selection succeeds without any Electricity results and blocks
  incomplete requested runs. Changing MAE cannot change MSE-selected rates.
  Both MSE and MAE validation curves are generated from retained records.
- Preprocessing identities now verify scaler and split metadata as well as
  array bytes. Re-preparation of all three actual datasets produced identical
  numerical arrays and counts; the integrity format is `td-v7-data-2`.
- Selection pools 80/90/100 across all five seeds; any failed seed invalidates its
  configuration. Missing/incompatible records block selection. Source-stream
  tampering is detected. All-failed methods remain explicit. Readout tests verify
  exact asymmetric boundary weights, transformed bias, rank-deficient minimum-norm
  behavior, chunk invariance and theta-zero equivalence to OLS.
- Summaries and descriptive paired comparisons are independently recomputed from
  CSV records. PDF/SVG figure generation and CSV sources pass. Persistence remains
  one deterministic result without invented seed uncertainty. Slurm tests check
  all 630 tasks exactly once, dependencies, immutable run maps and quoted paths.
- Endpoint caches bind to complete group selection evidence, final checkpoints,
  data and evaluation code while permitting the same selected group to be
  analyzed alone or with other datasets. Nonfinite predictions, overflowing RUL
  scores and numerical readout failures remain explicit in the five-seed tables.
  Independent summary checks cover finite values, infinities and NaNs.

## Actual data verification (no application results)

All original data sources downloaded successfully. SHA-256 values, acquisition
sources and measured details are in `fulltd_data_verification.json`.

| Dataset | Fitting windows | Validation windows | Test windows |
|---|---:|---:|---:|
| Electricity | 6,578,267 | 1,212,432 | 2,432,520 |
| METR-LA | 23,967 | 3,416 | 6,844 |
| FD001 | 13,989 | 100 | 100 |

Electricity is the original 140,256 × 370 quarter-hour matrix, aggregated to
35,064 hourly rows. It retains 348 customers; 22 have no nonzero training reading.
No incomplete hours/gaps were found. The amended budget is 274,095 draws per round,
27,409,500 total draws, or **4.166674900851547 effective original-data passes**.
There are 107,100 batch256 update opportunities or 214,200 batch128 opportunities
over 100 rounds. METR-LA has 9,400/18,800 opportunities and FD001 5,500/11,000.
Those are declared opportunities, not measured completed campaign updates.

Every training source's small partner set was enumerated for all three actual
datasets, verifying exact probabilities and no isolates. For Electricity seed0,
round1 has 137,006 lag1 and 137,089 lag24 pairs; 262,833 partner draws lie outside
that round's sampled source set. Both batch groupings replay the same sequence.

All 15 primary method/backbone combinations passed one finite Adam update on
three actual source windows and predictions on three validation windows. A local
CPU preflight also completed three throwaway logical updates per backbone and
verified all prepared hashes/counts. No test labels were used by these fitting
checks. They are correctness/throughput checks, not campaign evidence or tuning.

The Windows CPU tests disable the host's faulty oneDNN convolution backend; the
ordinary PyTorch CPU implementation is used for those checks. Scientific CUDA
training is unchanged. Exact deterministic replay presumes the same software,
device and numerical backend; cross-device bitwise equivalence is not asserted.
`runtime_seconds` covers the retained trajectory. Graceful interruptions record
discarded epoch work separately; `total_execution_seconds` includes retained and
discarded time. A hard kill can lose time since the last durable save or receipt;
Slurm accounting/logs retain allocation evidence. No partial checkpoint is written
to avoid that loss under the user's epoch-end-only policy.

## Completion boundary and prior runs

**Scientific candidates executed for this release: 0 of 630.** The code, data
acquisition, complete manifests, selection/evaluation/refit/plot pipeline and
correctness checks are ready for the requested HPC execution. No result numbers,
selected scientific rates, learned final theta, or superiority claims are supplied.
The user deferred manuscript/LaTeX generation until HPC results are returned.
The user subsequently chose to restart the HPC campaign under the 1 November
protocol. This update neither deletes existing HPC data/results nor migrates old
full-validation checkpoint records into the sampled-validation protocol.

The four earlier FD001 runs were not present in the repository or this workspace.
They have not been deleted, modified, counted, or imported. Compatibility cannot
be established from their names alone. The new loader requires protocol, data,
configuration and code identities and refuses to overwrite incompatible records.
Retain earlier checkpoints separately until their provenance is explicitly audited.
