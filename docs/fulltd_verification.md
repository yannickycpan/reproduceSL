# Verification before release

Checked on 29 October 2026 with Python 3.12.14, PyTorch 2.5.1 CPU and the core
versions in `requirements-fulltd.txt`. **112 focused tests passed.** The three
shell scripts pass `sh -n`; Python compilation and Git whitespace checks pass.
Slurm submission was exercised with mocked receipts/dry runs; no HPC jobs were
submitted from this workstation. CUDA and the actual CityU resource settings
must pass the supplied compute-node preflight.

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
- Wall-time recovery also resumes partial validation from float64 sufficient
  sums, including interruption at its final batch; it never selects on a partial
  score. Mocked Slurm tests verify that only the interrupted array element is
  requeued, and cancellation/numerical failure cannot cause a retry loop. Site
  requeue permission and actual delivery of the warning remain untested here.
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
Recorded runtime covers persisted execution segments. A hard kill can lose time
spent since the last recovery save; Slurm accounting/logs retain the corresponding
allocation evidence. Graceful wall-time continuation saves the current segment.

## Completion boundary and prior runs

**Scientific candidates executed for this release: 0 of 630.** The code, data
acquisition, complete manifests, selection/evaluation/refit/plot pipeline and
correctness checks are ready for the requested HPC execution. No result numbers,
selected scientific rates, learned final theta, or superiority claims are supplied.
The user deferred manuscript/LaTeX generation until HPC results are returned.

The four earlier FD001 runs were not present in the repository or this workspace.
They have not been deleted, modified, counted, or imported. Compatibility cannot
be established from their names alone. The new loader requires protocol, data,
configuration and code identities and refuses to overwrite incompatible records.
Retain earlier checkpoints separately until their provenance is explicitly audited.
