# Frozen scientific protocol

Identifier: `td-v7-neural-e24-v10-final3-ev20-epochckpt-v2`. The user's amendments
supersede exhaustive Electricity training, the five-point selector, full
Electricity validation, and mid-epoch checkpoints. The 1 November 2026 revision
changes only the authorized validation sampling, checkpoint cadence and CPU
execution layout; all 630 candidates and training objectives/budgets remain.
The authoritative theory is *Learning Statistically Efficient TD Operators:
Prediction with Sampled Neighbors and Simultaneous Full Gradients*, v7, 25 October
2026. Its setup, full-gradient objective/updates, main results, scope and notation
were read. The manuscript itself is not copied or modified in this repository.

## Objective and weights

For observed residuals `e_i = y_i - f_w(x_i)`, TD minimizes
`0.5 * mean((e_i - theta[type] * e_j)^2)` over valid scalar pairs. TD-scalar shares
one signed coefficient over the same pair sets. Both predictions participate in
the same gradient graph; both parameter groups take one simultaneous Adam step.
No partner residual averaging, MSE anchor, target network or test correction.
The two TD-full coefficients start exactly at zero and remain unrestricted.

One source is a customer/origin (Electricity), whole-network origin (traffic), or
engine/endpoint (FD001). Each partner keeps the same entity and scalar horizon;
traffic uses one origin partner/type for its entire sensor/horizon output. Pair
lags are `(1,24)` hours, `(1,3)` five-minute steps and `(1,5)` cycles. Choose a
supported type uniformly, then a supported earlier/later direction uniformly.
No wrapping, crossing splits or entity boundaries, or compressing timestamp gaps.
Isolates are removed from the common source universe for every method and logged.
The finite boundary law is row-stochastic, generally asymmetric/non-balanced.

Sources have mass `1/N`. Electricity samples `M=ceil(N/24)` iid sources each round
from the complete universe; partners also come from that universe. Conditional
on fixed parameters, the expected complete-target sample loss equals
`(1/N) sum_i sum_j P(j|i) loss(i,j)`. The 24 divisor is fixed by the forecast
horizon. Repeated draws are legitimate, including repeated sources within one
logical batch. No customers or origins are permanently thinned. The reduced
budget does not reproduce exhaustive optimization trajectories.

METR-LA/FD001 use full random permutations. Stable separate NumPy SeedSequence
streams depend only on dataset, training seed, round/epoch and source/partner
purpose; neither backbone, loss, rates nor batch size enters the source stream.
Partner choices are likewise shared across all TD configurations and batch sizes.
Type frequencies and source/partner sequence SHA-256 values are recorded per round.

Traffic source masks are determined in raw units (zero is missing). TD uses the
source/partner intersection and divides by its actual valid scalar count. Ordinary
objectives use every valid source target. Empty TD batches skip Adam and count
toward source passes. This ratio reduction is prescribed and is not asserted to
be an unbiased fixed-denominator masked estimator. At zero theta, gradient
equality with MSE holds on complete targets; missing-pair masks qualify it.

MSE is half mean squared residual; MAE is mean absolute residual. NLL is half
`e^2/v + log(v)`, with `v=1e-6+softplus(a)`, an input-dependent linear head sharing
the predictor representation. Initial head weights are zero and bias gives
variance one. All point inference uses only the mean network.

## Fixed matrix and optimization

| Dataset/backbone | Primary methods | Additional batch 128 |
|---|---|---|
| Electricity/TCN | MSE, TD-full, TD-scalar, NLL | both TD methods |
| Electricity/PatchTST | MSE, TD-full | TD-full |
| Electricity/DLinear | DLinear-MSE | none |
| METR-LA/Graph WaveNet | MSE, TD-full, TD-scalar, NLL, MAE | both TD methods |
| FD001/TCN | MSE, TD-full, NLL | TD-full |

Each combination has six configurations and seeds 0,1,2,3,4: 630 total. Nontrained
daily seasonal Electricity and last-observation METR-LA persistence use identical
held-out targets and causal input filling.

TD `(alpha,beta)`: `(1e-4,1e-4)`, `(1e-4,1e-3)`, `(3e-4,3e-4)`, `(3e-4,3e-3)`,
`(1e-3,1e-3)`, `(1e-3,1e-2)`. Ordinary alpha: `1e-4,2e-4,3e-4,5e-4,7e-4,1e-3`.
Adam uses `(0.9,0.999)`, epsilon `1e-8`, weight decay zero, AMSGrad false. Rates
are constant. Float32 fitting, float64 metric/normal-equation aggregation; no
mixed precision, clipping, constraints, schedule, augmentation or regularizer.
The final short logical batch is retained. Microbatches accumulate loss sums
divided by the full logical batch's valid count before one optimizer step.

All finite candidates train for 100 declared units. Batch128 has more updates
under the same source budget, so its comparison is not solely batch variance.
Numerical nonfinite loss/gradient/parameter is retained as a failed run; OOM or
data corruption is an operational error. Any failed seed gives its configuration
score infinity. No hyperparameter pruning or grid expansion.

## Validation, endpoint metrics and diagnostics

Exactly ten validation evaluations: 10,20,...,100. METR-LA and FD001 use their
complete fixed validation sets. Electricity independently samples each customer's
eligible validation windows uniformly without replacement at every evaluation,
using `ceil(n_customer/5)` windows. The score averages the sampled MSE within each
customer, then equally across customers. For a fixed predictor this estimates the
original equal-customer validation MSE without bias; it has sampling variability
and need not select the same configuration as full validation. No such unbiasedness
claim is made for the selected minimum score.

During training each validation pass accumulates only the all-horizon standardized
MSE and MAE in float64. Its masks, per-customer/per-engine weighting and reduction
order are unchanged; per-horizon statistics and RMSE are omitted from this inner
loop. Both measures are recorded at all ten evaluations and can be plotted. MSE
alone selects the common configuration; MAE is descriptive. Final endpoint
evaluation still supplies all previously prescribed horizon-specific/raw metrics.

Electricity's PCG64 stream uses a fixed validation seed, stable SHA-256 run ID,
evaluation round and customer ID. Different candidates/seeds have independently
reproducible samples, as authorized by the user's latest correction. These streams
never consume training/source/partner RNG state. Each record retains the parent
index hash and per-customer population/sample counts. On the verified data this
means 697 of 3,484 windows for each of 348 customers: 242,556 of 1,212,432 windows
per evaluation (20.0057%, due only to integer rounding).

For each fixed
dataset/backbone/method/batch, `Score(h) = mean_{s=0..4} mean_{e in {80,90,100}}
ValMSE(s,h,e)`. Exact ties use smaller alpha, then beta. The common selected h
reports all five actual final models without restart, averaging or refit. All
dataset selections must be fixed before the separate endpoint command opens test
targets. No best-checkpoint selection or early stopping.

Endpoint Electricity validation and its two frozen-readout diagnostics reuse the
selected run's round-100 validation sample; persistence has its own reproducible
validation sample. Every test target and the full final training evaluation set
remain unchanged. METR-LA/FD001 endpoint validation remains complete.

Weights and Adam state are saved only after an entire epoch/round and any scheduled
validation finish: one rolling recovery checkpoint for epochs/rounds 1--99 and
one final checkpoint at 100. No weights, optimizer state or validation partial
sums are saved inside training or validation, including on a time-limit warning.
An interruption resumes at the previous complete epoch and replays the incomplete
work with the same sequences. Small interruption receipts separately record
discarded computation. `runtime_seconds` measures the retained trajectory;
`discarded_runtime_seconds` records graceful-interruption work that must be
replayed; `total_execution_seconds` includes both. Hard kills can still lose
unrecorded runtime, for which scheduler accounting is the external evidence.

CPU execution uses seven Slurm array elements, one node per element and 90
independent one-CPU/one-thread candidate processes per node. This is not
distributed training of a model. It does not change logical batch sizes,
gradients, sampling, rates or update counts. CPU/GPU bitwise equality is not
asserted. Legacy results retain their old protocol identity and are not converted
or mixed into this restart campaign.

The campaign may also be submitted independently as Electricity (300 candidates)
and METR-LA plus FD001 (330 candidates). Both use the same original manifest indices
and run IDs. Default array concurrency is three and four nodes respectively, so
both submissions together use at most seven candidate nodes. The respective
72-hour and 12-hour allocation limits are operational defaults, not measured
completion times or changes to the 100-epoch/round budget. Either group can be
selected/evaluated after all its own candidates finish or fail numerically;
unrequested datasets are not read. A later combined analysis uses the same chosen
models and compatible endpoint caches.

Electricity validation and reported aggregates give each retained customer equal
weight over its available origins. Horizon1, horizon24 and all24 are retained in
standardized/kW MAE/MSE/RMSE. RMSE is the square root after squared-error
aggregation. Traffic averages valid scalar targets; horizons3/6/12 represent
15/30/60 minutes, plus all12. FD001 validation first averages fixed prefixes within
engine, then engines. Training-window error uses the source-window population;
test uses one endpoint per official test engine. Raw cycles and the asymmetric
score sum are retained. For `z=prediction-target`, score is `expm1(-z/13)` when
negative and `expm1(z/10)` otherwise; no clipping, overflow preserved.

Final training error is recomputed over all fitting windows, including all
Electricity sources, regardless of stochastic training budget. Only selected
primary TD-full TCN models on Electricity/FD001 receive frozen-feature/frozen-
theta OLS and exact-local-law TD linear readout fits. They stream all training
sources in float64, include the transformed bias `1-Gamma`, enumerate actual
boundary probabilities, use no ridge, and report rank, conditioning and equation
residual with a fixed Gram SVD relative tolerance `1e-12`. Original neural results
remain the headline predictions.

Five-seed summaries use sample SD and SE. Paired TD-full comparisons pair seed,
with descriptive t intervals (df4) and relative improvement from means. These
describe training randomness under one fixed split/shared tuning; overlapping
windows are not independent confidence samples. Neural learned representations,
constant Adam, dependent inputs and application pair laws are empirical extensions
outside the manuscript's linear statistical/convergence guarantees. Test observed
MSE is not the conditional-mean excess risk, and learned theta is not theta_C.

## Source and operational references

- [UCI original Electricity](https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014)
- [NASA C-MAPSS](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data)
- [NASA score definition](https://ntrs.nasa.gov/api/citations/20090029214/downloads/20090029214.pdf)
- [DCRNN author data and graph](https://github.com/liyaguang/DCRNN)
- [CityU HPC](https://www.cityu.edu.hk/its/news-events/announcements/2024/08/centralised-high-performance-computing-hpc)
- [CityU account information](https://www.cityu.edu.hk/its/services-facilities/list-of-services-facilities/h/hpc-account-request)
- [Slurm job arrays](https://slurm.schedmd.com/job_array.html)
- [Slurm submission options](https://slurm.schedmd.com/sbatch.html)
