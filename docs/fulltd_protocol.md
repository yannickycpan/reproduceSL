# Current Electricity and METR-LA protocol

The sole supported application protocol is
[`dscpuscripts/application_protocol.py`](../dscpuscripts/application_protocol.py).
The data processing, fixed network architectures and temporal partner laws
remain in `fulltd/`. FD001 keeps its existing protocol, code, checkpoint
identity and reporting commands unchanged.

## Matched method comparisons

| Dataset / backbone | Methods | Candidates |
|---|---|---:|
| Electricity / TCN | MSE, Gaussian NLL, typed TD, pairwise TD | 350 |
| Electricity / PatchTST | MSE, Gaussian NLL, typed TD, pairwise TD | 350 |
| METR-LA / Graph WaveNet | same four, plus masked MAE and pairwise TD-MAE | 525 |

Every comparison within a backbone uses the same initialized mean predictor
for a given seed. DLinear is a distinct seasonal/trend linear forecaster, not
an MSE head on TCN/PatchTST, and is therefore excluded from this campaign.
Gaussian NLL adds only its variance head and loss to the same mean network.
There is no stop-gradient path, target network, regularization, clipping,
learning-rate schedule, early stopping, or post-training refit.

All runs use Adam (betas 0.9/0.999, epsilon 1e-8, weight decay zero), seeds
0–4, and 100 training units. Baselines use logical batch 256 and predictor
rates 0.0001, 0.0003, 0.0005, 0.0007, 0.0009, 0.001, 0.003, 0.005,
0.007 and 0.01. All application methods use logical batch 256 only. TD
predictor rates are 0.00003, 0.0001, 0.0003, 0.001 and 0.003; TD operator
rates are 0.00003, 0.0001, 0.001, 0.003 and 0.01, as a full 5×5 grid.
FD001 retains its original beta grid and both TD batch sizes unchanged.

## TD definitions and Graph WaveNet adaptation

Typed TD learns two unrestricted zero-initialized coefficients, one per
native temporal lag type. Pairwise TD reuses FD001's exact
`PairwiseGammaHead`: a type-specific affine function of the normalized
hidden-state inner product, coordinatewise absolute difference, and
normalized difference norm. Both endpoint predictions and the representation
used by the operator remain in the gradient graph.

Graph WaveNet provides hidden features `[B, sensors, 192]`. Only the input
to the Gamma head is flattened to `[B, sensors * 192]`; the mean prediction
head still operates independently on each sensor's hidden vector. One Gamma
per window pair is broadcast over all sensors and forecast horizons.
For 207 sensors the operator sees 39,744 coordinates and has 79,494
parameters. This retains sensor identity without introducing a new pooling
rule, per-sensor Gamma family, or additional forward pass. It assumes the
fixed sensor order already recorded by the dataset and adjacency.

Writing source/neighbor prediction residuals as e_i/e_j, squared TD uses
0.5 * (e_i − Gamma_ij * e_j)^2. The extra METR-LA pairwise-TD-MAE method
uses |e_i − Gamma_ij * e_j| and otherwise has the same head, initialization,
sampling, optimizer and hyperparameter grid. Both use the source/partner
mask intersection. Ordinary MSE/NLL/MAE use all valid source targets.
An empty mask produces no optimizer step and is recorded. Gradient
accumulation divides by the global logical-batch valid count, including
unequal final microbatches.

## Data exposure and justified dataset exceptions

METR-LA uses 100 complete reshuffled passes, as FD001 does. Electricity
retains 100 independent stochastic rounds, each drawing ceil(N/24) sources
uniformly with replacement. Every method/backbone with a given seed sees
the same source sequence; all TD variants share the same partner stream,
independent of batch grouping. Partners come from the entire eligible
training set. Isolated sources are excluded from every method equally.
Native lag types remain (1,24) hours for Electricity and (1,3) five-minute
steps for METR-LA, with dataset gaps and entity boundaries respected.

Electricity evaluates every 5,000 actual Adam updates, METR-LA every 500.
Both evaluate update zero and the actual final state even when the endpoint
falls between regular measurements. Checkpoints contain the complete
optimizer, RNG, source cursor and measurement state; interruptions resume
without replaying or skipping sampled pairs.

Electricity validation uses one fixed customer-stratified 20% sample:
ceil(n_customer/5) origins without replacement for each customer. An
independent deterministic seed fixes the same view across methods, rates,
training seeds and evaluation times. This is the explicit computation
exception to FD001's full validation, and avoids candidate-dependent
validation noise. METR-LA uses all validation windows. Test evaluation uses
the complete test split in both datasets.

## Selection and requested reports

For each backbone and method at B256, ignore update zero, take the final
ceil(25% × number of positive-update measurements), average validation MSE
within a seed and then across all five seeds, and select the best common
rate setting. This keeps FD001's validation selection principle. The
candidate must reach the recorded actual update budget.

Every candidate must be complete or have a recorded numerical failure.
An unfinished run is not a poor score and does not permit selection. A
numerical failure invalidates the full-budget setting across all five seeds;
no separate matched-budget application reports are produced. Selection never reads test results. The actual
last iterate at the chosen budget is evaluated, not a best validation epoch.

Reports are deliberately restricted to RMSE/MAE tables and learning curves.
Each metric/backbone table has algorithms as rows and Test / Validation
columns, mean ± sample-standard-deviation/sqrt(5). Every row uses B256;
selected rates are written to CSV.
Electricity uses standardized equal-customer errors; METR-LA uses masked
all-horizon errors in mph. Curves have test/validation panels and standard
errors, retain the complete update axis, and are reconstructed from saved
states only after validation selections are locked.

## Entry points and FD001 compatibility

Generate the committed manifests with:

```bash
python -m dscpuscripts.application_protocol --output configs/fulltd
```

Run `dscpuscripts/run_electricity.sh` or `run_metr_la.sh`; report through
`dscpuscripts/analyze_standard.sh`. Detailed AMD commands are in
[`dscpuscripts/README.md`](../dscpuscripts/README.md).

The earlier application analysis/refit/Slurm entrypoints have been removed,
its manifests replaced, and `td_full` / `td_scalar` constructors now reject
use. A few shared source files (`fulltd/protocol.py`, `fulltd/train.py` and
common objective definitions) remain byte-identical because FD001 hashes
the entire files as part of its existing checkpoint identity and imports
shared helpers from them. They are frozen compatibility dependencies, not
application entrypoints; do not regenerate manifests through `fulltd.protocol`
or launch `fulltd.train`. Import-only retired class names remain solely
because the frozen FD001 baseline factory imports them eagerly.
The application scientific identity has changed; existing incompatible
application checkpoints must be preserved under a separate result root.
`tests/test_checkpoint_identity.py` enforces the original FD001 file hashes
and aggregate scientific hash from commit 24b9296.
