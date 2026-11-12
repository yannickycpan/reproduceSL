# METR-LA source-fraction research variant

This branch introduces a **compute-budget variant**, not a new fixed
dataset subset. It does not alter the FD001 or Electricity experiments
on the parent `fullTD` branch. Do **not** switch the working tree that
is currently training or analyzing those experiments.

## Protocol

Keep the complete METR-LA dataset (207 sensors), standard 70/10/20
chronological splits, full valid-target masks, 12-to-12 forecasting,
the same compact Graph WaveNet, all six methods, the complete TD partner
universe and temporal lags, Adam without clipping or weight decay, the
same learning-rate grids, five seeds and logical batch 256. The number
of rounds remains 100.

The scientific parameter `FULLTD_METR_LA_SOURCE_FRACTION` defaults
to `0.10`; `0.25` and `1.0` are alternatives (any finite value in
`(0,1]` is valid). For fractions below 1, every round draws
`ceil(fraction * N)` eligible **source windows** independently and
uniformly **with replacement**, using deterministic seed/round-specific
streams independent of the batch grouping. Partners for the TD methods
are still sampled from all eligible training windows. All methods with
the same seed use the same source sequence.

**At fraction exactly 1.0**, each of the 100 rounds calls the original
frozen METR-LA random-permutation sampler, restoring its exact source
sequence and 500-update validation cadence. Hence 100% recovers the
original *scientific sampling/optimization setup*, not old checkpoint
identity or stored results: protocol ID, source-file hash and run IDs
intentionally change in this branch.

Approximate budgets for 23,967 eligible source windows:

| Fraction | Draws/round | Updates/round | Updates/100 rounds | Validation cadence |
|---|---:|---:|---:|---:|
| 0.10 | 2,397 | 10 | 1,000 | 100 updates |
| 0.25 | 5,992 | 24 | 2,400 | 250 updates |
| 1.00 | 23,967 | 94 | 9,400 | 500 updates |

The 10% budget is approximately **ten effective full source passes**
in total draws, not 100 complete epochs and not 10% of the available
labels. Stochastic gradient noise differs from a full random-permutation
training run, and the valid-count normalization in masked TD is not
claimed to be a globally unbiased ratio estimator. Validation selection
still averages the final ceil(25%) of positive-update validation
measurements across five seeds and never accesses test labels before
locking the selection. The same fraction/cadence applies to all methods.

The 9.4x drop in optimizer update count is *not a measured runtime
speedup*. Loading the graph, 100 recovery checkpoints, 207-node
validation and test inference remain. Before using 10% as the main-paper
protocol, check late-stage validation convergence for MSE, Gaussian NLL,
masked MAE and both TD variants, and compare with an adequate full-budget
reference. Do not claim 10%-label sample efficiency.

## Independent checkout and output roots

From the original repository:

```bash
cd ~/reproduceSL
git fetch origin fullTD-metrla-sample10
git worktree add -b metrla-sample10 ~/reproduceSL-metrla10 origin/fullTD-metrla-sample10
cd ~/reproduceSL-metrla10
$HOME/scratch/fulltd/.venv-fulltd/bin/python -m pytest -q tests dscpuscripts
bash dscpuscripts/run_metr_la_sampled.sh --dry-run
```

Default paths:
`~/scratch/fulltd/metr_la_sampling/f10pct/metr_la_results/`
and
`~/scratch/fulltd/metr_la_sampling/f10pct/metr_la_outputs/`.
Choose a fraction by prefixing the commands with
`FULLTD_METR_LA_SOURCE_FRACTION=0.25` (separate `f25pct` root) or
`FULLTD_METR_LA_SOURCE_FRACTION=1.0` (separate `f100pct` root).
The wrappers explicitly reject a conflicting `FULLTD_METR_LA_ROOT`,
so cannot silently write into the original `metr_la_results`.

Once tests pass and the server has capacity:

```bash
nohup bash dscpuscripts/run_metr_la_sampled.sh --run --workers 64 --adaptive \
 > "$HOME/metr_la_sample10.log" 2>&1 < /dev/null &
bash dscpuscripts/run_metr_la_sampled.sh --status
# Only after all 525 runs are complete or documented numerical failures:
bash dscpuscripts/analyze_metr_la_sampled.sh check
bash dscpuscripts/analyze_metr_la_sampled.sh run
```

Do not start another 525-run campaign if the original Graph WaveNet
training already saturates the server. The original campaign can continue
in its original checkout without source or checkpoint changes.
