# reproduceSL — fullTD

The current FD001 experiment has **one** four-method protocol:
[`dscpuscripts/fd001_protocol.py`](dscpuscripts/fd001_protocol.py).
It covers 600 TCN candidates, with MSE, Gaussian NLL, typed TD and
pairwise TD. Each TD variant uses batch sizes 128 and 256. No historical
FD001 versioned protocol, extension study or duplicate reporting pipeline
is kept in the current tree.

Run and analysis commands are in the
[direct-server README](dscpuscripts/README.md):

```bash
bash dscpuscripts/run_standard.sh fd001 --dry-run
bash dscpuscripts/run_standard.sh fd001 --run --workers 48 --adaptive
bash dscpuscripts/analyze_standard.sh fd001 check
bash dscpuscripts/analyze_standard.sh fd001 run
```

Results use sibling directories
`~/scratch/fulltd/fd001_results/` (training states) and
`~/scratch/fulltd/fd001_outputs/` (analysis).
The model is evaluated initially and every 500 optimizer updates; the
last 25% of positive-update validation evaluations select each method's HPs.
Test labels are never used for hyperparameter tuning.

Electricity and METR-LA retain their existing model/training implementations
under `fulltd/` and `agents/`. Their 510 candidate configurations are
maintained separately in `configs/fulltd/`. The FD001 data/model interfaces
are shared with those implementations, but no obsolete FD001 method is
offered by the older `fulltd` manifest.

The sources include code, dataset verification, regression tests, and
licensing metadata. Training checkpoints and generated reports are stored
outside the tracked repository; updating Git code never deletes them.
