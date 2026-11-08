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

Electricity and METR-LA use the FD001-aligned application protocol in
[`dscpuscripts/application_protocol.py`](dscpuscripts/application_protocol.py).
Electricity has 1,200 candidates (TCN and PatchTST, each with MSE, Gaussian
NLL, typed TD and pairwise TD); METR-LA has 900 Graph WaveNet candidates,
adding masked MAE and pairwise TD with an absolute-error objective.
The current manifests replace the earlier application campaign.

```bash
bash dscpuscripts/run_electricity.sh --dry-run
bash dscpuscripts/run_metr_la.sh --dry-run
# Production launch: replace --dry-run with --run --workers 8 --adaptive.
bash dscpuscripts/analyze_standard.sh electricity run
bash dscpuscripts/analyze_standard.sh metr_la run
```

The [application protocol](docs/fulltd_protocol.md) explains the common
selection rule, Electricity's computational exceptions, exact matched-update
comparisons, and FD001 compatibility. Application reports contain RMSE/MAE
tables and validation/test learning curves only; they do not train or refit.

The sources include code, dataset verification, regression tests, and
licensing metadata. Training checkpoints and generated reports are stored
outside the tracked repository; updating Git code never deletes them.
