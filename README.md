# reproduceSL — fullTD

The TD v7 neural application campaign on branch `fullTD` provides PyTorch
objectives and a [frozen protocol](docs/fulltd_protocol.md) for Electricity,
METR-LA and FD001. Each explicit JSON manifest index identifies one candidate
run. HPC operating instructions are provided directly in the accompanying chat.

The branch contains only this campaign:

- `agents/fulltd_*.py`: the six prescribed objectives and their common interface.
- `fulltd/`: data preparation, fixed backbones, training, evaluation and analysis.
- `configs/fulltd/`: the complete 630-candidate manifest and dataset manifests.
- `scripts/`: environment/data setup, two dataset-group submission entries and analysis.
- `tests/`: reproducible mathematical, data, training and result-integrity checks.
- `docs/`: protocol, architecture/source attribution, verification and licenses.

The earlier experiment harness and unrelated bundled datasets remain available
in Git history. Current public data, checkpoints and generated results are stored
outside the tracked source tree. See [verification](docs/fulltd_verification.md)
for checks actually executed; this repository does not claim completed HPC results.
