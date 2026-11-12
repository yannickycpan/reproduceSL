# Verification and reproducibility notes

This repository now has one FD001 experiment protocol and an independent
Electricity/METR-LA study. It does not retain the obsolete historical
FD001 Gamma, extension, or baseline protocols.

## FD001 verification

The current 600-run FD001 protocol is tested by
`dscpuscripts/test_fd001.py`, which exercises the complete cartesian
hyperparameter grid, correct four-method selection across batch 128 and
256, recorded numerical-failure handling, deterministic interrupted
checkpoint recovery, measured step-zero and periodic update evaluations,
and Gamma diagnostics. Other integration checks verify the single
dispatcher and sibling result/output paths.

Run:

```bash
python -m pytest -q dscpuscripts/test_fd001.py \
  dscpuscripts/test_result_paths.py dscpuscripts/test_standard_workflow.py
```

This synthetic suite does not itself prove the real 600-run campaign has
completed: use `run_standard.sh fd001 --status`, followed by
`analyze_standard.sh fd001 check`. Held-out test evaluation occurs only
after all four validation-based choices are locked.

## Electricity and METR-LA

This experimental branch preserves the 350/525 manifest counts but uses
10% source sampling by default for METR-LA; all prior completion and test
claims belong to the parent fullTD branch. Full pytest and dry-run checks
must be rerun before using the experimental campaign.

The parent-branch application manifests have 350 Electricity and 525 METR-LA
candidates. The common model/data utilities are unchanged; application
protocol, objectives, training and reporting live in `dscpuscripts/application_*.py`.
The replaced application campaign's analysis/refit/Slurm entrypoints and its
obsolete tests have been removed. Current coverage is in
`dscpuscripts/test_application_training.py`,
`dscpuscripts/test_application_analysis.py`, and `tests/test_fulltd_*.py`.

Verification on 2026-11-08 used Python 3.12 and the repository's pinned
PyTorch 2.5.1 CPU / NumPy 1.26.4 requirements:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m pytest -q tests dscpuscripts
```

The historical 130-test result belongs to revision 5597f90; the current
PatchTST-only 240-round protocol must be tested on the AMD server before launch. Matplotlib/Pyparsing deprecation warnings did not
cause failures. The suite includes:

- Full-gradient finite-difference checks for typed, pairwise and pairwise-MAE
  objectives; full logical batch versus microbatch gradient equivalence.
- Real TCN, PatchTST and Graph WaveNet forward/backward and short training
  tests, including Gaussian NLL and the 207-sensor Graph WaveNet shape.
- Exact interrupted-versus-uninterrupted recovery on both application
  datasets, interrupted validation, empty masks, nonfinite failures and
  non-grid final checkpoint retention.
- Fixed common customer-stratified validation, complete grids, common source
  and partner streams, and data preparation/leakage checks.
- Five-seed final-quarter selection,
  missing/incomplete/altered-evidence rejection, correct standard errors,
  validation lock before test access, cached test measurements and actual
  table/PNG/PDF generation.
- Existing FD001 regression tests and byte-for-byte checks of all thirteen
  protected scientific sources against commit 24b9296.

The FD001 scientific identity remains
`c6820c53e26b4ab47be164cb0e5e3a4c62a8f865e01a8b918148980ea840ef21`.
At the earlier 5597f90 revision, all three direct dispatcher dry-runs succeeded (using one worker for FD001
in the smaller verification environment). Shell syntax and Git whitespace
checks also passed.

This verification uses synthetic fixtures, including the real network
implementations; it does not claim that the full application sweeps have
run, that the AMD host was benchmarked, or that unseen server checkpoints
have been inspected. On the server, prepare/verify data, inspect the dry-run,
then launch the current scripts. The selection check verifies the complete
candidate campaign before reporting tests.
