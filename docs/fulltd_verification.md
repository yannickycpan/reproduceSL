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

The `fulltd` study's two remaining dataset manifests contain 300 and
210 runs respectively. Their data preparation and architecture modules,
validation strategies, full-gradient TD objectives, and analysis routines
remain available. Run relevant `tests/test_fulltd_*.py` plus the
direct-server regression suite before production use.

Do not claim that an unexecuted test suite passed. Source changes and
GitHub commit status alone cannot confirm on-server checkpoint validity
or cross-hardware numerical equivalence.
