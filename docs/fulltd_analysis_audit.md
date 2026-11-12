# Application analysis audit and performance

**Current branch:** `fullTD`. This report concerns the existing Electricity
and METR-LA test analysis; it does not alter any training scientific sources,
FD001 objectives, hyperparameter selection, masks, data splits, or forecast
metrics. Results must still be verified with server-side regression tests.

## Confirmed and repaired correctness bug

The old `application_analysis.evaluate_selected()` compared the per-candidate
status `identity["code_sha256"]` against the *aggregate*
`plan["code_sha256"]`. The two values use different hash domains and should
not normally equal each other. The corrected code independently verifies
the complete manifest's aggregate against the locked validation-only plan,
and each selected experiment's status against its own scoped code hash,
config SHA, dataset/validation identities and frozen endpoint SHA.

Endpoint selection still uses the five-seed mean validation MSE averaged
over the final ceil(25%) positive-update observations, with no test data
opened until the plan has been locked. The retained full learning curves
still include **every saved checkpoint**, including step zero and the
actual training endpoint.

## Why Electricity full-test reporting takes time

The four selected methods each have five seed runs, around 53 checkpoint
observations per run. Thus a full report performs about 1,060 complete
test-split inferences; it is much more expensive than just plotting.
The former analysis was sequential. Evaluation is now split into two
optional phases and can process independent runs in separate CPU processes.

```bash
# Step 1: validation-only selection and checkpoint audit
bash dscpuscripts/analyze_standard.sh electricity check

# Step 2: ~20 final-checkpoint test evaluations, tables immediately
nohup bash dscpuscripts/analyze_standard.sh electricity endpoints --workers 4 \
  > "$HOME/electricity_endpoints.log" 2>&1 < /dev/null &

# After endpoints finish: evaluate all selected intermediate test checkpoints
# and regenerate complete learning curves. The 20 cached endpoints are reused.
nohup bash dscpuscripts/analyze_standard.sh electricity run --workers 4 \
  > "$HOME/electricity_analysis.log" 2>&1 < /dev/null &
```

Do not run endpoints and full analysis *concurrently* on the same result
output directory; a process-level flock forbids duplicate writers. A
single full `run` still works if endpoint-first tables are unnecessary.
The `run` command retains full test and validation learning curves,
RMSE/MAE tables, mean and standard error across exactly five seeds,
and the same held-out test metrics. There is **no fitting** during analysis.

Default `--workers 1` preserves the old sequential behavior. A conservative
starting point on the shared CPU server is `--workers 4`; benchmark and
monitor memory bandwidth and the concurrently running METR-LA training
before increasing to 8 or more. Each child sets one PyTorch CPU thread,
reads the same read-only mmap data from disk, and owns a distinct per-run
checkpoint and point-metric cache. Processes use Python's `spawn` start
method to avoid inheriting PyTorch thread pools. No speedup multiplier
is guaranteed; it depends on data IO, CPU, NUMA and memory contention.

The default inference `--batch-size 128` is preserved. Another batch size
can be passed explicitly, but it changes test-cache identity and forces
new inference. Full-test metrics should be compared at a consistent batch
size if exact cached floating-point values matter.

**Optional outputs:** `endpoints` writes
`endpoint_per_seed_metrics.csv`, `endpoint_manifest.json`,
`performance_summary.csv` and the usual RMSE/MAE tables.
The `evaluation_manifest.json`, `learning_curves_per_seed.csv` and
figures are produced only after `run` succeeds. A cached test observation
is reused only after the checkpoint's config, cursor and validation history
are checked; the previous aggregate-hash cache layout is migrated when
all other evidence hashes agree.

The existing `plots` command uses full completed cached tables; it does
not fabricate curves after endpoint-only evaluation. Source code changes
in these analysis files do **not** change the training checkpoint code hash.

## Other reviewed invariants and limitations

- The fixed 10%-METR-LA training setting changes source exposure; 100%
  calls the original full random reshuffling sampler. Neither fraction,
  optimizer, original 100-round budget, or any training function is altered
  by this analysis update.
- Checkpoint identity is verified separately per method and dataset; the
  FD001 scientific hash remains frozen.
- Evaluation receipts hash the results CSVs; rerunning plots refuses
  tampered or incomplete full reports.
- Recorded numerical failures disqualify their HP setting in the existing
  whole-grid five-seed selection.
- The analysis currently rejects a full candidate when recorded optimizer
  steps are below the nominal budget (e.g., an empty-valid-mask batch).
  This is an intentional strict full-budget comparison, not silently counted
  as a comparable full-budget result.
- Reproducible process speedup cannot be claimed from static source review.
  Production Python tests and a one-worker/four-worker equality comparison
  on the AMD host are still required before considering the parallel path
  operationally verified.

## Verification commands

```bash
cd ~/reproduceSL
git switch fullTD
git pull --ff-only origin fullTD
$HOME/scratch/fulltd/.venv-fulltd/bin/python -m pytest -q tests dscpuscripts
bash dscpuscripts/analyze_standard.sh electricity check
```

No training-state deletion, renaming, or retraining is performed by this
reporting change. The full 350-run Electricity audit must succeed with the
current scoped scientific identity before touching the test split.
