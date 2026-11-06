# Direct CPU servers: FD001 Gamma campaign

All new operational scripts live in **dscpuscripts/**. These servers do not need
Slurm, HPC modules, job arrays, requested memory, or allocation time limits.
The scripts reuse the **unchanged** `fd001_gamma.train`, `fd001_gamma.analysis`,
`fd001_gamma.protocol`, `fulltd.data`, and `fulltd.download`. They do
not modify the original TD/Graph WaveNet experiment or any scientific code hash.

## Commands on either server

```bash
git clone --branch fullTD https://github.com/yannickycpan/reproduceSL.git
cd reproduceSL
bash dscpuscripts/inspect_server.sh
bash dscpuscripts/setup_server.sh
bash dscpuscripts/download_data.sh fd001
```

If the repository is already cloned, use `git switch fullTD` then
`git pull --ff-only origin fullTD` instead of cloning.

The setup requires Python **3.12** (e.g. `python3.12`) and installs the
existing pinned PyTorch 2.5.1 CPU environment. To use a specific interpreter:

```bash
FULLTD_PYTHON=/path/to/python3.12 bash dscpuscripts/setup_server.sh
```

If the server cannot reach the NASA download URL, copy the original
`raw/fd001/` archive, extracted files, and `*.source.json` provenance files
from a trusted machine into the same `FULLTD_DATA_DIR`, then rerun
`download_data.sh fd001`; do not invent source sidecars. The downloader uses
the NASA official CMAPSSData.zip location already prescribed by the project.

`download_data.sh` uses the existing downloader and preprocessing functions.
It verifies prepared split hashes and refuses to silently reprepare a populated
processed dataset. To acquire other datasets with the same code, run
`bash dscpuscripts/download_data.sh electricity`, `metr_la`, or `all`.
**Only FD001 has the direct CPU training runner here.**

### One server only

```bash
bash dscpuscripts/run_fd001_gamma.sh --dry-run --workers 16
bash dscpuscripts/run_fd001_gamma.sh --run --workers 16
```

The runner skips completed and numerical-failure candidates, resumes compatible
epoch checkpoints, and writes a separate log per run under
`~/scratch/fulltd/fd001_gamma_v1/results/direct_logs/`. There is **no time
limit**. A Ctrl-C or SIGTERM requests a graceful pause at a batch boundary;
the current partial epoch is discarded, as in the original protocol.

Use `tmux` or `nohup` to survive SSH disconnection, for example:

```bash
nohup bash dscpuscripts/run_fd001_gamma.sh --run --workers 16 \
  > "$HOME/fd001_gamma_direct.log" 2>&1 < /dev/null &
echo "Dispatcher PID: $!"
```

A short operational trial can select `--max-runs 1`. This completes the
chosen candidate's full 100 epochs; rerunning without the limit fills the
remaining configurations. No incomplete-epoch trial results are used.

### Two servers: deterministic split (4:1)

The canonical manifest iterates over **five seeds fastest**, so modulo-five
shards assign the same hardware to the same seed across all hyperparameters
and three Gamma variants.

**Dual EPYC 9754 server** (144 candidates, seeds 0,1,2,3):

```bash
bash dscpuscripts/run_fd001_gamma.sh --dry-run --shards 5 --shard-indices 0-3 --workers 16
bash dscpuscripts/run_fd001_gamma.sh --run     --shards 5 --shard-indices 0-3 --workers 16
```

**Dual Xeon Gold 6226R server** (36 candidates, seed 4):

```bash
bash dscpuscripts/run_fd001_gamma.sh --dry-run --shards 5 --shard-indices 4 --workers 4
bash dscpuscripts/run_fd001_gamma.sh --run     --shards 5 --shard-indices 4 --workers 4
```

These are conservative initial concurrency settings, **not automatic
hardware benchmarks**. Compare `free -h`, CPU utilization, and timings before
raising to, for example, 32 processes on EPYC and 8 on Xeon. Each worker runs
PyTorch with exactly one CPU thread and its own independent Python process.
Avoid scheduling another CPU-intensive workload at the same time. The launcher
checks available CPU affinity and refuses worker counts beyond it.

Do not run the two shards into the same results directory on a shared
filesystem unless you intentionally coordinate dispatchers; the top-level
direct-runner lock prevents simultaneous launchers in one directory. The
recommended setup uses each server's own local `~/scratch/fulltd/` folder.

## Bring the Xeon results onto the EPYC machine

Do this **after** both servers have finished their assigned runs. Replace
`USER@XEON_HOST` and the remote path with your actual SSH address.
On the EPYC server:

```bash
cd ~/reproduceSL
bash dscpuscripts/pull_remote_results.sh USER@XEON_HOST
```

If the secondary server uses a different results path, supply it as the second
argument. The helper uses `rsync` to stage only canonical run directories,
performs a dry-run verification, then imports them. Staging files are retained
for audit and safe retry. You may instead transfer the directory manually and
call `python -m dscpuscripts.import_results --source /path/to/staging`.

The importer checks every run's canonical configuration, frozen protocol,
code fingerprint, and prepared-data identities against the local data. It
imports only completed or terminal numerical-failure runs, compares hashes
when a run already exists, and **never overwrites a different local result**.
It copies run directories atomically. No user data is deleted.

Both servers must run the **same git scientific code** and produce the
**same processed-data fingerprint**. Check the value printed by
`download_data.sh` on each server. Identical raw datasets from the same
source are required. Different CPUs can still produce small floating-point
differences; your paired comparisons must retain seed/hardware provenance.

## Analyze the complete 180-run campaign

On the machine with all results (EPYC after import):

```bash
bash dscpuscripts/analyze_fd001_gamma.sh
```

The existing analysis refuses to open test targets before all 180 runs have
terminal audit records. It selects one joint `(gamma_variant, alpha, beta)`
setting from five-seed pooled validation MSE at epochs 80/90/100; evaluates
the five selected seeds **per variant** (15 models) and plots validation,
Gamma and test metrics. Outputs appear in
`~/scratch/fulltd/fd001_gamma_v1/outputs/`.

## Diagnosis and safety

- Run `bash dscpuscripts/inspect_server.sh` and send its output if Python,
  RAM, number of visible CPUs, or disk layout is uncertain.
- An operational worker error makes the dispatcher stop starting new jobs,
  signal running workers to checkpoint at their next safe boundary, and
  return nonzero. Inspect `results/direct_logs/<run_id>.log` and rerun.
- Scientific numerical failures remain terminal and are included in
  validation selection as failed candidates; do not silently retry them.
- To change paths without editing code, export `FULLTD_ENV`,
  `FULLTD_DATA_DIR`, `FULLTD_FD001_GAMMA_RESULTS_DIR`, or
  `FULLTD_FD001_GAMMA_OUTPUT_DIR` **on each server**.
- `--dry-run` never writes experiment files or starts training.
- Never delete `~/scratch/fulltd/results/`: it holds unrelated earlier
  FD001/METR-LA campaigns.

Targeted tests, after setup:

```bash
"$HOME/scratch/fulltd/.venv-fulltd/bin/python" -m pytest -q dscpuscripts/test_direct_servers.py
```
