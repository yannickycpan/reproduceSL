# One AMD EPYC server: FD001 Gamma sweep

All operational scripts are inside `dscpuscripts/`. The scientific model, the
three Gamma variants, the 3-by-4 learning-rate grid, five seeds, original
sampler, training objective, Adam, and validation/test protocol are unchanged.
**All 180 configurations run on one machine.** No Slurm, cluster job time
limit, second server, sharding, copying or result merging is involved.

## First-time setup

On the dual AMD EPYC 9754 Ubuntu server:

```bash
git clone --branch fullTD https://github.com/yannickycpan/reproduceSL.git
cd reproduceSL
bash dscpuscripts/inspect_server.sh
bash dscpuscripts/setup_server.sh
bash dscpuscripts/download_data.sh fd001
```

If you already cloned the repository, instead run:

```bash
cd ~/reproduceSL
git switch fullTD
git pull --ff-only origin fullTD
```

`setup_server.sh` installs the existing version-pinned CPU requirements.
It accepts Python 3.10, 3.11 or 3.12 (Ubuntu 22.04 commonly uses 3.10).
If needed, specify an interpreter with `FULLTD_PYTHON=/path/to/python`.
The script will not upgrade system Python or need administrator access.

`download_data.sh fd001` reuses the original NASA C-MAPSS downloader,
retains SHA-256/source sidecars, performs the original train/val/test split
and training-only standardization, and verifies every processed file. It
does not overwrite already prepared data. It can optionally download
`electricity`, `metr_la` or `all`, but this direct training campaign is
**FD001-only**.

## CPU concurrency: choose from the actual server state

AMD EPYC 9754 provides 128 physical cores and 256 SMT threads per socket:
dual socket means 256 physical cores and up to 512 logical CPUs.

The old 16-process suggestion was a conservative check, **not the capacity**.
An idle machine may support 96, 128 or more independent single-thread runs,
but the practical optimum depends on memory bandwidth, process memory and
other active users.

The login report showed load 246.91. On a shared 256-core machine, that is
a reason to measure current utilization rather than blindly start 128 jobs.
Linux load is not identical to instantaneous CPU utilization; it can include
I/O-waiting tasks. The script prints both a short CPU-idle measurement and
the load average. **It warns rather than blocking on a high load**.

`--workers auto` is the default. It chooses **up to 128 concurrent processes**
using approximately 75% of the measured idle physical-core capacity (a
starting heuristic, not a benchmark). If you know you have capacity, override
the estimate explicitly with `--workers 96` or `--workers 128`.
The logical-batch size remains 128 regardless of CPU worker count.

Run the preflight/dry-run first:

```bash
bash dscpuscripts/inspect_server.sh
bash dscpuscripts/run_fd001_gamma.sh --dry-run
```

For an initial **full 100-epoch single-run smoke test**, use:

```bash
bash dscpuscripts/run_fd001_gamma.sh --run --workers 1 --max-runs 1
```

It trains one canonical configuration and leaves its checkpoint/result
available for the next invocation. If the server is not busy, run the
remaining candidates with the default auto worker selection:

```bash
nohup bash dscpuscripts/run_fd001_gamma.sh --run \
  > "$HOME/fd001_gamma_direct.log" 2>&1 < /dev/null &
echo "Dispatcher PID: $!"
```

Or explicitly choose a higher concurrency on an otherwise idle machine:

```bash
nohup bash dscpuscripts/run_fd001_gamma.sh --run --workers 96 \
  > "$HOME/fd001_gamma_direct.log" 2>&1 < /dev/null &
echo "Dispatcher PID: $!"
```

Do not launch two dispatchers simultaneously. A filesystem lock enforces
this. All workers run as independent one-thread Python CPU processes.
One completed run gets its own log in
`~/scratch/fulltd/fd001_gamma_v1/results/direct_logs/`.
Progress and restart receipts are in `results/direct_launches/`.
No hidden job time limit, requeue or Slurm command is involved.
On Ctrl-C/SIGTERM, in-progress runs are asked to stop gracefully and
restart later from their most recent **completed epoch**. Repeating
`--run` skips completed or terminal numerical-failure records.

Monitor:

```bash
tail -f "$HOME/fd001_gamma_direct.log"
vmstat 1 5
free -h
```

The `vmstat` columns `r`, `b`, `si`, `so`, `us`, `sy`,
`id`, `wa` help distinguish CPU contention, I/O wait, and swapping.
Pay particular attention to sustained `si/so` activity and low
`MemAvailable`. `--workers auto` measures availability at startup,
not dynamically throughout the campaign.

## Analyze after all 180 candidates finish

```bash
cd ~/reproduceSL
bash dscpuscripts/analyze_fd001_gamma.sh
```

The existing analysis requires all candidates to have terminal records,
checks the scientific/protocol hashes, and selects hyperparameters **only**
from five-seed validation MSE at epochs 80, 90 and 100. It evaluates the
five chosen seeds for each variant (15 trained models), saving score tables,
test engine predictions, uncertainty summaries, and vector figures.

Results:

```text
~/scratch/fulltd/fd001_gamma_v1/results/
~/scratch/fulltd/fd001_gamma_v1/outputs/
```

No original FD001 or ongoing METR-LA results are changed or deleted. Do not
delete `~/scratch/fulltd/results/`.

## Verification

```bash
"$HOME/scratch/fulltd/.venv-fulltd/bin/python" -m pytest -q \
  dscpuscripts/test_direct_servers.py
```

This verifies one-server manifest coverage, worker selection, checkpoint
identities and dry-run behavior. Synthetic tests and code review do **not**
replace a real-machine dry run and one-config smoke test.

If the setup or performance is unclear, send the output of:

```bash
bash dscpuscripts/inspect_server.sh
```

The default paths are exported by `dscpuscripts/env.sh` and may be
overridden through `FULLTD_ENV`, `FULLTD_DATA_DIR`,
`FULLTD_FD001_GAMMA_RESULTS_DIR` and
`FULLTD_FD001_GAMMA_OUTPUT_DIR`. Use fast local storage if available.
