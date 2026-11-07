"""Run the unchanged 180-config FD001 Gamma campaign on a direct-access Linux server.

No Slurm, job array, allocation time, or HPC module system is involved. Each
candidate runs in its own one-thread Python subprocess and writes its ordinary
fd001_gamma checkpoint files. A rerun skips terminal candidates and resumes
compatible epoch checkpoints. All five seeds run on one AMD EPYC server.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time
import uuid

from fd001_gamma.common import (THREAD_VARIABLES, atomic_json, campaign_paths,
                                cpu_environment, file_lock, read_json, scientific_hash)
from fd001_gamma.protocol import digest, make_manifest

STOP_SIGNALS = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)


def available_physical_cores() -> int:
    """Count unique physical cores accessible through Linux CPU affinity."""
    affinity = (os.sched_getaffinity(0) if hasattr(os, "sched_getaffinity")
                else set(range(os.cpu_count() or 1)))
    cores = set()
    for cpu in affinity:
        base = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        try:
            package = int((base / "physical_package_id").read_text().strip())
            core = int((base / "core_id").read_text().strip())
        except (OSError, ValueError):
            # Linux sysfs is expected on Ubuntu; don't double-count SMT in a
            # nonstandard environment where topology data are missing.
            return max(1, len(affinity) // 2)
        cores.add((package, core))
    return max(1, len(cores))


def cpu_idle_fraction(seconds: float = 0.75) -> float:
    """Estimate idle *physical* cores from both SMT siblings of each core.

    Using a system-wide fraction of 512 logical threads can overestimate free
    cores when one thread on each of 256 physical cores is already busy.
    Taking the minimum idle fraction across each core's siblings avoids that
    particular overcommit. This is a short snapshot, not a scheduler.
    """
    affinity = (os.sched_getaffinity(0) if hasattr(os, "sched_getaffinity")
                else set(range(os.cpu_count() or 1)))
    groups = {}
    for cpu in affinity:
        base = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        try:
            package = int((base / "physical_package_id").read_text().strip())
            core = int((base / "core_id").read_text().strip())
        except (OSError, ValueError):
            # Unknown topology: treat each allowed logical CPU as a group.
            groups = {(cpu,): [cpu] for cpu in affinity}
            break
        groups.setdefault((package, core), []).append(cpu)

    def snapshots():
        counters = {}
        with Path("/proc/stat").open(encoding="utf-8") as source:
            for line in source:
                fields = line.split()
                if not fields or not fields[0].startswith("cpu"):
                    break
                if fields[0] == "cpu":
                    continue
                name = fields[0][3:]
                if not name.isdigit():
                    continue
                values = [int(v) for v in fields[1:9]]
                counters[int(name)] = (sum(values), values[3] + values[4])
        return counters

    before = snapshots()
    time.sleep(seconds)
    after = snapshots()
    physical_idle = []
    for sibling_cpus in groups.values():
        sib_idle = []
        for cpu in sibling_cpus:
            if cpu not in before or cpu not in after:
                continue
            total0, idle0 = before[cpu]
            total1, idle1 = after[cpu]
            if total1 <= total0:
                continue
            sib_idle.append(max(0., min(1., (idle1 - idle0) / (total1 - total0))))
        if sib_idle:
            physical_idle.append(min(sib_idle))
    if not physical_idle:
        raise ValueError("Could not measure Linux per-core CPU idle time")
    return sum(physical_idle) / len(physical_idle)


def choose_workers(requested) -> tuple[int, int, float | None]:
    """Auto uses at most 128 processes and ~75% of currently idle cores.

    Explicit --workers N overrides the heuristic. It is a throughput starting
    point, not an ongoing resource reservation or dynamic scheduler.
    """
    physical = available_physical_cores()
    if str(requested) == "auto":
        idle = cpu_idle_fraction()
        workers = max(1, min(128, int(0.75 * physical * idle)))
    else:
        try:
            workers = int(str(requested))
        except ValueError as exc:
            raise ValueError("--workers must be a positive integer or 'auto'") from exc
        if workers < 1:
            raise ValueError("--workers must be positive")
        idle = None
    logical = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    if workers > logical:
        raise ValueError(f"Requested {workers} workers, but only {logical} logical CPUs are available")
    return workers, physical, idle



def adaptive_worker_target(max_workers: int, active: int, physical: int,
                           idle: float) -> int:
    """80% of estimated idle physical cores, plus existing active workers.

    If 100 physical cores are currently idle and there is no active worker,
    admit 80 jobs, capped by max_workers and the pending-run count (the
    latter is handled by the launch loop). If 80 are already active, retain
    them and add up to 80% of the NEW idle capacity at the next check.
    Do not terminate active jobs when capacity decreases.
    """
    if (max_workers < 1 or active < 0 or physical < 1
            or not 0 <= idle <= 1 or active > max_workers):
        raise ValueError("Invalid adaptive worker sizing inputs")
    new_slots = int(0.80 * physical * idle)
    return min(max_workers, active + new_slots)



def classify(config: dict, results: Path, code_hash: str) -> str:
    """Never trust an old terminal status with a different scientific identity."""
    directory = results / config["run_id"]
    status_file = directory / "status.json"
    if status_file.is_file():
        status = read_json(status_file)
        identity = status.get("identity") or {}
        if (status.get("config") != config
                or identity.get("config_sha256") != digest(config)
                or identity.get("code_sha256") != code_hash):
            raise RuntimeError(f"Incompatible existing record preserved: {directory}")
        state = status.get("state")
        if state == "numerical_failure":
            return "numerical_failure"
        if (state == "complete" and status.get("epochs_completed") == 100
                and all((directory / name).is_file()
                        for name in ("final.pt", "validation.json", "rounds.json"))):
            return "complete"
    if (directory / "final.pt").is_file():
        return "repair_final"
    if (directory / "recovery.pt").is_file():
        return "resume_checkpoint"
    return "start_epoch_1"


def validate_prepared_data(configs: list[dict], results: Path,
                           train_hash: str, val_hash: str) -> None:
    """An existing terminal result must describe the SAME prepared FD001 data."""
    for config in configs:
        path = results / config["run_id"] / "status.json"
        if not path.is_file():
            continue
        identity = (read_json(path).get("identity") or {})
        if (identity.get("data_sha256") != train_hash
                or identity.get("validation_sha256") != val_hash):
            raise RuntimeError(f"Existing Gamma result uses different FD001 processed data: {path}")


def worker(args) -> int:
    cpu_environment()  # Before torch or numpy imports.
    os.environ["PYTHONUNBUFFERED"] = "1"
    if scientific_hash() != args.expected_code_hash:
        raise RuntimeError("Scientific code differs from the dispatcher; refusing training")
    configs = {c["run_id"]: c for c in make_manifest()}
    config = configs.get(args.run_id)
    if config is None:
        raise ValueError(f"Unknown canonical run ID: {args.run_id}")

    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    from fd001_gamma.train import run_config

    stopped = False

    def request_stop(signum, frame):
        nonlocal stopped
        stopped = True

    for sig in STOP_SIGNALS:
        signal.signal(sig, request_stop)
    result = run_config(config, args.data_dir, args.results_dir, device="cpu",
                        microbatch=args.microbatch, stop_requested=lambda: stopped)
    state = result.get("state")
    print(f"DIRECT WORKER: {config['run_id']} -> {state}", flush=True)
    if state in ("complete", "numerical_failure"):
        return 0
    if stopped and state in ("checkpointed", "stopped_no_checkpoint"):
        return 75
    raise RuntimeError(f"Unexpected nonterminal worker outcome: {state}")


def run_campaign(args) -> int:
    cpu_environment()
    if not 1 <= args.microbatch <= 128 or args.max_runs < 0:
        raise ValueError("Use --microbatch in 1..128 and --max-runs >= 0")
    workers, physical, idle = choose_workers(args.workers)
    if args.adaptive:
        if not 5 <= args.check_seconds <= 86400:
            raise ValueError("Adaptive checks require --check-seconds between 5 and 86400")
        max_workers = min(128, physical) if str(args.workers) == "auto" else workers
        if idle is None:
            idle = cpu_idle_fraction()
        initial_target = adaptive_worker_target(max_workers, 0, physical, idle)
    else:
        max_workers, initial_target = workers, workers
    data, results, _ = campaign_paths()
    code_hash = scientific_hash()
    chosen = make_manifest()
    counts = Counter(classify(c, results, code_hash) for c in chosen)
    pending = [c for c in chosen if classify(c, results, code_hash)
               not in ("complete", "numerical_failure")]
    if args.max_runs:
        pending = pending[:args.max_runs]

    # Fail safely if an older nested/per-campaign directory contains run
    # checkpoints absent from the actual flat FD001 results directory.
    if args.mode == "run":
        from dscpuscripts.result_paths import dataset_root, guard_unmigrated_runs
        previous = [dataset_root("fd001") / "results"]
        if getattr(args, "campaign", "original") == "extension":
            previous.append(Path.home() / "scratch/fulltd/fd001_gamma_extension_v1/results")
        else:
            previous.append(Path.home() / "scratch/fulltd/fd001_gamma_v1/results")
        guard_unmigrated_runs(results, chosen, previous)

    label = "FD001 Gamma extension" if getattr(args, "campaign", "original") == "extension" else "FD001 Gamma"
    print(f"{label}: all {len(chosen)} runs assigned to this AMD server; "
          f"{len(pending)} to process now", flush=True)
    print(f"States: {dict(counts)}", flush=True)
    print(f"Data: {data}; results: {results}", flush=True)
    print(f"CPU capacity: {physical} physical cores available; "
          f"1-minute load average: {os.getloadavg()[0]:.1f}", flush=True)
    if idle is not None:
        print(f"Measured system-wide idle CPU fraction: {idle:.1%}", flush=True)
    if args.adaptive:
        print(f"Adaptive concurrency: ceiling {max_workers} one-thread workers; "
              f"initial launch budget {initial_target}; recheck every {args.check_seconds:g}s; "
              "admit 80% of currently idle physical cores (bounded by remaining runs)",
              flush=True)
    else:
        print(f"Workers: {workers} fixed one-thread processes; "
              f"physical microbatch: {args.microbatch}", flush=True)
    if os.getloadavg()[0] >= 0.75 * physical:
        print("WARNING: this shared server is heavily loaded; use --workers auto "
              "or a small explicit value until contention subsides.", flush=True)
    if args.mode == "dry-run" or not pending:
        print("Dry-run / nothing pending: no files or jobs changed.", flush=True)
        return 0

    from fulltd.data import verify_dataset, load_dataset
    from fulltd.sampling import training_universe
    from fd001_gamma.common import data_identity
    from fd001_gamma.protocol import LAGS
    fingerprint = verify_dataset("fd001", data)
    # Verifying the processed test file's SHA is an integrity check, not reading
    # test targets into the training or validation-selection procedure.
    train, _, _ = training_universe(load_dataset("fd001", data, "train"), LAGS)
    val = load_dataset("fd001", data, "val")
    validate_prepared_data(chosen, results, data_identity(train), data_identity(val))
    print(f"Verified FD001 processed-data fingerprint: {fingerprint}", flush=True)
    results.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    with file_lock(results / ".direct_campaign.lock"):
        # Recheck under the cross-process lock. No second dispatcher may race.
        pending = [c for c in pending if classify(c, results, code_hash)
                   not in ("complete", "numerical_failure")]
        if not pending:
            print("All selected runs became terminal before launch.")
            return 0
        logs = results / "direct_logs"
        launches = results / "direct_launches"
        logs.mkdir(parents=True, exist_ok=True)
        launches.mkdir(parents=True, exist_ok=True)
        receipt_path = launches / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_")
                                   + uuid.uuid4().hex[:12] + ".json")
        receipt = dict(state="running", hostname=platform.node(), launcher_pid=os.getpid(),
                       started_utc=datetime.now(timezone.utc).isoformat(), code_sha256=code_hash,
                       data_fingerprint=fingerprint, workers=max_workers, physical_cores=physical,
                       adaptive=bool(args.adaptive), check_seconds=args.check_seconds if args.adaptive else None,
                       admission_fraction=0.80 if args.adaptive else None,
                       idle_fraction_at_launch=idle, microbatch=args.microbatch,
                       selected_runs=len(chosen), queued_runs=[c["run_id"] for c in pending],
                       adaptive_samples=[], completed=[], failed=[], interrupted=[])
        atomic_json(receipt_path, receipt)
        active = {}
        submitted = iter(pending)
        exhausted = False
        stopping = False
        forwarded = False
        launch_limit = initial_target
        next_capacity_check = time.monotonic() + args.check_seconds

        def request_stop(signum, frame):
            nonlocal stopping
            stopping = True

        previous_handlers = {sig: signal.getsignal(sig) for sig in STOP_SIGNALS}
        for sig in STOP_SIGNALS:
            signal.signal(sig, request_stop)
        try:
            while active or (not stopping and not exhausted):
                if args.adaptive and not stopping and not exhausted and time.monotonic() >= next_capacity_check:
                    measured_idle = cpu_idle_fraction()
                    launch_limit = adaptive_worker_target(
                        max_workers, len(active), physical, measured_idle)
                    next_capacity_check = time.monotonic() + args.check_seconds
                    sample = dict(time_utc=datetime.now(timezone.utc).isoformat(),
                                  active=len(active), idle_fraction=measured_idle,
                                  estimated_idle_cores=physical * measured_idle,
                                  new_admission_budget=max(0, launch_limit - len(active)),
                                  allowed_concurrent=launch_limit)
                    receipt["adaptive_samples"].append(sample)
                    atomic_json(receipt_path, receipt)
                    print(f"Adaptive check: {len(active)} active; physical-core idle "
                          f"{measured_idle:.1%}; permitted concurrency {launch_limit}/{max_workers}",
                          flush=True)
                while not stopping and len(active) < launch_limit and not exhausted:
                    config = next(submitted, None)
                    if config is None:
                        exhausted = True
                        break
                    run_id = config["run_id"]
                    command = [sys.executable, "-m", "dscpuscripts.direct_runner",
                               "--campaign", getattr(args, "campaign", "original"),
                               "--worker", "--run-id", run_id,
                               "--data-dir", str(data), "--results-dir", str(results),
                               "--expected-code-hash", code_hash,
                               "--microbatch", str(args.microbatch)]
                    env = os.environ.copy()
                    for name in THREAD_VARIABLES:
                        env[name] = "1"
                    env["PYTHONUNBUFFERED"] = "1"
                    with (logs / (run_id + ".log")).open("a", encoding="utf-8") as log:
                        log.write(f"\n===== Launch {datetime.now(timezone.utc).isoformat()} =====\n")
                        log.flush()
                        proc = subprocess.Popen(command, env=env, cwd=root,
                                                stdout=log, stderr=subprocess.STDOUT)
                    active[run_id] = proc
                    print(f"Started {run_id} (PID {proc.pid})", flush=True)
                if stopping and not forwarded:
                    for proc in active.values():
                        if proc.poll() is None:
                            proc.send_signal(signal.SIGTERM)
                    forwarded = True
                if not active:
                    if stopping or exhausted:
                        break
                    # Adaptive mode can have zero free slots. Keep queued runs
                    # pending until CPU capacity becomes available.
                    time.sleep(1)
                    continue
                any_finished = False
                for run_id, proc in list(active.items()):
                    return_code = proc.poll()
                    if return_code is None:
                        continue
                    any_finished = True
                    del active[run_id]
                    if return_code == 0:
                        config = next(c for c in chosen if c["run_id"] == run_id)
                        terminal = classify(config, results, code_hash)
                        if terminal in ("complete", "numerical_failure"):
                            receipt["completed"].append(dict(run_id=run_id, state=terminal))
                            print(f"Finished {run_id}: {terminal}", flush=True)
                            atomic_json(receipt_path, receipt)
                            continue
                        return_code = 1
                    if return_code == 75 and stopping:
                        receipt["interrupted"].append(run_id)
                        print(f"Stopped at safe epoch boundary: {run_id}", flush=True)
                    else:
                        receipt["failed"].append(dict(run_id=run_id, exit_code=return_code))
                        print(f"ERROR: {run_id} exited {return_code}; inspect {logs / (run_id + '.log')}",
                              file=sys.stderr, flush=True)
                        stopping = True
                    atomic_json(receipt_path, receipt)
                if not any_finished:
                    time.sleep(1)
        except BaseException as error:
            receipt["failed"].append(dict(stage="dispatcher", error=f"{type(error).__name__}: {error}"))
            raise
        finally:
            # Do not leave orphan workers if dispatcher fails unexpectedly.
            for proc in active.values():
                if proc.poll() is None:
                    proc.send_signal(signal.SIGTERM)
            for proc in active.values():
                proc.wait()
            for sig, previous in previous_handlers.items():
                signal.signal(sig, previous)
            receipt["state"] = "failed" if receipt["failed"] else "interrupted" if stopping else "complete"
            receipt["ended_utc"] = datetime.now(timezone.utc).isoformat()
            atomic_json(receipt_path, receipt)
            print(f"Receipt: {receipt_path}", flush=True)
        return 1 if receipt["failed"] or stopping else 0


def main():
    from dscpuscripts.result_paths import configure_environment
    configure_environment()
    global make_manifest, scientific_hash, campaign_paths
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", choices=("original", "extension"), default="original",
                        help="Original 180 Gamma runs or isolated 140-run extension")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", dest="mode", action="store_const", const="dry-run")
    mode.add_argument("--run", dest="mode", action="store_const", const="run")
    mode.add_argument("--worker", dest="mode", action="store_const", const="worker")
    parser.add_argument("--workers", default="auto",
                        help="Number of single-thread processes, or auto (up to 128 based on idle physical cores)")
    parser.add_argument("--adaptive", action="store_true",
                        help="Treat --workers as a ceiling; start more jobs as CPU capacity frees up")
    parser.add_argument("--check-seconds", type=float, default=600.0,
                        help="Seconds between adaptive CPU checks (default: 600 = 10 minutes)")
    parser.add_argument("--microbatch", type=int, default=128)
    parser.add_argument("--max-runs", type=int, default=0,
                        help="Optional limited first batch; rerun without this to finish all.")
    parser.add_argument("--run-id")
    parser.add_argument("--data-dir")
    parser.add_argument("--results-dir")
    parser.add_argument("--expected-code-hash")
    args = parser.parse_args()
    if args.campaign == "extension":
        from fd001_gamma_ext import protocol as ext_protocol
        from fd001_gamma_ext import runtime as ext_runtime
        make_manifest = ext_protocol.make_manifest
        scientific_hash = ext_runtime.scientific_hash
        campaign_paths = ext_runtime.campaign_paths
    if args.mode == "worker":
        if not all((args.run_id, args.data_dir, args.results_dir, args.expected_code_hash)):
            parser.error("Internal --worker requires run-id, data-dir, results-dir, expected-code-hash")
        if args.campaign == "extension":
            from fd001_gamma_ext.runtime import worker as extension_worker
            return extension_worker(args)
        return worker(args)
    return run_campaign(args)


if __name__ == "__main__":
    raise SystemExit(main())
