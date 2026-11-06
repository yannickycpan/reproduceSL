"""Separate CPU Slurm array for the 180-run FD001 Gamma sweep."""
from __future__ import annotations

import argparse
from collections import Counter
import math
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import uuid

from .common import (atomic_json, campaign_paths, cpu_environment, file_lock,
                     read_json, scientific_hash)
from .protocol import digest, load_manifest, make_manifest

JOB_NAME = "fulltd_fd001_gamma_v1"
ANALYSIS_JOB = "fulltd_fd001_gamma_analysis"


def queued_jobs():
    """Check only this campaign; unrelated Slurm jobs must not break submission.

    %i is the normal job/array-task identifier. --name filters server-side, so
    there is no need to parse every unrelated job's name, state, or array ID.
    Reject any unexpected output rather than risk a duplicate submission.
    """
    command = ["squeue", "--array", "--noheader", "--user", os.environ["USER"],
               "--name", JOB_NAME, "--format", "%i"]
    result = subprocess.run(command, check=True, text=True, capture_output=True)
    rows = []
    for line in result.stdout.splitlines():
        job_id = line.strip()
        if not job_id:
            continue
        if not re.fullmatch(r"\d+(?:_(?:\d+|\[[0-9,%\-]+\]))?", job_id):
            raise RuntimeError(f"Unrecognized squeue job ID {job_id!r}; submission refused. "
                               f"Raw response: {result.stdout!r}")
        rows.append((job_id, JOB_NAME, "queued"))
    return rows


def pending_configs(results, expected_hash):
    pending, counts = [], Counter()
    for config in make_manifest():
        directory = results / config["run_id"]
        path = directory / "status.json"
        status = read_json(path) if path.exists() else None
        if status:
            if (status.get("config") != config or status.get("identity", {}).get("code_sha256") != expected_hash
                    or status.get("identity", {}).get("config_sha256") != digest(config)):
                raise ValueError(f"Incompatible existing result preserved: {directory}")
            if status.get("state") == "numerical_failure":
                counts["numerical_failure"] += 1
                continue
            if (status.get("state") == "complete" and status.get("epochs_completed") == 100
                    and all((directory / f).is_file() for f in
                            ("final.pt", "validation.json", "rounds.json"))):
                counts["complete"] += 1
                continue
        counts["resume_checkpoint" if (directory / "recovery.pt").is_file()
               else "repair_final" if (directory / "final.pt").is_file() else "start_epoch_1"] += 1
        pending.append(config)
    return pending, counts


def wrapped(command, root):
    body = "set -euo pipefail; source " + shlex.quote(str(root / "scripts/fulltd_environment.sh"))
    body += "; exec " + shlex.join(command)
    return "exec bash -c " + shlex.quote(body)


def submit(args):
    cpu_environment()
    if min(args.workers, args.concurrency, args.microbatch, args.signal_seconds) < 1 or args.signal_seconds > 65535:
        raise ValueError("Invalid resource count or Slurm warning interval")
    data, results, _ = campaign_paths()
    root = Path(__file__).resolve().parents[1]
    jobs = queued_jobs()  # Fail closed on scheduler errors, not just empty results.
    existing = sorted({j for j, name, _ in jobs if name == JOB_NAME})
    if existing and args.mode == "--submit":
        raise RuntimeError(f"Gamma sweep already queued/running: {', '.join(existing)}; nothing submitted")
    code = scientific_hash()
    pending, counts = pending_configs(results, code)
    print("FD001 Gamma sweep: 3 variants x 12 alpha/beta pairs x 5 seeds = 180 runs")
    print("Counts:", dict(counts), "| pending:", len(pending))
    print(f"Resources: {args.workers} workers/node, {args.memory}/node, {args.time}/allocation, "
          f"concurrency {args.concurrency}, logical batch 128, physical microbatch {min(128, args.microbatch)}")
    print("Results:", results)
    if existing:
        print("Live Gamma sweep array(s):", ", ".join(existing))
    if args.mode == "--dry-run" or not pending:
        print("No files changed or jobs submitted.")
        return None
    # A lock closes the race between simultaneous invocations of this launcher.
    with file_lock(results / ".submission.lock"):
        if any(name == JOB_NAME for _, name, _ in queued_jobs()):
            raise RuntimeError("Gamma sweep appeared during submission check; nothing submitted")
        pending, counts = pending_configs(results, code)
        if not pending:
            return None
        from fulltd.data import load_dataset
        # Validate train/validation metadata without opening test values.
        load_dataset("fd001", data, "train")
        load_dataset("fd001", data, "val")
        key = digest(pending)
        manifest = results / "submissions" / f"{key}_manifest.json"
        if manifest.exists() and read_json(manifest) != pending:
            raise ValueError("Immutable submission manifest changed")
        if not manifest.exists():
            atomic_json(manifest, pending)
        logs = results / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        supervisor = [sys.executable, "-m", "fd001_gamma.hpc", "supervise",
                      "--manifest", str(manifest), "--digest", key, "--expected-code-hash", code,
                      "--results-dir", str(results), "--data-dir", str(data),
                      "--workers", str(args.workers), "--microbatch", str(args.microbatch)]
        count = math.ceil(len(pending) / args.workers)
        command = ["sbatch", "--parsable", "--partition", args.partition,
                   "--job-name", JOB_NAME, "--nodes=1", f"--ntasks={args.workers}",
                   f"--ntasks-per-node={args.workers}", "--cpus-per-task=1", "--ntasks-per-core=1",
                   "--mem", args.memory, "--time", args.time, "--array", f"0-{count - 1}%{args.concurrency}",
                   "--requeue", f"--signal=B:USR1@{args.signal_seconds}", "--export=ALL",
                   "--open-mode=append", "--chdir", str(root),
                   "--output", str(logs / "gamma_%A_%a.out"), "--error", str(logs / "gamma_%A_%a.err"),
                   "--wrap", wrapped(supervisor, root)]
        receipt_path = results / "submissions" / f"receipt_{uuid.uuid4().hex}.json"
        receipt = dict(state="submitting", command=command, scientific_code_sha256=code,
                       candidates=len(pending), counts=dict(counts))
        atomic_json(receipt_path, receipt)
        try:
            job = subprocess.check_output(command, text=True).strip().split(";")[0]
            if not job.isdigit():
                raise ValueError(f"Unexpected sbatch receipt: {job!r}")
        except Exception as error:
            receipt.update(state="submission_failed", error=str(error))
            atomic_json(receipt_path, receipt)
            raise
        receipt.update(state="submitted", job_id=job)
        atomic_json(receipt_path, receipt)
        print(f"Submitted Gamma sweep array {job}; {count} elements. Receipt: {receipt_path}")
        return receipt


def frozen_configs(args):
    configs = load_manifest(args.manifest)
    if digest(configs) != args.digest:
        raise ValueError("Submission manifest changed")
    if scientific_hash() != args.expected_code_hash:
        raise ValueError("Scientific code changed after submission; fitting refused")
    return configs


def with_warning(action):
    requested = False
    def warn(signum, frame):
        nonlocal requested
        requested = True
    old = signal.signal(signal.SIGUSR1, warn)
    try:
        return action(lambda: requested)
    finally:
        signal.signal(signal.SIGUSR1, old)


def worker(args, stop_requested):
    cpu_environment()
    rank = int(os.environ["SLURM_PROCID"])
    attempt = Path(args.attempt)
    outcome = dict(rank=rank, state="starting")
    try:
        configs = frozen_configs(args)
        index = args.offset + rank
        if not 0 <= rank < args.task_count or not 0 <= index < len(configs):
            raise ValueError("Worker rank/index is outside the immutable submission")
        if int(os.environ.get("SLURM_CPUS_PER_TASK", "0")) != 1:
            raise ValueError("Expected one CPU per worker")
        config = configs[index]
        prior_path = Path(args.results_dir) / config["run_id"] / "status.json"
        prior = read_json(prior_path) if prior_path.exists() else {}
        outcome.update(run_id=config["run_id"], before_epoch=prior.get("next_epoch", 1))
        # The signal handler is installed before readiness is published.
        atomic_json(attempt / f"ready_{rank}.json", outcome)
        import torch
        torch.set_num_threads(1)
        if torch.get_num_interop_threads() != 1:
            torch.set_num_interop_threads(1)
        from .train import run_config
        result = run_config(config, args.data_dir, args.results_dir, device="cpu",
                            microbatch=args.microbatch, stop_requested=stop_requested)
        outcome.update(state=result["state"], next_epoch=result.get("next_epoch", 1))
        if outcome["state"] not in ("complete", "numerical_failure", "checkpointed", "stopped_no_checkpoint"):
            raise RuntimeError(f"Unexpected worker outcome: {outcome['state']}")
        if outcome["state"] in ("checkpointed", "stopped_no_checkpoint") and not stop_requested():
            raise RuntimeError("Worker paused without a scheduler warning")
        atomic_json(attempt / f"rank_{rank}.json", outcome)
        return outcome
    except BaseException as error:
        outcome.update(state="operational_failure", error=f"{type(error).__name__}: {error}")
        atomic_json(attempt / f"rank_{rank}.json", outcome)
        raise


def completion_action(outcomes, returncode, warned):
    terminal = ("complete", "numerical_failure")
    paused = ("checkpointed", "stopped_no_checkpoint")
    if returncode or any(o["state"] not in terminal + paused for o in outcomes):
        raise RuntimeError("Operational worker failure; inspect rank logs before resubmission")
    unfinished = [o for o in outcomes if o["state"] in paused]
    if not unfinished:
        return "complete"
    if not warned:
        raise RuntimeError("Incomplete workers without a scheduler warning; no requeue")
    if any(o["state"] == "stopped_no_checkpoint" or o["next_epoch"] <= o["before_epoch"] for o in unfinished):
        raise RuntimeError("No durable epoch progress; increase allocation time before resubmission")
    return "requeue"


def supervise(args, stop_requested):
    configs = frozen_configs(args)
    array = os.environ["SLURM_ARRAY_JOB_ID"]
    task = int(os.environ["SLURM_ARRAY_TASK_ID"])
    offset = task * args.workers
    count = min(args.workers, len(configs) - offset)
    if (not array.isdigit() or task < 0 or count < 1
            or int(os.environ.get("SLURM_JOB_NUM_NODES", "0")) != 1
            or int(os.environ.get("SLURM_NTASKS", "0")) != args.workers
            or int(os.environ.get("SLURM_CPUS_PER_TASK", "0")) != 1):
        raise ValueError("Allocation layout differs from the submitted campaign")
    job = f"{array}_{task}"
    attempt = Path(args.results_dir) / "allocation_attempts" / f"{job}_{uuid.uuid4().hex}"
    attempt.mkdir(parents=True)
    logs = Path(args.results_dir) / "logs"
    command = ["srun", "--nodes=1", f"--ntasks={count}", f"--ntasks-per-node={count}",
               "--cpus-per-task=1", "--ntasks-per-core=1", "--cpu-bind=cores", "--kill-on-bad-exit=0",
               "--wait=0", "--mpi=none", "--open-mode=append",
               "--output", str(logs / f"{job}_rank_%t.out"), "--error", str(logs / f"{job}_rank_%t.err"),
               sys.executable, "-m", "fd001_gamma.hpc", "worker", "--manifest", args.manifest,
               "--digest", args.digest, "--expected-code-hash", args.expected_code_hash,
               "--results-dir", args.results_dir, "--data-dir", args.data_dir,
               "--attempt", str(attempt), "--offset", str(offset), "--task-count", str(count),
               "--microbatch", str(args.microbatch)]
    receipt = dict(job_id=job, restart_count=int(os.environ.get("SLURM_RESTART_COUNT", "0")),
                   state="running", warning_forwarded=False, command=command)
    atomic_json(attempt / "coordinator.json", receipt)
    process = subprocess.Popen(command)
    try:
        while True:
            if stop_requested() and not receipt["warning_forwarded"]:
                if all((attempt / f"ready_{r}.json").exists() or (attempt / f"rank_{r}.json").exists() for r in range(count)):
                    process.send_signal(signal.SIGUSR1)
                    receipt["warning_forwarded"] = True
                    atomic_json(attempt / "coordinator.json", receipt)
            try:
                code = process.wait(timeout=1)
                break
            except subprocess.TimeoutExpired:
                pass
        outcomes = [read_json(attempt / f"rank_{r}.json") if (attempt / f"rank_{r}.json").exists()
                    else dict(rank=r, state="missing") for r in range(count)]
        receipt.update(returncode=code, outcomes=outcomes)
        action = completion_action(outcomes, code, stop_requested() and receipt["warning_forwarded"])
        if action == "requeue":
            receipt["state"] = "requeue_requested"
            atomic_json(attempt / "coordinator.json", receipt)
            subprocess.run(["scontrol", "requeue", job], check=True, timeout=30)
            receipt["state"] = "requeue_accepted"
        else:
            receipt["state"] = "complete"
        atomic_json(attempt / "coordinator.json", receipt)
        return receipt
    except BaseException as error:
        receipt.update(state="operational_failure", error=f"{type(error).__name__}: {error}")
        atomic_json(attempt / "coordinator.json", receipt)
        if process.poll() is None:
            process.terminate()
        raise


def main():
    cpu_environment()
    parser = argparse.ArgumentParser(description=__doc__)
    if len(sys.argv) > 1 and sys.argv[1] in ("worker", "supervise"):
        parser.add_argument("action", choices=("worker", "supervise"))
        for name in ("manifest", "digest", "expected-code-hash", "results-dir", "data-dir"):
            parser.add_argument("--" + name, required=True)
        parser.add_argument("--workers", type=int, default=90)
        parser.add_argument("--microbatch", type=int, default=128)
        parser.add_argument("--attempt")
        parser.add_argument("--offset", type=int, default=0)
        parser.add_argument("--task-count", type=int)
        args = parser.parse_args()
        return with_warning(lambda requested: worker(args, requested) if args.action == "worker"
                            else supervise(args, requested))
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--submit", dest="mode", action="store_const", const="--submit")
    mode.add_argument("--dry-run", dest="mode", action="store_const", const="--dry-run")
    parser.add_argument("--workers", type=int, default=90)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--microbatch", type=int, default=128)
    parser.add_argument("--memory", default="400G")
    parser.add_argument("--time", default="02:00:00")
    parser.add_argument("--partition", default="batch")
    parser.add_argument("--signal-seconds", type=int, default=180)
    return submit(parser.parse_args())


if __name__ == "__main__":
    main()
