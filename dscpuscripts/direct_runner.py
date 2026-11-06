"""Run the unchanged 180-config FD001 Gamma campaign on a direct-access Linux server.

No Slurm, job array, allocation time, or HPC module system is involved. Each
candidate runs in its own one-thread Python subprocess and writes its ordinary
fd001_gamma checkpoint files. A rerun skips terminal candidates and resumes
compatible epoch checkpoints. Use disjoint deterministic shards on two servers.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import os
from pathlib import Path
import platform
import re
import signal
import subprocess
import sys
import time
import uuid

from fd001_gamma.common import (THREAD_VARIABLES, atomic_json, campaign_paths,
                                cpu_environment, file_lock, read_json, scientific_hash)
from fd001_gamma.protocol import digest, make_manifest

STOP_SIGNALS = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)


def parse_shards(spec: str, count: int) -> tuple[int, ...]:
    if count < 1:
        raise ValueError("--shards must be positive")
    found = set()
    for part in spec.split(","):
        value = part.strip()
        match = re.fullmatch(r"(\d+)(?:-(\d+))?", value)
        if match is None:
            raise ValueError(f"Invalid shard range {value!r}; use e.g. 0-3 or 4")
        start = int(match.group(1))
        end = int(match.group(2)) if match.group(2) is not None else start
        if start > end or end >= count:
            raise ValueError(f"Shard {value!r} is outside 0..{count - 1}")
        found.update(range(start, end + 1))
    if not found:
        raise ValueError("Select at least one shard")
    return tuple(sorted(found))


def selected_configs(shards: int, indices: tuple[int, ...]) -> list[dict]:
    return [config for i, config in enumerate(make_manifest()) if i % shards in indices]


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
    if args.workers < 1 or not 1 <= args.microbatch <= 128 or args.max_runs < 0:
        raise ValueError("Use positive --workers, --microbatch in 1..128, and --max-runs >= 0")
    if args.mode == "run":
        available = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
        if args.workers > available:
            raise ValueError(f"Requested {args.workers} workers but only {available} logical CPUs are available")
    shard_ids = parse_shards(args.shard_indices, args.shards)
    data, results, _ = campaign_paths()
    code_hash = scientific_hash()
    chosen = selected_configs(args.shards, shard_ids)
    counts = Counter(classify(c, results, code_hash) for c in chosen)
    pending = [c for c in chosen if classify(c, results, code_hash)
               not in ("complete", "numerical_failure")]
    if args.max_runs:
        pending = pending[:args.max_runs]

    print(f"FD001 Gamma: {len(chosen)} assigned runs; shards {shard_ids} of {args.shards}; "
          f"{len(pending)} to process now", flush=True)
    print(f"States: {dict(counts)}", flush=True)
    print(f"Data: {data}; results: {results}", flush=True)
    print(f"Workers: {args.workers} CPU processes; physical microbatch: {args.microbatch}", flush=True)
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
                       data_fingerprint=fingerprint, shards=args.shards, shard_indices=list(shard_ids),
                       workers=args.workers, microbatch=args.microbatch,
                       selected_runs=len(chosen), queued_runs=[c["run_id"] for c in pending],
                       completed=[], failed=[], interrupted=[])
        atomic_json(receipt_path, receipt)
        active = {}
        submitted = iter(pending)
        stopping = False
        forwarded = False

        def request_stop(signum, frame):
            nonlocal stopping
            stopping = True

        previous_handlers = {sig: signal.getsignal(sig) for sig in STOP_SIGNALS}
        for sig in STOP_SIGNALS:
            signal.signal(sig, request_stop)
        try:
            while active or not stopping:
                while not stopping and len(active) < args.workers:
                    config = next(submitted, None)
                    if config is None:
                        break
                    run_id = config["run_id"]
                    command = [sys.executable, "-m", "dscpuscripts.direct_runner",
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
                    break
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
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", dest="mode", action="store_const", const="dry-run")
    mode.add_argument("--run", dest="mode", action="store_const", const="run")
    mode.add_argument("--worker", dest="mode", action="store_const", const="worker")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--shard-indices", default="0")
    parser.add_argument("--microbatch", type=int, default=128)
    parser.add_argument("--max-runs", type=int, default=0,
                        help="Optional limited first batch; rerun without this to finish all.")
    parser.add_argument("--run-id")
    parser.add_argument("--data-dir")
    parser.add_argument("--results-dir")
    parser.add_argument("--expected-code-hash")
    args = parser.parse_args()
    if args.mode == "worker":
        if not all((args.run_id, args.data_dir, args.results_dir, args.expected_code_hash)):
            parser.error("Internal --worker requires run-id, data-dir, results-dir, expected-code-hash")
        return worker(args)
    return run_campaign(args)


if __name__ == "__main__":
    raise SystemExit(main())
