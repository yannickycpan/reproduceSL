"""Independent resumable CPU dispatcher for representation-based Pairwise TD."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

from .pairwise_protocol import DATASETS, make_manifest
from .pairwise_train import code_hash


def paths(dataset):
    from .result_paths import base_root, paths as original_paths
    from .typed_ablation_runner import paths as ablation_paths
    from .lag579_runner import paths as lag_paths
    if dataset not in DATASETS:
        raise ValueError("Unsupported Pairwise campaign")
    data, original, _ = original_paths(dataset)
    _, ablation = ablation_paths(dataset)
    if dataset == "electricity":
        _, lag = lag_paths(dataset)
    else:
        lag = None
    from .metr_la_sampling import SOURCE_FRACTION, fraction_tag
    name = (f"pairwise_{dataset}_results" if dataset == "electricity"
            else f"pairwise_metr_la_{fraction_tag(SOURCE_FRACTION)}_results")
    key = "FULLTD_PAIRWISE_" + dataset.upper() + "_ROOT"
    root = Path(os.environ.get(key, base_root() / name)).expanduser().resolve()
    for protected in (original, ablation, lag):
        if protected is None:
            continue
        previous = Path(protected).resolve()
        if (root == previous or root.is_relative_to(previous)
                or previous.is_relative_to(root)):
            raise ValueError("Pairwise result root overlaps an earlier campaign")
    return data, root


def _scoped_codes(configs):
    codes = {}
    for c in configs:
        key = (c["dataset"], c["batch_size"], tuple(c["lags"]))
        if key not in codes:
            codes[key] = code_hash(c)
    return {c["run_id"]: codes[(c["dataset"], c["batch_size"], tuple(c["lags"]))]
            for c in configs}


def worker(args):
    import torch
    from fulltd.data import load_dataset
    from .common import file_lock, read_json
    from .standard_runner import classify, stop_flag
    from .pairwise_train import train_candidate

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    configs = {c["run_id"]: c for c in make_manifest(args.dataset)}
    if args.run_id not in configs:
        raise ValueError("Unknown Pairwise run identifier")
    c = configs[args.run_id]
    if args.expected_code != code_hash(c):
        raise ValueError("Scientific code changed since launch")
    data, root = paths(args.dataset)
    with stop_flag() as stopping, file_lock(root / c["run_id"] / ".worker.lock"):
        state = classify(c, root, args.expected_code)
        if state not in ("complete", "numerical_failure"):
            train_candidate(c, load_dataset(args.dataset, data, "train"),
                            load_dataset(args.dataset, data, "val"),
                            root, device="cpu", microbatch=args.microbatch,
                            stop_requested=stopping)
        state = read_json(root / c["run_id"] / "status.json")["state"]
        if state in ("complete", "numerical_failure"):
            return 0
        return 75 if stopping() else 1


def dispatch(args):
    from .common import atomic_json, file_lock
    from .standard_runner import choose_workers, classify, stop_flag
    from fulltd.data import verify_dataset

    configs = make_manifest(args.dataset)
    code_by_id = _scoped_codes(configs)
    data, root = paths(args.dataset)
    states = {c["run_id"]: classify(c, root, code_by_id[c["run_id"]])
              for c in configs}
    pending = [c for c in configs if states[c["run_id"]] in ("missing", "resume")]
    if args.max_runs:
        pending = pending[:args.max_runs]
    workers, _, _ = choose_workers(args.workers)
    print(f"{args.dataset}: representation Pairwise TD; "
          f"{len(configs)} new configurations; states={dict(Counter(states.values()))}; "
          f"pending={len(pending)}; workers={workers}", flush=True)
    print(f"Data: {data}\nNew output: {root}", flush=True)
    if args.mode in ("status", "dry-run") or not pending:
        return 0
    verify_dataset(args.dataset, data)
    root.mkdir(parents=True, exist_ok=True)
    with file_lock(root / ".pairwise_campaign.lock"), stop_flag() as stopping:
        logs = root / "direct_logs"
        logs.mkdir(exist_ok=True)
        launches = root / "direct_launches"
        launches.mkdir(exist_ok=True)
        receipt = launches / f"{uuid.uuid4().hex}.json"
        history = dict(dataset=args.dataset,
                       timestamp=datetime.now(timezone.utc).isoformat(),
                       workers=workers, completed=[], failures=[])
        atomic_json(receipt, history)
        active, index, stopping_all = {}, 0, False
        while active or (index < len(pending) and not stopping_all):
            stopping_all = stopping_all or stopping()
            for run_id, proc in list(active.items()):
                ret = proc.poll()
                if ret is None:
                    continue
                del active[run_id]
                config = next(c for c in configs if c["run_id"] == run_id)
                if ret == 0 and classify(config, root, code_by_id[run_id]) in (
                        "complete", "numerical_failure"):
                    history["completed"].append(run_id)
                    print(f"Finished {len(history['completed'])}/{len(pending)}: {run_id}", flush=True)
                elif not (ret == 75 and stopping_all):
                    history["failures"].append(dict(run_id=run_id, exit_code=ret))
                    stopping_all = True
                atomic_json(receipt, history)
            if stopping_all:
                for proc in active.values():
                    if proc.poll() is None:
                        proc.send_signal(signal.SIGTERM)
            else:
                while len(active) < workers and index < len(pending) and not stopping():
                    config = pending[index]
                    index += 1
                    cmd = [sys.executable, "-m", "dscpuscripts.pairwise_runner",
                           args.dataset, "--worker", "--run-id", config["run_id"],
                           "--expected-code", code_by_id[config["run_id"]],
                           "--microbatch", str(args.microbatch)]
                    with (logs / (config["run_id"] + ".log")).open("a", encoding="utf-8") as handle:
                        active[config["run_id"]] = subprocess.Popen(
                            cmd, cwd=Path(__file__).resolve().parents[1],
                            stdout=handle, stderr=subprocess.STDOUT,
                            start_new_session=True,
                            env=dict(os.environ, PYTHONUNBUFFERED="1"))
            if active or index < len(pending):
                time.sleep(1)
        for proc in active.values():
            if proc.poll() is None:
                proc.send_signal(signal.SIGTERM)
            proc.wait()
        history["state"] = ("failed" if history["failures"] else
                            "interrupted" if stopping_all or stopping()
                            else "complete")
        atomic_json(receipt, history)
    return 0 if history["state"] == "complete" else 1


def main():
    from .common import cpu_environment
    from .standard_runner import DEFAULTS
    cpu_environment()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=DATASETS)
    choice = parser.add_mutually_exclusive_group(required=True)
    for mode in ("dry-run", "status", "run", "worker"):
        choice.add_argument("--" + mode, dest="mode", action="store_const", const=mode)
    parser.add_argument("--workers", default="8")
    parser.add_argument("--microbatch", type=int)
    parser.add_argument("--max-runs", type=int, default=0)
    parser.add_argument("--run-id")
    parser.add_argument("--expected-code")
    args = parser.parse_args()
    if args.microbatch is None:
        args.microbatch = DEFAULTS[args.dataset][1]
    if args.microbatch < 1 or args.max_runs < 0:
        parser.error("Invalid microbatch/max-runs")
    if args.mode == "worker":
        if not args.run_id or not args.expected_code:
            parser.error("Worker requires --run-id and --expected-code")
        return worker(args)
    return dispatch(args)


if __name__ == "__main__":
    raise SystemExit(main())
