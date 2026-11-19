"""Direct CPU runner for the 7- and 9-lag additions (the 5-lag results are reused)."""
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

from .lag579_protocol import DATASETS, METHODS, make_manifest
from .lag579_train import code_hash


def paths(dataset):
    from .result_paths import base_root, paths as original_paths
    from .typed_ablation_runner import paths as previous_paths
    data, original_root, _ = original_paths(dataset)
    _, previous_root = previous_paths(dataset)
    key = "FULLTD_LAG579_" + dataset.upper() + "_ROOT"
    root = Path(os.environ.get(key, base_root() / f"lag579_{dataset}_results")).expanduser().resolve()
    for protected in (original_root, previous_root):
        protected = Path(protected).resolve()
        if (root == protected or root.is_relative_to(protected)
                or protected.is_relative_to(root)):
            raise ValueError("New run directory overlaps an existing campaign")
    return data, root


def worker(args):
    import torch
    from fulltd.data import load_dataset
    from .common import file_lock, read_json
    from .standard_runner import classify, stop_flag
    from .lag579_train import train_candidate
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    by_id = {c["run_id"]: c for c in make_manifest(args.dataset)}
    if args.run_id not in by_id:
        raise ValueError("Unknown new-run identifier")
    config = by_id[args.run_id]
    if args.expected_code != code_hash(config):
        raise ValueError("Scientific code changed since dispatch")
    data, root = paths(args.dataset)
    with stop_flag() as stopping, file_lock(root / config["run_id"] / ".worker.lock"):
        if classify(config, root, args.expected_code) not in ("complete", "numerical_failure"):
            train_candidate(config, load_dataset(args.dataset, data, "train"),
                            load_dataset(args.dataset, data, "val"), root,
                            device="cpu", microbatch=args.microbatch,
                            stop_requested=stopping)
        current = read_json(root / config["run_id"] / "status.json")["state"]
        if current in ("complete", "numerical_failure"):
            return 0
        return 75 if stopping() else 1


def dispatch(args):
    from .common import atomic_json, file_lock
    from .standard_runner import classify, choose_workers, stop_flag
    from fulltd.data import verify_dataset

    configs = make_manifest(args.dataset)
    data, root = paths(args.dataset)
    code_by_method = {
        method: code_hash(next(c for c in configs if c["method"] == method))
        for method in METHODS
    }
    states = {c["run_id"]: classify(c, root, code_by_method[c["method"]])
              for c in configs}
    pending = [c for c in configs if states[c["run_id"]] in ("missing", "resume")]
    if args.max_runs:
        pending = pending[:args.max_runs]
    scan = "7/9-lag additions (5-lag reused)" if args.dataset == "fd001" else "5-lag only"
    print(f"{args.dataset}: {scan}; "
          f"{len(configs)} configurations; states={dict(Counter(states.values()))}; "
          f"pending={len(pending)}", flush=True)
    print(f"Data: {data}\nNEW output: {root}", flush=True)
    if args.mode in ("dry-run", "status") or not pending:
        return 0

    verify_dataset(args.dataset, data)
    workers, _, _ = choose_workers(args.workers)
    root.mkdir(parents=True, exist_ok=True)
    with file_lock(root / ".lag579_campaign.lock"), stop_flag() as stopping:
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
                code = proc.poll()
                if code is None:
                    continue
                del active[run_id]
                config = next(c for c in configs if c["run_id"] == run_id)
                if code == 0 and classify(config, root, code_by_method[config["method"]]) in (
                        "complete", "numerical_failure"):
                    history["completed"].append(run_id)
                elif not (code == 75 and stopping_all):
                    history["failures"].append(dict(run_id=run_id, exit_code=code))
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
                    command = [sys.executable, "-m", "dscpuscripts.lag579_runner",
                               args.dataset, "--worker", "--run-id", config["run_id"],
                               "--expected-code", code_by_method[config["method"]],
                               "--microbatch", str(args.microbatch)]
                    with (logs / (config["run_id"] + ".log")).open("a", encoding="utf-8") as handle:
                        active[config["run_id"]] = subprocess.Popen(
                            command, cwd=Path(__file__).resolve().parents[1],
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
                            "interrupted" if stopping_all or stopping() else "complete")
        atomic_json(receipt, history)
    return 0 if history["state"] == "complete" else 1


def main():
    from .common import cpu_environment
    from .standard_runner import DEFAULTS
    cpu_environment()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=DATASETS)
    group = parser.add_mutually_exclusive_group(required=True)
    for mode in ("dry-run", "status", "run", "worker"):
        group.add_argument("--" + mode, action="store_const", dest="mode", const=mode)
    parser.add_argument("--workers", default="8")
    parser.add_argument("--microbatch", type=int)
    parser.add_argument("--max-runs", type=int, default=0)
    parser.add_argument("--run-id")
    parser.add_argument("--expected-code")
    args = parser.parse_args()
    if args.microbatch is None:
        args.microbatch = DEFAULTS[args.dataset][1]
    if args.microbatch < 1 or args.max_runs < 0:
        parser.error("Invalid microbatch or max-runs")
    if args.mode == "worker":
        if not args.run_id or not args.expected_code:
            parser.error("Worker requires --run-id and --expected-code")
        return worker(args)
    return dispatch(args)


if __name__ == "__main__":
    raise SystemExit(main())
