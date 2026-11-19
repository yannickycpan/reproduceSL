"""Run ONLY two new typed-TD mechanism ablations without touching old runs.

Usage:
    python -m dscpuscripts.typed_ablation_runner fd001 --dry-run
    python -m dscpuscripts.typed_ablation_runner electricity --run --workers 8
    python -m dscpuscripts.typed_ablation_runner metr_la --status

Each worker delegates to the original dataset's unchanged training function.
Old MSE and full typed TD states remain in their original results folders.
"""
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

from .typed_ablation_protocol import DATASETS, make_manifest
from .typed_ablation_train import code_hash


def paths(dataset):
    from .result_paths import base_root
    from .result_paths import paths as original_paths
    data, old_root, _ = original_paths(dataset)
    key = "FULLTD_TYPED_ABLATION_" + dataset.upper() + "_ROOT"
    root = Path(os.environ.get(key, base_root() / f"typed_ablation_{dataset}_results")).expanduser().resolve()
    if root == old_root or root.is_relative_to(old_root) or old_root.is_relative_to(root):
        raise ValueError("Ablation results root may not overlap the original campaign")
    return data, root


def worker(args):
    from .standard_runner import stop_flag, classify
    from .common import file_lock, read_json
    from fulltd.data import load_dataset
    from .typed_ablation_train import train_candidate
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    configs = {c["run_id"]: c for c in make_manifest(args.dataset)}
    if args.run_id not in configs:
        raise ValueError("Not an ablation run ID")
    c = configs[args.run_id]
    if args.expected_code != code_hash(c):
        raise ValueError("Code changed after launch; refusing an incompatible worker")
    data, root = paths(args.dataset)
    with stop_flag() as stopped, file_lock(root / c["run_id"] / ".worker.lock"):
        if classify(c, root, args.expected_code) not in ("complete", "numerical_failure"):
            train_candidate(c, load_dataset(args.dataset, data, "train"),
                            load_dataset(args.dataset, data, "val"), root,
                            device="cpu", microbatch=args.microbatch,
                            stop_requested=stopped)
        state = read_json(root / c["run_id"] / "status.json")["state"]
        if state in ("complete", "numerical_failure"):
            return 0
        return 75 if stopped() else 1


def dispatch(args):
    from .common import file_lock, atomic_json
    from .standard_runner import classify, choose_workers, stop_flag
    from fulltd.data import verify_dataset
    configs = make_manifest(args.dataset)
    data, root = paths(args.dataset)
    scoped = {m: code_hash(next(c for c in configs if c["method"] == m))
              for m in ("td_semi_typed", "td_nonnegative_typed")}
    states = {c["run_id"]: classify(c, root, scoped[c["method"]]) for c in configs}
    pending = [c for c in configs if states[c["run_id"]] in ("missing", "resume")]
    if args.max_runs:
        pending = pending[:args.max_runs]
    print(f"{args.dataset}: {len(configs)} new typed-ablation configurations; "
          f"states={dict(Counter(states.values()))}; pending={len(pending)}", flush=True)
    print(f"Data: {data}\nNEW output: {root}", flush=True)
    if args.mode in ("dry-run", "status") or not pending:
        return 0
    verify_dataset(args.dataset, data)
    workers, _, _ = choose_workers(args.workers)
    root.mkdir(parents=True, exist_ok=True)
    with file_lock(root / ".typed_ablation_campaign.lock"), stop_flag() as stopped:
        logs = root / "direct_logs"
        logs.mkdir(exist_ok=True)
        launches = root / "direct_launches"
        launches.mkdir(exist_ok=True)
        receipt = launches / f"{uuid.uuid4().hex}.json"
        history = dict(dataset=args.dataset, timestamp=datetime.now(timezone.utc).isoformat(),
                       workers=workers, completed=[], failures=[])
        atomic_json(receipt, history)
        active, next_index, stopping = {}, 0, False
        while active or (next_index < len(pending) and not stopping):
            stopping = stopping or stopped()
            for run_id, proc in list(active.items()):
                exitcode = proc.poll()
                if exitcode is None:
                    continue
                del active[run_id]
                c = next(c for c in configs if c["run_id"] == run_id)
                if exitcode == 0 and classify(c, root, scoped[c["method"]]) in ("complete", "numerical_failure"):
                    history["completed"].append(run_id)
                elif not (exitcode == 75 and stopping):
                    history["failures"].append(dict(run_id=run_id, exit_code=exitcode))
                    stopping = True
                atomic_json(receipt, history)
            if stopping:
                for proc in active.values():
                    if proc.poll() is None:
                        proc.send_signal(signal.SIGTERM)
            else:
                while len(active) < workers and next_index < len(pending) and not stopped():
                    c = pending[next_index]
                    next_index += 1
                    command = [sys.executable, "-m", "dscpuscripts.typed_ablation_runner",
                               args.dataset, "--worker", "--run-id", c["run_id"],
                               "--expected-code", scoped[c["method"]],
                               "--microbatch", str(args.microbatch)]
                    with (logs / (c["run_id"] + ".log")).open("a", encoding="utf8") as log:
                        active[c["run_id"]] = subprocess.Popen(
                            command, cwd=Path(__file__).resolve().parents[1],
                            stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True, env=dict(os.environ, PYTHONUNBUFFERED="1"))
            if active or next_index < len(pending):
                time.sleep(1)
        for proc in active.values():
            if proc.poll() is None:
                proc.send_signal(signal.SIGTERM)
            proc.wait()
        history["state"] = ("failed" if history["failures"] else
                            "interrupted" if stopping or stopped() else "complete")
        atomic_json(receipt, history)
    return 0 if history["state"] == "complete" else 1


def main():
    from .common import cpu_environment
    from .standard_runner import DEFAULTS
    cpu_environment()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=DATASETS)
    action = parser.add_mutually_exclusive_group(required=True)
    for mode in ("dry-run", "status", "run", "worker"):
        action.add_argument("--" + mode, action="store_const", dest="mode", const=mode)
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
