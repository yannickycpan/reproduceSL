"""Slurm submission for CityU Burgundy; queue/resource names are supplied by you."""
import argparse
import json
import os
import signal
from pathlib import Path
import shlex
import subprocess
import sys
import uuid

from .protocol import load_manifest, digest


def submit(args):
    root = Path(__file__).resolve().parents[1]
    manifest_path = Path(args.manifest).resolve()
    manifest = load_manifest(manifest_path)
    indices = [i for i, r in enumerate(manifest) if args.dataset == "all" or r["dataset"] == args.dataset]
    if not indices:
        raise ValueError("No candidates selected")
    from .train import code_hash
    scientific_hash = code_hash()
    data, results, logs = (Path(p).resolve() for p in (args.data_dir, args.results_dir, args.log_dir))
    logs.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    pending = []
    for index in indices:
        config = manifest[index]
        status_path = results / config["run_id"] / "status.json"
        if status_path.exists():
            status = json.loads(status_path.read_text())
            if status.get("config") != config or status.get("identity", {}).get("code_sha256") != scientific_hash:
                raise ValueError(f"Incompatible existing run preserved: {config['run_id']}; use a fresh results directory")
            if status.get("state") == "numerical_failure":
                continue
            if status.get("state") == "complete" and (status_path.parent / "final.pt").is_file():
                continue
        pending.append(index)
    if not pending:
        print("All requested candidates already have terminal records. Run the selection/analysis audit next.")
        return dict(candidate_count=0, candidate_array_jobs=[], state="nothing_to_submit")
    indices = pending
    # Queued jobs must never follow a later edit to the caller's manifest/map.
    submission_key = digest(dict(manifest=manifest, indices=indices))[:24]
    manifest_path = results / f"submission_{submission_key}_manifest.json"
    mapping = results / f"submission_{submission_key}_indices.json"
    for destination, payload in ((manifest_path, manifest), (mapping, indices)):
        serialized = json.dumps(payload, indent=2) + "\n"
        if destination.exists() and json.loads(destination.read_text()) != payload:
            raise ValueError("Immutable submission record changed")
        if not destination.exists():
            destination.write_text(serialized)
    common = ["sbatch", "--parsable", "--partition", args.partition, "--gres", args.gres,
              "--nodes=1", "--ntasks=1", "--open-mode=append",
              "--cpus-per-task", str(args.cpus), "--mem", args.memory, "--time", args.time,
              "--chdir", str(root), "--output", str(logs / "%x_%A_%a.out"),
              "--error", str(logs / "%x_%A_%a.err"), "--export=ALL"]
    for key in ("account", "qos", "constraint"):
        if getattr(args, key):
            common.extend(["--" + key, getattr(args, key)])

    commands = []
    receipt = dict(preflight_job=None, candidate_array_jobs=[], candidate_count=len(indices),
                   dataset=args.dataset, dry_run=args.dry_run, commands=commands,
                   scientific_code_sha256=scientific_hash, state="submitting",
                   note="Chunks run sequentially; concurrency caps this submission. Account limits remain site-specific.")
    receipt_path = results / f"submission_{args.dataset}_{uuid.uuid4().hex[:12]}.json"

    def record():
        from .train import atomic_json
        atomic_json(receipt_path, receipt)

    def send(command):
        commands.append(command)
        record()
        print(shlex.join(command))
        if args.dry_run:
            return "PREFLIGHT_JOB_ID" if len(commands) == 1 else "ARRAY_JOB_ID"
        try:
            job_id = subprocess.check_output(command, text=True).strip().split(";")[0]
        except (OSError, subprocess.CalledProcessError) as error:
            receipt.update(state="submission_failed", error=str(error))
            record()
            raise RuntimeError(f"Submission stopped; already accepted job IDs are retained in {receipt_path}") from error
        if not job_id.isdigit():
            receipt.update(state="submission_failed", error=f"Unexpected sbatch receipt: {job_id!r}")
            record()
            raise RuntimeError(f"Unexpected sbatch job receipt {job_id!r}")
        return job_id

    preflight_cmd = [sys.executable, "-m", "fulltd.preflight", "--dataset", args.dataset,
                     "--data-dir", str(data), "--output", str(results / f"preflight_{args.dataset}.json"),
                     "--device", "cuda", "--threads", str(args.cpus)]
    preflight = send(common + ["--job-name=fulltd_preflight", "--wrap", shlex.join(preflight_cmd)])
    receipt["preflight_job"] = preflight
    record()
    # The explicit map permits chunking under MaxArraySize without changing run IDs.
    job_ids = []
    for offset in range(0, len(indices), args.array_chunk):
        count = min(args.array_chunk, len(indices)-offset)
        command = [sys.executable, "-m", "fulltd.hpc", "worker", "--manifest", str(manifest_path),
                   "--mapping", str(mapping), "--offset", str(offset), "--data-dir", str(data),
                   "--results-dir", str(results), "--threads", str(args.cpus),
                   "--microbatch", str(args.microbatch), "--expected-code-hash", scientific_hash]
        if args.no_auto_requeue:
            command.append("--no-auto-requeue")
        dependency = "afterok:" + preflight
        if job_ids:
            # Chain chunks to keep the requested concurrency a campaign-wide
            # cap. afterany allows independent later candidates after a failed job.
            dependency += ",afterany:" + job_ids[-1]
        job_ids.append(send(common + ["--job-name=fulltd_" + args.dataset,
                        "--requeue", f"--signal=B:USR1@{args.signal_seconds}",
                        "--dependency=" + dependency,
                        "--array", f"0-{count-1}%{args.concurrency}", "--wrap", "exec " + shlex.join(command)]))
        receipt["candidate_array_jobs"] = list(job_ids)
        record()
    receipt["state"] = "submitted" if not args.dry_run else "dry_run"
    record()
    print(f"Submission receipt: {receipt_path}")
    return receipt


def worker(args):
    # exec in the batch wrapper makes this process the recipient of Slurm's B:
    # warning. Run training in this process; there is no child signal race.
    requested = False
    def request_checkpoint(signum, frame):
        nonlocal requested
        requested = True
    warning_signal = getattr(signal, "SIGUSR1", None)
    previous_handler = None
    if warning_signal is not None:
        previous_handler = signal.signal(warning_signal, request_checkpoint)
    try:
        return _run_worker(args, lambda: requested)
    finally:
        if warning_signal is not None:
            signal.signal(warning_signal, previous_handler)


def _run_worker(args, stop_requested):
    mapping = json.loads(Path(args.mapping).read_text())
    task = int(os.environ["SLURM_ARRAY_TASK_ID"])
    if args.offset < 0 or task < 0 or args.offset + task >= len(mapping):
        raise ValueError("Array offset/task ID is outside the submitted mapping")
    index = mapping[args.offset + task]
    from .train import run_index, code_hash, atomic_json
    manifest = load_manifest(args.manifest)
    if type(index) is not int or not 0 <= index < len(manifest):
        raise ValueError("Mapped manifest index is out of range")
    if args.expected_code_hash and args.expected_code_hash != code_hash():
        raise ValueError("Scientific code changed after submission; queued task stopped before fitting")
    run_dir = Path(args.results_dir) / manifest[index]["run_id"]
    def progress(record):
        return (record.get("next_epoch", 1), record.get("cursor", 0), record.get("validation_cursor", 0))
    status_path = run_dir / "status.json"
    before = progress(json.loads(status_path.read_text())) if status_path.exists() else (1, 0, 0)
    try:
        result = run_index(args.manifest, index, args.data_dir, args.results_dir, device="cuda",
                           microbatch=args.microbatch, threads=args.threads, stop_requested=stop_requested)
        if result.get("state") != "checkpointed":
            return result
        status = json.loads(status_path.read_text())
        if args.no_auto_requeue:
            raise RuntimeError("Time-limit checkpoint saved; automatic requeue disabled. Resubmit the same campaign command.")
        if progress(status) <= before:
            raise RuntimeError("No progress within this allocation; increase its time limit before resubmitting.")
        array_job = os.environ.get("SLURM_ARRAY_JOB_ID", "")
        if not array_job.isdigit():
            raise RuntimeError("Automatic continuation requires a valid Slurm array job ID")
        element = f"{array_job}_{task}"
        requeue_record = dict(array_element=element, restart_count=int(os.environ.get("SLURM_RESTART_COUNT", 0)),
                              progress=progress(status), state="requested")
        atomic_json(run_dir / "requeue.json", requeue_record)
        print(f"Checkpoint saved; requesting continuation of array task {element}", flush=True)
        try:
            subprocess.run(["scontrol", "requeue", element], check=True, timeout=30)
        except (OSError, subprocess.SubprocessError) as error:
            requeue_record.update(state="requeue_failed", error=str(error))
            atomic_json(run_dir / "requeue.json", requeue_record)
            raise RuntimeError("Slurm did not accept requeue. Checkpoint preserved; resubmit the same campaign command.") from error
        # Slurm may terminate this process before returning from scontrol.
        requeue_record["state"] = "accepted"
        atomic_json(run_dir / "requeue.json", requeue_record)
        return result
    except BaseException:
        # Do not translate cancellation, numerical failure, or arbitrary errors
        # into a requeue. Only the explicit checkpointed result above qualifies.
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    # Hidden internal worker keeps the public shell entry point small.
    if len(sys.argv) > 1 and sys.argv[1] == "worker":
        parser.add_argument("worker")
        for key in ("manifest", "mapping", "data-dir", "results-dir"):
            parser.add_argument("--" + key, required=True)
        parser.add_argument("--offset", type=int, required=True)
        parser.add_argument("--threads", type=int, default=4)
        parser.add_argument("--microbatch", type=int, default=256)
        parser.add_argument("--expected-code-hash")
        parser.add_argument("--no-auto-requeue", action="store_true")
        return worker(parser.parse_args())
    parser.add_argument("--manifest", default="configs/fulltd/manifest.json")
    parser.add_argument("--dataset", choices=("all", "electricity", "metr_la", "fd001"), default="all")
    parser.add_argument("--partition", required=True, help="Your approved CityU GPU partition (verify with sinfo)")
    parser.add_argument("--gres", default="gpu:1", help="Set the GPU resource string supported by your partition")
    for key in ("account", "qos", "constraint"):
        parser.add_argument("--" + key)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--log-dir", default="logs/fulltd")
    parser.add_argument("--memory", default="32G")
    parser.add_argument("--time", default="24:00:00")
    parser.add_argument("--cpus", type=int, default=4)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--array-chunk", type=int, default=100, help="Must fit site MaxArraySize and submission limits")
    parser.add_argument("--microbatch", type=int, default=256)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--signal-seconds", type=int, default=180,
                        help="Slurm warning before wall time; allow enough time to save a checkpoint")
    parser.add_argument("--no-auto-requeue", action="store_true",
                        help="Checkpoint on warning but require manual resubmission")
    args = parser.parse_args()
    if min(args.cpus, args.concurrency, args.array_chunk, args.microbatch, args.signal_seconds) < 1:
        parser.error("Resource counts must be positive")
    submit(args)


if __name__ == "__main__":
    main()
