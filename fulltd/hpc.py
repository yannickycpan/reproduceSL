"""CPU Slurm campaign: one candidate per rank; one coordinator per allocation."""
import argparse
import json
import math
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import uuid

from .protocol import DATASET_SCOPES, datasets_for_scope, digest, load_manifest

THREAD_VARIABLES = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS")


def cpu_environment():
    # Set before importing NumPy/PyTorch, including in each fresh srun process.
    for name in THREAD_VARIABLES:
        os.environ[name] = "1"


def allocation_command(command, root):
    # Slurm --wrap uses /bin/sh; explicitly enter Bash for the module setup.
    # Both execs preserve the batch PID for --signal=B:USR1 on the coordinator.
    environment = Path(root) / "scripts" / "fulltd_environment.sh"
    body = "set -euo pipefail; source " + shlex.quote(str(environment)) + "; exec " + shlex.join(command)
    return "exec bash -c " + shlex.quote(body)


def sbatch_base(args, root, logs):
    command = ["sbatch", "--parsable", "--cpus-per-task=1", "--ntasks-per-core=1",
               "--open-mode=append", "--chdir", str(root),
               "--output", str(logs / "%x_%j.out"), "--error", str(logs / "%x_%j.err"), "--export=ALL"]
    for name in ("account", "qos", "constraint"):
        if getattr(args, name):
            command += ["--" + name, getattr(args, name)]
    return command


def submit_preflight(args):
    """Submit only the explicitly requested, independent preparation/check job."""
    cpu_environment()
    from .train import atomic_json
    root = Path(__file__).resolve().parents[1]
    data, results, logs = (Path(p).resolve() for p in (args.data_dir, args.results_dir, args.log_dir))
    logs.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    preflight = [sys.executable, "-m", "fulltd.preflight", "--dataset", args.dataset,
                 "--data-dir", str(data), "--output", str(results / f"preflight_{args.dataset}.json"),
                 "--device", "cpu", "--threads", "1"]
    command = sbatch_base(args, root, logs) + ["--partition", args.partition, "--time", args.time,
        "--nodes=1", "--ntasks=1", "--ntasks-per-node=1", "--mem", args.memory,
        "--job-name=fulltd_preflight_" + args.dataset, "--wrap", allocation_command(preflight, root)]
    receipt = dict(preflight_job=None, dataset=args.dataset, dry_run=args.dry_run,
                   commands=[command], state="submitting")
    path = results / f"preflight_submission_{args.dataset}_{uuid.uuid4().hex[:12]}.json"
    atomic_json(path, receipt)
    print(shlex.join(command))
    try:
        job_id = "PREFLIGHT_JOB_ID" if args.dry_run else subprocess.check_output(command, text=True).strip().split(";")[0]
        if not args.dry_run and not job_id.isdigit():
            raise ValueError(f"Unexpected sbatch receipt: {job_id!r}")
    except (OSError, subprocess.CalledProcessError, ValueError) as error:
        receipt.update(state="submission_failed", error=str(error))
        atomic_json(path, receipt)
        raise RuntimeError(f"Preflight submission stopped; receipt retained in {path}") from error
    receipt.update(preflight_job=job_id, state="dry_run" if args.dry_run else "submitted")
    atomic_json(path, receipt)
    print(f"Preflight submission receipt: {path}")
    return receipt


def load_submission(args):
    manifest = load_manifest(args.manifest)
    mapping = json.loads(Path(args.mapping).read_text())
    if not isinstance(mapping, list) or any(type(i) is not int or not 0 <= i < len(manifest) for i in mapping):
        raise ValueError("Mapped manifest indices must be valid integers")
    if not mapping or len(set(mapping)) != len(mapping):
        raise ValueError("Submission mapping must contain unique candidates")
    if digest(dict(manifest=manifest, indices=mapping)) != args.submission_digest:
        raise ValueError("Immutable submission manifest/mapping changed")
    return manifest, mapping


def submit(args):
    cpu_environment()
    if min(args.concurrency, args.tasks_per_node, args.microbatch, args.signal_seconds) < 1:
        raise ValueError("Concurrency, processes per node, microbatch and warning interval must be positive")
    if args.signal_seconds > 65535:
        raise ValueError("Slurm warning interval must be at most 65535 seconds")
    root = Path(__file__).resolve().parents[1]
    manifest = load_manifest(Path(args.manifest).resolve())
    datasets = datasets_for_scope(args.dataset)
    indices = [i for i, r in enumerate(manifest) if r["dataset"] in datasets]
    if not indices:
        raise ValueError("No candidates selected")
    array_count = math.ceil(len(indices) / args.tasks_per_node)
    from .train import atomic_json, code_hash
    scientific_hash = code_hash()
    data, results, logs = (Path(p).resolve() for p in (args.data_dir, args.results_dir, args.log_dir))
    logs.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    pending = 0
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
        pending += 1
    if not pending:
        print("All requested candidates have terminal records. Run selection and analysis next.")
        return dict(candidate_count=len(indices), pending_count=0, state="nothing_to_submit")
    # Retain all ranks on requeue. Terminal candidates exit without fitting.
    submission_digest = digest(dict(manifest=manifest, indices=indices))
    key = submission_digest[:24]
    manifest_path = results / f"submission_{key}_manifest.json"
    mapping = results / f"submission_{key}_indices.json"
    for destination, payload in ((manifest_path, manifest), (mapping, indices)):
        if destination.exists() and json.loads(destination.read_text()) != payload:
            raise ValueError("Immutable submission record changed")
        if not destination.exists():
            atomic_json(destination, payload)
    common = sbatch_base(args, root, logs)
    commands = []
    receipt = dict(campaign_job=None, candidate_count=len(indices), pending_count=pending,
                   array_elements=array_count, array_concurrency=args.concurrency, nodes_per_element=1,
                   tasks_per_node=args.tasks_per_node, cpus_per_task=1, threads_per_process=1,
                   memory_per_node=args.memory, dataset=args.dataset, datasets=list(datasets),
                   dry_run=args.dry_run, commands=commands,
                   scientific_code_sha256=scientific_hash, state="submitting")
    receipt_path = results / f"submission_{args.dataset}_{uuid.uuid4().hex[:12]}.json"

    def send(command):
        commands.append(command)
        atomic_json(receipt_path, receipt)
        print(shlex.join(command))
        if args.dry_run:
            return "CAMPAIGN_JOB_ID"
        try:
            job_id = subprocess.check_output(command, text=True).strip().split(";")[0]
            if not job_id.isdigit():
                raise ValueError(f"Unexpected sbatch receipt: {job_id!r}")
        except (OSError, subprocess.CalledProcessError, ValueError) as error:
            receipt.update(state="submission_failed", error=str(error))
            atomic_json(receipt_path, receipt)
            raise RuntimeError(f"Submission stopped; accepted job IDs are retained in {receipt_path}") from error
        return job_id

    supervisor = [sys.executable, "-m", "fulltd.hpc", "supervise", "--manifest", str(manifest_path),
                  "--mapping", str(mapping), "--submission-digest", submission_digest,
                  "--expected-code-hash", scientific_hash, "--data-dir", str(data),
                  "--results-dir", str(results), "--log-dir", str(logs),
                  "--tasks-per-node", str(args.tasks_per_node), "--microbatch", str(args.microbatch)]
    if args.no_auto_requeue:
        supervisor.append("--no-auto-requeue")
    receipt["campaign_job"] = send(common + ["--partition", args.partition, "--time", args.time,
        "--nodes=1", f"--ntasks={args.tasks_per_node}",
        f"--ntasks-per-node={args.tasks_per_node}", "--mem", args.memory, "--job-name=fulltd_cpu_" + args.dataset,
        "--array", f"0-{array_count - 1}%{args.concurrency}",
        "--requeue", f"--signal=B:USR1@{args.signal_seconds}",
        "--wrap", allocation_command(supervisor, root)])
    receipt["state"] = "dry_run" if args.dry_run else "submitted"
    atomic_json(receipt_path, receipt)
    print(f"Submission receipt: {receipt_path}")
    return receipt


def with_warning_handler(action):
    requested = False
    def warn(signum, frame):
        nonlocal requested
        requested = True
    warning = getattr(signal, "SIGUSR1", None)
    if warning is None:
        raise RuntimeError("Slurm CPU workers require POSIX SIGUSR1 support")
    previous = signal.signal(warning, warn)
    try:
        return action(lambda: requested, warning)
    finally:
        signal.signal(warning, previous)


def worker(args):
    return with_warning_handler(lambda requested, warning: _worker(args, requested))


def _worker(args, stop_requested):
    cpu_environment()
    from .train import atomic_json, code_hash, run_index
    import torch
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    manifest, mapping = load_submission(args)
    rank = int(os.environ["SLURM_PROCID"])
    if args.offset < 0 or not 0 <= rank < args.task_count or args.offset + rank >= len(mapping):
        raise ValueError("Slurm rank is outside the submitted mapping")
    if int(os.environ.get("SLURM_CPUS_PER_TASK", "0")) != 1:
        raise ValueError("CPU campaign requires one allocated CPU per candidate")
    index = mapping[args.offset + rank]
    attempt = Path(args.attempt_dir)
    outcome_path = attempt / f"rank_{rank}.json"
    record = dict(rank=rank, index=index, run_id=manifest[index]["run_id"], state="starting")
    try:
        if code_hash() != args.expected_code_hash:
            raise ValueError("Scientific code changed after submission; queued task stopped before fitting")
        run_dir = Path(args.results_dir) / manifest[index]["run_id"]
        status_path = run_dir / "status.json"
        prior = json.loads(status_path.read_text()) if status_path.exists() else {}
        record["before_epoch"] = prior.get("next_epoch", 1)
        atomic_json(attempt / f"ready_{rank}.json", dict(rank=rank, index=index))
        if prior and (prior.get("config") != manifest[index]
                      or prior.get("identity", {}).get("code_sha256") != args.expected_code_hash):
            raise ValueError("Existing run has an incompatible configuration or scientific code identity")
        terminal = prior.get("state") == "numerical_failure" or (
            prior.get("state") == "complete" and (run_dir / "final.pt").is_file())
        result = prior if terminal else run_index(args.manifest, index, args.data_dir, args.results_dir,
            device="cpu", microbatch=args.microbatch, threads=1, stop_requested=stop_requested)
        record.update(state=result["state"], next_epoch=result.get("next_epoch", prior.get("next_epoch", 1)))
        if record["state"] not in ("complete", "numerical_failure", "checkpointed", "stopped_no_checkpoint"):
            raise RuntimeError(f"Unexpected training result: {record['state']}")
        if record["state"] in ("checkpointed", "stopped_no_checkpoint") and not stop_requested():
            raise RuntimeError("Worker stopped without a scheduler warning")
        atomic_json(outcome_path, record)
        return record
    except BaseException as error:
        record.update(state="operational_failure", error=f"{type(error).__name__}: {error}")
        atomic_json(outcome_path, record)
        raise


def supervise(args):
    return with_warning_handler(lambda requested, warning: _supervise(args, requested, warning))


def _supervise(args, stop_requested, warning):
    cpu_environment()
    from .train import atomic_json, code_hash
    _, mapping = load_submission(args)
    if code_hash() != args.expected_code_hash:
        raise ValueError("Scientific code changed after submission")
    array_job = os.environ.get("SLURM_ARRAY_JOB_ID", "")
    array_task = int(os.environ["SLURM_ARRAY_TASK_ID"])
    offset = array_task * args.tasks_per_node
    if array_task < 0 or offset >= len(mapping):
        raise ValueError("Array element is outside the submitted mapping")
    task_count = min(args.tasks_per_node, len(mapping) - offset)
    if (int(os.environ.get("SLURM_JOB_NUM_NODES", "0")) != 1
            or int(os.environ.get("SLURM_NTASKS", "0")) != args.tasks_per_node
            or int(os.environ.get("SLURM_CPUS_PER_TASK", "0")) != 1):
        raise RuntimeError("Allocated nodes/tasks/CPUs do not match the requested CPU campaign layout")
    if not array_job.isdigit():
        raise RuntimeError("CPU coordinator requires a valid Slurm array job ID")
    job = f"{array_job}_{array_task}"
    attempt = Path(args.results_dir) / "allocation_attempts" / f"{job}_{uuid.uuid4().hex}"
    attempt.mkdir(parents=True, exist_ok=False)
    command = ["srun", "--nodes=1", f"--ntasks={task_count}",
               f"--ntasks-per-node={task_count}", "--cpus-per-task=1", "--ntasks-per-core=1",
               "--cpu-bind=cores", "--kill-on-bad-exit=0", "--wait=0", "--mpi=none",
               "--output", str(Path(args.log_dir) / f"{job}_rank_%t.out"),
               "--error", str(Path(args.log_dir) / f"{job}_rank_%t.err"), "--open-mode=append",
               sys.executable, "-m", "fulltd.hpc", "worker", "--manifest", args.manifest,
               "--mapping", args.mapping, "--submission-digest", args.submission_digest,
               "--expected-code-hash", args.expected_code_hash, "--data-dir", args.data_dir,
               "--results-dir", args.results_dir, "--attempt-dir", str(attempt),
               "--offset", str(offset), "--task-count", str(task_count),
               "--microbatch", str(args.microbatch)]
    receipt = dict(job_id=job, restart_count=int(os.environ.get("SLURM_RESTART_COUNT", "0")),
                   command=command, state="running", warning_forwarded=False)
    atomic_json(attempt / "coordinator.json", receipt)
    process = subprocess.Popen(command)
    try:
        while True:
            if stop_requested() and not receipt["warning_forwarded"]:
                # Signals sent to synchronous srun are forwarded to its tasks.
                # Readiness avoids sending USR1 before Python installs handlers.
                if all((attempt / f"ready_{rank}.json").exists() or (attempt / f"rank_{rank}.json").exists()
                       for rank in range(task_count)):
                    process.send_signal(warning)
                    receipt["warning_forwarded"] = True
                    atomic_json(attempt / "coordinator.json", receipt)
            try:
                returncode = process.wait(timeout=1)
                break
            except subprocess.TimeoutExpired:
                pass
        outcomes = []
        for rank in range(task_count):
            path = attempt / f"rank_{rank}.json"
            outcomes.append(json.loads(path.read_text()) if path.exists() else dict(rank=rank, state="missing"))
        receipt.update(returncode=returncode, states={str(o["rank"]): o["state"] for o in outcomes})
        failures = [o for o in outcomes if o["state"] not in
                    ("complete", "numerical_failure", "checkpointed", "stopped_no_checkpoint")]
        if returncode != 0 or failures:
            raise RuntimeError("One or more CPU workers had an operational failure; no automatic requeue. Inspect rank logs.")
        unfinished = [o for o in outcomes if o["state"] in ("checkpointed", "stopped_no_checkpoint")]
        if not unfinished:
            receipt["state"] = "complete"
        else:
            if not stop_requested() or not receipt["warning_forwarded"]:
                raise RuntimeError("Incomplete workers without a scheduler warning; automatic requeue refused")
            stalled = [o["rank"] for o in unfinished if o["state"] == "stopped_no_checkpoint"
                       or o.get("next_epoch", 1) <= o.get("before_epoch", 1)]
            if stalled:
                raise RuntimeError(f"No durable epoch progress for ranks {stalled}; increase wall time to fit an epoch and scheduled validation")
            if args.no_auto_requeue:
                raise RuntimeError("Epoch checkpoints preserved; automatic requeue disabled. Resubmit manually.")
            receipt["state"] = "requeue_requested"
            atomic_json(attempt / "coordinator.json", receipt)
            # Only this coordinator requeues its element, after its ranks drain.
            subprocess.run(["scontrol", "requeue", job], check=True, timeout=30)
            receipt["state"] = "requeue_accepted"
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
        for name in ("manifest", "mapping", "submission-digest", "expected-code-hash", "data-dir", "results-dir"):
            parser.add_argument("--" + name, required=True)
        parser.add_argument("--microbatch", type=int, default=256)
        parser.add_argument("--attempt-dir")
        parser.add_argument("--log-dir")
        parser.add_argument("--tasks-per-node", type=int, default=90)
        parser.add_argument("--offset", type=int, default=0)
        parser.add_argument("--task-count", type=int)
        parser.add_argument("--no-auto-requeue", action="store_true")
        args = parser.parse_args()
        return worker(args) if args.action == "worker" else supervise(args)
    preflight_mode = len(sys.argv) > 1 and sys.argv[1] == "preflight"
    if preflight_mode:
        parser.add_argument("action", choices=("preflight",))
    else:
        parser.add_argument("--manifest", default="configs/fulltd/manifest.json")
        parser.add_argument("--concurrency", type=int, default=7)
        parser.add_argument("--tasks-per-node", type=int, default=90)
        parser.add_argument("--microbatch", type=int, default=256)
        parser.add_argument("--signal-seconds", type=int, default=180)
        parser.add_argument("--no-auto-requeue", action="store_true")
    parser.add_argument("--dataset", choices=DATASET_SCOPES, default="all")
    parser.add_argument("--partition", default="tiny" if preflight_mode else "batch", help="Your approved CPU partition")
    for name in ("account", "qos", "constraint"):
        parser.add_argument("--" + name)
    storage = Path.home() / "scratch" / "fulltd"
    parser.add_argument("--data-dir", default=os.environ.get("FULLTD_DATA_DIR", str(storage / "data")))
    parser.add_argument("--results-dir", default=os.environ.get("FULLTD_RESULTS_DIR", str(storage / "results")))
    parser.add_argument("--log-dir", help="Defaults to RESULTS_DIR/logs")
    parser.add_argument("--memory", default="32G" if preflight_mode else "400G", help="Total RAM PER NODE")
    parser.add_argument("--time", default="01:00:00" if preflight_mode else "3-00:00:00")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.log_dir is None:
        args.log_dir = str(Path(args.results_dir) / "logs")
    return submit_preflight(args) if preflight_mode else submit(args)


if __name__ == "__main__":
    main()
