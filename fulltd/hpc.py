"""Slurm submission for CityU Burgundy; queue/resource names are supplied by you."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from .protocol import load_manifest, digest


def submit(args):
    root = Path(__file__).resolve().parents[1]
    manifest_path = Path(args.manifest).resolve()
    manifest = load_manifest(manifest_path)
    indices = [i for i, r in enumerate(manifest) if args.dataset == "all" or r["dataset"] == args.dataset]
    if not indices:
        raise ValueError("No candidates selected")
    data, results, logs = (Path(p).resolve() for p in (args.data_dir, args.results_dir, args.log_dir))
    logs.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
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
              "--cpus-per-task", str(args.cpus), "--mem", args.memory, "--time", args.time,
              "--chdir", str(root), "--output", str(logs / "%x_%A_%a.out"),
              "--error", str(logs / "%x_%A_%a.err"), "--export=ALL"]
    for key in ("account", "qos", "constraint"):
        if getattr(args, key):
            common.extend(["--" + key, getattr(args, key)])

    commands = []
    def send(command):
        commands.append(command)
        print(shlex.join(command))
        if args.dry_run:
            return "PREFLIGHT_JOB_ID" if len(commands) == 1 else "ARRAY_JOB_ID"
        job_id = subprocess.check_output(command, text=True).strip().split(";")[0]
        if not job_id.isdigit():
            raise RuntimeError(f"Unexpected sbatch job receipt {job_id!r}")
        return job_id

    preflight_cmd = [sys.executable, "-m", "fulltd.preflight", "--dataset", args.dataset,
                     "--data-dir", str(data), "--output", str(results / f"preflight_{args.dataset}.json"),
                     "--device", "cuda", "--threads", str(args.cpus)]
    preflight = send(common + ["--job-name=fulltd_preflight", "--wrap", shlex.join(preflight_cmd)])
    # The explicit map permits chunking under MaxArraySize without changing run IDs.
    job_ids = []
    for offset in range(0, len(indices), args.array_chunk):
        count = min(args.array_chunk, len(indices)-offset)
        command = [sys.executable, "-m", "fulltd.hpc", "worker", "--manifest", str(manifest_path),
                   "--mapping", str(mapping), "--offset", str(offset), "--data-dir", str(data),
                   "--results-dir", str(results), "--threads", str(args.cpus),
                   "--microbatch", str(args.microbatch)]
        dependency = "afterok:" + preflight
        if job_ids:
            # Chain chunks to keep the requested concurrency a campaign-wide
            # cap. afterany allows independent later candidates after a failed job.
            dependency += ",afterany:" + job_ids[-1]
        job_ids.append(send(common + ["--job-name=fulltd_" + args.dataset,
                        "--dependency=" + dependency,
                        "--array", f"0-{count-1}%{args.concurrency}", "--wrap", shlex.join(command)]))
    receipt = dict(preflight_job=preflight, candidate_array_jobs=job_ids, candidate_count=len(indices),
                   dataset=args.dataset, dry_run=args.dry_run, commands=commands,
                   note="Chunks run sequentially; concurrency caps this submission. Account limits remain site-specific.")
    (results / f"submission_{args.dataset}.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def worker(args):
    mapping = json.loads(Path(args.mapping).read_text())
    index = mapping[args.offset + int(os.environ["SLURM_ARRAY_TASK_ID"])]
    cmd = [sys.executable, "-m", "fulltd.train", "--manifest", args.manifest,
           "--index", str(index), "--data-dir", args.data_dir, "--results-dir", args.results_dir,
           "--device", "cuda", "--threads", str(args.threads), "--microbatch", str(args.microbatch)]
    subprocess.run(cmd, check=True)


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
    args = parser.parse_args()
    if min(args.cpus, args.concurrency, args.array_chunk, args.microbatch) < 1:
        parser.error("Resource counts must be positive")
    submit(args)


if __name__ == "__main__":
    main()
