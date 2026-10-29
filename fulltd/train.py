"""Train a single prespecified candidate with deterministic rolling recovery."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
import hashlib
import json
import math
import platform
import random
import subprocess
import time
import uuid
from pathlib import Path

import numpy as np
import torch

from . import PROTOCOL_ID
from .protocol import EVALUATIONS, LAGS, digest, load_manifest, validate_config
from .sampling import rng_for, sources_for_round, sequence_hash, training_universe
from .metrics import validation_mse


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=True) + "\n")
    replace_file(temp, path)


def atomic_save(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temp)
    replace_file(temp, path)


def replace_file(source, destination):
    # Windows file indexers can transiently hold a just-written sidecar. This
    # bounded retry changes no experiment state and never discards a checkpoint.
    for attempt in range(6):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(.02 * 2 ** attempt)


def code_hash():
    root = Path(__file__).resolve().parents[1]
    files = sorted((root / "fulltd").glob("*.py")) + sorted((root / "agents").glob("fulltd_*.py"))
    h = hashlib.sha256()
    for file in files:
        h.update(str(file.relative_to(root)).replace("\\", "/").encode())
        h.update(file.read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()


def state_hash(state):
    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        h.update(key.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def data_identity(dataset):
    metadata = dataset.metadata
    # Acquisition dates are provenance, not a scientific preprocessing change.
    return digest(dict(fingerprint=metadata.get("fingerprint", digest(metadata)),
                       split=getattr(dataset, "split", "unknown"),
                       common_source_count=metadata.get("common_source_count"),
                       source_index_sha256=metadata.get("source_index_sha256"),
                       isolated_excluded_count=metadata.get("isolated_excluded_count")))


def tensor_batch(batch, start, stop, device):
    return {key: torch.as_tensor(np.asarray(batch[key][start:stop]), device=device,
                                dtype=torch.bool if key == "mask" else torch.float32)
            for key in ("x", "y", "mask")}


def logical_backward(objective, batch, partner, types, device, microbatch):
    """Accumulate sums / GLOBAL valid count, including unequal final chunks."""
    mask = batch["mask"] if partner is None else batch["mask"] & partner["mask"]
    count = int(np.sum(mask))
    if count == 0:
        return 0., 0
    total = 0.
    for start in range(0, len(batch["x"]), microbatch):
        stop = min(start + microbatch, len(batch["x"]))
        source_t = tensor_batch(batch, start, stop, device)
        partner_t = None if partner is None else tensor_batch(partner, start, stop, device)
        type_t = None if types is None else torch.as_tensor(types[start:stop], dtype=torch.long, device=device)
        loss_sum, valid = objective.loss_sum(source_t, partner_t, type_t)
        if not torch.isfinite(loss_sum):
            raise FloatingPointError("Nonfinite loss")
        if valid:
            (loss_sum / count).backward()
        total += float(loss_sum.detach())
    return total, count


def finite_gradients(objective):
    return all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in objective.parameters())


def seed_runtime(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def make_objective_for(config, train_data):
    from .models import build_model
    from agents.fulltd_objectives import make_objective
    model = build_model(config["dataset"], config["backbone"], config["seed"],
                        adjacency=getattr(train_data, "adjacency", None))
    return make_objective(config["method"], model)


def train_candidate(config, train_data, val_data, result_root, device="cpu", microbatch=None,
                    recovery_every=100, stop_after_batches=None):
    """stop_after_batches is solely a test interruption hook; absent from CLI."""
    validate_config(config)
    seed_runtime(config["seed"])
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    run_dir = Path(result_root) / config["run_id"]
    run_dir.mkdir(parents=True, exist_ok=True)
    train_data, pair_law, excluded = training_universe(train_data, LAGS[config["dataset"]])
    n = len(train_data)
    rounds_size = math.ceil(n / 24) if config["dataset"] == "electricity" else n
    identity = dict(config_sha256=digest(config), code_sha256=code_hash(),
                    data_sha256=data_identity(train_data), validation_sha256=data_identity(val_data), protocol_id=PROTOCOL_ID)
    objective = make_objective_for(config, train_data)
    initial_mean = objective.model.state_dict()
    initial_hash = state_hash(initial_mean)
    # One identical, inspectable initial mean state per backbone/seed (safe concurrent creation).
    init_dir = Path(result_root) / "initial_states"
    init_dir.mkdir(exist_ok=True)
    init_file = init_dir / f"{config['dataset']}_{config['backbone']}_s{config['seed']}_{initial_hash[:16]}.pt"
    if not init_file.exists():
        temp = init_dir / f".{init_file.name}.{uuid.uuid4().hex}.tmp"
        torch.save(initial_mean, temp)
        os.replace(temp, init_file)
    objective.to(device)
    optimizer = torch.optim.Adam(objective.parameter_groups(config["alpha"], config["beta"]),
                                 betas=(.9, .999), eps=1e-8, weight_decay=0, amsgrad=False)
    td = config["method"].startswith("td_")
    physical = min(config["batch_size"], microbatch or config["batch_size"])
    counters = dict(optimizer_steps=0, source_draws=0, partner_draws=0, source_valid_count=0,
                    pair_valid_count=0, empty_mask_batches=0, update_opportunities=0, oom_retries=0)
    validation, round_records = [], []
    epoch, cursor, elapsed = 1, 0, 0.
    recovery = run_dir / "recovery.pt"
    final = run_dir / "final.pt"
    existing = final if final.exists() else recovery
    completed_checkpoint = None
    checkpoint = None
    if existing.exists():
        checkpoint = torch.load(existing, map_location=device, weights_only=False)
        if checkpoint["identity"] != identity:
            raise ValueError(f"Incompatible checkpoint preserved at {existing}; use a fresh results directory.")
        objective.load_state_dict(checkpoint["objective"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        counters = checkpoint["counters"]
        validation, round_records = checkpoint["validation"], checkpoint["round_records"]
        epoch, cursor, elapsed = checkpoint["next_epoch"], checkpoint["cursor"], checkpoint["runtime_seconds"]
        physical = min(physical, checkpoint["physical_microbatch"])
        random.setstate(checkpoint["python_rng"])
        np.random.set_state(checkpoint["numpy_rng"])
        torch.set_rng_state(checkpoint["torch_rng"].cpu())
        if device.type == "cuda" and checkpoint["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all([r.cpu() for r in checkpoint["cuda_rng"]])
        if existing == final:
            completed_checkpoint = checkpoint
    if (run_dir / "status.json").exists():
        previous = json.loads((run_dir / "status.json").read_text())
        if previous.get("state") == "numerical_failure" and completed_checkpoint is None:
            if previous.get("identity") != identity:
                raise ValueError("Incompatible prior numerical-failure record; preserved")
            print(f"Preserved numerical failure: {config['run_id']}")
            return previous
    start_clock = time.perf_counter()
    hardware = dict(platform=platform.platform(), python=platform.python_version(), torch=torch.__version__,
                    numpy=np.__version__, device=str(device), cuda_runtime=torch.version.cuda,
                    accelerator=torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
                    slurm_job_id=os.environ.get("SLURM_JOB_ID"))
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parents[1], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = "unavailable"
    saved_peak = (checkpoint or {}).get("peak_accelerator_memory_bytes") or 0
    saved_segments = list((checkpoint or {}).get("execution_segments", []))
    current_segment = dict(hardware=hardware, code_revision=revision)
    original_hardware = (checkpoint or {}).get("hardware", hardware)
    original_revision = (checkpoint or {}).get("code_revision", revision)

    def runtime():
        return elapsed if completed_checkpoint is not None else elapsed + time.perf_counter() - start_clock

    def peak_memory():
        current = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        return max(saved_peak, current) or None

    def segments():
        if completed_checkpoint is not None:
            return saved_segments
        return saved_segments + [dict(current_segment, runtime_seconds=time.perf_counter() - start_clock)]
    atomic_json(run_dir / "config.json", config)
    atomic_json(run_dir / "source_universe.json", dict(n=n, isolated_excluded=excluded, data=train_data.metadata,
                draws_per_round=rounds_size, nominal_updates_per_round=math.ceil(rounds_size / config["batch_size"]),
                effective_passes_at_completion=100 * rounds_size / n))

    def snapshot():
        return dict(identity=identity, config=config, objective=objective.state_dict(),
                    optimizer=optimizer.state_dict(), next_epoch=epoch, cursor=cursor,
                    counters=counters.copy(), validation=list(validation), round_records=list(round_records),
                    runtime_seconds=runtime(), physical_microbatch=physical,
                    python_rng=random.getstate(), numpy_rng=np.random.get_state(), torch_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
                    initial_mean_sha256=initial_hash, hardware=original_hardware,
                    code_revision=original_revision, execution_segments=segments(),
                    peak_accelerator_memory_bytes=peak_memory())

    def write_status(state, error=None):
        atomic_json(run_dir / "status.json", dict(state=state, protocol_id=PROTOCOL_ID, config=config,
                    identity=identity, epochs_completed=epoch-1, next_epoch=epoch, cursor=cursor,
                    **counters, runtime_seconds=runtime(),
                    effective_original_passes=counters["source_draws"] / n, n_sources=n,
                    training_unit=config["training_unit"], physical_microbatch=physical,
                    initial_mean_sha256=initial_hash, mean_parameter_count=sum(p.numel() for p in objective.model.parameters()),
                    total_parameter_count=sum(p.numel() for p in objective.parameters()), hardware=original_hardware,
                    code_revision=original_revision, execution_segments=segments(), error=error,
                    peak_accelerator_memory_bytes=peak_memory()))
    if completed_checkpoint is not None:
        # A preemption can occur after atomic final.pt but before its sidecars.
        # Repair those records from the authoritative complete checkpoint.
        atomic_json(run_dir / "validation.json", validation)
        atomic_json(run_dir / "rounds.json", round_records)
        write_status("complete")
        print(f"Already complete and compatible: {config['run_id']}")
        return completed_checkpoint
    write_status("running")
    try:
        while epoch <= 100:
            sources = sources_for_round(config["dataset"], n, config["seed"], epoch)
            partners, types = pair_law.sample(sources, rng_for(config["dataset"], config["seed"], epoch, 22)) if td else (None, None)
            source_sha = sequence_hash(sources)
            objective.train()
            while cursor < len(sources):
                end = min(cursor + config["batch_size"], len(sources))
                batch = train_data.get_batch(sources[cursor:end])
                partner = train_data.get_batch(partners[cursor:end]) if td else None
                batch_types = types[cursor:end] if td else None
                # These count actual attempted logical batches even if numerical
                # failure interrupts this update. Recovery restores its earlier
                # saved counters and cursor for operational interruptions.
                counters["update_opportunities"] += 1
                counters["source_draws"] += end - cursor
                counters["partner_draws"] += end - cursor if td else 0
                counters["source_valid_count"] += int(np.sum(batch["mask"]))
                counters["pair_valid_count"] += int(np.sum(batch["mask"] & partner["mask"])) if td else 0
                while True:
                    optimizer.zero_grad(set_to_none=True)
                    try:
                        loss_sum, valid_count = logical_backward(objective, batch, partner, batch_types, device, physical)
                        break
                    except torch.cuda.OutOfMemoryError:
                        optimizer.zero_grad(set_to_none=True)
                        if physical <= 1:
                            raise
                        physical = max(1, physical // 2)
                        counters["oom_retries"] += 1
                        torch.cuda.empty_cache()
                if not finite_gradients(objective):
                    raise FloatingPointError("Nonfinite gradient")
                if valid_count:
                    optimizer.step()
                    counters["optimizer_steps"] += 1
                    if not all(bool(torch.isfinite(p).all()) for p in objective.parameters()):
                        raise FloatingPointError("Nonfinite parameter")
                else:
                    counters["empty_mask_batches"] += 1
                cursor = end
                if counters["update_opportunities"] % recovery_every == 0:
                    atomic_save(recovery, snapshot())
                    write_status("running")
                if stop_after_batches is not None and counters["update_opportunities"] >= stop_after_batches:
                    atomic_save(recovery, snapshot())
                    write_status("interrupted_test_only")
                    return snapshot()
            if epoch in EVALUATIONS:
                mse = validation_mse(objective.model, val_data, physical, device)
                if not math.isfinite(mse):
                    raise FloatingPointError("Nonfinite validation MSE")
                validation.append(dict(epoch=epoch, mse=mse, optimizer_steps=counters["optimizer_steps"],
                                       theta=objective.theta.detach().cpu().tolist() if td else None))
                atomic_json(run_dir / "validation.json", validation)
            round_records.append(dict(epoch=epoch, source_draws=len(sources), source_sha256=source_sha,
                partner_sha256=sequence_hash(partners) if td else None,
                type_counts=np.bincount(types, minlength=2).tolist() if td else None,
                optimizer_steps_cumulative=counters["optimizer_steps"],
                runtime_seconds_cumulative=elapsed + time.perf_counter() - start_clock))
            epoch += 1
            cursor = 0
            atomic_save(recovery, snapshot())
            atomic_json(run_dir / "rounds.json", round_records)
            write_status("running")
        atomic_save(final, snapshot())
        write_status("complete")
        return snapshot()
    except FloatingPointError as error:
        write_status("numerical_failure", str(error))
        # Keep all validation evidence and the previous finite recovery checkpoint.
        atomic_json(run_dir / "validation.json", validation)
        return {"state": "numerical_failure", "error": str(error)}
    except Exception as error:
        write_status("operational_failure", f"{type(error).__name__}: {error}")
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="configs/fulltd/manifest.json")
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--microbatch", type=int)
    parser.add_argument("--recovery-every", type=int, default=100)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.microbatch is not None and args.microbatch < 1 or args.recovery_every < 1:
        parser.error("Microbatch and checkpoint interval must be positive")
    torch.set_num_threads(args.threads)
    config = load_manifest(args.manifest)[args.index]
    from .data import load_dataset
    train = load_dataset(config["dataset"], args.data_dir, "train")
    val = load_dataset(config["dataset"], args.data_dir, "validation")
    from filelock import FileLock
    lock_dir = Path(args.results_dir) / config["run_id"]
    lock_dir.mkdir(parents=True, exist_ok=True)
    with FileLock(str(lock_dir / ".run.lock"), timeout=0):
        train_candidate(config, train, val, args.results_dir, args.device, args.microbatch, args.recovery_every)


if __name__ == "__main__":
    main()
