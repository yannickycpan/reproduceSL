"""FD001-only training with the legacy sampling, Adam and epoch-only recovery rules.

Kept separate so adding this campaign does not invalidate running legacy jobs or
old scientific code hashes. The shared model, data, sampler and metrics are imported
unchanged. No test split is loaded here; Gamma summaries use a fixed training probe.
"""
from __future__ import annotations

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import hashlib
import math
from pathlib import Path
import platform
import random
import subprocess
import time
import uuid

import numpy as np
import torch

from fulltd.data import load_dataset
from fulltd.metrics import validation_scores
from fulltd.sampling import rng_for, sources_for_round, sequence_hash, training_universe
from . import PROTOCOL_ID
from .common import atomic_json, read_json, file_lock, scientific_hash, data_identity
from .objectives import make_objective
from .protocol import EVALUATIONS, LAGS, digest, validate_config


def seed_runtime(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def state_hash(state):
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def atomic_save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with tmp.open("wb") as stream:
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def tensor_batch(batch, start, stop, device):
    return {key: torch.as_tensor(np.asarray(batch[key][start:stop]), device=device,
                                dtype=torch.bool if key == "mask" else torch.float32)
            for key in ("x", "y", "mask")}


def logical_backward(objective, batch, partner, types, device, microbatch):
    """One logical-batch gradient: every chunk divides by the SAME valid count."""
    if microbatch < 1:
        raise ValueError("Physical microbatch must be positive")
    count = int(np.sum(batch["mask"] & partner["mask"]))
    if not count:
        return 0., 0
    total = 0.
    for start in range(0, len(batch["x"]), microbatch):
        stop = min(start + microbatch, len(batch["x"]))
        source = tensor_batch(batch, start, stop, device)
        other = tensor_batch(partner, start, stop, device)
        kind = torch.as_tensor(types[start:stop], device=device, dtype=torch.long)
        loss, valid = objective.loss_sum(source, other, kind)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite loss")
        if valid:
            (loss / count).backward()
        total += float(loss.detach())
    return total, count


def checkpoint_identity(config, train, val):
    return dict(protocol_id=PROTOCOL_ID, config_sha256=digest(config), code_sha256=scientific_hash(),
                data_sha256=data_identity(train), validation_sha256=data_identity(val))


def validate_checkpoint(checkpoint, config, identity, *, final=False):
    if checkpoint.get("config") != config or checkpoint.get("identity") != identity:
        raise ValueError("Incompatible checkpoint; existing results preserved")
    epoch = checkpoint.get("next_epoch")
    if type(epoch) is not int or not 1 <= epoch <= 101 or checkpoint.get("cursor") != 0:
        raise ValueError("Only completed epoch-boundary checkpoints can be resumed")
    if final and epoch != 101:
        raise ValueError("final.pt is not the actual epoch-100 checkpoint")
    if [r["epoch"] for r in checkpoint["round_records"]] != list(range(1, epoch)):
        raise ValueError("Checkpoint epoch records are incomplete")
    if [r["epoch"] for r in checkpoint["validation"]] != [e for e in EVALUATIONS if e < epoch]:
        raise ValueError("Checkpoint validation records are incomplete")


@torch.no_grad()
def gamma_diagnostics(objective, dataset, pair_law, microbatch, device, stop_requested):
    """Fixed label-free training-pair probe; never consumes training RNG state."""
    ids = np.unique(np.linspace(0, len(dataset) - 1, min(len(dataset), 512), dtype=np.int64))
    partners, kinds, probabilities = pair_law.enumerate(ids)
    source_row, slot = np.nonzero(probabilities > 0)
    sources, targets = ids[source_row], partners[source_row, slot]
    types, weights = kinds[source_row, slot], probabilities[source_row, slot]
    values = []
    was_training = objective.training
    objective.eval()
    try:
        for start in range(0, len(sources), microbatch):
            if stop_requested():
                return None
            sl = slice(start, start + microbatch)
            x = torch.as_tensor(dataset.get_batch(sources[sl])["x"], device=device)
            other = torch.as_tensor(dataset.get_batch(targets[sl])["x"], device=device)
            t = torch.as_tensor(types[sl].copy(), device=device, dtype=torch.long)
            values.append(objective.gamma_values(x, other, t).cpu().double().numpy())
    finally:
        objective.train(was_training)
    values = np.concatenate(values)
    # Logging must not change scientific stopping: preserve nonfinite probe
    # values rather than turn an otherwise finite training update into a failure.
    records = []
    for kind, lag in enumerate(LAGS):
        take = types == kind
        if not take.any():
            continue
        v, w = values[take], weights[take]
        with np.errstate(invalid="ignore", over="ignore"):
            mean = float(np.average(v, weights=w))
            sd = float(np.sqrt(np.average((v - mean) ** 2, weights=w)))
        records.append(dict(type=kind + 1, lag=lag, mean=mean,
                            std=sd, finite_pair_count=int(np.isfinite(v).sum()),
                            minimum=float(v.min()), maximum=float(v.max()),
                            negative_fraction=float(np.average(v < 0, weights=w)),
                            near_one_fraction=float(np.average(abs(v - 1) < .05, weights=w)),
                            pair_count=int(take.sum())))
    return dict(source_count=len(ids), source_indices_sha256=sequence_hash(ids),
                weighting="exact conditional pair probabilities on fixed training-source probe",
                types=records)


def train_candidate(config, train_data, val_data, results_dir, *, device="cpu", microbatch=128,
                    stop_requested=None, stop_after_batches=None):
    """The batch stop is a test hook only; no production early stopping exists."""
    validate_config(config)
    if microbatch < 1:
        raise ValueError("Physical microbatch must be positive")
    if train_data.split != "train" or val_data.split not in ("val", "validation"):
        raise ValueError("Training accepts training and validation splits only")
    directory = Path(results_dir) / config["run_id"]
    with file_lock(directory / ".worker.lock"):
        return _train(config, train_data, val_data, directory, device, microbatch,
                      stop_requested or (lambda: False), stop_after_batches)


def _train(config, train_data, val_data, directory, device, microbatch, stop_requested, stop_after_batches):
    seed_runtime(config["seed"])
    device = torch.device(device)
    train, law, excluded = training_universe(train_data, LAGS)
    n = len(train)
    identity = checkpoint_identity(config, train, val_data)
    status_path = directory / "status.json"
    previous = read_json(status_path) if status_path.exists() else None
    if previous and (previous.get("config") != config or previous.get("identity") != identity):
        raise ValueError("Incompatible existing status; nothing overwritten")
    final, recovery = directory / "final.pt", directory / "recovery.pt"
    if previous and previous["state"] == "numerical_failure" and not final.exists():
        return previous
    objective = make_objective(config).to(device)
    initial_hash = state_hash(objective.model.state_dict())
    init_file = directory.parent / "initial_states" / f"fd001_tcn_s{config['seed']}_{initial_hash[:16]}.pt"
    if not init_file.exists():
        atomic_save(init_file, {k: v.detach().cpu() for k, v in objective.model.state_dict().items()})
    optimizer = torch.optim.Adam(objective.parameter_groups(config["alpha"], config["beta"]),
                                 betas=(.9, .999), eps=1e-8, weight_decay=0, amsgrad=False)
    physical = min(config["batch_size"], microbatch)
    counters = dict(optimizer_steps=0, source_draws=0, partner_draws=0, source_valid_count=0,
                    pair_valid_count=0, empty_mask_batches=0, update_opportunities=0, oom_retries=0)
    validations, rounds, epoch, elapsed = [], [], 1, 0.
    interruptions_path = directory / "interruptions.json"
    interruptions = read_json(interruptions_path) if interruptions_path.exists() else []
    discarded = sum(item["uncheckpointed_runtime_seconds"] for item in interruptions)
    checkpoint = None
    source = final if final.exists() else recovery
    if source.exists():
        checkpoint = torch.load(source, map_location=device, weights_only=False)
        validate_checkpoint(checkpoint, config, identity, final=source == final)
        objective.load_state_dict(checkpoint["objective"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        epoch, elapsed = checkpoint["next_epoch"], checkpoint["runtime_seconds"]
        validations, rounds = checkpoint["validation"], checkpoint["round_records"]
        counters = checkpoint["counters"].copy()
        if checkpoint["initial_mean_sha256"] != initial_hash:
            raise ValueError("Initial predictor identity changed")
        physical = min(physical, checkpoint["physical_microbatch"])
        random.setstate(checkpoint["python_rng"])
        np.random.set_state(checkpoint["numpy_rng"])
        torch.set_rng_state(checkpoint["torch_rng"].cpu())
        if device.type == "cuda" and checkpoint["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all([v.cpu() for v in checkpoint["cuda_rng"]])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    hardware = dict(platform=platform.platform(), python=platform.python_version(), torch=torch.__version__,
                    numpy=np.__version__, device=str(device), threads=torch.get_num_threads(),
                    slurm_job_id=os.environ.get("SLURM_JOB_ID"))
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1],
                                           stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = "unavailable"
    original_hardware = (checkpoint or {}).get("hardware", hardware)
    segments = list((checkpoint or {}).get("execution_segments", []))
    clock = time.perf_counter()
    durable = dict(next_epoch=epoch, counters=counters.copy(), runtime_seconds=elapsed)

    def runtime():
        return elapsed + time.perf_counter() - clock

    def status(state, error=None):
        record = dict(state=state, protocol_id=PROTOCOL_ID, config=config, identity=identity,
                      next_epoch=durable["next_epoch"], epochs_completed=durable["next_epoch"] - 1, cursor=0,
                      **durable["counters"], n_sources=n, initial_mean_sha256=initial_hash,
                      runtime_seconds=durable["runtime_seconds"], discarded_runtime_seconds=discarded,
                      total_execution_seconds=durable["runtime_seconds"] + discarded,
                      physical_microbatch=physical, mean_parameter_count=sum(p.numel() for p in objective.model.parameters()),
                      total_parameter_count=sum(p.numel() for p in objective.parameters()),
                      hardware=original_hardware, code_revision=revision, error=error)
        atomic_json(status_path, record)
        return record

    def publish_sidecars():
        atomic_json(directory / "validation.json", validations)
        atomic_json(directory / "rounds.json", rounds)

    def pause(cursor):
        nonlocal discarded
        wasted = max(0., runtime() - durable["runtime_seconds"])
        interruptions.append(dict(epoch_interrupted=epoch, source_cursor_interrupted=cursor,
                                  resume_epoch=durable["next_epoch"],
                                  discarded_counters={k: v - durable["counters"][k] for k, v in counters.items()},
                                  uncheckpointed_runtime_seconds=wasted, slurm_job_id=os.environ.get("SLURM_JOB_ID")))
        discarded += wasted
        atomic_json(interruptions_path, interruptions)
        # Do not write partial weights or partial validation/round records.
        return status("checkpointed" if recovery.exists() else "stopped_no_checkpoint")

    atomic_json(directory / "config.json", config)
    atomic_json(directory / "source_universe.json", dict(n=n, isolated_excluded=excluded,
                data=train.metadata, draws_per_round=n, nominal_updates_per_round=math.ceil(n / 128)))
    if epoch == 101:
        publish_sidecars()
        return status("complete")
    status("running")
    try:
        while epoch <= 100:
            if stop_requested():
                return pause(0)
            sources = sources_for_round("fd001", n, config["seed"], epoch)
            partners, kinds = law.sample(sources, rng_for("fd001", config["seed"], epoch, 22))
            objective.train()
            cursor, loss_sum_epoch, valid_epoch = 0, 0., 0
            while cursor < n:
                if stop_requested():
                    return pause(cursor)
                end = min(cursor + 128, n)
                batch, partner = train.get_batch(sources[cursor:end]), train.get_batch(partners[cursor:end])
                counters["update_opportunities"] += 1
                counters["source_draws"] += end - cursor
                counters["partner_draws"] += end - cursor
                counters["source_valid_count"] += int(batch["mask"].sum())
                counters["pair_valid_count"] += int((batch["mask"] & partner["mask"]).sum())
                while True:
                    optimizer.zero_grad(set_to_none=True)
                    try:
                        total, valid = logical_backward(objective, batch, partner, kinds[cursor:end], device, physical)
                        break
                    except torch.cuda.OutOfMemoryError:
                        optimizer.zero_grad(set_to_none=True)
                        if physical <= 1:
                            raise
                        physical //= 2
                        counters["oom_retries"] += 1
                        torch.cuda.empty_cache()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in objective.parameters()):
                    raise FloatingPointError("Nonfinite gradient")
                if valid:
                    optimizer.step()
                    counters["optimizer_steps"] += 1
                    if any(not torch.isfinite(p).all() for p in objective.parameters()):
                        raise FloatingPointError("Nonfinite parameter")
                else:
                    counters["empty_mask_batches"] += 1
                loss_sum_epoch += total
                valid_epoch += valid
                cursor = end
                if stop_after_batches is not None and counters["update_opportunities"] >= stop_after_batches:
                    return pause(cursor)
            if stop_requested():
                return pause(cursor)
            if epoch in EVALUATIONS:
                scores = validation_scores(objective.model, val_data, physical, device, stop_requested)
                if scores is None:
                    return pause(cursor)
                gamma = gamma_diagnostics(objective, train, law, physical, device, stop_requested)
                if gamma is None:
                    return pause(cursor)
                validations.append(dict(epoch=epoch, **scores, optimizer_steps=counters["optimizer_steps"],
                                        validation_population_size=len(val_data), validation_sampling="full",
                                        operator_state=objective.operator_state(), gamma_probe=gamma))
            rounds.append(dict(epoch=epoch, source_draws=n, source_sha256=sequence_hash(sources),
                               partner_sha256=sequence_hash(partners), type_counts=np.bincount(kinds, minlength=2).tolist(),
                               optimizer_steps_cumulative=counters["optimizer_steps"],
                               sampled_half_pair_mse=loss_sum_epoch / valid_epoch if valid_epoch else None))
            epoch += 1
            retained = runtime()
            state = dict(config=config, identity=identity, next_epoch=epoch, cursor=0,
                         objective=objective.state_dict(), optimizer=optimizer.state_dict(), counters=counters.copy(),
                         validation=list(validations), round_records=list(rounds), runtime_seconds=retained,
                         physical_microbatch=physical, initial_mean_sha256=initial_hash,
                         hardware=original_hardware, code_revision=revision,
                         execution_segments=segments + [dict(hardware=hardware, runtime_seconds=retained - elapsed)],
                         python_rng=random.getstate(), numpy_rng=np.random.get_state(), torch_rng=torch.get_rng_state(),
                         cuda_rng=torch.cuda.get_rng_state_all() if device.type == "cuda" else None)
            atomic_save(final if epoch == 101 else recovery, state)
            durable = dict(next_epoch=epoch, counters=counters.copy(), runtime_seconds=retained)
            publish_sidecars()
            record = status("complete" if epoch == 101 else "running")
            print(f"{config['run_id']}: epoch {epoch - 1}/100 saved; steps={counters['optimizer_steps']}", flush=True)
        return record
    except FloatingPointError as error:
        # Keep partial evidence and failure, never restart it as a fresh candidate.
        publish_sidecars()
        record = status("numerical_failure", str(error))
        record.update(counters, failed_epoch=epoch)
        atomic_json(status_path, record)
        return record


def run_config(config, data_dir, results_dir, **kwargs):
    validate_config(config)
    return train_candidate(config, load_dataset("fd001", data_dir, "train"),
                           load_dataset("fd001", data_dir, "val"), results_dir, **kwargs)
