"""Train the sole current FD001 four-method sweep, with update-based observations.

100 epochs, unchanged Adam/objectives/sampling. Retained update snapshots
include the source cursor so restarts never repeat or omit a pair sample.
"""
from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
import random
import time
import uuid

import numpy as np
import torch

from dscpuscripts.common import atomic_json, read_json, data_identity
from fulltd.data import load_dataset
from fulltd.metrics import evaluate
from fulltd.models import build_model
from fulltd.sampling import (LocalPairLaw, rng_for, sources_for_round, sequence_hash,
                             training_universe)
from fulltd.train import (seed_runtime, state_hash, logical_backward)
from .fd001_protocol import (PROTOCOL_ID, EPOCHS, EVAL_UPDATE_PERIOD, LAG_SETS,
                                 digest, validate_config)


SCIENTIFIC_SOURCES = (
    "dscpuscripts/fd001_protocol.py", "dscpuscripts/fd001_train.py",
    "dscpuscripts/fd001_objectives.py", "dscpuscripts/common.py",
    "agents/fulltd_objectives.py", "agents/fulltd_mse.py",
    "agents/fulltd_gaussian_nll.py", "fulltd/protocol.py", "fulltd/train.py",
    "fulltd/data.py", "fulltd/metrics.py", "fulltd/models.py",
    "fulltd/sampling.py",
)


def code_hash():
    root = Path(__file__).resolve().parents[1]
    h = hashlib.sha256()
    for name in SCIENTIFIC_SOURCES:
        h.update(name.encode())
        h.update((root / name).read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()


def build_objective(config):
    model = build_model("fd001", "tcn", config["seed"])
    if config["method"] == "td_typed":
        from dscpuscripts.fd001_objectives import TypedGammaTD
        return TypedGammaTD(model, num_types=len(config["lags"]))
    if config["method"] == "td_pairwise":
        from dscpuscripts.fd001_objectives import PairwiseGammaTD
        return PairwiseGammaTD(model, num_types=len(config["lags"]))
    if config["method"] in ("mse", "gaussian_nll"):
        from agents.fulltd_objectives import make_objective
        return make_objective(config["method"], model)
    raise ValueError("Unprescribed objective")


def atomic_torch(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("wb") as f:
            torch.save(value, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def rng_snapshot():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state().clone())


def rng_restore(s):
    random.setstate(s["python"])
    np.random.set_state(s["numpy"])
    torch.set_rng_state(s["torch"].cpu())


def json_safe(value):
    """Use identical nonfinite markers in retained checkpoints and JSON sidecars."""
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(v) for v in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return "NaN" if math.isnan(value) else "Infinity" if value > 0 else "-Infinity"
    if isinstance(value, np.generic):
        return value.item()
    return value


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
    for kind, lag in enumerate(pair_law.lags):
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


def capture_validation(objective, validation, training, pair_law, batch_size,
                       optimizer_steps, epoch, td):
    """Observation-only, including full FD001 RUL metrics, with no RNG effects."""
    was_training = objective.training
    state = rng_snapshot()
    try:
        objective.eval()
        with torch.no_grad():
            scores, _ = evaluate(objective.model, validation,
                                 batch_size=batch_size, device="cpu", allow_nonfinite=True)
            fields = {(r["units"], str(r["horizon"]), r["metric"]): float(r["value"])
                      for r in scores}
            mse = fields[("standardized", "all", "mse")]
            mae = fields[("cycles", "all", "mae")]
            rmse = fields[("cycles", "all", "rmse")]
            rul = fields[("score", "all", "rul_score_sum")]
            if not all(math.isfinite(v) for v in (mse, mae, rmse)):
                raise FloatingPointError("Nonfinite validation point predictions")
            record = dict(epoch=epoch, optimizer_steps=optimizer_steps,
                          mse_standardized=mse, rmse_cycles=rmse,
                          mae_cycles=mae, rul_score_sum=rul)
            if td:
                probe = gamma_diagnostics(objective, training, pair_law,
                                          batch_size, "cpu", lambda: False)
                if probe is None:
                    raise RuntimeError("Unexpected interrupted gamma probe")
                record["gamma_probe"] = probe
            return json_safe(record)
    finally:
        objective.train(was_training)
        rng_restore(state)


def train_candidate(config, train_data, val_data, results_dir, *, device="cpu",
                    microbatch=128, stop_requested=None, stop_after_updates=None):
    """Produce canonical update/epoch evidence; test hook stops after an update."""
    validate_config(config)
    if device != "cpu":
        raise ValueError("The direct fresh FD001 sweep uses CPU only")
    stop_requested = stop_requested or (lambda: False)
    seed_runtime(config["seed"])
    # Do not add previously excluded source windows when adding lag types.
    train, _, excluded = training_universe(train_data, LAG_SETS[0])
    law = LocalPairLaw(train.entity_ids, train.origins, tuple(config["lags"]))
    if len(law.nonisolated()) != len(train):
        raise ValueError("Lag set isolates a source from the fixed common source universe")
    n = len(train)
    batch_size = config["batch_size"]
    physical = min(batch_size, microbatch)
    if physical < 1:
        raise ValueError("Positive physical microbatch required")
    td = config["method"].startswith("td_")
    root = Path(results_dir) / config["run_id"]
    root.mkdir(parents=True, exist_ok=True)
    identity = dict(protocol_id=PROTOCOL_ID, config_sha256=digest(config),
                    code_sha256=code_hash(), data_sha256=data_identity(train),
                    validation_sha256=data_identity(val_data))
    status_file, recovery, final = (root / "status.json", root / "recovery.pt", root / "final.pt")
    if status_file.exists():
        old_status = read_json(status_file)
        if old_status.get("config") != config or old_status.get("identity") != identity:
            raise ValueError("Existing FD001 run belongs to a different protocol/code/data")
        if old_status["state"] == "numerical_failure" and not final.exists():
            return old_status
    objective = build_objective(config).cpu()
    mean_hash = state_hash(objective.model.state_dict())
    optimizer = torch.optim.Adam(objective.parameter_groups(config["alpha"], config["beta"]),
                                 betas=(.9, .999), eps=1e-8, weight_decay=0, amsgrad=False)
    expected_updates = math.ceil(n / batch_size) * EPOCHS
    initial = dict(epoch=1, cursor=0, count=0, optimizer_steps=0, epoch_loss=0.,
                   epoch_valid=0, curve=[], rounds=[])
    progress = initial
    checkpoint_file = final if final.exists() else recovery
    if checkpoint_file.exists():
        ckpt = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
        if ckpt["identity"] != identity or ckpt["config"] != config or ckpt["initial_mean_sha256"] != mean_hash:
            raise ValueError("Saved FD001 checkpoint is incompatible; never silently restart")
        objective.load_state_dict(ckpt["objective"])
        optimizer.load_state_dict(ckpt["optimizer"])
        rng_restore(ckpt["rng"])
        progress = ckpt["progress"]
        physical = min(physical, int(ckpt["physical_microbatch"]))
        if progress["count"] != progress["optimizer_steps"]:
            raise ValueError("FD001 expected one actual Adam update per logical batch")
    else:
        if status_file.exists():
            raise ValueError("Existing status without a checkpoint; preserve and investigate")
        # Record initial identical predictor states without accessing validation/test labels.
        initfile = Path(results_dir) / "initial_states" / f"fd001_tcn_s{config['seed']}_{mean_hash[:16]}.pt"
        if not initfile.exists():
            atomic_torch(initfile, {k: v.detach().cpu() for k, v in objective.model.state_dict().items()})
        progress["curve"].append(capture_validation(objective, val_data, train, law,
                            physical, 0, 0, td))
    atomic_json(root / "config.json", config)
    atomic_json(root / "source_universe.json", dict(n_sources=n, isolated_excluded=excluded,
                  updates_per_epoch=math.ceil(n/batch_size), total_updates=expected_updates,
                  evaluation_period=EVAL_UPDATE_PERIOD))
    started = time.perf_counter()

    def checkpoint(state, *, retain_step=False):
        value = dict(config=config, identity=identity, initial_mean_sha256=mean_hash,
                     objective=objective.state_dict(), optimizer=optimizer.state_dict(),
                     progress=progress, rng=rng_snapshot(), physical_microbatch=physical)
        # Preserve every scheduled full state, including the actual initial state.
        if retain_step:
            path = root / "checkpoints" / f"step_{progress['optimizer_steps']:07d}.pt"
            atomic_torch(path, value)
        atomic_torch(final if state == "complete" else recovery, value)
        atomic_json(root / "update_curve.json", progress["curve"])
        atomic_json(root / "rounds.json", progress["rounds"])
        report = dict(config=config, identity=identity, state=state,
                      epochs_completed=progress["epoch"]-1,
                      next_epoch=progress["epoch"], cursor=progress["cursor"],
                      optimizer_steps=progress["optimizer_steps"],
                      update_opportunities=progress["count"], n_sources=n,
                      initial_mean_sha256=mean_hash,
                      source_draws=(progress["epoch"]-1)*n+progress["cursor"],
                      runtime_seconds=time.perf_counter()-started,
                      error=None)
        atomic_json(status_file, report)
        return report

    # Initial snapshot is saved once; all remaining snapshots are cumulative.
    if not checkpoint_file.exists():
        checkpoint("running", retain_step=True)
    if final.exists():
        return checkpoint("complete")
    if progress["epoch"] == EPOCHS+1:
        return checkpoint("complete", retain_step=True)
    try:
        while progress["epoch"] <= EPOCHS:
            epoch = progress["epoch"]
            sources = sources_for_round("fd001", n, config["seed"], epoch)
            partners, types = law.sample(sources, rng_for("fd001", config["seed"], epoch, 22)) if td else (None,None)
            objective.train()
            while progress["cursor"] < n:
                if stop_requested():
                    return checkpoint("checkpointed")
                cursor = progress["cursor"]
                end = min(cursor+batch_size, n)
                batch = train.get_batch(sources[cursor:end])
                other = train.get_batch(partners[cursor:end]) if td else None
                kinds = types[cursor:end] if td else None
                optimizer.zero_grad(set_to_none=True)
                loss, valid = logical_backward(objective, batch, other, kinds, "cpu", physical)
                if not math.isfinite(loss) or not all(
                        p.grad is None or bool(torch.isfinite(p.grad).all())
                        for p in objective.parameters()):
                    raise FloatingPointError("Nonfinite loss or gradient")
                if valid == 0:
                    raise ValueError("FD001 must have nonempty training masks")
                optimizer.step()
                if any(not bool(torch.isfinite(p).all()) for p in objective.parameters()):
                    raise FloatingPointError("Nonfinite parameter")
                progress["count"] += 1
                progress["optimizer_steps"] += 1
                progress["cursor"] = end
                progress["epoch_loss"] += loss
                progress["epoch_valid"] += valid
                steps = progress["optimizer_steps"]
                if steps % EVAL_UPDATE_PERIOD == 0:
                    progress["curve"].append(capture_validation(
                        objective, val_data, train, law, physical, steps, epoch, td))
                    checkpoint("running", retain_step=True)
                if stop_after_updates is not None and steps >= stop_after_updates:
                    return checkpoint("checkpointed")
            if stop_requested():
                return checkpoint("checkpointed")
            progress["rounds"].append(dict(epoch=epoch, source_draws=n,
                                          source_sha256=sequence_hash(sources),
                                          partner_sha256=sequence_hash(partners) if td else None,
                                          type_counts=np.bincount(types, minlength=len(config["lags"])).tolist() if td else [],
                                          optimizer_steps_cumulative=progress["optimizer_steps"],
                                          sampled_half_pair_mse=progress["epoch_loss"]/progress["epoch_valid"]))
            progress["epoch"] = epoch+1
            progress["cursor"] = 0
            progress["epoch_loss"] = 0.
            progress["epoch_valid"] = 0
            checkpoint("complete" if epoch == EPOCHS else "running")
        return read_json(status_file)
    except FloatingPointError as error:
        # No favorable restart or suppression; all five seeds must pass.
        report = checkpoint("numerical_failure")
        report["error"] = str(error)
        atomic_json(status_file, report)
        return report
