"""Train the current Electricity/METR-LA methods with resumable update records.

Only validation data are read here. Test evaluation is a separate operation,
after validation-only hyperparameter selection. The FD001 training files and
their checkpoint identity remain unchanged.
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
from fulltd.metrics import evaluate
from .electricity_model import build_model as build_electricity_model
from .metr_la_model import build_model as build_metr_la_model
from .electricity_validation import selection as electricity_validation_selection
from .metr_la_validation import selection as metr_la_validation_selection
from fulltd.sampling import rng_for, sequence_hash, training_universe
from fulltd.train import seed_runtime, state_hash, logical_backward
from .application_objectives import make_objective
from .application_protocol import (PROTOCOL_ID, digest, validate_config,
                                   update_budgets, evaluation_steps,
                                   sources_for_application_round)


def code_hash(config):
    from .application_identity import code_hash as scientific_code_hash
    return scientific_code_hash(config)


def build_objective(config, dataset=None):
    adjacency = getattr(dataset, "adjacency", None)
    if isinstance(dataset, (np.ndarray, torch.Tensor)):
        adjacency = dataset
    if config["dataset"] == "electricity" and config["backbone"] == "patchtst":
        model = build_electricity_model(config["seed"])
    elif config["dataset"] == "metr_la" and config["backbone"] == "graph_wavenet":
        model = build_metr_la_model(config["seed"], adjacency)
    else:
        raise ValueError("Unsupported application dataset/backbone")
    return make_objective(config["method"], model, num_types=len(config["lags"]))


def atomic_torch(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("wb") as stream:
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def rng_snapshot():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state().clone())


def rng_restore(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())


class _ValidationView:
    """Lazy view preserving the parent's customer normalization and masks."""

    def __init__(self, parent, indices, metadata):
        self.parent = parent
        self.indices = np.asarray(indices, dtype=np.int64)
        self.indices.flags.writeable = False
        self.entity_ids = np.asarray(parent.entity_ids)[self.indices]
        self.origins = np.asarray(parent.origins)[self.indices]
        self.validation_metadata = metadata

    def __getattr__(self, name):
        return getattr(self.parent, name)

    def __len__(self):
        return len(self.indices)

    def get_batch(self, indices):
        return self.parent.get_batch(self.indices[np.asarray(indices, dtype=np.int64)])

    def inverse(self, values, indices):
        return self.parent.inverse(values, self.indices[np.asarray(indices, dtype=np.int64)])


def evaluation_validation_view(dataset):
    """A fixed common validation view, with dataset policies in separate modules."""
    if (dataset.dataset not in ("electricity", "metr_la") or
            dataset.split not in ("val", "validation") or len(dataset) < 1):
        raise ValueError("A nonempty Electricity/METR-LA validation split is required")
    if dataset.dataset == "metr_la":
        return metr_la_validation_selection(dataset)
    indices, record = electricity_validation_selection(dataset)
    return _ValidationView(dataset, indices, record), record


class _EvaluationInterrupted(Exception):
    pass


class _InterruptibleView:
    def __init__(self, parent, stop_requested):
        self.parent, self.stop_requested = parent, stop_requested

    def __getattr__(self, name):
        return getattr(self.parent, name)

    def __len__(self):
        return len(self.parent)

    def get_batch(self, indices):
        if self.stop_requested():
            raise _EvaluationInterrupted
        return self.parent.get_batch(indices)


def capture_validation(objective, validation, subset_record, batch_size,
                       optimizer_steps, epoch, stop_requested=None):
    """An observation leaves both the training mode and RNG state unchanged."""
    stop_requested = stop_requested or (lambda: False)
    was_training, state = objective.training, rng_snapshot()
    try:
        objective.eval()
        scores, _ = evaluate(objective.model, _InterruptibleView(validation, stop_requested),
                             batch_size=batch_size, device="cpu")
        fields = {(r["units"], str(r["horizon"]), r["metric"]): float(r["value"]) for r in scores}
        units = {"electricity": "kW", "metr_la": "mph"}[validation.dataset]
        record = dict(epoch=int(epoch), optimizer_steps=int(optimizer_steps),
                      raw_units=units, validation_subset=subset_record)
        for metric in ("mse", "mae", "rmse"):
            record[metric + "_standardized"] = fields[("standardized", "all", metric)]
            record[metric + "_raw"] = fields[(units, "all", metric)]
        if not all(math.isfinite(record[m + s]) for m in ("mse", "mae", "rmse")
                   for s in ("_standardized", "_raw")):
            raise FloatingPointError("Nonfinite validation point predictions")
        return record
    except _EvaluationInterrupted:
        return None
    finally:
        objective.train(was_training)
        rng_restore(state)


def train_candidate(config, train_data, val_data, results_dir, *, device="cpu",
                    microbatch=128, stop_requested=None, stop_after_updates=None):
    """Fit one frozen candidate; recover at its exact next source/partner draw."""
    validate_config(config)
    if str(device) != "cpu":
        raise ValueError("The direct application sweep uses CPU only")
    if train_data.dataset != config["dataset"] or train_data.split != "train" or val_data.dataset != config["dataset"]:
        raise ValueError("Candidate dataset and train/validation split do not agree")
    stop_requested = stop_requested or (lambda: False)
    seed_runtime(config["seed"])
    train, law, excluded = training_universe(train_data, tuple(config["lags"]))
    validation, subset_record = evaluation_validation_view(val_data)
    n, batch_size = len(train), config["batch_size"]
    physical = min(batch_size, microbatch)
    if physical < 1:
        raise ValueError("Positive physical microbatch required")
    td = config["method"].startswith("td_")
    budgets = update_budgets(config, n)
    draws = budgets["draws_per_round"]
    scheduled = set(evaluation_steps(budgets["total_updates"], config["learning_curve_updates"]))
    root = Path(results_dir) / config["run_id"]
    root.mkdir(parents=True, exist_ok=True)
    identity = dict(protocol_id=config["protocol_id"], config_sha256=digest(config), code_sha256=code_hash(config),
                    data_sha256=data_identity(train), validation_sha256=data_identity(val_data),
                    validation_subset_sha256=digest(subset_record))
    status_file, recovery, final = root / "status.json", root / "recovery.pt", root / "final.pt"
    if status_file.exists():
        old_status = read_json(status_file)
        if old_status.get("config") != config or old_status.get("identity") != identity:
            raise ValueError("Existing application run belongs to a different protocol/code/data")
        if old_status["state"] == "numerical_failure" and not final.exists():
            return old_status
    objective = build_objective(config, train).cpu()
    mean_hash = state_hash(objective.model.state_dict())
    optimizer = torch.optim.Adam(objective.parameter_groups(config["alpha"], config["beta"]),
                                 betas=(.9, .999), eps=1e-8, weight_decay=0, amsgrad=False)
    progress = dict(epoch=1, cursor=0, count=0, optimizer_steps=0,
                    epoch_loss=0., epoch_valid=0, empty_mask_batches=0,
                    curve=[], rounds=[], pending_evaluation=0)
    elapsed = 0.
    checkpoint_file = final if final.exists() else recovery
    resumed = checkpoint_file.exists()
    if resumed:
        ckpt = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
        if (ckpt["identity"] != identity or ckpt["config"] != config
                or ckpt["initial_mean_sha256"] != mean_hash):
            raise ValueError("Saved application checkpoint is incompatible; never silently restart")
        objective.load_state_dict(ckpt["objective"])
        optimizer.load_state_dict(ckpt["optimizer"])
        rng_restore(ckpt["rng"])
        progress = ckpt["progress"]
        physical = min(physical, int(ckpt["physical_microbatch"]))
        elapsed = float(ckpt.get("runtime_seconds", 0.))
        if (progress["count"] != progress["optimizer_steps"] + progress["empty_mask_batches"]
                or not 1 <= progress["epoch"] <= config["epochs"] + 1
                or not 0 <= progress["cursor"] <= draws):
            raise ValueError("Malformed saved training cursor/update accounting")
    elif status_file.exists():
        raise ValueError("Existing status without a checkpoint; preserve and investigate")
    else:
        initfile = Path(results_dir) / "initial_states" / f"{config['dataset']}_{config['backbone']}_s{config['seed']}_{mean_hash[:16]}.pt"
        if not initfile.exists():
            atomic_torch(initfile, {k: v.detach().cpu() for k, v in objective.model.state_dict().items()})
    atomic_json(root / "config.json", config)
    atomic_json(root / "source_universe.json", dict(n_sources=n, isolated_excluded=excluded,
                **budgets, evaluation_period=config["learning_curve_updates"],
                scheduled_evaluations=sorted(scheduled), validation_subset=subset_record,
                effective_passes_at_completion=config["epochs"] * draws / n))
    started = time.perf_counter()

    def checkpoint(state, *, retain_step=False, error=None):
        runtime = elapsed + time.perf_counter() - started
        value = dict(config=config, identity=identity, initial_mean_sha256=mean_hash,
                     objective=objective.state_dict(), optimizer=optimizer.state_dict(),
                     progress=progress, rng=rng_snapshot(), physical_microbatch=physical,
                     runtime_seconds=runtime)
        if retain_step:
            path = root / "checkpoints" / f"step_{progress['optimizer_steps']:07d}.pt"
            atomic_torch(path, value)
        atomic_torch(final if state == "complete" else recovery, value)
        atomic_json(root / "update_curve.json", progress["curve"])
        atomic_json(root / "rounds.json", progress["rounds"])
        source_draws = (progress["epoch"] - 1) * draws + progress["cursor"]
        report = dict(config=config, identity=identity, state=state, protocol_id=config["protocol_id"],
                      epochs_completed=progress["epoch"] - 1, next_epoch=progress["epoch"],
                      cursor=progress["cursor"], optimizer_steps=progress["optimizer_steps"],
                      update_opportunities=progress["count"], n_sources=n, **budgets,
                      empty_mask_batches=progress["empty_mask_batches"],
                      initial_mean_sha256=mean_hash, source_draws=source_draws,
                      partner_draws=source_draws if td else 0,
                      validation_subset=subset_record, runtime_seconds=runtime, error=error)
        atomic_json(status_file, report)
        return report

    def measure_pending():
        pending = progress["pending_evaluation"]
        if pending is None:
            return True
        if pending != progress["optimizer_steps"]:
            raise ValueError("Pending validation does not match the saved predictor")
        epoch = min(config["epochs"], progress["epoch"]) if pending else 0
        record = capture_validation(objective, validation, subset_record, physical,
                                    pending, epoch, stop_requested)
        if record is None:
            return False
        if progress["curve"] and progress["curve"][-1]["optimizer_steps"] >= pending:
            raise ValueError("Duplicate or unordered update measurement")
        progress["curve"].append(record)
        progress["pending_evaluation"] = None
        checkpoint("running", retain_step=True)
        return True

    if final.exists():
        if progress["epoch"] != config["epochs"] + 1 or progress["pending_evaluation"] is not None:
            raise ValueError("Final checkpoint is not a fully evaluated final iterate")
        return checkpoint("complete")
    if not resumed:
        checkpoint("running")
    try:
        if not measure_pending():
            return checkpoint("checkpointed")
        while progress["epoch"] <= config["epochs"]:
            epoch = progress["epoch"]
            sources = sources_for_application_round(config["dataset"], n, config["seed"], epoch,
                                                    config.get("source_fraction"))
            if len(sources) != draws:
                raise ValueError("Source sampler differs from the frozen draw budget")
            partners, types = law.sample(sources, rng_for(config["dataset"], config["seed"], epoch, 22)) if td else (None, None)
            objective.train()
            while progress["cursor"] < draws:
                if stop_requested():
                    return checkpoint("checkpointed")
                cursor = progress["cursor"]
                end = min(cursor + batch_size, draws)
                batch = train.get_batch(sources[cursor:end])
                other = train.get_batch(partners[cursor:end]) if td else None
                kinds = types[cursor:end] if td else None
                optimizer.zero_grad(set_to_none=True)
                loss, valid = logical_backward(objective, batch, other, kinds, "cpu", physical)
                if not math.isfinite(loss) or not all(
                        p.grad is None or bool(torch.isfinite(p.grad).all()) for p in objective.parameters()):
                    raise FloatingPointError("Nonfinite loss or gradient")
                if valid:
                    optimizer.step()
                    if any(not bool(torch.isfinite(p).all()) for p in objective.parameters()):
                        raise FloatingPointError("Nonfinite parameter")
                    progress["optimizer_steps"] += 1
                else:
                    # Missing observations must not cause a zero-gradient Adam
                    # step (which would still change moments/parameters).
                    progress["empty_mask_batches"] += 1
                progress["count"] += 1
                progress["cursor"] = end
                progress["epoch_loss"] += loss
                progress["epoch_valid"] += valid
                steps = progress["optimizer_steps"]
                if valid and steps in scheduled:
                    progress["pending_evaluation"] = steps
                    if not measure_pending():
                        return checkpoint("checkpointed")
                if stop_after_updates is not None and steps >= stop_after_updates:
                    return checkpoint("checkpointed")
            if stop_requested():
                return checkpoint("checkpointed")
            progress["rounds"].append(dict(epoch=epoch, source_draws=draws,
                source_sha256=sequence_hash(sources),
                partner_sha256=sequence_hash(partners) if td else None,
                type_counts=np.bincount(types, minlength=len(config["lags"])).tolist() if td else [],
                optimizer_steps_cumulative=progress["optimizer_steps"],
                update_opportunities_cumulative=progress["count"],
                empty_mask_batches_cumulative=progress["empty_mask_batches"],
                sampled_loss=(progress["epoch_loss"] / progress["epoch_valid"] if progress["epoch_valid"] else None)))
            progress["epoch"] = epoch + 1
            progress["cursor"] = 0
            progress["epoch_loss"] = 0.
            progress["epoch_valid"] = 0
            checkpoint("running")
        if not progress["curve"] or progress["curve"][-1]["optimizer_steps"] != progress["optimizer_steps"]:
            # Empty-mask skips can make the true final update off the nominal
            # schedule. Record it; downstream comparison checks actual budgets.
            progress["pending_evaluation"] = progress["optimizer_steps"]
            if not measure_pending():
                return checkpoint("checkpointed")
        return checkpoint("complete", retain_step=True)
    except FloatingPointError as error:
        return checkpoint("numerical_failure", error=str(error))
