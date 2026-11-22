"""Sequential MSE -> frozen-residual Gamma -> full-gradient predictor training.

Stage 1 reuses the exact 100-epoch, same-seed MSE candidate from the FD001
campaign, never silently fitting a different predictor. Stage 2 learns the
original PairwiseGammaHead with predictor/encoder frozen. Stage 3 keeps both
the stage-1 encoder and Gamma head frozen, updating a separate predictor
through BOTH current residuals. All records remain in fd001_results.
"""
from __future__ import annotations

import copy
import math
from pathlib import Path
import time

import numpy as np
import torch

from agents.fulltd_objectives import masked_residual
from dscpuscripts.common import atomic_json, data_identity, file_hash, read_json
from dscpuscripts.fd001_objectives import PairwiseGammaTD, check_pairs
from dscpuscripts.fd001_protocol import (
    BASELINE_ALPHAS, EPOCHS, EVAL_UPDATE_PERIOD, LAG_SETS, digest, validate_config,
)
from fulltd.sampling import LocalPairLaw, rng_for, sequence_hash, sources_for_round, training_universe
from fulltd.train import logical_backward, seed_runtime, state_hash


class ComboObjective(PairwiseGammaTD):
    """The teacher used by Gamma never follows stage-3 predictor updates."""

    gamma_variant = "frozen_representation_pairwise"

    def __init__(self, model, num_types):
        super().__init__(model, num_types=num_types)
        self.teacher = copy.deepcopy(model)
        self.phase = "gamma"
        self.set_phase("gamma")

    def set_phase(self, phase):
        if phase not in ("gamma", "predictor"):
            raise ValueError("Unknown sequential TD phase")
        self.phase = phase
        for parameter in self.model.parameters():
            parameter.requires_grad_(phase == "predictor")
        for parameter in self.teacher.parameters():
            parameter.requires_grad_(False)
        for parameter in self.gamma_head.parameters():
            parameter.requires_grad_(phase == "gamma")

    def gamma_values(self, x, other, types, *, hidden=None, neighbor_hidden=None):
        check_pairs(x, other, types, self.gamma_head.num_types)
        # The frozen teacher is an independent copy of the stage-1 encoder.
        with torch.no_grad():
            hi = self.teacher.features(x)
            hj = self.teacher.features(other)
        return self.gamma_head(hi, hj, types)

    def loss_sum(self, batch, partner=None, types=None):
        if partner is None:
            raise ValueError("Sequential TD needs a paired training example")
        x, other = batch["x"], partner["x"]
        check_pairs(x, other, types, self.gamma_head.num_types)
        gamma = self.gamma_values(x, other, types)
        if self.phase == "gamma":
            with torch.no_grad():
                prediction_i = self.model(x)
                prediction_j = self.model(other)
        else:
            # Full-gradient TD: BOTH prediction residuals backpropagate.
            prediction_i = self.model(x)
            prediction_j = self.model(other)
        source, source_mask = masked_residual(prediction_i, batch)
        neighbor, neighbor_mask = masked_residual(prediction_j, partner)
        valid = source_mask & neighbor_mask
        delta = torch.where(valid, source - gamma[:, None] * neighbor,
                            torch.zeros_like(source))
        return 0.5 * delta.square().sum(), int(valid.sum().item())

    def parameter_groups(self, alpha, beta=None):
        if self.phase == "gamma":
            return [{"params": list(self.gamma_head.parameters()), "lr": beta}]
        return [{"params": list(self.model.parameters()), "lr": alpha}]


def build_objective(config):
    from fulltd.models import build_model
    validate_config(config)
    if config["method"] != "td_mse_combo":
        raise ValueError("Not a TD-MSE combo configuration")
    return ComboObjective(build_model("fd001", "tcn", config["seed"]), len(config["lags"]))


def _mse_source(config, results, initial_hash, data_sha, val_sha, code_sha):
    """Only the matching original, fully completed 100-epoch MSE candidate."""
    h = BASELINE_ALPHAS.index(config["mse_alpha"])
    reference_id = f"fd001_mse_b256_h{h}_s{config['seed']}"
    folder = Path(results) / reference_id
    status_file, final_path = folder / "status.json", folder / "final.pt"
    if not status_file.is_file() or not final_path.is_file():
        raise RuntimeError(
            f"Stage 1 needs completed {reference_id}. Run FD001 MSE first; "
            "do not skip its 100-epoch training.")
    status = read_json(status_file)
    identity = status.get("identity", {})
    if (status.get("state") != "complete" or
            status.get("epochs_completed") != EPOCHS or
            status.get("config", {}).get("method") != "mse" or
            status["config"].get("alpha") != config["mse_alpha"] or
            status["config"].get("seed") != config["seed"] or
            identity.get("code_sha256") != code_sha or
            identity.get("data_sha256") != data_sha or
            identity.get("validation_sha256") != val_sha or
            status.get("initial_mean_sha256") != initial_hash):
        raise ValueError(f"Stage-1 MSE reference is incompatible: {reference_id}")
    state = torch.load(final_path, map_location="cpu", weights_only=False)
    if (state.get("config") != status["config"] or
            state.get("identity") != identity or
            state.get("initial_mean_sha256") != initial_hash or
            state.get("progress", {}).get("epoch") != EPOCHS + 1):
        raise ValueError(f"Stage-1 MSE checkpoint is not complete: {reference_id}")
    mean_state = {}
    for name, tensor in state["objective"].items():
        if not name.startswith("model."):
            raise ValueError("Unexpected MSE checkpoint state keys")
        mean_state[name[6:]] = tensor
    return dict(run_id=reference_id,
                sha256=file_hash(final_path),
                runtime_seconds=float(status["runtime_seconds"]),
                mean_update_seconds=float(status.get("mean_update_seconds", math.nan)),
                optimizer_steps=int(status["optimizer_steps"])), mean_state


def _optimizer(objective, config):
    if objective.phase == "gamma":
        rate = config["beta"]
    else:
        rate = config["alpha"]
    return torch.optim.Adam(objective.parameter_groups(config["alpha"], config["beta"]),
                            betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False)


def _plateau(progress, phase, config):
    loss = progress["epoch_loss"] / progress["epoch_valid"]
    previous = progress["best_loss"]
    threshold = config["stop_relative_min_decrease"]
    if not math.isfinite(previous) or loss < previous - threshold * max(abs(previous), 1e-8):
        progress["best_loss"], progress["bad_epochs"] = loss, 0
    else:
        progress["bad_epochs"] += 1
    progress["phase_epochs"][phase] += 1
    progress[phase + "_rounds"].append(dict(
        epoch=progress["phase_epochs"][phase],
        training_half_pair_mse=loss,
        best_training_half_pair_mse=progress["best_loss"],
        consecutive_nonimproving_epochs=progress["bad_epochs"],
        optimizer_steps=progress["phase_steps"][phase],
        source_sha256=progress["source_sha256"],
        partner_sha256=progress["partner_sha256"]))
    return progress["bad_epochs"] >= config["stop_patience"]


def train_candidate(config, train_data, val_data, results_dir, *, device="cpu",
                    microbatch=128, stop_requested=None, stop_after_updates=None):
    """Resume either training phase exactly; stage-1 MSE is independently trained."""
    from . import fd001_train as regular
    validate_config(config)
    if config["method"] != "td_mse_combo" or device != "cpu":
        raise ValueError("Only CPU FD001 td_mse_combo is supported")
    stop_requested = stop_requested or (lambda: False)
    seed_runtime(config["seed"])
    train, _, excluded = training_universe(train_data, LAG_SETS[0])
    law = LocalPairLaw(train.entity_ids, train.origins, tuple(config["lags"]))
    n, batch_size = len(train), config["batch_size"]
    physical = min(batch_size, microbatch)
    if physical < 1:
        raise ValueError("Positive physical microbatch required")
    objective = build_objective(config).cpu()
    initial_hash = state_hash(objective.model.state_dict())
    data_sha, val_sha = data_identity(train), data_identity(val_data)
    mse_ref, mse_state = _mse_source(
        config, results_dir, initial_hash, data_sha, val_sha, regular.code_hash())
    identity = dict(protocol_id=config["protocol_id"], config_sha256=digest(config),
                    code_sha256=regular.code_hash(), data_sha256=data_sha,
                    validation_sha256=val_sha, mse_reference_sha256=mse_ref["sha256"])
    root = Path(results_dir) / config["run_id"]
    root.mkdir(parents=True, exist_ok=True)
    status_file, recovery, final = root / "status.json", root / "recovery.pt", root / "final.pt"
    progress = dict(phase="gamma", epoch=1, cursor=0, optimizer_steps=0,
                    count=0, epoch_loss=0., epoch_valid=0, best_loss=math.inf,
                    bad_epochs=0, phase_steps={"gamma": 0, "predictor": 0},
                    phase_epochs={"gamma": 0, "predictor": 0},
                    gamma_rounds=[], predictor_rounds=[], curve=[],
                    timings={"gamma_update_seconds": 0., "predictor_update_seconds": 0.,
                             "gamma_training_wall_seconds": 0.,
                             "predictor_training_wall_seconds": 0.},
                    source_sha256=None, partner_sha256=None, stop_reasons={})
    elapsed = 0.
    checkpoint_file = final if final.is_file() else recovery
    if status_file.is_file():
        status = read_json(status_file)
        if status.get("config") != config or status.get("identity") != identity:
            raise ValueError(f"Existing sequential checkpoint is incompatible: {root}")
        if status["state"] == "numerical_failure" and not final.exists():
            return status
    if checkpoint_file.is_file():
        saved = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
        if (saved["config"] != config or saved["identity"] != identity or
                saved["initial_mean_sha256"] != initial_hash):
            raise ValueError("Sequential restart would change the model/data")
        objective.load_state_dict(saved["objective"])
        progress = saved["progress"]
        if progress["phase"] not in ("gamma", "predictor", "complete"):
            raise ValueError("Corrupted stage cursor")
        objective.set_phase("gamma" if progress["phase"] == "gamma" else "predictor")
        optimizer = _optimizer(objective, config)
        optimizer.load_state_dict(saved["optimizer"])
        regular.rng_restore(saved["rng"])
        elapsed = float(saved["runtime_seconds"])
        physical = min(physical, int(saved["physical_microbatch"]))
    else:
        if status_file.exists():
            raise ValueError("Existing status without recoverable checkpoint")
        objective.model.load_state_dict(mse_state, strict=True)
        objective.teacher.load_state_dict(mse_state, strict=True)
        objective.set_phase("gamma")
        optimizer = _optimizer(objective, config)
    atomic_json(root / "config.json", config)
    atomic_json(root / "source_universe.json", dict(
        n_sources=n, isolated_excluded=excluded, updates_per_epoch=math.ceil(n / batch_size),
        total_updates=math.ceil(n / batch_size) * config["predictor_max_epochs"],
        evaluation_period=EVAL_UPDATE_PERIOD, mse_reference=mse_ref))
    started = time.perf_counter()

    def checkpoint(state, *, retain_step=False, error=None):
        runtime = elapsed + time.perf_counter() - started
        timing = dict(progress["timings"])
        for phase in ("gamma", "predictor"):
            key = phase + "_update_seconds"
            timing[phase + "_mean_update_seconds"] = (
                timing[key] / progress["phase_steps"][phase]
                if progress["phase_steps"][phase] else None)
        timing["mse_runtime_seconds"] = mse_ref["runtime_seconds"]
        timing["mse_mean_update_seconds"] = (
            mse_ref["mean_update_seconds"] if math.isfinite(mse_ref["mean_update_seconds"])
            else None)
        timing["sequential_runtime_seconds"] = runtime
        timing["attributed_total_seconds"] = runtime + mse_ref["runtime_seconds"]
        value = dict(config=config, identity=identity, initial_mean_sha256=initial_hash,
                     objective=objective.state_dict(), optimizer=optimizer.state_dict(),
                     progress=progress, rng=regular.rng_snapshot(),
                     physical_microbatch=physical, runtime_seconds=runtime,
                     timing=timing, mse_reference=mse_ref)
        if retain_step:
            regular.atomic_torch(
                root / "checkpoints" / f"step_{progress['phase_steps']['predictor']:07d}.pt",
                value)
        regular.atomic_torch(final if state == "complete" else recovery, value)
        atomic_json(root / "update_curve.json", progress["curve"])
        atomic_json(root / "rounds.json", progress["predictor_rounds"])
        atomic_json(root / "gamma_rounds.json", progress["gamma_rounds"])
        report = dict(config=config, identity=identity, state=state,
                      epochs_completed=progress["phase_epochs"]["predictor"],
                      next_epoch=progress["phase_epochs"]["predictor"] + 1,
                      cursor=progress["cursor"], optimizer_steps=progress["phase_steps"]["predictor"],
                      update_opportunities=progress["phase_steps"]["predictor"],
                      n_sources=n, source_draws=progress["phase_epochs"]["predictor"] * n,
                      initial_mean_sha256=initial_hash, mse_reference=mse_ref,
                      phase=progress["phase"], phase_epochs=progress["phase_epochs"],
                      phase_steps=progress["phase_steps"],
                      stop_reasons=progress["stop_reasons"], timing=timing,
                      runtime_seconds=runtime, error=error)
        atomic_json(status_file, report)
        return report

    if not checkpoint_file.is_file():
        checkpoint("running")
    if final.is_file():
        return checkpoint("complete")
    try:
        while progress["phase"] != "complete":
            phase = progress["phase"]
            max_epochs = (config["gamma_max_epochs"] if phase == "gamma"
                          else config["predictor_max_epochs"])
            if progress["epoch"] > max_epochs:
                raise ValueError("Stage exceeds its declared maximum epochs")
            epoch = progress["epoch"]
            sources = sources_for_round("fd001", n, config["seed"], epoch)
            partners, types = law.sample(
                sources, rng_for("fd001", config["seed"], epoch, 22))
            progress["source_sha256"], progress["partner_sha256"] = (
                sequence_hash(sources), sequence_hash(partners))
            objective.train()
            epoch_started = time.perf_counter()
            while progress["cursor"] < n:
                if stop_requested():
                    return checkpoint("checkpointed")
                cursor = progress["cursor"]
                end = min(cursor + batch_size, n)
                batch = train.get_batch(sources[cursor:end])
                other = train.get_batch(partners[cursor:end])
                kind = types[cursor:end]
                step_started = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                loss, valid = logical_backward(objective, batch, other, kind, "cpu", physical)
                if valid < 1 or not math.isfinite(loss):
                    raise FloatingPointError("Empty/nonfinite sequential TD batch")
                if any(p.grad is not None and not bool(torch.isfinite(p.grad).all())
                       for p in objective.parameters()):
                    raise FloatingPointError("Nonfinite sequential TD gradient")
                optimizer.step()
                if any(not bool(torch.isfinite(p).all()) for p in objective.parameters()):
                    raise FloatingPointError("Nonfinite sequential TD parameter")
                progress["timings"][phase + "_update_seconds"] += time.perf_counter() - step_started
                progress["phase_steps"][phase] += 1
                progress["epoch_loss"] += loss
                progress["epoch_valid"] += valid
                progress["cursor"] = end
                progress["count"] += 1
                if phase == "predictor" and progress["phase_steps"][phase] % EVAL_UPDATE_PERIOD == 0:
                    steps = progress["phase_steps"][phase]
                    progress["curve"].append(regular.capture_validation(
                        objective, val_data, train, law, physical, steps, epoch, True))
                    checkpoint("running", retain_step=True)
                if stop_after_updates is not None and progress["count"] >= stop_after_updates:
                    return checkpoint("checkpointed")
            if stop_requested():
                return checkpoint("checkpointed")
            progress["timings"][phase + "_training_wall_seconds"] += (
                time.perf_counter() - epoch_started)
            stopped = _plateau(progress, phase, config)
            # Preserve at least one periodic validation measurement before
            # declaring stage-3 convergence under the 500-update protocol.
            if phase == "predictor" and (
                    progress["phase_steps"]["predictor"] < EVAL_UPDATE_PERIOD):
                stopped = False
            progress["cursor"] = 0
            progress["epoch_loss"], progress["epoch_valid"] = 0., 0
            progress["epoch"] += 1
            if stopped or progress["phase_epochs"][phase] >= max_epochs:
                progress["stop_reasons"][phase] = (
                    "training_loss_plateau" if stopped else "max_epochs")
                if phase == "gamma":
                    objective.set_phase("predictor")
                    optimizer = _optimizer(objective, config)
                    progress["phase"] = "predictor"
                    progress["epoch"] = 1
                    progress["best_loss"], progress["bad_epochs"] = math.inf, 0
                    progress["curve"] = [regular.capture_validation(
                        objective, val_data, train, law, physical, 0, 0, True)]
                    checkpoint("running", retain_step=True)
                else:
                    steps = progress["phase_steps"]["predictor"]
                    if not progress["curve"] or progress["curve"][-1]["optimizer_steps"] != steps:
                        progress["curve"].append(regular.capture_validation(
                            objective, val_data, train, law, physical, steps,
                            progress["phase_epochs"]["predictor"], True))
                    progress["phase"] = "complete"
                    checkpoint("complete", retain_step=True)
                    break
            else:
                checkpoint("running")
        return read_json(status_file)
    except FloatingPointError as error:
        return checkpoint("numerical_failure", error=str(error))
