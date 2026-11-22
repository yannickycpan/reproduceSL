"""FD001 TD-MSE combo: fresh MSE, frozen-residual Gamma, full-gradient TD.

Independent three-stage training in the *same* run and existing FD001 result
root. Stage 1: exactly 100 epochs of MSE with B256 and its selected alpha.
Stage 2: learn the ORIGINAL representation-based PairwiseGammaHead while
freezing predictor/residuals, and stop on training-loss plateau or 100 epochs.
Stage 3: freeze both the stage-1 teacher and Gamma head, train the current
predictor for exactly 100 epochs, differentiating BOTH residuals.

All stages use the same seeded FD001 training sources and local pair law as
the original FD001 campaign. No previous MSE checkpoint is required.
"""
from __future__ import annotations

import copy
import math
from pathlib import Path
import time

import numpy as np
import torch

from agents.fulltd_objectives import masked_residual
from dscpuscripts.common import atomic_json, data_identity, read_json
from dscpuscripts.fd001_objectives import PairwiseGammaTD, check_pairs
from dscpuscripts.fd001_protocol import EPOCHS, EVAL_UPDATE_PERIOD, LAG_SETS, digest, validate_config
from fulltd.sampling import LocalPairLaw, rng_for, sequence_hash, sources_for_round, training_universe
from fulltd.train import logical_backward, seed_runtime, state_hash


class ComboObjective(PairwiseGammaTD):
    """A frozen teacher supplies Gamma features throughout stage 3."""

    gamma_variant = "frozen_representation_pairwise"

    def __init__(self, model, num_types):
        super().__init__(model, num_types=num_types)
        self.teacher = copy.deepcopy(model)
        self.phase = "mse"
        self.set_phase("mse")

    def set_phase(self, phase):
        if phase not in ("mse", "gamma", "predictor"):
            raise ValueError("Unknown sequential TD phase")
        self.phase = phase
        for parameter in self.model.parameters():
            parameter.requires_grad_(phase in ("mse", "predictor"))
        for parameter in self.teacher.parameters():
            parameter.requires_grad_(False)
        for parameter in self.gamma_head.parameters():
            parameter.requires_grad_(phase == "gamma")
        # Do not retain gradients from an earlier stage on frozen blocks.
        for parameter in self.parameters():
            parameter.grad = None
        self.teacher.eval()

    def gamma_values(self, x, other, types, *, hidden=None, neighbor_hidden=None):
        check_pairs(x, other, types, self.gamma_head.num_types)
        with torch.no_grad():
            hi = self.teacher.features(x)
            hj = self.teacher.features(other)
        return self.gamma_head(hi, hj, types)

    def loss_sum(self, batch, partner=None, types=None):
        if self.phase == "mse":
            if partner is not None or types is not None:
                raise ValueError("MSE stage cannot receive TD partners")
            residual, mask = masked_residual(self.model(batch["x"]), batch)
            return .5 * residual.square().sum(), int(mask.sum().item())

        if partner is None:
            raise ValueError("Sequential TD needs a sampled partner")
        x, other = batch["x"], partner["x"]
        check_pairs(x, other, types, self.gamma_head.num_types)
        gamma = self.gamma_values(x, other, types)
        if self.phase == "gamma":
            # Stage-1 residuals cannot change; gradients train Gamma only.
            with torch.no_grad():
                prediction_i, prediction_j = self.model(x), self.model(other)
        else:
            # Stage 3: both LIVE residuals contribute to the full gradient.
            prediction_i, prediction_j = self.model(x), self.model(other)
        source, mask_i = masked_residual(prediction_i, batch)
        neighbor, mask_j = masked_residual(prediction_j, partner)
        valid = mask_i & mask_j
        delta = torch.where(valid, source - gamma[:, None] * neighbor,
                            torch.zeros_like(source))
        return .5 * delta.square().sum(), int(valid.sum().item())

    def parameter_groups(self, alpha, beta=None):
        if self.phase == "mse":
            return [{"params": list(self.model.parameters()), "lr": alpha}]
        if self.phase == "gamma":
            return [{"params": list(self.gamma_head.parameters()), "lr": beta}]
        return [{"params": list(self.model.parameters()), "lr": alpha}]


def build_objective(config):
    from fulltd.models import build_model
    validate_config(config)
    if config["method"] != "td_mse_combo":
        raise ValueError("Not a TD-MSE combo configuration")
    return ComboObjective(build_model("fd001", "tcn", config["seed"]), len(config["lags"]))


def _optimizer(objective, config):
    rate = (config["mse_alpha"] if objective.phase == "mse" else
            config["beta"] if objective.phase == "gamma" else config["alpha"])
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError("Invalid stage learning rate")
    return torch.optim.Adam(objective.parameter_groups(rate, rate),
                            betas=(.9, .999), eps=1e-8,
                            weight_decay=0, amsgrad=False)


def _plateau(progress, phase, config):
    """Stage-2 training-loss criterion; evaluated once per completed epoch."""
    if phase != "gamma":
        raise ValueError("Only Gamma training may early-stop")
    loss = progress["epoch_loss"] / progress["epoch_valid"]
    previous = progress["best_loss"]
    threshold = config["stop_relative_min_decrease"]
    if not math.isfinite(previous) or loss < previous - threshold * max(abs(previous), 1e-8):
        progress["best_loss"], progress["bad_epochs"] = loss, 0
    else:
        progress["bad_epochs"] += 1
    return progress["bad_epochs"] >= config["stop_patience"]


def _record_round(progress, phase, epoch, sampled_loss):
    progress["phase_epochs"][phase] += 1
    row = dict(epoch=epoch, source_draws=progress["n_sources"],
               source_sha256=progress["source_sha256"],
               partner_sha256=progress["partner_sha256"] if phase != "mse" else None,
               optimizer_steps=progress["phase_steps"][phase],
               training_loss=sampled_loss)
    if phase == "gamma":
        row.update(best_training_loss=progress["best_loss"],
                   consecutive_nonimproving_epochs=progress["bad_epochs"])
    progress[phase + "_rounds"].append(row)


def train_candidate(config, train_data, val_data, results_dir, *, device="cpu",
                    microbatch=128, stop_requested=None, stop_after_updates=None):
    """All three stages restart at the exact source cursor and optimizer state."""
    from . import fd001_train as regular
    validate_config(config)
    if config["method"] != "td_mse_combo" or device != "cpu":
        raise ValueError("Only CPU FD001 td_mse_combo is supported")
    stop_requested = stop_requested or (lambda: False)
    seed_runtime(config["seed"])
    train, _, excluded = training_universe(train_data, LAG_SETS[0])
    law = LocalPairLaw(train.entity_ids, train.origins, tuple(config["lags"]))
    n = len(train)
    stage_batch = {"mse": 256, "gamma": config["batch_size"],
                   "predictor": config["batch_size"]}
    stage_epochs = {"mse": config["mse_epochs"],
                    "gamma": config["gamma_max_epochs"],
                    "predictor": config["predictor_max_epochs"]}
    if stage_epochs["mse"] != EPOCHS or stage_epochs["predictor"] != EPOCHS:
        raise ValueError("MSE and final predictor stage must each train exactly 100 epochs")

    objective = build_objective(config).cpu()
    initial_hash = state_hash(objective.model.state_dict())
    identity = dict(protocol_id=config["protocol_id"], config_sha256=digest(config),
                    code_sha256=regular.code_hash(),
                    data_sha256=data_identity(train),
                    validation_sha256=data_identity(val_data))
    root = Path(results_dir) / config["run_id"]
    root.mkdir(parents=True, exist_ok=True)
    status_file, recovery, final = root / "status.json", root / "recovery.pt", root / "final.pt"
    progress = dict(phase="mse", epoch=1, cursor=0, count=0,
                    n_sources=n, epoch_loss=0., epoch_valid=0,
                    best_loss=math.inf, bad_epochs=0,
                    phase_steps={"mse": 0, "gamma": 0, "predictor": 0},
                    phase_epochs={"mse": 0, "gamma": 0, "predictor": 0},
                    mse_rounds=[], gamma_rounds=[], predictor_rounds=[],
                    curve=[], source_sha256=None, partner_sha256=None,
                    stage1_model_sha256=None, stop_reasons={},
                    timings={f"{phase}_{kind}": 0.
                             for phase in ("mse", "gamma", "predictor")
                             for kind in ("update_seconds", "wall_seconds")})
    elapsed = 0.
    checkpoint_file = final if final.is_file() else recovery
    if status_file.is_file():
        status = read_json(status_file)
        if status.get("config") != config or status.get("identity") != identity:
            raise ValueError(f"Incompatible earlier sequential run: {root}")
        if status.get("state") == "numerical_failure" and not final.is_file():
            return status
    if checkpoint_file.is_file():
        saved = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
        if (saved.get("config") != config or saved.get("identity") != identity
                or saved.get("initial_mean_sha256") != initial_hash):
            raise ValueError("Saved model/data differs; refusing silent restart")
        objective.load_state_dict(saved["objective"])
        progress = saved["progress"]
        if progress["phase"] not in ("mse", "gamma", "predictor", "complete"):
            raise ValueError("Invalid staged training cursor")
        if not 0 <= progress["cursor"] <= n or progress["n_sources"] != n:
            raise ValueError("Saved dataset cursor is inconsistent")
        actual_phase = "predictor" if progress["phase"] == "complete" else progress["phase"]
        objective.set_phase(actual_phase)
        optimizer = _optimizer(objective, config)
        optimizer.load_state_dict(saved["optimizer"])
        regular.rng_restore(saved["rng"])
        elapsed = float(saved["runtime_seconds"])
        physical = min(stage_batch[actual_phase], microbatch,
                       int(saved["physical_microbatch"]))
    else:
        if status_file.exists():
            raise ValueError("Status exists without a recoverable checkpoint")
        optimizer = _optimizer(objective, config)
        physical = min(256, microbatch)
    if physical < 1:
        raise ValueError("Physical microbatch must be positive")

    atomic_json(root / "config.json", config)
    atomic_json(root / "source_universe.json", dict(
        n_sources=n, isolated_excluded=excluded, mse_batch=256,
        pairwise_batch=config["batch_size"],
        mse_total_updates=math.ceil(n / 256) * EPOCHS,
        gamma_max_updates=math.ceil(n / config["batch_size"]) * config["gamma_max_epochs"],
        predictor_total_updates=math.ceil(n / config["batch_size"]) * EPOCHS,
        evaluation_period=EVAL_UPDATE_PERIOD))
    started = time.perf_counter()
    phase_clock = started

    def account_phase():
        nonlocal phase_clock
        now = time.perf_counter()
        phase = progress["phase"]
        if phase in ("mse", "gamma", "predictor"):
            progress["timings"][phase + "_wall_seconds"] += now - phase_clock
        phase_clock = now

    def checkpoint(state, *, retain_step=False, error=None):
        account_phase()
        runtime = elapsed + time.perf_counter() - started
        timing = dict(progress["timings"])
        for stage in ("mse", "gamma", "predictor"):
            count = progress["phase_steps"][stage]
            timing[stage + "_mean_update_seconds"] = (
                timing[stage + "_update_seconds"] / count if count else None)
            timing[stage + "_optimizer_steps"] = count
        timing["attributed_total_seconds"] = runtime
        if retain_step:
            # The normal analysis expects only stage-3 update-indexed snapshots.
            regular.atomic_torch(
                root / "checkpoints" / f"step_{progress['phase_steps']['predictor']:07d}.pt",
                dict(config=config, identity=identity, initial_mean_sha256=initial_hash,
                     objective=objective.state_dict(), optimizer=optimizer.state_dict(),
                     progress=progress, rng=regular.rng_snapshot(),
                     physical_microbatch=physical, runtime_seconds=runtime))
        value = dict(config=config, identity=identity, initial_mean_sha256=initial_hash,
                     objective=objective.state_dict(), optimizer=optimizer.state_dict(),
                     progress=progress, rng=regular.rng_snapshot(),
                     physical_microbatch=physical, runtime_seconds=runtime, timing=timing)
        regular.atomic_torch(final if state == "complete" else recovery, value)
        atomic_json(root / "update_curve.json", progress["curve"])
        for phase in ("mse", "gamma", "predictor"):
            atomic_json(root / ("rounds.json" if phase == "predictor" else phase + "_rounds.json"),
                        progress[phase + "_rounds"])
        report = dict(config=config, identity=identity, state=state,
                      epochs_completed=progress["phase_epochs"]["predictor"],
                      next_epoch=progress["phase_epochs"]["predictor"] + 1,
                      cursor=progress["cursor"],
                      optimizer_steps=progress["phase_steps"]["predictor"],
                      update_opportunities=progress["phase_steps"]["predictor"],
                      n_sources=n, source_draws=progress["phase_epochs"]["predictor"] * n,
                      initial_mean_sha256=initial_hash, phase=progress["phase"],
                      phase_steps=progress["phase_steps"], phase_epochs=progress["phase_epochs"],
                      stage1_model_sha256=progress["stage1_model_sha256"],
                      stop_reasons=progress["stop_reasons"], timing=timing,
                      runtime_seconds=runtime, error=error)
        atomic_json(status_file, report)
        return report

    def log_stage_end(stage):
        updates = progress["phase_steps"][stage]
        compute = progress["timings"][stage + "_update_seconds"]
        wall = progress["timings"][stage + "_wall_seconds"]
        mean_update = compute / updates if updates else None
        print(f"{config['run_id']}: stage={stage} "
              f"epochs={progress['phase_epochs'][stage]} updates={updates} "
              f"mean_optimizer_update_seconds={mean_update:.6f} "
              f"stage_wall_seconds={wall:.3f}", flush=True)

    if not checkpoint_file.is_file():
        checkpoint("running")
    if final.is_file():
        if progress["phase"] != "complete":
            raise ValueError("Final state does not correspond to 100 predictor epochs")
        return checkpoint("complete")

    try:
        while progress["phase"] != "complete":
            phase, epoch = progress["phase"], progress["epoch"]
            if not 1 <= epoch <= stage_epochs[phase]:
                raise ValueError("Stage epoch cursor exceeds configured horizon")
            batch_size = stage_batch[phase]
            physical = min(physical, batch_size)
            sources = sources_for_round("fd001", n, config["seed"], epoch)
            if phase == "mse":
                partners, types = None, None
            else:
                partners, types = law.sample(
                    sources, rng_for("fd001", config["seed"], epoch, 22))
            progress["source_sha256"] = sequence_hash(sources)
            progress["partner_sha256"] = sequence_hash(partners) if partners is not None else None
            objective.train()
            objective.teacher.eval()

            while progress["cursor"] < n:
                if stop_requested():
                    return checkpoint("checkpointed")
                cursor = progress["cursor"]
                end = min(cursor + batch_size, n)
                batch = train.get_batch(sources[cursor:end])
                other = train.get_batch(partners[cursor:end]) if partners is not None else None
                kind = types[cursor:end] if types is not None else None
                update_start = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                loss, valid = logical_backward(objective, batch, other, kind, "cpu", physical)
                if not math.isfinite(loss) or valid <= 0:
                    raise FloatingPointError("Empty/nonfinite sequential batch")
                if any(p.grad is not None and not bool(torch.isfinite(p.grad).all())
                       for p in objective.parameters()):
                    raise FloatingPointError("Nonfinite gradient")
                optimizer.step()
                if any(not bool(torch.isfinite(p).all()) for p in objective.parameters()):
                    raise FloatingPointError("Nonfinite parameters")
                progress["timings"][phase + "_update_seconds"] += (
                    time.perf_counter() - update_start)
                progress["phase_steps"][phase] += 1
                progress["epoch_loss"] += loss
                progress["epoch_valid"] += valid
                progress["cursor"] = end
                progress["count"] += 1
                if (phase == "predictor" and
                        progress["phase_steps"]["predictor"] % EVAL_UPDATE_PERIOD == 0):
                    t = progress["phase_steps"]["predictor"]
                    progress["curve"].append(regular.capture_validation(
                        objective, val_data, train, law, physical, t, epoch, True))
                    checkpoint("running", retain_step=True)
                if stop_after_updates is not None and progress["count"] >= stop_after_updates:
                    return checkpoint("checkpointed")
            if stop_requested():
                return checkpoint("checkpointed")

            observed_loss = progress["epoch_loss"] / progress["epoch_valid"]
            plateau = _plateau(progress, phase, config) if phase == "gamma" else False
            _record_round(progress, phase, epoch, observed_loss)
            progress["cursor"] = 0
            progress["epoch_loss"], progress["epoch_valid"] = 0., 0
            progress["epoch"] += 1
            if phase == "mse" and progress["phase_epochs"]["mse"] == EPOCHS:
                account_phase()
                log_stage_end("mse")
                progress["stage1_model_sha256"] = state_hash(objective.model.state_dict())
                objective.teacher.load_state_dict(objective.model.state_dict(), strict=True)
                objective.set_phase("gamma")
                optimizer = _optimizer(objective, config)
                progress.update(phase="gamma", epoch=1, best_loss=math.inf, bad_epochs=0)
            elif phase == "gamma" and (
                    plateau or progress["phase_epochs"]["gamma"] == stage_epochs["gamma"]):
                account_phase()
                log_stage_end("gamma")
                progress["stop_reasons"]["gamma"] = (
                    "training_loss_plateau" if plateau else "max_epochs")
                objective.set_phase("predictor")
                optimizer = _optimizer(objective, config)
                progress.update(phase="predictor", epoch=1,
                                best_loss=math.inf, bad_epochs=0)
                progress["curve"] = [regular.capture_validation(
                    objective, val_data, train, law, physical, 0, 0, True)]
                checkpoint("running", retain_step=True)
            elif phase == "predictor" and progress["phase_epochs"]["predictor"] == EPOCHS:
                account_phase()
                t = progress["phase_steps"]["predictor"]
                if not progress["curve"] or progress["curve"][-1]["optimizer_steps"] != t:
                    progress["curve"].append(regular.capture_validation(
                        objective, val_data, train, law, physical, t, EPOCHS, True))
                log_stage_end("predictor")
                progress["stop_reasons"]["predictor"] = "max_epochs"
                progress["phase"] = "complete"
                checkpoint("complete", retain_step=True)
                break
            else:
                checkpoint("running")
        return read_json(status_file)
    except FloatingPointError as error:
        return checkpoint("numerical_failure", error=str(error))
