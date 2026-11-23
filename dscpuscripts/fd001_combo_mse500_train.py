"""MSE(alpha=.007, exactly 500 updates) -> frozen Gamma -> Full TD.

Separated from the legacy trainer so its source SHA256 and 1600 checkpoint
identities remain unchanged. This file lives on the SAME fullTD branch.
"""
from __future__ import annotations
import hashlib
import math
from pathlib import Path
import time

import torch

from dscpuscripts.common import atomic_json, data_identity, read_json
from dscpuscripts.fd001_combo import ComboObjective, _optimizer, _plateau, _record_round
from dscpuscripts.fd001_combo_mse500_protocol import validate_config
from dscpuscripts.fd001_protocol import EPOCHS, EVAL_UPDATE_PERIOD, LAG_SETS, digest
from fulltd.sampling import LocalPairLaw, rng_for, sequence_hash, sources_for_round, training_universe
from fulltd.train import logical_backward, seed_runtime, state_hash


def code_hash():
    """Legacy code hash + NEW files only. Do not change legacy code_hash()."""
    from dscpuscripts.fd001_train import code_hash as original
    h = hashlib.sha256(original().encode("ascii"))
    root = Path(__file__).resolve().parents[1]
    for name in ("dscpuscripts/fd001_combo_mse500_protocol.py",
                 "dscpuscripts/fd001_combo_mse500_train.py"):
        h.update(name.encode())
        h.update((root / name).read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()


def build_objective(config):
    from fulltd.models import build_model
    validate_config(config)
    return ComboObjective(build_model("fd001", "tcn", config["seed"]),
                          num_types=len(config["lags"]))


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
    stage_epochs = {"mse": math.ceil(config["mse_max_updates"] / math.ceil(n / 256)),
                    "gamma": config["gamma_max_epochs"],
                    "predictor": config["predictor_max_epochs"]}
    if config["mse_max_updates"] != 500 or config["mse_alpha"] != .007 or stage_epochs["predictor"] != EPOCHS:
        raise ValueError("Stage 1 must train 500 updates at alpha .007; Stage 3 100 epochs")

    objective = build_objective(config).cpu()
    initial_hash = state_hash(objective.model.state_dict())
    identity = dict(protocol_id=config["protocol_id"], config_sha256=digest(config),
                    code_sha256=code_hash(),
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
        mse_total_updates=500,
        mse_stop_rule="fixed_optimizer_updates_not_full_epochs",
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
                      mse_max_updates=500,
                      mse_partial_final_epoch=any(
                          r.get("partial_epoch", False) for r in progress["mse_rounds"]),
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

            while (progress["cursor"] < n and
                   (phase != "mse" or progress["phase_steps"]["mse"] < 500)):
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
            partial_mse_round = phase == "mse" and progress["cursor"] < n
            _record_round(progress, phase, epoch, observed_loss)
            if partial_mse_round:
                # Precisely 500 updates: the last MSE round is only partial.
                progress["mse_rounds"][-1]["source_draws"] = progress["cursor"]
                progress["mse_rounds"][-1]["partial_epoch"] = True
            progress["cursor"] = 0
            progress["epoch_loss"], progress["epoch_valid"] = 0., 0
            progress["epoch"] += 1
            if phase == "mse" and progress["phase_steps"]["mse"] == 500:
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
