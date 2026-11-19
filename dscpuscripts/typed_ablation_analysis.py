"""Validation-only selection and held-out endpoint metrics for new typed ablations.

This does NOT select or rewrite historical MSE / Full TD results. Use their
matching-lag, B256 selected outputs as comparison references.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import math
from pathlib import Path

from .common import atomic_json, file_hash, read_json, write_csv
from .typed_ablation_protocol import DATASETS, METHODS, make_manifest
from .typed_ablation_runner import paths
from .typed_ablation_train import code_hash, build_objective


def _digest(value):
    from fulltd.protocol import digest
    return digest(value)



def _verify_completed_record(config, folder, status, curve):
    """Reject incomplete endpoints before using validation scores for selection."""
    if status.get("state") != "complete":
        raise ValueError("Only complete candidates can be scored")
    identity = status.get("identity", {})
    if (identity.get("protocol_id") != config["protocol_id"]
            or identity.get("config_sha256") != _digest(config)):
        raise ValueError(f"Checkpoint protocol/config mismatch: {config['run_id']}")
    epochs = config["epochs"]
    n = status.get("n_sources")
    if type(n) is not int or n <= 0:
        raise ValueError(f"Invalid source count: {config['run_id']}")
    if config["dataset"] == "fd001":
        per_epoch = math.ceil(n / config["batch_size"])
        total = epochs * per_epoch
        period = config["learning_curve_updates"]
        schedule = [0, *range(period, total + 1, period)]
        expected_opportunities = total
    else:
        from .application_protocol import update_budgets, evaluation_steps
        budget = update_budgets(config, n)
        total = budget["total_updates"]
        schedule = list(evaluation_steps(total, config["learning_curve_updates"]))
        expected_opportunities = total + int(status.get("empty_mask_batches", 0))
    if (status.get("epochs_completed") != epochs
            or status.get("next_epoch") != epochs + 1
            or status.get("optimizer_steps") != total
            or status.get("update_opportunities") != expected_opportunities
            or not (folder / "final.pt").is_file()):
        raise ValueError(f"Incomplete final training budget/checkpoint: {config['run_id']}")
    actual = [r.get("optimizer_steps") for r in curve]
    if actual != schedule:
        raise ValueError(f"Missing, duplicated or truncated validation history: {config['run_id']}")
    if any(not math.isfinite(float(r["mse_standardized"]))
           or float(r["mse_standardized"]) < 0 for r in curve):
        raise ValueError(f"Invalid validation MSE: {config['run_id']}")
    if not (folder / "source_universe.json").is_file():
        raise ValueError(f"Missing source-universe evidence: {config['run_id']}")
    rounds_path = folder / "rounds.json"
    if not rounds_path.is_file():
        raise ValueError(f"Missing training-round evidence: {config['run_id']}")
    rounds = read_json(rounds_path)
    if (len(rounds) != epochs or [r.get("epoch") for r in rounds] != list(range(1, epochs + 1))
            or rounds[-1].get("optimizer_steps_cumulative") != total):
        raise ValueError(f"Missing or incomplete training rounds: {config['run_id']}")
    for step in schedule:
        if step == total:
            continue   # final.pt is the authoritative endpoint snapshot
        if not (folder / "checkpoints" / f"step_{step:07d}.pt").is_file():
            raise ValueError(f"Missing intermediate checkpoint {step}: {config['run_id']}")


def _final_quarter(config, curve):
    """Exactly the original rule: mean noisy validation MSE, last 25%."""
    points = [r for r in curve if int(r["optimizer_steps"]) > 0]
    if not points or any(not math.isfinite(float(r["mse_standardized"])) for r in points):
        return math.inf
    points.sort(key=lambda r: r["optimizer_steps"])
    length = math.ceil(len(points) * .25)
    return sum(float(p["mse_standardized"]) for p in points[-length:]) / length


def select(dataset):
    from .standard_runner import classify
    configs = make_manifest(dataset)
    _, root = paths(dataset)
    hash_by_method = {m: code_hash(next(c for c in configs if c["method"] == m))
                      for m in METHODS}
    by_setting = defaultdict(list)
    for config in configs:
        key = (config["method"], tuple(config["lags"]), config["configuration"])
        state = classify(config, root, hash_by_method[config["method"]])
        if state not in ("complete", "numerical_failure"):
            raise RuntimeError(f"Cannot select before all candidates terminate: {config['run_id']}")
        if state == "numerical_failure":
            by_setting[key].append((config, math.inf))
            continue
        status = read_json(root / config["run_id"] / "status.json")
        if status["config"] != config or status["identity"]["code_sha256"] != hash_by_method[config["method"]]:
            raise ValueError(f"Incompatible results: {config['run_id']}")
        folder = root / config["run_id"]
        curve = read_json(folder / "update_curve.json")
        _verify_completed_record(config, folder, status, curve)
        by_setting[key].append((config, _final_quarter(config, curve)))
    winners, score_rows = [], []
    lag_sets = list(dict.fromkeys(tuple(c["lags"]) for c in configs))
    for method in METHODS:
        for lag_set in lag_sets:
            choices = []
            for (kind, lags, setting), records in by_setting.items():
                if kind != method or lags != lag_set:
                    continue
                seeds = sorted(c["seed"] for c, _ in records)
                if seeds != list(range(5)):
                    raise ValueError(f"Five seed records required for {method}, setting={setting}")
                scores = [score for _, score in records]
                score = sum(scores) / 5 if all(math.isfinite(x) for x in scores) else math.inf
                c = records[0][0]
                score_rows.append(dict(method=method, lag_set="-".join(map(str, lag_set)), configuration=setting,
                                       alpha=c["alpha"], beta=c["beta"], score=score))
                if math.isfinite(score):
                    choices.append((score, setting, c))
            if not choices:
                raise RuntimeError(f"No valid full-budget configuration for {method}/{lag_set}")
            _, _, chosen = min(choices, key=lambda x: (x[0], x[1]))
            run_ids = [next(c["run_id"] for c in configs
                            if c["method"] == method and tuple(c["lags"]) == lag_set
                            and c["configuration"] == chosen["configuration"]
                            and c["seed"] == seed) for seed in range(5)]
            winners.append(dict(method=method, lag_set="-".join(map(str, lag_set)),
                                 configuration=chosen["configuration"],
                                alpha=chosen["alpha"], beta=chosen["beta"],
                                run_ids=run_ids,
                                endpoint_sha256={run_id: file_hash(root / run_id / "final.pt")
                                                 for run_id in run_ids}))
    plan = dict(dataset=dataset, method_scope=list(METHODS),
                 lag_sets=["-".join(map(str, x)) for x in lag_sets], batch_size=256,
                selection="last 25% of positive-update noisy validation MSE, pooled across five seeds",
                code_by_method=hash_by_method, manifest_sha256=_digest(configs),
                selected=winners)
    out = root / "typed_ablation_outputs"
    out.mkdir(parents=True, exist_ok=True)
    target = out / "selection.json"
    if target.exists() and read_json(target) != plan:
        raise RuntimeError("Previous validation selection differs; do not overwrite")
    atomic_json(target, plan)
    write_csv(out / "candidate_validation_scores.csv", sorted(score_rows,
              key=lambda x: (x["method"], x["lag_set"], x["configuration"])))
    return plan


def evaluate(dataset):
    """Open held-out test data only after the locked validation plan exists."""
    import torch
    from fulltd.data import load_dataset, verify_dataset
    from fulltd.metrics import evaluate as evaluate_model
    configs = make_manifest(dataset)
    by_id = {c["run_id"]: c for c in configs}
    data, root = paths(dataset)
    output = root / "typed_ablation_outputs"
    plan = read_json(output / "selection.json")
    if (plan["manifest_sha256"] != _digest(configs)
            or plan["code_by_method"] != {m: code_hash(next(c for c in configs if c["method"] == m))
                                                for m in METHODS}):
        raise ValueError("Locked ablation selection belongs to different code/configs")
    torch.set_num_threads(1)
    verify_dataset(dataset, data)
    test = load_dataset(dataset, data, "test")
    result = []
    for selected in plan["selected"]:
        for run_id in selected["run_ids"]:
            c = by_id[run_id]
            checkpoint_path = root / run_id / "final.pt"
            if file_hash(checkpoint_path) != selected["endpoint_sha256"][run_id]:
                raise ValueError("A selected checkpoint changed since validation selection")
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if checkpoint["config"] != c or checkpoint["identity"]["code_sha256"] != plan["code_by_method"][c["method"]]:
                raise ValueError("Selected checkpoint does not match validation lock")
            # Loading the original shared mean architecture; operator does not
            # participate in held-out point prediction.
            obj = build_objective(c, test)
            obj.load_state_dict(checkpoint["objective"])
            obj.eval()
            metrics, _ = evaluate_model(obj.model, test, batch_size=128, device="cpu")
            for row in metrics:
                result.append(dict(dataset=dataset, method=c["method"],
                                   lag_set="-".join(map(str, c["lags"])), seed=c["seed"],
                                   alpha=c["alpha"], beta=c["beta"],
                                   horizon=row["horizon"], units=row["units"],
                                   metric=row["metric"], value=row["value"]))
    filename = output / "test_per_seed_metrics.csv"
    # Results are append-only in meaning; rerun is permitted only when identical.
    if filename.exists():
        with filename.open(newline="", encoding="utf8") as f:
            earlier = list(csv.DictReader(f))
        if len(earlier) != len(result) or any(
                str(old[k]) != str(new[k]) for old, new in zip(earlier, result)
                for k in new):
            raise RuntimeError("Existing held-out report differs; do not overwrite")
    else:
        write_csv(filename, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=DATASETS)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--select", action="store_true")
    group.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()
    if args.select:
        plan = select(args.dataset)
        print(f"Selection locked for {args.dataset}: {plan['selected']}")
    else:
        rows = evaluate(args.dataset)
        print(f"Held-out endpoint metrics: {len(rows)} values for {args.dataset}")


if __name__ == "__main__":
    main()
