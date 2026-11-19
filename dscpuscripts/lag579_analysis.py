"""Validation-locked 5/7/9 comparisons using existing 5-lag checkpoints."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import math
from pathlib import Path
from statistics import mean, stdev

from .common import atomic_json, file_hash, read_json, write_csv
from .lag579_protocol import DATASETS, METHODS, LAG_SETS, make_manifest
from .lag579_runner import paths
from .lag579_train import code_hash, build_objective
from .typed_ablation_runner import paths as old_ablation_paths
from .typed_ablation_analysis import _verify_completed_record, _final_quarter
from .result_paths import paths as old_paths

VERSION = "lag579-analysis-v1"


def digest(value):
    from fulltd.protocol import digest as current
    return current(value)


def key_of(config):
    return "-".join(str(x) for x in config["lags"])


def _read_csv(path):
    with Path(path).open(newline="", encoding="utf8") as handle:
        return list(csv.DictReader(handle))


def _metric_keys(dataset):
    if dataset == "fd001":
        return {("cycles", "rmse"), ("cycles", "mae"), ("score", "rul_score_sum")}
    return {("kW", "rmse"), ("kW", "mae")}


def _locked_five(dataset):
    """Identify the independent old/new five-lag validation selections."""
    _, old_root, _ = old_paths(dataset)
    _, ablation_root = old_ablation_paths(dataset)
    base = ablation_root / "typed_ablation_outputs"
    combined_path = base / "four_method_selection.json"
    selected_path = base / "selection.json"
    for p in (combined_path, selected_path):
        if not p.is_file():
            raise FileNotFoundError(
                f"Five-lag comparison not locked: {p}. "
                f"Run the previous typed-ablation selection/reporting workflow first.")
    old = read_json(combined_path)
    recent = read_json(selected_path)
    if (old.get("dataset") != dataset or recent.get("dataset") != dataset
            or old.get("batch_size") != 256 or recent.get("batch_size") != 256):
        raise ValueError("Five-lag selections are for the wrong campaign")
    five = "-".join(map(str, LAG_SETS[dataset][0]))
    selected = {}
    for c in old["old_selections"]:
        if c["method"] == "td_typed" and c["lag_set"] == five:
            selected[c["method"]] = c
    for c in recent["selected"]:
        if c["method"] in METHODS[1:] and c["lag_set"] == five:
            selected[c["method"]] = c
    if set(selected) != set(METHODS):
        raise ValueError("Five-lag three-method selection is incomplete")
    for method, choice in selected.items():
        if len(choice["run_ids"]) != 5:
            raise ValueError(f"Five-lag selection missing seeds: {method}")
        directory = old_root if method == "td_typed" else ablation_root
        for run_id in choice["run_ids"]:
            if file_hash(directory / run_id / "final.pt") != choice["endpoint_sha256"][run_id]:
                raise ValueError(f"Five-lag checkpoint changed: {run_id}")
    return (file_hash(combined_path), file_hash(selected_path))


def _audit_fd001_candidate(config, folder, status, shared, source_hashes):
    """Cross-check the frozen FD001 source stream and equal-seed comparisons."""
    from fulltd.sampling import sequence_hash, sources_for_round

    run_id = config["run_id"]
    identity = status["identity"]
    for field in ("data_sha256", "validation_sha256"):
        value = identity.get(field)
        if not value or (field in shared and shared[field] != value):
            raise ValueError(f"FD001 data identity mismatch: {run_id}/{field}")
        shared[field] = value
    seed = config["seed"]
    initial = status.get("initial_mean_sha256")
    key = ("initial_mean", seed)
    if not initial or (key in shared and shared[key] != initial):
        raise ValueError(f"FD001 predictor initialization differs: {run_id}")
    shared[key] = initial
    n = status["n_sources"]
    if "n_sources" in shared and shared["n_sources"] != n:
        raise ValueError(f"FD001 source count differs: {run_id}")
    shared["n_sources"] = n
    per_epoch = math.ceil(n / config["batch_size"])
    universe = read_json(folder / "source_universe.json")
    if (universe.get("n_sources") != n
            or universe.get("updates_per_epoch") != per_epoch
            or universe.get("total_updates") != per_epoch * config["epochs"]
            or universe.get("evaluation_period") != config["learning_curve_updates"]):
        raise ValueError(f"FD001 source universe/budget mismatch: {run_id}")
    excluded = universe.get("isolated_excluded")
    if excluded is None or ("isolated_excluded" in shared
                            and shared["isolated_excluded"] != excluded):
        raise ValueError(f"FD001 training sources differ: {run_id}")
    shared["isolated_excluded"] = excluded
    rounds = read_json(folder / "rounds.json")
    for row in rounds:
        epoch = row["epoch"]
        source_key = (seed, epoch)
        if source_key not in source_hashes:
            source_hashes[source_key] = sequence_hash(
                sources_for_round("fd001", n, seed, epoch))
        if (row.get("source_draws") != n
                or row.get("source_sha256") != source_hashes[source_key]
                or row.get("optimizer_steps_cumulative") != epoch * per_epoch):
            raise ValueError(f"FD001 source/update mismatch: {run_id}/epoch={epoch}")
        types = row.get("type_counts")
        if (not isinstance(types, list) or len(types) != len(config["lags"])
                or any(type(v) is not int or v < 0 for v in types)
                or sum(types) != n):
            raise ValueError(f"FD001 lag-pair counts mismatch: {run_id}/epoch={epoch}")
        partner_hash = row.get("partner_sha256")
        if not isinstance(partner_hash, str) or len(partner_hash) != 64:
            raise ValueError(f"FD001 partner-pair hash missing: {run_id}/epoch={epoch}")
        partner_key = ("partner", tuple(config["lags"]), seed, epoch)
        if partner_key in shared and shared[partner_key] != partner_hash:
            raise ValueError(f"FD001 paired source/partner sequence differs: {run_id}/epoch={epoch}")
        shared[partner_key] = partner_hash


def select(dataset):
    from .standard_runner import classify
    if dataset not in DATASETS:
        raise ValueError("Unsupported dataset")
    frozen = _locked_five(dataset)
    configs = make_manifest(dataset)
    _, root = paths(dataset)
    hashes = {m: code_hash(next(c for c in configs if c["method"] == m)) for m in METHODS}
    grouped = defaultdict(list)
    shared, source_hashes = {}, {}
    for c in configs:
        state = classify(c, root, hashes[c["method"]])
        if state not in ("complete", "numerical_failure"):
            raise RuntimeError(f"Training incomplete: {c['run_id']} ({state})")
        score = math.inf
        if state == "complete":
            folder = root / c["run_id"]
            status = read_json(folder / "status.json")
            curve = read_json(folder / "update_curve.json")
            _verify_completed_record(c, folder, status, curve)
            if c["dataset"] == "fd001":
                _audit_fd001_candidate(c, folder, status, shared, source_hashes)
            if c["dataset"] == "electricity":
                from .application_analysis import _audit_candidate
                _audit_candidate(c, root, hashes[c["method"]], shared, source_hashes)
            score = _final_quarter(c, curve)
        grouped[(c["method"], key_of(c), c["configuration"])].append((c, score))
    settings, winners = [], []
    for method in METHODS:
        for lags in LAG_SETS[dataset][1:]:
            lag = "-".join(map(str, lags))
            choices = []
            for (m, lag_key, setting), rows in grouped.items():
                if m != method or lag_key != lag:
                    continue
                rows = sorted(rows, key=lambda item: item[0]["seed"])
                if [c["seed"] for c, _ in rows] != list(range(5)):
                    raise ValueError(f"Missing seeds: {method}/{lag}/{setting}")
                scores = [s for _, s in rows]
                pooled = mean(scores) if all(math.isfinite(s) for s in scores) else math.inf
                c = rows[0][0]
                settings.append(dict(method=method, lag_set=lag, configuration=setting,
                                     alpha=c["alpha"], beta=c["beta"], score=pooled))
                if math.isfinite(pooled):
                    choices.append((pooled, setting, rows))
            if not choices:
                raise ValueError(f"No complete, finite five-seed grid: {method}/{lag}")
            _, setting, rows = min(choices, key=lambda x: (x[0], x[1]))
            run_ids = [c["run_id"] for c, _ in rows]
            winners.append(dict(method=method, lag_set=lag, configuration=setting,
                                alpha=rows[0][0]["alpha"], beta=rows[0][0]["beta"],
                                run_ids=run_ids,
                                endpoint_sha256={rid: file_hash(root / rid / "final.pt")
                                                 for rid in run_ids}))
    if len(winners) != 6 or len(settings) != 150:
        raise ValueError("Unexpected 7/9-lag validation grid size")
    plan = dict(version=VERSION, dataset=dataset, batch_size=256,
                five_lag_lock_hashes=list(frozen),
                manifest_sha256=digest(configs), code_hashes=hashes,
                criterion="five-seed pooled mean validation MSE over the final quarter",
                selected=winners)
    output = root / "lag579_outputs"
    output.mkdir(parents=True, exist_ok=True)
    target = output / "selection.json"
    if target.exists() and read_json(target) != plan:
        raise RuntimeError("Selection changed; refusing to overwrite old validation lock")
    atomic_json(target, plan)
    write_csv(output / "candidate_validation_scores.csv", settings)
    print(f"Locked {len(winners)} new 7/9-lag configurations: {target}")
    return plan


def evaluate(dataset):
    """Held-out evaluation is performed only after validation selection is locked."""
    import torch
    from fulltd.data import load_dataset, verify_dataset
    from fulltd.metrics import evaluate as evaluate_model
    configs = {c["run_id"]: c for c in make_manifest(dataset)}
    data, root = paths(dataset)
    output = root / "lag579_outputs"
    plan = read_json(output / "selection.json")
    if (plan.get("version") != VERSION or plan.get("dataset") != dataset
            or plan.get("manifest_sha256") != digest(list(configs.values()))
            or plan.get("five_lag_lock_hashes") != list(_locked_five(dataset))
            or plan.get("code_hashes") != {
                m: code_hash(next(c for c in configs.values() if c["method"] == m))
                for m in METHODS}):
        raise ValueError("Saved validation selection is no longer compatible")
    _, old_ablation_root = old_ablation_paths(dataset)
    previous_file = (old_ablation_root / "typed_ablation_outputs" /
                     "four_method_per_seed.csv")
    if not previous_file.is_file():
        raise FileNotFoundError(
            f"Five-lag test summary missing: {previous_file}. "
            "Complete the earlier four-method comparison first.")
    five = "-".join(map(str, LAG_SETS[dataset][0]))
    keys = _metric_keys(dataset)
    prior = []
    for r in _read_csv(previous_file):
        method, lag = r["method"], r["lag_set"]
        if not (method == "mse" and lag == "none" or
                method in METHODS and lag == five):
            continue
        if (int(r["batch_size"]) != 256 or (r["units"], r["metric"]) not in keys):
            continue
        prior.append(dict(dataset=dataset, method=method, lag_set=lag,
                          batch_size=256, seed=int(r["seed"]),
                          units=r["units"], metric=r["metric"],
                          value=float(r["value"]), source="reused_5_lag"))
    previous_groups = defaultdict(list)
    for row in prior:
        previous_groups[(row["method"],row["lag_set"],row["units"],row["metric"])].append(row)
    if (len(previous_groups) != 4 * len(keys) or
            any(sorted(r["seed"] for r in rows) != list(range(5))
                for rows in previous_groups.values())):
        raise ValueError("Previous five-lag test summary is incomplete")
    if any(not math.isfinite(r["value"]) for r in prior):
        raise ValueError("Previous test summary has nonfinite metrics")

    verify_dataset(dataset, data)
    torch.set_num_threads(1)
    test = load_dataset(dataset, data, "test")
    fresh = []
    if len(plan["selected"]) != 6:
        raise ValueError("Expected six validation-locked 7/9-lag settings")
    for choice in plan["selected"]:
        method, lag = choice["method"], choice["lag_set"]
        if method not in METHODS or len(choice["run_ids"]) != 5:
            raise ValueError("Unexpected selected method or seed count")
        seen = set()
        for run_id in choice["run_ids"]:
            c = configs[run_id]
            if c["method"] != method or key_of(c) != lag or c["seed"] in seen:
                raise ValueError("Selected run identity does not match lag/method/seed")
            seen.add(c["seed"])
            ckpt_path = root / run_id / "final.pt"
            if file_hash(ckpt_path) != choice["endpoint_sha256"][run_id]:
                raise ValueError(f"Selected checkpoint changed: {run_id}")
            saved = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            if (saved["config"] != c or
                    saved["identity"]["code_sha256"] != plan["code_hashes"][method]):
                raise ValueError(f"Selected checkpoint does not match lock: {run_id}")
            objective = build_objective(c, test)
            objective.load_state_dict(saved["objective"])
            objective.eval()
            measured, _ = evaluate_model(objective.model, test, batch_size=128, device="cpu")
            found = {(r["units"],r["metric"]): float(r["value"]) for r in measured
                     if str(r["horizon"]) == "all" and
                     (r["units"],r["metric"]) in keys}
            if set(found) != keys or any(not math.isfinite(v) for v in found.values()):
                raise ValueError(f"Missing/nonfinite test endpoint metrics: {run_id}")
            for (unit, metric), value in found.items():
                fresh.append(dict(dataset=dataset, method=method, lag_set=lag,
                                  batch_size=256, seed=c["seed"], units=unit,
                                  metric=metric, value=value, source="new_7_9_lag"))
    combined = prior + fresh
    grouped = defaultdict(list)
    for row in combined:
        grouped[(row["method"],row["lag_set"],row["units"],row["metric"])].append(row)
    if (len(grouped) != 10 * len(keys)
            or any(sorted(r["seed"] for r in rows) != list(range(5))
                   for rows in grouped.values())):
        raise ValueError("Incomplete 5/7/9 summary")
    summary = []
    for (method, lag, unit, metric), rows in sorted(grouped.items()):
        values = [r["value"] for r in sorted(rows,key=lambda r:r["seed"])]
        summary.append(dict(dataset=dataset, method=method, lag_set=lag,
                            units=unit, metric=metric, n=5, mean=mean(values),
                            standard_error=stdev(values) / (5 ** 0.5)))
    paired = []
    for lags in LAG_SETS[dataset]:
        lag = "-".join(map(str,lags))
        for unit, metric in sorted(keys):
            full = sorted(grouped[("td_typed",lag,unit,metric)],key=lambda r:r["seed"])
            for comparator in ("mse","td_semi_typed","td_nonnegative_typed"):
                other_lag = "none" if comparator=="mse" else lag
                other = sorted(grouped[(comparator,other_lag,unit,metric)],key=lambda r:r["seed"])
                differences = [a["value"]-b["value"] for a,b in zip(full,other)]
                paired.append(dict(dataset=dataset,lag_set=lag,units=unit,metric=metric,
                                   method_a="td_typed",method_b=comparator,n=5,
                                   mean_difference=mean(differences),
                                   standard_error=stdev(differences)/(5**0.5),
                                   wins=sum(x<0 for x in differences)))
    for filename, rows in (
        ("per_seed.csv",combined),
        ("summary.csv",summary),
        ("paired.csv",paired),
    ):
        target = output / filename
        if target.exists():
            existing = _read_csv(target)
            expected = [{k:str(v) for k,v in row.items()} for row in rows]
            if existing != expected:
                raise RuntimeError(f"Existing analysis differs; refusing overwrite: {target}")
        else:
            write_csv(target,rows)
    print(f"Results saved: {output}")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=DATASETS)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--select",action="store_true")
    action.add_argument("--evaluate",action="store_true")
    args = parser.parse_args()
    select(args.dataset) if args.select else evaluate(args.dataset)


if __name__=="__main__":
    main()
