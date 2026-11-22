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


def _select_electricity_mse():
    """Validation-only selection of the already trained original MSE B256 grid."""
    from .application_analysis import (
        manifest_for, _expected_codes, _audit_candidate, _candidate_score
    )

    _, old_root, _ = old_paths("electricity")
    candidates = [c for c in manifest_for("electricity")
                  if c["method"] == "mse" and c["batch_size"] == 256]
    if len(candidates) != 50:
        raise ValueError("Expected 50 original Electricity MSE candidates")
    expected = _expected_codes(candidates, old_root)
    shared, sources, by_setting = {}, {}, defaultdict(list)
    for config in candidates:
        checked = _audit_candidate(config, old_root, expected[config["run_id"]],
                                   shared, sources)
        score, _ = _candidate_score(
            checked, checked["budgets"]["total_updates"], "full_budget")
        by_setting[config["configuration"]].append((config, score))
    choices = []
    for setting, rows in by_setting.items():
        rows.sort(key=lambda row: row[0]["seed"])
        if [c["seed"] for c, _ in rows] != list(range(5)):
            raise ValueError("Original MSE five-seed grid is incomplete")
        config = rows[0][0]
        scores = [v for _, v in rows]
        pooled = mean(scores) if all(math.isfinite(x) for x in scores) else math.inf
        choices.append((pooled, config["alpha"], setting, rows))
    if len(choices) != 10:
        raise ValueError("Original Electricity MSE learning-rate grid incomplete")
    score, alpha, setting, rows = min(choices, key=lambda x: x[:3])
    if not math.isfinite(score):
        raise ValueError("No valid five-seed original Electricity MSE setting")
    return dict(method="mse", lag_set="none", configuration=setting,
                alpha=alpha, beta=None, score=score, 
                run_ids=[c["run_id"] for c, _ in rows],
                endpoint_sha256={c["run_id"]: file_hash(old_root / c["run_id"] / "final.pt")
                                 for c, _ in rows})



def _available_electricity_mse():
    """Select the historical MSE only when its original 50-run evidence exists.

    A missing historical campaign must never block validation/test of the
    375 completed new TD candidates. Missing MSE is explicitly represented by
    None, not replaced by a made-up score or an unverified historical table.
    """
    from .application_analysis import manifest_for

    _, old_root, _ = old_paths("electricity")
    originals = [c for c in manifest_for("electricity")
                 if c["method"] == "mse" and c["batch_size"] == 256]
    if len(originals) != 50:
        raise ValueError("Expected 50 original Electricity MSE candidates")
    missing = [c["run_id"] for c in originals
               if not (old_root / c["run_id"] / "status.json").is_file()]
    if missing:
        print(f"NOTICE: historical MSE reference unavailable: {len(missing)}/50 "
              f"candidate status files absent from {old_root}. "
              "Continuing with three TD methods only; no MSE test values "
              "will be manufactured. Set FULLTD_ELECTRICITY_ROOT to the "
              "actual original MSE run directory to include that baseline.",
              flush=True)
        return None
    return _select_electricity_mse()



def select(dataset):
    from .standard_runner import classify
    if dataset not in DATASETS:
        raise ValueError("Unsupported dataset")
    frozen = _locked_five(dataset) if dataset == "fd001" else ()
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
        for lags in (LAG_SETS[dataset][1:] if dataset == "fd001" else LAG_SETS[dataset]):
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
    expected_winners = 6 if dataset == "fd001" else 3
    expected_settings = 150 if dataset == "fd001" else 75
    if len(winners) != expected_winners or len(settings) != expected_settings:
        raise ValueError("Unexpected lag validation grid size")
    plan = dict(version=VERSION, dataset=dataset, batch_size=256,
                five_lag_lock_hashes=list(frozen),
                manifest_sha256=digest(configs), code_hashes=hashes,
                criterion="five-seed pooled mean validation MSE over the final quarter",
                selected=winners)
    if dataset == "electricity":
        plan["mse_reference"] = _available_electricity_mse()
        if plan["mse_reference"] is not None:
            _, old_root, _ = old_paths(dataset)
            reference_status = read_json(
                old_root / plan["mse_reference"]["run_ids"][0] / "status.json")
            for identity_key in ("data_sha256", "validation_sha256"):
                if (reference_status.get("identity", {}).get(identity_key)
                        != shared.get(identity_key)):
                    raise ValueError(
                        f"Old MSE and new TD use different {identity_key} data identities")
    output = root / "lag579_outputs"
    output.mkdir(parents=True, exist_ok=True)
    target = output / "selection.json"
    if target.exists():
        old_plan = read_json(target)
        if old_plan != plan:
            promoted = (dataset == "electricity"
                        and old_plan.get("mse_reference") is None
                        and plan.get("mse_reference") is not None
                        and dict(old_plan, mse_reference=plan["mse_reference"]) == plan)
            if not promoted:
                raise RuntimeError(
                    "Validation selection changed; refusing to overwrite old TD locks")
            print("Added independently validation-selected MSE reference "
                  "without changing any locked TD choice.", flush=True)
    atomic_json(target, plan)
    write_csv(output / "candidate_validation_scores.csv", settings)
    print(f"Locked {len(winners)} method/lag configurations: {target}")
    return plan



class _TestProgress:
    """Print streaming progress without changing samples, metrics, or model."""
    def __init__(self, dataset, run_id):
        self.parent, self.run_id = dataset, run_id
        self.done, self.next_report = 0, 256000

    def __len__(self):
        return len(self.parent)

    def __getattr__(self, name):
        return getattr(self.parent, name)

    def get_batch(self, indices):
        data = self.parent.get_batch(indices)
        self.done += len(indices)
        if self.done >= self.next_report or self.done == len(self.parent):
            print(f"TEST {self.run_id}: {self.done}/{len(self.parent)} samples",
                  flush=True)
            self.next_report = self.done + 256000
        return data


def _electricity_test_worker(task):
    """One selected held-out endpoint, with independent CPU-only model inference."""
    import torch
    from fulltd.data import load_dataset
    from fulltd.metrics import evaluate as evaluate_model
    from .common import data_identity

    torch.set_num_threads(1)
    config = task["config"]
    run_id = config["run_id"]
    checkpoint = Path(task["root"]) / run_id / "final.pt"
    identity = task["identity"]
    if file_hash(checkpoint) != identity["endpoint_sha256"]:
        raise ValueError(f"Selected checkpoint changed: {run_id}")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if (saved["config"] != config or
            saved["identity"]["code_sha256"] != identity["code_sha256"] or
            saved["identity"]["config_sha256"] != identity["config_sha256"]):
        raise ValueError(f"Checkpoint identity differs from validation selection: {run_id}")
    dataset = load_dataset("electricity", Path(task["data"]), "test")
    if data_identity(dataset) != identity["test_sha256"]:
        raise ValueError(f"Test dataset identity changed: {run_id}")
    if config["method"] == "mse":
        from .application_train import build_objective as build_mse
        objective = build_mse(config, dataset)
    else:
        objective = build_objective(config, dataset)
    objective.load_state_dict(saved["objective"])
    objective.eval()
    measured, _ = evaluate_model(objective.model, _TestProgress(dataset, run_id),
                                 batch_size=128, device="cpu")
    keys = {tuple(pair) for pair in task["metric_keys"]}
    found = {(r["units"], r["metric"]): float(r["value"]) for r in measured
             if str(r["horizon"]) == "all" and (r["units"], r["metric"]) in keys}
    if set(found) != keys or any(not math.isfinite(v) for v in found.values()):
        raise ValueError(f"Incomplete/nonfinite selected test metrics: {run_id}")
    return sorted((dict(units=unit, metric=metric, value=value)
                   for (unit, metric), value in found.items()),
                  key=lambda r: (r["units"], r["metric"]))


def _electricity_test_tasks(tasks, output, workers):
    """Parallel full-test inference with checkpoint+test-identity-bound resume cache."""
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from multiprocessing import get_context

    if not 1 <= workers <= 32:
        raise ValueError("Electricity evaluation --workers must be between 1 and 32")
    cached, pending = {}, []
    cache_root = output / "test_cache"
    for task in tasks:
        run_id = task["config"]["run_id"]
        path = cache_root / f"{run_id}.json"
        if path.is_file():
            record = read_json(path)
            if record.get("identity") != task["identity"]:
                raise ValueError(f"Old test cache identity differs: {path}")
            rows = record.get("metrics")
            expected = set(map(tuple, task["metric_keys"]))
            if (not isinstance(rows, list)
                    or {(r["units"], r["metric"]) for r in rows} != expected
                    or len(rows) != len(expected)
                    or any(not math.isfinite(float(r["value"])) for r in rows)):
                raise ValueError(f"Malformed test cache: {path}")
            cached[run_id] = rows
        else:
            pending.append(task)
    print(f"Electricity full-test endpoints: {len(tasks)} selected models; "
          f"{len(cached)} cached; {len(pending)} to evaluate; workers={workers}.",
          flush=True)
    if pending:
        # spawn avoids forking after PyTorch has initialized runtime threads.
        with ProcessPoolExecutor(max_workers=min(workers, len(pending)),
                                 mp_context=get_context("spawn")) as pool:
            futures = {pool.submit(_electricity_test_worker, task): task
                       for task in pending}
            for future in as_completed(futures):
                task = futures[future]
                run_id = task["config"]["run_id"]
                rows = future.result()
                atomic_json(cache_root / f"{run_id}.json",
                            dict(identity=task["identity"], metrics=rows))
                cached[run_id] = rows
                print(f"Completed held-out test {len(cached)}/{len(tasks)}: {run_id}",
                      flush=True)
    return cached


def _complete_saved_electricity_td(output, selected, keys):
    """Reuse an already completed TD report, never infer missing seed metrics."""
    file = output / "per_seed.csv"
    if not file.is_file():
        return None
    existing = _read_csv(file)
    five = "-".join(map(str, LAG_SETS["electricity"][0]))
    desired = {(method, five, seed, unit, metric)
               for method in METHODS for seed in range(5) for unit, metric in keys}
    actual = {}
    for row in existing:
        if row["method"] not in METHODS:
            continue
        key = (row["method"], row["lag_set"], int(row["seed"]),
               row["units"], row["metric"])
        if (key in actual or row["dataset"] != "electricity"
                or int(row["batch_size"]) != 256 or row["source"] != "new_5_lag"
                or not math.isfinite(float(row["value"]))):
            raise ValueError("Existing TD test report cannot be safely reused")
        actual[key] = float(row["value"])
    if set(actual) != desired:
        return None
    for choice in selected:
        if (choice["method"] not in METHODS or choice["lag_set"] != five
                or len(choice["run_ids"]) != 5):
            raise ValueError("Saved TD selections changed")
    print("Reusing complete previously generated Electricity TD per-seed test report.",
          flush=True)
    return [dict(dataset="electricity", method=m, lag_set=lag, batch_size=256,
                 seed=seed, units=u, metric=k, value=value, source="new_5_lag")
            for (m, lag, seed, u, k), value in sorted(actual.items())]



def evaluate(dataset, workers=4):
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
            or plan.get("five_lag_lock_hashes") != list(_locked_five(dataset) if dataset == "fd001" else ())
            or plan.get("code_hashes") != {
                m: code_hash(next(c for c in configs.values() if c["method"] == m))
                for m in METHODS}):
        raise ValueError("Saved validation selection is no longer compatible")
    keys = _metric_keys(dataset)
    prior = []
    if dataset == "fd001":
        _, old_ablation_root = old_ablation_paths(dataset)
        previous_file = (old_ablation_root / "typed_ablation_outputs" /
                         "four_method_per_seed.csv")
        if not previous_file.is_file():
            raise FileNotFoundError(
                f"Five-lag test summary missing: {previous_file}. "
                "Complete the earlier four-method comparison first.")
        five = "-".join(map(str, LAG_SETS[dataset][0]))
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
            previous_groups[(row["method"], row["lag_set"],
                             row["units"], row["metric"])].append(row)
        if (len(previous_groups) != 4 * len(keys)
                or any(sorted(r["seed"] for r in rows) != list(range(5))
                       for rows in previous_groups.values())):
            raise ValueError("Previous FD001 five-lag test summary is incomplete")
        if any(not math.isfinite(r["value"]) for r in prior):
            raise ValueError("Previous FD001 test summary contains nonfinite metrics")
    else:
        if plan.get("mse_reference") is not None:
            if plan["mse_reference"] != _select_electricity_mse():
                raise ValueError("Original MSE selection/checkpoint evidence changed")
        else:
            print("NOTICE: MSE is absent from the locked comparison. "
                  "Testing only the three new Electricity TD methods.",
                  flush=True)

    verify_dataset(dataset, data)
    torch.set_num_threads(1)
    test = load_dataset(dataset, data, "test")
    fresh = []
    expected_selections = 6 if dataset == "fd001" else 3
    if len(plan["selected"]) != expected_selections:
        raise ValueError("Unexpected number of validation-locked TD selections")
    if dataset == "fd001":
        # Preserve the original FD001 scientific endpoint and comparison protocol.
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
                found = {(r["units"], r["metric"]): float(r["value"]) for r in measured
                         if str(r["horizon"]) == "all" and
                         (r["units"], r["metric"]) in keys}
                if set(found) != keys or any(not math.isfinite(v) for v in found.values()):
                    raise ValueError(f"Missing/nonfinite test metrics: {run_id}")
                for (unit, metric), value in found.items():
                    fresh.append(dict(dataset=dataset, method=method, lag_set=lag,
                                      batch_size=256, seed=c["seed"], units=unit,
                                      metric=metric, value=value, source="new_7_9_lag"))
    else:
        from .common import data_identity
        existing_td = _complete_saved_electricity_td(output, plan["selected"], keys)
        test_sha = data_identity(test)
        metrics_sha = file_hash(Path(__file__).resolve().parents[1] / "fulltd/metrics.py")
        tasks = []

        def register(config, directory, code_sha, endpoint_sha, source, lag):
            record = dict(
                run_id=config["run_id"], config_sha256=digest(config),
                endpoint_sha256=endpoint_sha, code_sha256=code_sha,
                test_sha256=test_sha, metrics_code_sha256=metrics_sha,
                evaluation_batch_size=128)
            tasks.append(dict(config=config, data=str(data), root=str(directory),
                              identity=record, source=source, lag_set=lag,
                              metric_keys=sorted(keys)))

        # MSE is independently retrained and selected by validation; no TD
        # checkpoint or test result is reused to choose its hyperparameters.
        if plan.get("mse_reference") is not None:
            from .application_protocol import make_manifest as original_manifest
            _, mse_root, _ = old_paths(dataset)
            originals = {c["run_id"]: c for c in original_manifest()
                         if c["dataset"] == dataset and c["method"] == "mse"
                         and c["batch_size"] == 256}
            choice = plan["mse_reference"]
            if len(choice["run_ids"]) != 5:
                raise ValueError("Incomplete original MSE five-seed selection")
            for run_id in choice["run_ids"]:
                config = originals[run_id]
                status = read_json(mse_root / run_id / "status.json")
                register(config, mse_root, status["identity"]["code_sha256"],
                         choice["endpoint_sha256"][run_id], "fresh_mse", "none")

        if existing_td is not None:
            fresh.extend(existing_td)
        else:
            for choice in plan["selected"]:
                method, lag = choice["method"], choice["lag_set"]
                if method not in METHODS or len(choice["run_ids"]) != 5:
                    raise ValueError("Unexpected selected TD method or seed count")
                seen = set()
                for run_id in choice["run_ids"]:
                    config = configs[run_id]
                    if (config["method"] != method or key_of(config) != lag
                            or config["seed"] in seen):
                        raise ValueError(f"Selected TD configuration mismatch: {run_id}")
                    seen.add(config["seed"])
                    register(config, root, plan["code_hashes"][method],
                             choice["endpoint_sha256"][run_id], "new_5_lag", lag)

        # A recorded cache survives interruption, and a fully completed old TD
        # report avoids repeating the expensive 15 full-test inferences.
        results = _electricity_test_tasks(tasks, output, workers)
        for task in tasks:
            c = task["config"]
            for record in results[c["run_id"]]:
                row = dict(dataset=dataset, method=c["method"],
                           lag_set=task["lag_set"], batch_size=256,
                           seed=c["seed"], units=record["units"],
                           metric=record["metric"], value=float(record["value"]),
                           source=task["source"])
                if c["method"] == "mse":
                    prior.append(row)
                else:
                    fresh.append(row)
    combined = prior + fresh
    grouped = defaultdict(list)
    for row in combined:
        grouped[(row["method"],row["lag_set"],row["units"],row["metric"])].append(row)
    expected_groups = (10 if dataset == "fd001" else
                       4 if plan.get("mse_reference") is not None else 3)
    if (len(grouped) != expected_groups * len(keys)
            or any(sorted(r["seed"] for r in rows) != list(range(5))
                   for rows in grouped.values())):
        raise ValueError("Incomplete TD/MSE comparison summary")
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
                if comparator == "mse" and dataset == "electricity" and plan.get("mse_reference") is None:
                    continue
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
                if dataset != "electricity":
                    raise RuntimeError(f"Existing analysis differs; refusing overwrite: {target}")
                def canonical(row):
                    return tuple(sorted(row.items()))
                old_rows = [canonical(row) for row in existing]
                new_rows = [canonical(row) for row in expected]
                if len(set(old_rows)) != len(old_rows) or len(set(new_rows)) != len(new_rows):
                    raise RuntimeError(f"Duplicate analysis rows: {target}")
                if set(old_rows) == set(new_rows):
                    continue
                old_set = set(old_rows)
                new_set = set(new_rows)
                added = [row for row in expected if canonical(row) not in old_set]
                if (plan.get("mse_reference") is None
                        or not old_set.issubset(new_set) or not added):
                    raise RuntimeError(f"Non-append-only test report change: {target}")
                # Only the independently selected newly trained MSE may be
                # appended to the old TD-only report, never reselect TD.
                valid_append = all(
                    (row.get("method") == "mse" if filename != "paired.csv"
                     else row.get("method_b") == "mse")
                    for row in added)
                if not valid_append:
                    raise RuntimeError(f"Existing TD outcome changed: {target}")
                write_csv(target, rows)
                print(f"Added {len(added)} independently trained MSE rows: {target}",
                      flush=True)
        else:
            write_csv(target, rows)
    print(f"Results saved: {output}")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=DATASETS)
    parser.add_argument("--workers", type=int, default=4,
                        help="Electricity held-out test inference processes (1..32)")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--select",action="store_true")
    action.add_argument("--evaluate",action="store_true")
    args = parser.parse_args()
    select(args.dataset) if args.select else evaluate(args.dataset, workers=args.workers)


if __name__=="__main__":
    main()
