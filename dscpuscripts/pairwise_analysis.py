"""Validation-locked, reproducible Pairwise TD neural application comparisons.

Two batch sizes are reported separately. We reuse original MSE/Gaussian NLL
training runs and independently train Pairwise TD without altering the older
experiment protocol. Held-out test is read only after all validation choices
are frozen. Per-run inference caches are checkpoint/test-data bound.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import math
from multiprocessing import get_context
from pathlib import Path
from statistics import mean, stdev

from .common import atomic_json, data_identity, file_hash, read_json, write_csv
from .pairwise_protocol import DATASETS, LAG_SETS, BATCH_SIZES, make_manifest
from .pairwise_runner import paths as pairwise_paths, _scoped_codes
from .result_paths import paths as baseline_paths
from .application_protocol import make_manifest as application_manifest
from .application_analysis import (
    _audit_candidate, _candidate_score, _expected_codes,
)
from fulltd.protocol import digest

VERSION = "pairwise-neural-analysis-v1"
BASELINES = ("mse", "gaussian_nll")
METHOD = "td_pairwise"
METRICS = ("mse", "rmse", "mae")


def _csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _lag(config):
    return "-".join(map(str, config["lags"])) if config["method"] == METHOD else "none"


def _baseline_configs(dataset):
    configs = [c for c in application_manifest()
               if c["dataset"] == dataset and c["method"] in BASELINES
               and c["batch_size"] == 256]
    if (len(configs) != 100
            or {c["method"] for c in configs} != set(BASELINES)):
        raise ValueError("Expected 50 MSE and 50 Gaussian NLL candidates")
    return configs


def _selected_groups(dataset):
    return {(m, "none", 256) for m in BASELINES} | {
        (METHOD, "-".join(map(str, lags)), batch)
        for lags in LAG_SETS[dataset] for batch in BATCH_SIZES}


def select(dataset):
    if dataset not in DATASETS:
        raise ValueError("Unsupported dataset")
    old = _baseline_configs(dataset)
    pair = list(make_manifest(dataset))
    data, new_root = pairwise_paths(dataset)
    old_data, old_root, _ = baseline_paths(dataset)
    if data != old_data:
        raise ValueError("Old and new results point to different prepared data")
    old_missing = [c for c in old if not (old_root / c["run_id"] / "status.json").is_file()]
    new_missing = [c for c in pair if not (new_root / c["run_id"] / "status.json").is_file()]
    if old_missing or new_missing:
        baseline_missing = {m: sum(c["method"] == m for c in old_missing)
                            for m in BASELINES}
        raise RuntimeError(
            f"Training is not finished: MSE={baseline_missing['mse']}, "
            f"Gaussian NLL={baseline_missing['gaussian_nll']}, "
            f"Pairwise TD={len(new_missing)} missing status files. "
            "Check the original baseline launchers and "
            "bash dscpuscripts/run_pairwise.sh DATASET --status.")
    old_codes = _expected_codes(old, old_root)
    pair_codes = _scoped_codes(pair)
    shared, source_hashes = {}, {}
    grouped = defaultdict(list)
    # Original baselines are audited first, then the new TD models; the same
    # shared dictionary verifies source/data/initialization consistency.
    for config in old + pair:
        is_pair = config["method"] == METHOD
        folder = new_root if is_pair else old_root
        code = pair_codes[config["run_id"]] if is_pair else old_codes[config["run_id"]]
        result = _audit_candidate(config, folder, code, shared, source_hashes)
        score, window = _candidate_score(result, result["budgets"]["total_updates"],
                                         "full_budget")
        key = (config["method"], _lag(config), config["batch_size"], config["configuration"])
        grouped[key].append((config, score))
    choices, candidates = [], []
    for method, lag, batch in sorted(_selected_groups(dataset)):
        options = []
        for (m, l, b, setting), rows in grouped.items():
            if (m, l, b) != (method, lag, batch):
                continue
            rows.sort(key=lambda record: record[0]["seed"])
            if [c["seed"] for c, _ in rows] != list(range(5)):
                raise ValueError("Five distinct training seeds required per setting")
            scores = [score for _, score in rows]
            pooled = mean(scores) if all(math.isfinite(v) for v in scores) else math.inf
            c = rows[0][0]
            candidates.append(dict(method=method, lag_set=lag, batch_size=batch,
                                   configuration=setting, alpha=c["alpha"], beta=c["beta"],
                                   validation_mse=pooled))
            if math.isfinite(pooled):
                options.append((pooled, c["alpha"], c["beta"] or 0., setting, rows))
        expected = 25 if method == METHOD else 10
        if len([x for x in candidates if (x["method"], x["lag_set"],
                   x["batch_size"]) == (method, lag, batch)]) != expected:
            raise ValueError(f"Wrong hyperparameter grid size for {method}/{lag}/B{batch}")
        if not options:
            raise ValueError(f"No five-seed finite configuration: {method}/{lag}/B{batch}")
        score, alpha, beta, setting, rows = min(options, key=lambda t: t[:4])
        base = new_root if method == METHOD else old_root
        choices.append(dict(method=method, lag_set=lag, batch_size=batch,
                            alpha=alpha, beta=rows[0][0]["beta"], configuration=setting,
                            validation_mse=score,
                            run_ids=[c["run_id"] for c, _ in rows],
                            endpoint_sha256={
                                c["run_id"]: file_hash(base / c["run_id"] / "final.pt")
                                for c, _ in rows}))
    expected_total = 70 if dataset == "electricity" else 170
    if len(candidates) != expected_total or len(choices) != len(_selected_groups(dataset)):
        raise ValueError("Incomplete hyperparameter selection")
    plan = dict(version=VERSION, dataset=dataset, n_seeds=5,
                criterion="pooled five-seed standardized validation MSE, final 25% of positive-update observations",
                original_baseline_manifest_sha256=digest(old),
                pairwise_manifest_sha256=digest(pair),
                pairwise_codes=pair_codes,
                baseline_codes=old_codes,
                train_data_sha256=shared["data_sha256"],
                validation_sha256=shared["validation_sha256"],
                selections=choices)
    output = new_root / "pairwise_outputs"
    target = output / "selection.json"
    if target.is_file() and read_json(target) != plan:
        raise RuntimeError("Validation selection has changed: refusing to overwrite the lock")
    atomic_json(target, plan)
    write_csv(output / "candidate_validation_scores.csv",
              sorted(candidates, key=lambda r: (r["method"], r["lag_set"],
                                                r["batch_size"], r["configuration"])))
    print(f"Validation locked: {len(choices)} method/lag/batch choices, "
          f"{len(candidates)} hyperparameter settings: {target}", flush=True)
    return plan


class _Progress:
    def __init__(self, dataset, run_id):
        self.parent, self.run_id = dataset, run_id
        self.done, self.next_report = 0, 250000

    def __len__(self):
        return len(self.parent)

    def __getattr__(self, attr):
        return getattr(self.parent, attr)

    def get_batch(self, indices):
        batch = self.parent.get_batch(indices)
        self.done += len(indices)
        if self.done >= self.next_report or self.done == len(self.parent):
            print(f"TEST {self.run_id}: {self.done}/{len(self.parent)}",
                  flush=True)
            self.next_report = self.done + 250000
        return batch


def _inference(task):
    import torch
    from fulltd.data import load_dataset
    from fulltd.metrics import evaluate as run_evaluation
    from .application_train import build_objective

    torch.set_num_threads(1)
    config = task["config"]
    ident = task["identity"]
    root = Path(task["root"])
    checkpoint = root / config["run_id"] / "final.pt"
    if file_hash(checkpoint) != ident["endpoint_sha256"]:
        raise ValueError(f"Checkpoint changed since validation selection: {checkpoint}")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if (saved["config"] != config or
            saved["identity"]["code_sha256"] != ident["code_sha256"] or
            saved["identity"]["config_sha256"] != ident["config_sha256"] or
            saved["identity"]["data_sha256"] != ident["train_data_sha256"]):
        raise ValueError(f"Selected checkpoint scientific identity mismatch: {config['run_id']}")
    test = load_dataset(task["dataset"], Path(task["data"]), "test")
    if data_identity(test) != ident["test_data_sha256"]:
        raise ValueError("Test data changed since evaluation was locked")
    objective = build_objective(config, test)
    objective.load_state_dict(saved["objective"])
    objective.eval()
    records, _ = run_evaluation(objective.model, _Progress(test, config["run_id"]),
                                batch_size=128, device="cpu")
    units = {"standardized", task["raw_unit"]}
    found = sorted([dict(units=r["units"], metric=r["metric"], value=float(r["value"]))
                    for r in records
                    if str(r["horizon"]) == "all" and r["units"] in units
                    and r["metric"] in METRICS], key=lambda r: (r["units"], r["metric"]))
    expected = {(u, m) for u in units for m in METRICS}
    if (len(found) != len(expected) or
            {(r["units"], r["metric"]) for r in found} != expected
            or any(not math.isfinite(r["value"]) for r in found)):
        raise ValueError(f"Missing/nonfinite held-out metrics: {config['run_id']}")
    return found


def _evaluate_tasks(tasks, out, workers):
    if not 1 <= workers <= 32:
        raise ValueError("Evaluation workers must be between 1 and 32")
    cache_dir = out / "test_cache"
    already, pending = {}, []
    for task in tasks:
        run_id = task["config"]["run_id"]
        cache = cache_dir / f"{run_id}.json"
        if cache.is_file():
            saved = read_json(cache)
            if saved.get("identity") != task["identity"]:
                raise ValueError(f"Stale test cache for {run_id}")
            rows = saved.get("metrics")
            expected = {(u, m) for u in ("standardized", task["raw_unit"])
                        for m in METRICS}
            if (not isinstance(rows, list) or len(rows) != len(expected)
                    or {(r["units"], r["metric"]) for r in rows} != expected
                    or any(not math.isfinite(float(r["value"])) for r in rows)):
                raise ValueError(f"Malformed test cache for {run_id}")
            already[run_id] = rows
        else:
            pending.append(task)
    print(f"Full-test evaluation: {len(tasks)} selected endpoints; "
          f"{len(already)} cached; {len(pending)} pending; {workers} workers.",
          flush=True)
    if pending:
        with ProcessPoolExecutor(max_workers=min(workers, len(pending)),
                                 mp_context=get_context("spawn")) as executor:
            futures = {executor.submit(_inference, task): task for task in pending}
            for future in as_completed(futures):
                task = futures[future]
                run_id = task["config"]["run_id"]
                rows = future.result()
                atomic_json(cache_dir / f"{run_id}.json",
                            dict(identity=task["identity"], metrics=rows))
                already[run_id] = rows
                print(f"Completed {len(already)}/{len(tasks)}: {run_id}", flush=True)
    return already


def evaluate(dataset, workers=8):
    import torch
    from fulltd.data import verify_dataset, load_dataset

    data, root = pairwise_paths(dataset)
    old_data, old_root, _ = baseline_paths(dataset)
    output = root / "pairwise_outputs"
    plan = read_json(output / "selection.json")
    if plan.get("version") != VERSION or plan.get("dataset") != dataset:
        raise ValueError("Selection does not match Pairwise reporting version")
    originals = _baseline_configs(dataset)
    new = list(make_manifest(dataset))
    if (plan["original_baseline_manifest_sha256"] != digest(originals)
            or plan["pairwise_manifest_sha256"] != digest(new)
            or plan["pairwise_codes"] != _scoped_codes(new)
            or plan["baseline_codes"] != _expected_codes(originals, old_root)):
        raise ValueError("Selection does not match current scientific configurations")
    configurations = {c["run_id"]: c for c in originals + new}
    verify_dataset(dataset, data)
    test = load_dataset(dataset, data, "test")
    test_identity = data_identity(test)
    raw_unit = "kW" if dataset == "electricity" else "mph"
    metrics_source_hash = file_hash(Path(__file__).resolve().parents[1] / "fulltd/metrics.py")
    tasks = []
    for choice in plan["selections"]:
        if len(choice["run_ids"]) != 5:
            raise ValueError("Missing seed of a validation-selected method")
        for run_id in choice["run_ids"]:
            c = configurations[run_id]
            if ((c["method"], _lag(c), c["batch_size"])
                    != (choice["method"], choice["lag_set"], choice["batch_size"])):
                raise ValueError("Selected run doesn't match method/lag/batch")
            training_root = root if c["method"] == METHOD else old_root
            status = read_json(training_root / run_id / "status.json")
            identity = status["identity"]
            expected_code = (plan["pairwise_codes"][run_id] if c["method"] == METHOD
                             else plan["baseline_codes"][run_id])
            if (status["config"] != c or identity["code_sha256"] != expected_code
                    or identity["data_sha256"] != plan["train_data_sha256"]
                    or identity["validation_sha256"] != plan["validation_sha256"]):
                raise ValueError(f"Selected status mismatch: {run_id}")
            record = dict(config_sha256=digest(c), code_sha256=expected_code,
                          endpoint_sha256=choice["endpoint_sha256"][run_id],
                          train_data_sha256=plan["train_data_sha256"],
                          test_data_sha256=test_identity,
                          metrics_source_sha256=metrics_source_hash,
                          batch_size=128)
            tasks.append(dict(config=c, root=str(training_root), data=str(data),
                              dataset=dataset, raw_unit=raw_unit, identity=record,
                              runtime_seconds=float(status["runtime_seconds"])))
    if len(tasks) != (20 if dataset == "electricity" else 40):
        raise ValueError("Unexpected number of selected endpoints")
    values = _evaluate_tasks(tasks, output, workers)
    per_seed = []
    for task in tasks:
        c = task["config"]
        for record in values[c["run_id"]]:
            per_seed.append(dict(dataset=dataset, method=c["method"],
                                 lag_set=_lag(c), batch_size=c["batch_size"],
                                 seed=c["seed"], alpha=c["alpha"], beta=c["beta"],
                                 units=record["units"], metric=record["metric"],
                                 value=float(record["value"]),
                                 runtime_seconds=task["runtime_seconds"]))
    groups = defaultdict(list)
    for row in per_seed:
        key = (row["method"], row["lag_set"], row["batch_size"],
               row["units"], row["metric"])
        groups[key].append(row)
    expected_count = len(_selected_groups(dataset)) * 2 * len(METRICS)
    if (len(groups) != expected_count
            or any(sorted(r["seed"] for r in values) != list(range(5))
                   for values in groups.values())):
        raise ValueError("Incomplete paired five-seed test results")
    summary = []
    for (m, lag, batch, unit, metric), rows in sorted(groups.items()):
        scores = [r["value"] for r in rows]
        times = [r["runtime_seconds"] for r in rows]
        summary.append(dict(dataset=dataset, method=m, lag_set=lag,
                            batch_size=batch, units=unit, metric=metric, n=5,
                            mean=mean(scores), standard_error=stdev(scores)/(5**0.5),
                            mean_runtime_seconds=mean(times),
                            runtime_se=stdev(times)/(5**0.5)))
    contrasts = []
    for lag_tuple in LAG_SETS[dataset]:
        lag = "-".join(map(str, lag_tuple))
        for batch in BATCH_SIZES:
            for unit in ("standardized", raw_unit):
                for metric in METRICS:
                    td = {r["seed"]:r["value"] for r in groups[(METHOD, lag, batch, unit, metric)]}
                    for comparator in BASELINES:
                        old = {r["seed"]:r["value"] for r in groups[(comparator,"none",256,unit,metric)]}
                        diff = [td[s]-old[s] for s in range(5)]
                        contrasts.append(dict(dataset=dataset, lag_set=lag,
                                              batch_size=batch, units=unit, metric=metric,
                                              method_a=METHOD, method_b=comparator,
                                              n=5, mean_difference=mean(diff),
                                              standard_error=stdev(diff)/(5**0.5),
                                              wins=sum(d<0 for d in diff)))
    batch_contrasts = []
    for lag_tuple in LAG_SETS[dataset]:
        lag = "-".join(map(str, lag_tuple))
        for unit in ("standardized", raw_unit):
            for metric in METRICS:
                a = {r["seed"]:r["value"] for r in groups[(METHOD,lag,128,unit,metric)]}
                b = {r["seed"]:r["value"] for r in groups[(METHOD,lag,256,unit,metric)]}
                differences = [a[s]-b[s] for s in range(5)]
                batch_contrasts.append(dict(dataset=dataset, lag_set=lag, units=unit,
                                            metric=metric, method=METHOD,
                                            batch_a=128, batch_b=256, n=5,
                                            mean_difference=mean(differences),
                                            standard_error=stdev(differences)/(5**.5),
                                            wins_a=sum(x<0 for x in differences)))
    for name, rows in (("per_seed.csv", per_seed),("summary.csv",summary),
                       ("paired.csv",contrasts),("batch_paired.csv",batch_contrasts)):
        target = output / name
        if target.is_file():
            earlier = _csv(target)
            expected = [{k:str(v) for k,v in row.items()} for row in rows]
            if earlier != expected:
                raise RuntimeError(f"Existing held-out report differs: {target}")
        else:
            write_csv(target, rows)
    print(f"Finished {len(tasks)} endpoint evaluations; "
          f"{len(summary)} method/batch/lag/metric summary rows: {output}", flush=True)
    return summary


def main():
    from .common import cpu_environment
    cpu_environment()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=DATASETS)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--select", action="store_true")
    actions.add_argument("--evaluate", action="store_true")
    actions.add_argument("--all", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.workers <= 32:
        parser.error("--workers must be between 1 and 32")
    if args.select or args.all:
        select(args.dataset)
    if args.evaluate or args.all:
        evaluate(args.dataset, args.workers)


if __name__ == "__main__":
    main()
