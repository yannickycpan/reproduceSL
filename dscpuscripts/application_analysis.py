"""Validation-locked Electricity/METR-LA tables and learning curves only.

Each application method at B256 selects one rate setting across five seeds
by the mean validation MSE over the final ceil(25%) of positive-update
measurements. Only final-iterate metrics are reported; test labels never
select hyperparameters or checkpoints. FD001 remains untouched.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import math
from pathlib import Path
import statistics

from dscpuscripts.common import (atomic_json, data_identity, file_hash,
                                  file_lock, read_json, write_csv)
from dscpuscripts.result_paths import paths as result_paths
from .application_protocol import (PROTOCOL_ID, SEEDS, digest, evaluation_steps,
                                     make_manifest, update_budgets,
                                     sources_for_application_round)

ANALYSIS_VERSION = "application-comparison-v1"
EVALUATION_VERSION = "application-point-metrics-v1"
SELECTION_FRACTION = 0.25
METRICS = ("rmse", "mae")
LABELS = {"mse": "MSE", "gaussian_nll": "Gaussian NLL", "td_typed": "TD typed",
          "td_pairwise": "TD pairwise", "mae": "Masked MAE",
          "td_pairwise_mae": "TD pairwise (MAE)"}
BACKBONES = {"tcn": "TCN", "patchtst": "PatchTST", "graph_wavenet": "Graph WaveNet"}
DATASET_LABELS = {"electricity": "Electricity", "metr_la": "METR-LA"}
ROW_KEYS = ("backbone", "regime", "method", "batch_size", "split", "metric", "units")


def training_code_hash(config):
    from .application_train import code_hash
    return code_hash(config)


def manifest_for(campaign):
    if campaign not in DATASET_LABELS:
        raise ValueError(f"Unknown application campaign: {campaign}")
    return [row for row in make_manifest() if row["dataset"] == campaign]


def selection_update_steps(measured_steps, endpoint):
    """Apply the FD001 count-based final-quarter rule AFTER endpoint truncation."""
    if type(endpoint) is not int or endpoint < 1:
        raise ValueError("A positive integer selection endpoint is required")
    steps = list(measured_steps)
    if (any(type(step) is not int or step < 0 for step in steps)
            or steps != sorted(set(steps))):
        raise ValueError("Measurement steps must be unique increasing integers")
    positive = [step for step in steps if 0 < step <= endpoint]
    if not positive or positive[-1] != endpoint:
        raise ValueError("The exact final/matched endpoint must have a measurement")
    return tuple(positive[-math.ceil(SELECTION_FRACTION * len(positive)):])


def _match(shared, key, value):
    if value is None or (key in shared and shared[key] != value):
        raise ValueError(f"Incompatible campaign evidence: {key}")
    shared[key] = value


def _curve_value(record, campaign, metric):
    suffix = "standardized" if campaign == "electricity" else "raw"
    return float(record[f"{metric}_{suffix}"])


def _snapshot_path(folder, endpoint, total):
    return (folder / "final.pt" if endpoint == total and (folder / "final.pt").is_file()
            else folder / "checkpoints" / f"step_{endpoint:07d}.pt")


def _audit_candidate(config, results, expected_code, shared, source_hashes):
    from fulltd.sampling import sequence_hash

    run_id = config["run_id"]
    folder = Path(results) / run_id
    if not (folder / "status.json").is_file():
        raise ValueError(f"Missing candidate status: {run_id}")
    status = read_json(folder / "status.json")
    identity = status.get("identity", {})
    if (status.get("config") != config or identity.get("protocol_id") != config["protocol_id"]
            or identity.get("config_sha256") != digest(config)
            or identity.get("code_sha256") != expected_code):
        raise ValueError(f"Scientific configuration/code mismatch: {run_id}")
    state = status.get("state")
    if state not in ("complete", "numerical_failure"):
        raise ValueError(f"Operationally unfinished candidate: {run_id} ({state})")
    for field in ("data_sha256", "validation_sha256"):
        _match(shared, field, identity.get(field))
    subset = status.get("validation_subset")
    if not isinstance(subset, dict) or identity.get("validation_subset_sha256") != digest(subset):
        raise ValueError(f"Validation-subset identity mismatch: {run_id}")
    _match(shared, "validation_subset", subset)
    _match(shared, ("initial_mean", config["backbone"], config["seed"]),
           status.get("initial_mean_sha256"))
    n = status.get("n_sources")
    budgets = update_budgets(config, n)
    _match(shared, "n_sources", n)
    for key in ("total_updates", "matched_updates"):
        if status.get(key) != budgets[key]:
            raise ValueError(f"Incorrect declared {key}: {run_id}")
    universe = read_json(folder / "source_universe.json")
    if (universe.get("n_sources") != n or
            any(universe.get(k) != v for k, v in budgets.items()) or
            universe.get("evaluation_period") != config["learning_curve_updates"]
            or universe.get("validation_subset") != subset):
        raise ValueError(f"Source-budget sidecar mismatch: {run_id}")
    total, matched = budgets["total_updates"], budgets["matched_updates"]
    schedule = evaluation_steps(total, config["learning_curve_updates"])
    curve = read_json(folder / "update_curve.json")
    measured = [record.get("optimizer_steps") for record in curve]
    if ((not curve and state == "complete") or measured != list(schedule[:len(curve)])
            or any(type(step) is not int for step in measured)):
        raise ValueError(f"Missing, duplicated, or out-of-order measurements: {run_id}")
    for record in curve:
        values = [float(record["mse_standardized"]),
                  float(record["rmse_standardized"]), float(record["mae_standardized"]),
                  float(record["rmse_raw"]), float(record["mae_raw"])]
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError(f"Nonfinite validation measurement: {run_id}")
        if not math.isclose(values[1] ** 2, values[0], rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError(f"Inconsistent standardized RMSE/MSE: {run_id}")
        if record.get("raw_units") != ("kW" if config["dataset"] == "electricity" else "mph"):
            raise ValueError(f"Incorrect measurement units: {run_id}")
        _match(shared, "validation_subset", record.get("validation_subset"))
        path = folder / "checkpoints" / f"step_{record['optimizer_steps']:07d}.pt"
        if not path.is_file():
            raise FileNotFoundError(f"Missing retained measurement checkpoint: {path}")
    rounds = read_json(folder / "rounds.json")
    if ([r.get("epoch") for r in rounds] != list(range(1, len(rounds) + 1))
            or len(rounds) > config["epochs"]):
        raise ValueError(f"Missing/duplicated source-round evidence: {run_id}")
    td = config["method"].startswith("td_")
    for record in rounds:
        epoch = record["epoch"]
        key = (config["dataset"], n, config["seed"], epoch, config.get("source_fraction"))
        if key not in source_hashes:
            source_hashes[key] = sequence_hash(sources_for_application_round(config["dataset"], n,
                                   config["seed"], epoch, config.get("source_fraction")))
        if (record.get("source_draws") != budgets["draws_per_round"]
                or record.get("source_sha256") != source_hashes[key]):
            raise ValueError(f"Source stream/budget mismatch: {run_id}, round {epoch}")
        if record.get("optimizer_steps_cumulative") != epoch * budgets["updates_per_epoch"]:
            raise ValueError(f"Empty masks or inconsistent actual update counts: {run_id}; "
                             "the declared matched-update comparison cannot be used")
        if td:
            _match(shared, ("partner", config["seed"], epoch), record.get("partner_sha256"))
            counts = record.get("type_counts", [])
            if len(counts) != 2 or sum(counts) != budgets["draws_per_round"]:
                raise ValueError(f"Incorrect temporal-pair accounting: {run_id}")
    if state == "complete":
        if (status.get("epochs_completed") != config["epochs"] or status.get("next_epoch") != config["epochs"] + 1
                or status.get("cursor") != 0 or len(rounds) != config["epochs"]
                or status.get("optimizer_steps") != total
                or status.get("update_opportunities") != total
                or status.get("source_draws") != config["epochs"] * budgets["draws_per_round"]
                or measured != list(schedule) or not (folder / "final.pt").is_file()):
            raise ValueError(f"Incomplete final budget/evidence: {run_id}")
    evidence = dict(identity=identity, initial_mean_sha256=status["initial_mean_sha256"],
                    state=state, optimizer_steps=status.get("optimizer_steps"),
                    n_sources=n, universe=universe, curve=curve, rounds=rounds)
    return dict(config=config, state=state, curve=curve, budgets=budgets,
                evidence_sha256=digest(evidence))


def _candidate_score(candidate, endpoint, regime):
    if regime != "full_budget":
        raise ValueError("Application comparisons support full-budget B256 only")
    if candidate["state"] != "complete":
        return math.inf, ()
    curve = candidate["curve"]
    steps = [record["optimizer_steps"] for record in curve]
    if endpoint not in steps:
        return math.inf, ()
    window = selection_update_steps(steps, endpoint)
    lookup = {record["optimizer_steps"]: float(record["mse_standardized"]) for record in curve}
    return statistics.mean(lookup[step] for step in window), window


def _choice_order(choice):
    return (list(BACKBONES).index(choice["backbone"]),
            0 if choice["regime"] == "full_budget" else 1,
            list(LABELS).index(choice["method"]), choice["batch_size"])


def compute_plan(campaign, results):
    """Audit the entire prescribed grid using only training/validation records."""
    manifest = manifest_for(campaign)
    scoped_codes = {}
    code_by_id = {}
    for config in manifest:
        scope = (config["dataset"], config["backbone"], config["method"])
        if scope not in scoped_codes:
            scoped_codes[scope] = training_code_hash(config)
        code_by_id[config["run_id"]] = scoped_codes[scope]
    shared, source_hashes, candidates, by_hp = {}, {}, [], defaultdict(list)
    for config in manifest:
        candidate = _audit_candidate(config, results, code_by_id[config["run_id"]], shared, source_hashes)
        candidates.append(candidate)
        regime = "full_budget"
        endpoint = candidate["budgets"]["total_updates"]
        score, window = _candidate_score(candidate, endpoint, regime)
        key = (config["backbone"], regime, config["method"],
               config["batch_size"], config["configuration"])
        by_hp[key].append(dict(candidate=candidate, score=score,
                              optimizer_steps=endpoint, selection_steps=window))
    scores = []
    for key, members in by_hp.items():
        members.sort(key=lambda row: row["candidate"]["config"]["seed"])
        configs = [row["candidate"]["config"] for row in members]
        if [row["seed"] for row in configs] != list(SEEDS):
            raise ValueError(f"Every setting must contain all five distinct seeds: {key}")
        endpoints = {row["optimizer_steps"] for row in members}
        windows = {row["selection_steps"] for row in members if math.isfinite(row["score"])}
        if len(endpoints) != 1 or len(windows) > 1:
            raise ValueError(f"Inconsistent measurement/update budgets across seeds: {key}")
        valid = all(math.isfinite(row["score"]) for row in members)
        scores.append(dict(zip(("backbone", "regime", "method", "batch_size", "configuration"), key),
                           alpha=configs[0]["alpha"], beta=configs[0]["beta"],
                           score=statistics.mean(row["score"] for row in members) if valid else math.inf,
                           unavailable_seeds=[config["seed"] for config, row in zip(configs, members)
                                              if not math.isfinite(row["score"])],
                           optimizer_steps=next(iter(endpoints)),
                           selection_steps=list(next(iter(windows))) if windows else [],
                           run_ids=[config["run_id"] for config in configs]))
    groups = defaultdict(list)
    for score in scores:
        groups[tuple(score[k] for k in ("backbone", "regime", "method", "batch_size"))].append(score)
    choices = []
    by_id = {row["config"]["run_id"]: row for row in candidates}
    for key, options in groups.items():
        winner = min(options, key=lambda r: (r["score"], r["alpha"], r["beta"] or 0., r["configuration"]))
        if not math.isfinite(winner["score"]):
            raise ValueError(f"No finite five-seed hyperparameter setting for {key}; no test report produced")
        choice = {k: v for k, v in winner.items() if k not in ("score", "unavailable_seeds")}
        choice["selection_mse"] = winner["score"]
        choice["checkpoint_sha256"] = {}
        for run_id in choice["run_ids"]:
            total = by_id[run_id]["budgets"]["total_updates"]
            checkpoint = _snapshot_path(Path(results) / run_id, choice["optimizer_steps"], total)
            if not checkpoint.is_file():
                raise FileNotFoundError(f"Missing selected endpoint: {checkpoint}")
            choice["checkpoint_sha256"][run_id] = file_hash(checkpoint)
        choices.append(choice)
    choices.sort(key=_choice_order)
    plan = dict(analysis_version=ANALYSIS_VERSION,
                protocol_id=manifest[0]["protocol_id"], dataset=campaign,
                code_sha256=digest([[r["run_id"], code_by_id[r["run_id"]]] for r in manifest]),
                manifest_sha256=digest(manifest),
                data_sha256=shared["data_sha256"], validation_sha256=shared["validation_sha256"],
                validation_subset=shared["validation_subset"], n_sources=shared["n_sources"],
                evidence_sha256=digest([[r["config"]["run_id"], r["evidence_sha256"]] for r in candidates]),
                criterion="Five-seed mean standardized validation MSE over the final ceil(25%) "
                          "of positive-update measurements, after truncation to the reported endpoint; "
                          "select separately within each backbone/method/batch and report that endpoint.",
                counts=dict(Counter(row["state"] for row in candidates)), choices=choices)
    return plan, scores


def _validate_snapshot(checkpoint, config, status, curve, step, subset):
    progress = checkpoint.get("progress", {})
    if (checkpoint.get("config") != config or checkpoint.get("identity") != status.get("identity")
            or checkpoint.get("initial_mean_sha256") != status.get("initial_mean_sha256")
            or progress.get("optimizer_steps") != step):
        raise ValueError(f"Snapshot/configuration/status disagree: {config['run_id']} at {step}")
    observed = progress.get("curve", [])
    if (not observed or observed != curve[:len(observed)]
            or observed[-1]["optimizer_steps"] != step
            or observed[-1].get("validation_subset") != subset):
        raise ValueError(f"Snapshot validation evidence differs from selected history: {config['run_id']}")


def _point_metrics(model, dataset, batch_size):
    from fulltd.metrics import evaluate
    units = "standardized" if dataset.dataset == "electricity" else "mph"
    # Electricity's report uses standardized metrics only. Avoid computing
    # unused inverse-scaled kW metrics for every intermediate checkpoint.
    records, _ = evaluate(model, dataset, batch_size=batch_size, device="cpu",
                          allow_nonfinite=True, include_raw=dataset.dataset != "electricity")
    values = {}
    for metric in METRICS:
        matching = [r for r in records if r["units"] == units and str(r["horizon"]) == "all"
                    and r["metric"] == metric]
        if len(matching) != 1:
            raise ValueError(f"Missing unique {units}/{metric} measurement")
        values[metric] = float(matching[0]["value"])
    return values


def _evaluate_one_run(task, train, test, batch_size):
    """Verify one selected seed/run and evaluate its retained test checkpoints.

    Separate run-ID cache paths allow independent OS processes without shared
    mutable model/metric state or a second global analysis dispatcher.
    """
    import torch
    from .application_train import build_objective

    config = task["config"]
    run_id = config["run_id"]
    folder = Path(task["results"]) / run_id
    status = read_json(folder / "status.json")
    curve = read_json(folder / "update_curve.json")
    identity = status.get("identity", {})
    # A checkpoint code hash is *per candidate*. plan["code_sha256"] is
    # a digest across all candidates and MUST NOT be compared here.
    if (status.get("config") != config
            or identity.get("protocol_id") != config["protocol_id"]
            or identity.get("config_sha256") != digest(config)
            or identity.get("code_sha256") != task["expected_code"]
            or identity.get("data_sha256") != task["data_sha256"]
            or identity.get("validation_sha256") != task["validation_sha256"]
            or identity.get("validation_subset_sha256") != digest(task["subset"])):
        raise ValueError(f"Selected status no longer matches locked plan: {run_id}")
    endpoint, total = task["endpoint"], task["total"]
    endpoint_path = _snapshot_path(folder, endpoint, total)
    if file_hash(endpoint_path) != task["endpoint_sha256"]:
        raise ValueError(f"Selected endpoint weights changed after locking: {run_id}")
    units = "standardized" if task["campaign"] == "electricity" else "mph"
    per_seed, learning = [], []
    objective = None

    for record in curve:
        step = record["optimizer_steps"]
        if step > endpoint:
            break
        if task["endpoints_only"] and step != endpoint:
            continue
        checkpoint_path = _snapshot_path(folder, step, total)
        checksum = file_hash(checkpoint_path)
        cache_identity = dict(
            version=EVALUATION_VERSION,
            code_sha256=task["expected_code"],
            metric_code_sha256=task["metric_code_sha256"],
            checkpoint_sha256=checksum,
            test_sha256=task["test_sha256"],
            batch_size=batch_size,
            run_id=run_id, optimizer_steps=step)
        cache = (Path(task["output"]) / ".cache" / run_id /
                 f"step_{step:07d}_{digest(cache_identity)[:20]}.json")
        # Older reports keyed the cache by the *manifest aggregate* rather
        # than by the per-run hash. Those already-evaluated checkpoint metrics
        # remain valid when every other key and the checkpoint bytes match.
        # Migrate them without repeating expensive full-test inference.
        legacy_identity = dict(cache_identity, code_sha256=task["manifest_code_sha256"])
        legacy_cache = (Path(task["output"]) / ".cache" / run_id /
                        f"step_{step:07d}_{digest(legacy_identity)[:20]}.json")
        # Keep the existing scientific audit: even cached snapshots must have
        # consistent configuration, training cursor and validation records.
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        _validate_snapshot(checkpoint, config, status, curve, step, task["subset"])
        if cache.is_file():
            cached = read_json(cache)
            if cached.get("identity") != cache_identity:
                raise ValueError(f"Point-metric cache identity mismatch: {cache}")
            values = cached["metrics"]
        elif legacy_cache.is_file():
            cached = read_json(legacy_cache)
            if cached.get("identity") != legacy_identity:
                raise ValueError(f"Legacy point-metric cache identity mismatch: {legacy_cache}")
            values = cached["metrics"]
            atomic_json(cache, dict(identity=cache_identity, metrics=values))
            print(f"Reused legacy test cache {run_id} update {step}", flush=True)
        else:
            if objective is None:
                objective = build_objective(config, train).cpu()
            objective.load_state_dict(checkpoint["objective"])
            values = _point_metrics(objective.model, test, batch_size)
            atomic_json(cache, dict(identity=cache_identity, metrics=values))
            print(f"Evaluated test {run_id} update {step}", flush=True)
        del checkpoint
        for split in ("test", "validation"):
            for metric in METRICS:
                value = (float(values[metric]) if split == "test"
                         else _curve_value(record, task["campaign"], metric))
                row = dict(dataset=task["campaign"], backbone=task["backbone"],
                           regime=task["regime"], method=task["method"],
                           batch_size=config["batch_size"], seed=config["seed"],
                           run_id=run_id, optimizer_steps=step,
                           split=split, metric=metric, units=units, value=value)
                learning.append(row)
                if step == endpoint:
                    per_seed.append(row.copy())
    if len(per_seed) != len(METRICS) * 2:
        raise ValueError(f"Selected final checkpoint was not evaluated: {run_id}")
    return run_id, per_seed, learning


def _evaluate_one_run_worker(task):
    """Fresh process: one CPU thread; reopen read-only mmap datasets locally."""
    from dscpuscripts.common import cpu_environment
    cpu_environment()
    import torch
    from fulltd.data import load_dataset
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    campaign, data_dir = task["campaign"], task["data_dir"]
    train = load_dataset(campaign, data_dir, "train")
    test = load_dataset(campaign, data_dir, "test")
    if data_identity(test) != task["test_sha256"]:
        raise ValueError(f"Prepared test data changed while evaluating {task['config']['run_id']}")
    return _evaluate_one_run(task, train, test, task["batch_size"])


def evaluate_selected(plan, results, data_dir, output, batch_size=128,
                      *, workers=1, endpoints_only=False):
    """Evaluate validation-locked runs; parallelize independent test inference.

    Defaults preserve the old single-process behavior. --workers launches one
    process per selected run, capped by the 20/30 selected run count. Each run
    owns its cache directory. No fitting, refit or changed evaluation windows.
    """
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing
    import torch
    from fulltd.data import load_dataset
    from fulltd.sampling import training_universe
    from .application_train import evaluation_validation_view

    if type(workers) is not int or workers < 1:
        raise ValueError("Analysis workers must be a positive integer")
    if batch_size < 1:
        raise ValueError("Evaluation batch size must be positive")
    campaign = plan["dataset"]
    candidate_manifest = manifest_for(campaign)
    cached_hashes, codes = {}, []
    for c in candidate_manifest:
        scope = (c["dataset"], c["backbone"], c["method"])
        if scope not in cached_hashes:
            cached_hashes[scope] = training_code_hash(c)
        codes.append([c["run_id"], cached_hashes[scope]])
    if (plan.get("code_sha256") != digest(codes)
            or plan.get("manifest_sha256") != digest(candidate_manifest)):
        raise ValueError("Selection belongs to a different scientific implementation")
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    configs = {c["run_id"]: c for c in candidate_manifest}
    reference = next(iter(configs.values()))
    train, _, _ = training_universe(load_dataset(campaign, data_dir, "train"),
                                   tuple(reference["lags"]))
    validation = load_dataset(campaign, data_dir, "val")
    _, subset = evaluation_validation_view(validation)
    if (data_identity(train) != plan["data_sha256"]
            or data_identity(validation) != plan["validation_sha256"]
            or subset != plan["validation_subset"]):
        raise ValueError("Prepared training/validation data changed after selection")
    # First test access is AFTER the complete validation-only plan is locked.
    test = load_dataset(campaign, data_dir, "test")
    test_identity = data_identity(test)
    metric_code = file_hash(Path(__file__).resolve().parents[1] / "fulltd" / "metrics.py")
    tasks = []
    for choice in plan["choices"]:
        for run_id in choice["run_ids"]:
            config = configs[run_id]
            endpoint = choice["optimizer_steps"]
            total = update_budgets(config, plan["n_sources"])["total_updates"]
            scope = (config["dataset"], config["backbone"], config["method"])
            tasks.append(dict(
                campaign=campaign, config=config, results=str(results),
                output=str(output), data_dir=str(data_dir), endpoint=endpoint,
                total=total, expected_code=cached_hashes[scope],
                endpoint_sha256=choice["checkpoint_sha256"][run_id],
                subset=subset, data_sha256=plan["data_sha256"],
                validation_sha256=plan["validation_sha256"],
                test_sha256=test_identity, metric_code_sha256=metric_code,
                manifest_code_sha256=plan["code_sha256"],
                batch_size=batch_size, backbone=choice["backbone"],
                regime=choice["regime"], method=choice["method"],
                endpoints_only=endpoints_only))

    per_seed, learning = [], []
    if workers == 1:
        results_iterator = (_evaluate_one_run(task, train, test, batch_size)
                            for task in tasks)
        for run_id, endpoint_rows, curve_rows in results_iterator:
            per_seed.extend(endpoint_rows)
            learning.extend(curve_rows)
            print(f"Completed selected run: {run_id}", flush=True)
    else:
        # 'spawn' avoids importing an inherited threaded PyTorch/MKL state.
        # Read-only numpy mmap arrays are reopened in each child on demand.
        with ProcessPoolExecutor(
                max_workers=min(workers, len(tasks)),
                mp_context=multiprocessing.get_context("spawn")) as pool:
            for run_id, endpoint_rows, curve_rows in pool.map(
                    _evaluate_one_run_worker, tasks, chunksize=1):
                per_seed.extend(endpoint_rows)
                learning.extend(curve_rows)
                print(f"Completed selected run: {run_id}", flush=True)
    return per_seed, learning, test_identity



def summarize(rows, keys):
    """One seed is one observation; retain nonfinite outcomes without filtering."""
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    output = []
    for key, records in sorted(groups.items()):
        records.sort(key=lambda row: int(row["seed"]))
        if [int(row["seed"]) for row in records] != list(SEEDS):
            raise ValueError(f"Report requires all five distinct selected seeds: {key}")
        values = [float(row["value"]) for row in records]
        if all(math.isfinite(value) for value in values):
            mean, se = statistics.mean(values), statistics.stdev(values) / math.sqrt(len(SEEDS))
        else:
            mean, se = math.inf if any(math.isinf(v) for v in values) else math.nan, math.nan
        output.append(dict(zip(keys, key), n=len(values), mean=mean, standard_error=se,
                           finite_seeds=sum(math.isfinite(v) for v in values)))
    return output


def _row_label(choice):
    return f"{LABELS[choice['method']]} B{choice['batch_size']}"


def _format_metric(mean, se):
    def number(value):
        value = float(value)
        return f"{value:.4f}" if math.isfinite(value) else str(value)
    return f"{number(mean)} ± {number(se)}"


def write_tables(plan, summary, output):
    """Separate RMSE and MAE tables; columns are test and validation only."""
    output = Path(output)
    indexed = {tuple(row[k] for k in ROW_KEYS): row for row in summary}
    units = "standardized" if plan["dataset"] == "electricity" else "mph"
    validation_label = "Validation (fixed 20%)" if plan["dataset"] == "electricity" else "Validation"
    paths = []
    for backbone in dict.fromkeys(choice["backbone"] for choice in plan["choices"]):
        choices = [row for row in plan["choices"] if row["backbone"] == backbone]
        for metric in METRICS:
            title = f"{DATASET_LABELS[plan['dataset']]} · {BACKBONES[backbone]} · {metric.upper()} ({units})"
            lines = ["# " + title, "", "Mean ± standard error across five seeds; lower is better.", "",
                     f"| Algorithm | Test | {validation_label} |", "|---|---:|---:|"]
            table = []
            for choice in choices:
                row = {"algorithm": _row_label(choice)}
                formatted = []
                for split in ("test", "validation"):
                    key = (backbone, choice["regime"], choice["method"], choice["batch_size"], split, metric, units)
                    measurement = indexed[key]
                    row[f"{split}_mean"] = measurement["mean"]
                    row[f"{split}_standard_error"] = measurement["standard_error"]
                    row[f"{split}_n"] = measurement["n"]
                    formatted.append(_format_metric(measurement["mean"], measurement["standard_error"]))
                lines.append("| " + " | ".join([row["algorithm"], *formatted]) + " |")
                table.append(row)
            if plan["dataset"] == "electricity":
                exposure = "240 stochastic rounds (~10 effective passes)"
            else:
                fraction = manifest_for(plan["dataset"])[0].get("source_fraction", 1.0)
                exposure = ("100 full epochs" if fraction == 1.0
                            else f"100 stochastic rounds ({100*fraction:g}% source draws per round; "
                                 f"~{100*fraction:g} effective passes)")
            lines.extend(["", "Rows report the final iterate after " + exposure + ".",
                          "Hyperparameters are selected by standardized validation MSE, never test performance."])
            if plan["dataset"] == "electricity":
                lines.append("Metrics average within each customer and then across customers. "
                             "Validation uses the same fixed customer-stratified 20% sample at every update; "
                             "test uses the complete held-out set.")
            else:
                lines.append("Metrics pool all observed sensor/horizon targets across the full validation/test sets.")
            prefix = output / f"{backbone}_{metric}_table"
            prefix.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
            write_csv(prefix.with_suffix(".csv"), table)
            paths.append(prefix.with_suffix(".md"))
    return paths


def plot_curves(plan, summary, output):
    """One test and one validation panel per backbone and metric."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    figures = Path(output) / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    units = "standardized" if plan["dataset"] == "electricity" else "mph"
    for backbone in dict.fromkeys(c["backbone"] for c in plan["choices"]):
        choices = [c for c in plan["choices"] if c["backbone"] == backbone]
        for metric in METRICS:
            fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharex=True)
            for ax, split in zip(axes, ("test", "validation")):
                for choice in choices:
                    records = [r for r in summary
                               if r["backbone"] == backbone and r["metric"] == metric
                               and r["split"] == split and r["regime"] == "full_budget"
                               and r["method"] == choice["method"] and int(r["batch_size"]) == 256]
                    records.sort(key=lambda r: int(r["optimizer_steps"]))
                    x = np.asarray([int(r["optimizer_steps"]) for r in records])
                    y = np.asarray([float(r["mean"]) for r in records])
                    se = np.asarray([float(r["standard_error"]) for r in records])
                    finite = np.isfinite(y)
                    color = f"C{list(LABELS).index(choice['method'])}"
                    ax.plot(x, np.where(finite, y, np.nan), color=color,
                            label=LABELS[choice["method"]])
                    ax.fill_between(x, np.maximum(0., y-se), y+se,
                                    where=finite & np.isfinite(se), color=color, alpha=.12)
                ax.grid(alpha=.2)
                ax.set_xlabel("Optimizer updates")
                label = ("Validation (fixed 20%)" if split == "validation"
                         and plan["dataset"] == "electricity" else split.title())
                ax.set_ylabel(f"{label} {metric.upper()} ({units})")
                ax.legend(fontsize=8, loc="best")
            fig.suptitle(f"{DATASET_LABELS[plan['dataset']]} · {BACKBONES[backbone]} — "
                         f"mean ± standard error (5 seeds, B256)")
            fig.tight_layout(rect=(0, 0, 1, .93))
            for ext in ("png", "pdf"):
                fig.savefig(figures / f"{backbone}_{metric}_learning_curves.{ext}", dpi=160)
            plt.close(fig)


def _read_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    for row in rows:
        for field in ("seed", "batch_size", "optimizer_steps"):
            row[field] = int(row[field])
        row["value"] = float(row["value"])
    return rows


def report(plan, output):
    output = Path(output)
    receipt = read_json(output / "evaluation_manifest.json")
    if receipt.get("selection_sha256") != digest(plan):
        raise ValueError("Endpoint/curve files belong to a different locked selection")
    for filename in ("per_seed_metrics.csv", "learning_curves_per_seed.csv"):
        if receipt.get("files", {}).get(filename) != file_hash(output / filename):
            raise ValueError(f"Recorded performance values were changed or are incomplete: {filename}")
    endpoints = summarize(_read_rows(output / "per_seed_metrics.csv"), ROW_KEYS)
    curves = summarize(_read_rows(output / "learning_curves_per_seed.csv"), ROW_KEYS + ("optimizer_steps",))
    write_csv(output / "performance_summary.csv", endpoints)
    write_csv(output / "learning_curves.csv", curves)
    paths = write_tables(plan, endpoints, output)
    plot_curves(plan, curves, output)
    for path in paths:
        print(path.read_text(encoding="utf-8"))
    print("Wrote performance tables and learning curves:", output, flush=True)


def run(campaign, mode="all", *, data_dir=None, results_dir=None, output=None,
        batch_size=128, workers=1):
    """Direct-server API; check/run/plots aliases preserve the existing launcher."""
    mode = {"check": "select", "run": "all", "plots": "report"}.get(mode, mode)
    if mode not in ("select", "evaluate", "report", "all", "endpoints"):
        raise ValueError(f"Unknown application analysis mode: {mode}")
    defaults = result_paths(campaign)
    data_dir, results, output = [Path(value) if value is not None else fallback
                                for value, fallback in zip((data_dir, results_dir, output), defaults)]
    output.mkdir(parents=True, exist_ok=True)
    with file_lock(output / ".application_analysis.lock"):
        lock = output / "selections.json"
        if mode == "report":
            plan = read_json(lock)
            if plan.get("dataset") != campaign or plan.get("analysis_version") != ANALYSIS_VERSION:
                raise ValueError("Locked selection is not this application's current report")
            return report(plan, output)
        plan, scores = compute_plan(campaign, results)
        if lock.is_file():
            if read_json(lock) != plan:
                raise ValueError("Locked validation-only selection changed; preserve the existing report and investigate")
        else:
            if mode == "evaluate":
                raise ValueError("Run select first to lock all validation-only choices")
            if any((output / name).exists() for name in ("per_seed_metrics.csv", "evaluation_manifest.json",
                                                        "endpoint_per_seed_metrics.csv", "endpoint_manifest.json")):
                raise ValueError("Unlocked prior test results exist; preserve them separately before a new analysis")
            atomic_json(lock, plan)
        write_csv(output / "hyperparameter_scores.csv", scores)
        write_csv(output / "selected_hyperparameters.csv", [
            {key: value for key, value in row.items() if key != "checkpoint_sha256"}
            for row in plan["choices"]])
        if mode == "select":
            print(f"Locked {len(plan['choices'])} five-seed choices using validation only: {lock}", flush=True)
            return plan
        per_seed, learning, test_identity = evaluate_selected(
            plan, results, data_dir, output, batch_size,
            workers=workers, endpoints_only=(mode == "endpoints"))
        if mode == "endpoints":
            write_csv(output / "endpoint_per_seed_metrics.csv", per_seed)
            atomic_json(output / "endpoint_manifest.json", dict(
                selection_sha256=digest(plan), test_sha256=test_identity,
                files={"endpoint_per_seed_metrics.csv":
                       file_hash(output / "endpoint_per_seed_metrics.csv")}))
            summaries = summarize(per_seed, ROW_KEYS)
            write_csv(output / "performance_summary.csv", summaries)
            paths = write_tables(plan, summaries, output)
            for path in paths:
                print(path.read_text(encoding="utf-8"))
            print("Wrote endpoint performance tables (full learning curves not yet evaluated):",
                  output, flush=True)
            return plan
        write_csv(output / "per_seed_metrics.csv", per_seed)
        write_csv(output / "learning_curves_per_seed.csv", learning)
        atomic_json(output / "evaluation_manifest.json", dict(
            selection_sha256=digest(plan), test_sha256=test_identity,
            files={name: file_hash(output / name) for name in ("per_seed_metrics.csv", "learning_curves_per_seed.csv")}))
        if mode == "all":
            report(plan, output)
        return plan


def main():
    from dscpuscripts.common import cpu_environment
    cpu_environment()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", choices=tuple(DATASET_LABELS))
    parser.add_argument("mode", choices=("select", "evaluate", "report", "all",
                                         "endpoints", "check", "run", "plots"))
    parser.add_argument("--data-dir")
    parser.add_argument("--results-dir")
    parser.add_argument("--output")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=1,
                        help="Independent selected runs to evaluate in parallel; default 1")
    args = parser.parse_args()
    run(args.campaign, args.mode, data_dir=args.data_dir, results_dir=args.results_dir,
        output=args.output, batch_size=args.batch_size, workers=args.workers)


if __name__ == "__main__":
    main()
