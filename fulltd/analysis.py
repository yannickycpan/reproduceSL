"""Locked pooled selection and table-driven analysis for the TD v7 campaign.

Selection reads validation records only. Test evaluation is a separate command.
The figures and summaries below are regenerated from retained numerical records;
no training, new selectors, or favorable-seed filtering occurs here.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import statistics
import time
import uuid
from collections import defaultdict
from pathlib import Path

import numpy as np

PROTOCOL_ID = "td-v7-neural-e24-v10-final3-v1"
VALIDATION_EPOCHS = tuple(range(10, 101, 10))
SELECTION_EPOCHS = (80, 90, 100)
SEEDS = (0, 1, 2, 3, 4)
TD_RATES = ((1e-4, 1e-4), (1e-4, 1e-3), (3e-4, 3e-4),
            (3e-4, 3e-3), (1e-3, 1e-3), (1e-3, 1e-2))
OTHER_RATES = (1e-4, 2e-4, 3e-4, 5e-4, 7e-4, 1e-3)
MATRIX = {
    ("electricity", "tcn"): ("mse", "td_full", "td_scalar", "gaussian_nll"),
    ("electricity", "patchtst"): ("mse", "td_full"),
    ("electricity", "dlinear"): ("dlinear_mse",),
    ("metr_la", "graph_wavenet"): ("mse", "td_full", "td_scalar", "gaussian_nll", "mae"),
    ("fd001", "tcn"): ("mse", "td_full", "gaussian_nll"),
}
CONFIG_KEYS = ("run_id", "dataset", "backbone", "method", "batch_size", "alpha",
               "beta", "seed", "configuration", "protocol_id")
METRIC_KEYS = ("dataset", "backbone", "method", "batch_size", "split", "units",
               "horizon", "metric", "variant")
T_CRITICAL_DF4 = 2.7764451051977987
PERSISTENCE_METHODS = ("persistence", "daily_persistence", "last_observation")


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def _atomic_text(path, text):
    """Publish a complete sidecar only after its bytes have been flushed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        # Indexers may briefly hold a Windows file open after publication.
        for attempt in range(6):
            try:
                os.replace(temporary, path)
                break
            except PermissionError:
                if attempt == 5:
                    raise
                time.sleep(.02 * 2 ** attempt)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path, value):
    """Strict JSON: nonfinite values are explicit strings, never invalid JSON."""
    def clean(x):
        if isinstance(x, dict):
            return {k: clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [clean(v) for v in x]
        if isinstance(x, (float, np.floating)) and not math.isfinite(x):
            return "Infinity" if x > 0 else "-Infinity" if x < 0 else "NaN"
        if isinstance(x, np.generic):
            return x.item()
        return x
    _atomic_text(path, json.dumps(clean(value), indent=2, allow_nan=False) + "\n")


def write_csv(path, rows, fields=None):
    rows = list(rows)
    if fields is None:
        fields = list(dict.fromkeys(k for row in rows for k in row))
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    _atomic_text(path, stream.getvalue())


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def group_id(config):
    return "/".join(str(config[k]) for k in ("dataset", "backbone", "method", "batch_size"))


def validate_manifest(manifest):
    """Independent check of the prescribed 630 runs, not just their count."""
    if not isinstance(manifest, list):
        raise ValueError("Manifest must be a list of candidate configurations")
    expected = set()
    for (dataset, backbone), methods in MATRIX.items():
        for method in methods:
            for batch in ((256, 128) if method.startswith("td_") else (256,)):
                for setting in range(6):
                    for seed in SEEDS:
                        expected.add((dataset, backbone, method, batch, setting, seed))
    actual, run_ids = set(), set()
    for row in manifest:
        from .protocol import validate_config
        validate_config(row)
        missing = set(CONFIG_KEYS) - row.keys()
        if missing:
            raise ValueError(f"Manifest entry is missing {sorted(missing)}")
        run_id = row["run_id"]
        if not isinstance(run_id, str) or not run_id or Path(run_id).name != run_id or "\\" in run_id:
            raise ValueError("run_id must be a plain directory name")
        key = tuple(row[k] for k in ("dataset", "backbone", "method", "batch_size", "configuration", "seed"))
        if key in actual or run_id in run_ids:
            raise ValueError(f"Duplicate candidate: {run_id}")
        if row["protocol_id"] != PROTOCOL_ID:
            raise ValueError(f"Incompatible protocol: {run_id}")
        if key not in expected:
            raise ValueError(f"Unprescribed candidate: {key}")
        rates = TD_RATES[row["configuration"]] if row["method"].startswith("td_") else (OTHER_RATES[row["configuration"]], None)
        if (row["alpha"], row["beta"]) != rates:
            raise ValueError(f"Incorrect learning rates in {run_id}")
        actual.add(key)
        run_ids.add(run_id)
    if actual != expected:
        raise ValueError(f"Campaign must contain all 630 prescribed candidates; {len(expected - actual)} missing")
    return manifest


def _manifest_hash(manifest):
    return hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _validate_candidate(config, root):
    status_path = root / config["run_id"] / "status.json"
    if not status_path.exists():
        raise ValueError(f"Missing candidate status: {config['run_id']}; selection is blocked")
    status = read_json(status_path)
    if status.get("protocol_id") != PROTOCOL_ID:
        raise ValueError(f"Incompatible status: {config['run_id']}")
    recorded = status.get("config", {})
    if recorded != config:
        raise ValueError(f"Checkpoint/status configuration mismatch: {config['run_id']}")
    state = status.get("state")
    if state == "numerical_failure":
        record_path = root / config["run_id"] / "validation.json"
        records = read_json(record_path) if record_path.exists() else []
        if (not isinstance(records, list) or len(records) > len(VALIDATION_EPOCHS)
                or tuple(r.get("epoch") for r in records) != VALIDATION_EPOCHS[:len(records)]
                or any(not math.isfinite(float(r["mse"])) for r in records)):
            raise ValueError(f"Malformed partial validation evidence for failed candidate: {config['run_id']}")
        return math.inf, records, status
    if state != "complete":
        raise ValueError(f"Operationally incomplete candidate {config['run_id']}: {state}")
    if status.get("epochs_completed") != 100:
        raise ValueError(f"Candidate is not the actual epoch/round 100: {config['run_id']}")
    if not (root / config["run_id"] / "final.pt").is_file():
        raise ValueError(f"Missing final checkpoint: {config['run_id']}")
    records = read_json(root / config["run_id"] / "validation.json")
    if not isinstance(records, list) or tuple(r.get("epoch") for r in records) != VALIDATION_EPOCHS:
        raise ValueError(f"Expected exactly ten validation records at 10,...,100: {config['run_id']}")
    if any(not math.isfinite(float(r["mse"])) for r in records):
        raise ValueError(f"Nonfinite validation MSE labelled complete: {config['run_id']}")
    late = [float(r["mse"]) for r in records if r["epoch"] in SELECTION_EPOCHS]
    return math.fsum(late) / 3, records, status


def audit_campaign(manifest, results_dir, dataset="all"):
    """Check actual budgets and fair streams before any selection is published."""
    from .protocol import digest
    from .sampling import sources_for_round, sequence_hash
    shared, checked_sources = {}, {}
    counts = dict(runs=0, complete=0, numerical_failure=0, recorded_rounds=0)

    def match(key, value):
        if value is None:
            raise ValueError(f"Missing reproducibility identity: {key}")
        if key in shared and shared[key] != value:
            raise ValueError(f"Campaign mixes incompatible data/initializations/source streams: {key}")
        shared[key] = value

    for config in manifest:
        if dataset != "all" and config["dataset"] != dataset:
            continue
        root = Path(results_dir) / config["run_id"]
        status = read_json(root / "status.json")
        if status["state"] not in ("complete", "numerical_failure"):
            raise ValueError("All candidates must be complete or declared numerical failures")
        name, seed = config["dataset"], config["seed"]
        identity = status["identity"]
        if identity["config_sha256"] != digest(config) or identity["protocol_id"] != PROTOCOL_ID:
            raise ValueError("Configuration identity differs from the frozen manifest")
        match(("code",), identity["code_sha256"])
        match((name, "data"), identity["data_sha256"])
        match((name, "validation"), identity["validation_sha256"])
        match((name, config["backbone"], seed, "initial_mean"), status["initial_mean_sha256"])
        n = status["n_sources"]
        if not isinstance(n, int) or n < 1:
            raise ValueError("Invalid training-source universe size")
        match((name, "n_sources"), n)
        draws = math.ceil(n / 24) if name == "electricity" else n
        opportunities = math.ceil(draws / config["batch_size"])
        td = config["method"].startswith("td_")
        round_path = root / "rounds.json"
        rounds = read_json(round_path) if round_path.exists() else []
        if tuple(r["epoch"] for r in rounds) != tuple(range(1, len(rounds) + 1)) or len(rounds) > 100:
            raise ValueError("Round accounting contains gaps, duplicates or excessive rounds")
        previous_steps = 0
        for record in rounds:
            epoch = record["epoch"]
            if record["source_draws"] != draws:
                raise ValueError("Recorded source count violates the declared round/epoch budget")
            key = (name, seed, epoch, "source")
            if key not in checked_sources:
                checked_sources[key] = sequence_hash(sources_for_round(name, n, seed, epoch))
            if record["source_sha256"] != checked_sources[key]:
                raise ValueError("Source sequence does not reproduce the prespecified deterministic stream")
            match(key, record["source_sha256"])
            if td:
                match((name, seed, epoch, "partner"), record["partner_sha256"])
                if len(record["type_counts"]) != 2 or sum(record["type_counts"]) != draws:
                    raise ValueError("Each TD source must have exactly one typed partner")
            steps = record["optimizer_steps_cumulative"]
            if steps < previous_steps or steps - previous_steps > opportunities:
                raise ValueError("Impossible optimizer-update count in round accounting")
            previous_steps = steps
        if status["state"] == "complete":
            if len(rounds) != 100 or status["epochs_completed"] != 100:
                raise ValueError("Completed run lacks all 100 round/epoch records")
            if status["source_draws"] != 100 * draws or status["partner_draws"] != (100 * draws if td else 0):
                raise ValueError("Completed run has an incorrect source/partner budget")
            if status["update_opportunities"] != 100 * opportunities:
                raise ValueError("Completed run has an incorrect logical-batch budget")
            if status["optimizer_steps"] + status["empty_mask_batches"] != 100 * opportunities:
                raise ValueError("Optimizer steps and empty batches do not reconcile")
            if rounds[-1]["optimizer_steps_cumulative"] != status["optimizer_steps"]:
                raise ValueError("Final optimizer counter differs from round records")
            if name != "metr_la" and status["empty_mask_batches"] != 0:
                raise ValueError("Complete-target datasets must not skip logical batches")
            if not math.isclose(status["effective_original_passes"], 100 * draws / n, rel_tol=1e-12):
                raise ValueError("Effective original-dataset passes are misreported")
        counts["runs"] += 1
        counts[status["state"]] += 1
        counts["recorded_rounds"] += len(rounds)
    return {**counts, "source_streams_reproduced": len(checked_sources),
            "checks": "Exact source budgets, deterministic shared source sequences, shared TD partners, matched initial means, code/data identities"}


def select_candidates(manifest, results_dir, output_dir, dataset="all", write_outputs=True):
    """Require the entire dataset's campaign before locking its selections."""
    if isinstance(manifest, (str, Path)):
        manifest = read_json(manifest)
    validate_manifest(manifest)
    if dataset not in ("all", "electricity", "metr_la", "fd001"):
        raise ValueError("Unknown dataset scope")
    root, out = Path(results_dir), Path(output_dir)
    grouped, group_evidence = defaultdict(list), defaultdict(list)
    validation_rows, candidate_rows = [], []
    for config in manifest:
        if dataset != "all" and config["dataset"] != dataset:
            continue
        score, records, status = _validate_candidate(config, root)
        grouped[group_id(config)].append((config, score, status))
        group_evidence[group_id(config)].append(dict(config=config, state=status["state"],
            validation=records, identity=status.get("identity"), initial_mean_sha256=status.get("initial_mean_sha256")))
        for record in records:
            row = {k: config[k] for k in CONFIG_KEYS}
            row.update(record)
            if isinstance(row.get("theta"), list):
                row["theta"] = json.dumps(row["theta"])
            validation_rows.append(row)
        candidate_rows.append({**{k: config[k] for k in CONFIG_KEYS}, "state": status["state"],
                               "late_validation_mse": score,
                               **{k: status.get(k) for k in ("epochs_completed", "optimizer_steps", "source_draws",
                                                            "runtime_seconds", "peak_accelerator_memory_bytes")}})
    scores, choices, selected_rows = [], [], []
    for name, entries in sorted(grouped.items()):
        configurations = defaultdict(list)
        for config, score, status in entries:
            configurations[config["configuration"]].append((config, score, status))
        group_scores = []
        for h, values in sorted(configurations.items()):
            values.sort(key=lambda x: x[0]["seed"])
            if tuple(x[0]["seed"] for x in values) != SEEDS:
                raise ValueError(f"Missing seeds for {name}/{h}")
            config = values[0][0]
            score = math.inf if any(not math.isfinite(v[1]) for v in values) else math.fsum(v[1] for v in values) / 5
            row = {k: config[k] for k in ("dataset", "backbone", "method", "batch_size", "configuration", "alpha", "beta")}
            row.update(group_id=name, score=score,
                       failed_seeds=json.dumps([v[0]["seed"] for v in values if v[2]["state"] == "numerical_failure"]),
                       seed_scores=json.dumps([v[1] if math.isfinite(v[1]) else "Infinity" for v in values]))
            scores.append(row)
            group_scores.append((score, config["alpha"], config["beta"] or 0.0, h, values))
        best = min(group_scores, key=lambda x: x[:3])
        config = best[4][0][0]
        choice = {k: config[k] for k in ("dataset", "backbone", "method", "batch_size")}
        choice.update(group_id=name, selection_evidence_sha256=_manifest_hash(group_evidence[name]))
        if not math.isfinite(best[0]):
            choice.update(state="all_configurations_failed", score="Infinity", run_ids=[])
        else:
            choice.update(state="selected", configuration=best[3], alpha=best[1], beta=config["beta"],
                          score=best[0], run_ids=[v[0]["run_id"] for v in best[4]])
            for run_config, seed_score, run_status in best[4]:
                validations = read_json(root / run_config["run_id"] / "validation.json")
                record = {k: run_config[k] for k in CONFIG_KEYS}
                record.update(group_id=name, pooled_selection_score=best[0], seed_late_validation_mse=seed_score,
                              batch_definition="source_partner_pairs" if run_config["method"].startswith("td_") else "source_windows")
                for epoch in SELECTION_EPOCHS:
                    record[f"validation_mse_{epoch}"] = next(r["mse"] for r in validations if r["epoch"] == epoch)
                record["final_theta"] = json.dumps(validations[-1].get("theta"))
                for key in ("training_unit", "epochs_completed", "n_sources", "source_draws", "partner_draws",
                            "optimizer_steps", "update_opportunities", "empty_mask_batches", "effective_original_passes",
                            "source_valid_count", "pair_valid_count", "physical_microbatch", "runtime_seconds",
                            "mean_parameter_count", "total_parameter_count", "peak_accelerator_memory_bytes", "code_revision",
                            "initial_mean_sha256"):
                    record[key] = run_status.get(key)
                record["hardware"] = json.dumps(run_status.get("hardware", {}), sort_keys=True)
                selected_rows.append(record)
        choices.append(choice)
    selected = {"protocol_id": PROTOCOL_ID, "manifest_sha256": _manifest_hash(manifest),
                "dataset_scope": dataset, "selection_epochs": list(SELECTION_EPOCHS),
                "validation_epochs": list(VALIDATION_EPOCHS), "groups": choices}
    audit = audit_campaign(manifest, root, dataset)
    # Writes occur only after every relevant candidate has passed the audit.
    if write_outputs:
        write_csv(out / "scores.csv", scores)
        write_csv(out / "candidate_accounting.csv", candidate_rows)
        write_csv(out / "selected_runs.csv", selected_rows)
        write_csv(out / "validation_records.csv", validation_rows)
        write_json(out / "selections.json", selected)
        write_json(out / "campaign_audit.json", audit)
    return selected


def aggregate_metrics(records):
    """Mean, sample SD, and SE of all five paired seeds; preserve failures."""
    groups = defaultdict(list)
    normalized = []
    for original in records:
        row = dict(original)
        row.setdefault("variant", "trained")
        row["batch_size"] = int(row["batch_size"])
        row["horizon"] = str(row["horizon"])
        row["seed"] = int(row["seed"])
        row["value"] = float(row["value"])
        groups[tuple(row[k] for k in METRIC_KEYS)].append(row)
        normalized.append(row)
    summaries = []
    for key, rows in sorted(groups.items(), key=lambda kv: tuple(map(str, kv[0]))):
        rows.sort(key=lambda x: x["seed"])
        seeds = [r["seed"] for r in rows]
        deterministic = rows[0]["method"] in PERSISTENCE_METHODS
        expected = [-1] if deterministic else list(SEEDS)
        if seeds != expected:
            raise ValueError(f"Metrics must include exactly {expected}; got {seeds} for {key}")
        values = np.asarray([r["value"] for r in rows], dtype=np.float64)
        with np.errstate(invalid="ignore", over="ignore"):
            mean = float(values.mean())
            sd = float(values.std(ddof=1)) if len(values) > 1 else math.nan
        row = dict(zip(METRIC_KEYS, key))
        row.update(n=len(values), mean=mean, sample_sd=None if deterministic else sd,
                   standard_error=None if deterministic else sd / math.sqrt(len(values)),
                   uncertainty="deterministic; no training-seed uncertainty" if deterministic else "five training seeds",
                   finite_seeds=int(np.isfinite(values).sum()))
        summaries.append(row)
    return normalized, summaries


def paired_comparisons(records):
    """Primary within-backbone TD-full minus matched baselines, seed paired."""
    indexed = defaultdict(dict)
    for row in records:
        if int(row["batch_size"]) != 256 or row.get("variant", "trained") != "trained" or row["method"] in PERSISTENCE_METHODS:
            continue
        key = tuple(str(row[k]) for k in ("dataset", "backbone", "split", "units", "horizon", "metric"))
        indexed[key].setdefault(row["method"], {})[int(row["seed"])] = float(row["value"])
    output = []
    for key, methods in sorted(indexed.items()):
        if "td_full" not in methods:
            continue
        if sorted(methods["td_full"]) != list(SEEDS):
            raise ValueError("TD-full paired comparison has missing seeds")
        td = np.array([methods["td_full"][s] for s in SEEDS], dtype=np.float64)
        for baseline, by_seed in sorted(methods.items()):
            if baseline == "td_full":
                continue
            if sorted(by_seed) != list(SEEDS):
                raise ValueError("Baseline paired comparison has missing seeds")
            values = np.array([by_seed[s] for s in SEEDS], dtype=np.float64)
            with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
                diff = td - values
                mean, sd = float(diff.mean()), float(diff.std(ddof=1))
                half = T_CRITICAL_DF4 * sd / math.sqrt(5)
                improvement = float((values.mean() - td.mean()) / values.mean()) if values.mean() != 0 else math.nan
            row = dict(zip(("dataset", "backbone", "split", "units", "horizon", "metric"), key))
            row.update(baseline=baseline, difference="TD-full minus baseline", mean_difference=mean,
                       sample_sd_difference=sd, standard_error_difference=sd / math.sqrt(5),
                       descriptive_t95_lower=mean-half, descriptive_t95_upper=mean+half, degrees_of_freedom=4,
                       relative_improvement=improvement)
            output.append(row)
    return output


def _safe_name(value):
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in str(value))


def create_figures(table_dir, output_dir):
    """Vector-only, caption-free figures, each accompanied by its exact CSV."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42, "svg.fonttype": "none"})
    table_dir, out = Path(table_dir), Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    summaries = read_csv(table_dir / "summary_metrics.csv")
    panels = defaultdict(list)
    for row in summaries:
        if row["split"] != "test":
            continue
        batch = int(row["batch_size"])
        variant = row.get("variant", "trained")
        family = "refit" if variant != "trained" else "sensitivity" if batch == 128 else "primary"
        # All-horizon values are retained in tables; primary plots show declared horizons.
        wanted = {"electricity": {"1", "24"}, "metr_la": {"3", "6", "12"}, "fd001": {"all"}}
        if family == "primary" and row["horizon"] not in wanted[row["dataset"]]:
            continue
        allowed_primary = {
            "electricity": {("standardized", "rmse"), ("kW", "rmse")},
            "metr_la": {("mph", "rmse"), ("mph", "mae")},
            "fd001": {("cycles", "rmse"), ("cycles", "mae"), ("score", "rul_score_sum")},
        }
        if family == "primary" and (row["units"], row["metric"]) not in allowed_primary[row["dataset"]]:
            continue
        if family != "primary":
            diagnostic_units = {"electricity": "standardized", "metr_la": "mph", "fd001": "cycles"}
            if row["horizon"] != "all" or row["metric"] != "rmse" or row["units"] != diagnostic_units[row["dataset"]]:
                continue
        # Error magnitudes and the asymmetric RUL score stay on different axes.
        key = (family, row["dataset"], row["units"], row["metric"])
        panels[key].append(row)
    for key, values in sorted(panels.items()):
        family, dataset, units, metric = key
        if family == "sensitivity":
            # Include the prespecified primary counterpart in the same plot/table.
            counterparts = [r for r in summaries if r["split"] == "test" and r["dataset"] == dataset
                            and r["units"] == units and r["metric"] == metric and r["batch_size"] == "256"
                            and r["horizon"] == "all" and r["method"].startswith("td_") and r.get("variant", "trained") == "trained"]
            values = values + counterparts
        if family == "refit":
            values += [r for r in summaries if r["split"] == "test" and r["dataset"] == dataset
                       and r["units"] == units and r["metric"] == metric and r["batch_size"] == "256"
                       and r["backbone"] == "tcn" and r["horizon"] == "all" and r["method"] == "td_full"
                       and r.get("variant", "trained") == "trained"]
        values.sort(key=lambda r: (r["backbone"], r["method"], r["variant"], int(r["batch_size"]), r["horizon"]))
        def horizon_label(row):
            h = row["horizon"]
            return "all horizons" if h == "all" else f"{int(h) * 5} min" if dataset == "metr_la" else f"{h} h"
        labels = [f"{r['backbone']} {r['method']}" + (f" {r['variant']}" if r["variant"] != "trained" else "")
                  + (f" B={r['batch_size']}" if family == "sensitivity" else "")
                  + (f" {horizon_label(r)}" if dataset != "fd001" else "") for r in values]
        means = np.array([float(r["mean"]) for r in values])
        errors = np.array([float(r["standard_error"]) if r["standard_error"] else math.nan for r in values])
        fig, ax = plt.subplots(figsize=(max(6.0, 0.65 * len(values)), 4.4))
        positions = np.arange(len(values))
        finite = np.isfinite(means)
        valid_err = np.isfinite(errors)
        ax.bar(positions[finite], means[finite], color="#3478A3", label="Mean (persistence is deterministic)")
        with_error = finite & valid_err
        if with_error.any():
            ax.errorbar(positions[with_error], means[with_error], yerr=errors[with_error], fmt="none",
                        color="#222222", capsize=3, label="±1 standard error (training seeds)")
        for position in positions[~finite]:
            ax.text(position, 0, "nonfinite", rotation=90, va="bottom", ha="center")
        ax.set_xticks(positions, labels, rotation=55, ha="right")
        unit_label = "cycles" if units == "cycles" else units
        if metric == "mse":
            unit_label += "²"
        ax.set_ylabel(f"{metric.upper()} ({unit_label})" if metric != "rul_score_sum" else "Asymmetric RUL score (sum over engines)")
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
        ax.legend(fontsize=8)
        fig.tight_layout()
        stem = "_".join(_safe_name(k) for k in key)
        for extension in ("pdf", "svg"):
            fig.savefig(out / f"{stem}.{extension}", bbox_inches="tight")
        plt.close(fig)
        write_csv(out / f"{stem}.csv", values)
    selection_path = table_dir / "selections.json"
    validation_path = table_dir / "validation_records.csv"
    if not (selection_path.exists() and validation_path.exists()):
        return
    selected = read_json(selection_path)
    run_ids = {run_id for group in selected["groups"] for run_id in group["run_ids"]}
    curves = defaultdict(list)
    for row in read_csv(validation_path):
        if row["run_id"] in run_ids:
            curves[(row["dataset"], row["backbone"])].append(row)
    for key, values in sorted(curves.items()):
        grouped = defaultdict(list)
        for row in values:
            grouped[(row["method"], int(row["batch_size"]), int(row["epoch"]))].append(float(row["mse"]))
        plot_rows = []
        fig, ax = plt.subplots(figsize=(6.5, 4.1))
        labels = sorted({(m, b) for m, b, e in grouped})
        for method, batch in labels:
            y, se = [], []
            for epoch in VALIDATION_EPOCHS:
                v = np.asarray(grouped[(method, batch, epoch)], dtype=np.float64)
                if len(v) != 5:
                    raise ValueError("Selected validation curve is missing one or more seeds")
                y.append(v.mean())
                se.append(v.std(ddof=1) / math.sqrt(5))
                plot_rows.append(dict(dataset=key[0], backbone=key[1], method=method, batch_size=batch,
                                      epoch=epoch, mean=float(y[-1]), standard_error=float(se[-1]), n=5))
            line, = ax.plot(VALIDATION_EPOCHS, y, marker="o", ms=3, label=f"{method}, B={batch}")
            ax.fill_between(VALIDATION_EPOCHS, np.array(y)-se, np.array(y)+se, color=line.get_color(), alpha=.12)
        ax.set_xlabel("Stochastic training round" if key[0] == "electricity" else "Full training epoch")
        ax.set_ylabel("Ordinary validation MSE (standardized units)")
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
        fig.tight_layout()
        stem = "validation_" + "_".join(key)
        for extension in ("pdf", "svg"):
            fig.savefig(out / f"{stem}.{extension}", bbox_inches="tight")
        plt.close(fig)
        write_csv(out / f"{stem}.csv", plot_rows)
        theta_values = [r for r in values if r["method"].startswith("td_") and r.get("theta")]
        if theta_values:
            theta_groups = defaultdict(list)
            for row in theta_values:
                for coordinate, value in enumerate(json.loads(row["theta"])):
                    theta_groups[(row["method"], int(row["batch_size"]), int(row["epoch"]), coordinate)].append(float(value))
            fig, ax = plt.subplots(figsize=(6.5, 4.1))
            theta_rows = []
            for method, batch, coordinate in sorted({(m, b, c) for m, b, e, c in theta_groups}):
                means, errors = [], []
                for epoch in VALIDATION_EPOCHS:
                    v = np.asarray(theta_groups[(method, batch, epoch, coordinate)])
                    if len(v) != 5:
                        raise ValueError("Selected theta trajectory is missing seeds")
                    means.append(v.mean())
                    errors.append(v.std(ddof=1) / math.sqrt(5))
                    theta_rows.append(dict(dataset=key[0], backbone=key[1], method=method, batch_size=batch,
                                           epoch=epoch, coordinate=coordinate, mean=float(means[-1]),
                                           standard_error=float(errors[-1]), n=5))
                label = f"{method}, B={batch}, " + ("shared" if method == "td_scalar" else f"type {coordinate + 1}")
                line, = ax.plot(VALIDATION_EPOCHS, means, marker="o", ms=3, label=label)
                ax.fill_between(VALIDATION_EPOCHS, np.array(means)-errors, np.array(means)+errors,
                                color=line.get_color(), alpha=.12)
            ax.axhline(0, color="grey", linewidth=.7)
            ax.set_xlabel("Stochastic training round" if key[0] == "electricity" else "Full training epoch")
            ax.set_ylabel("Signed operator coefficient (mean ±1 SE)")
            ax.legend(fontsize=7)
            ax.grid(alpha=.2)
            fig.tight_layout()
            stem = "theta_" + "_".join(key)
            for extension in ("pdf", "svg"):
                fig.savefig(out / f"{stem}.{extension}", bbox_inches="tight")
            plt.close(fig)
            write_csv(out / f"{stem}.csv", theta_rows)


def analyze_results(metrics_file, table_dir, figures_dir=None):
    out = Path(table_dir)
    records, summaries = aggregate_metrics(read_csv(metrics_file))
    selection_path = out / "selections.json"
    if selection_path.exists():
        selection = read_json(selection_path)
        expected = {run for group in selection["groups"] for run in group["run_ids"]}
        observed = defaultdict(set)
        for row in records:
            if row.get("variant", "trained") == "trained" and row["method"] not in PERSISTENCE_METHODS:
                observed[row["run_id"]].add(row["split"])
        if set(observed) != expected or any(splits != {"train", "validation", "test"} for splits in observed.values()):
            raise ValueError("Endpoint records do not cover every selected run on train/validation/test")
    write_csv(out / "per_seed_metrics.csv", records)
    write_csv(out / "summary_metrics.csv", summaries)
    write_csv(out / "paired_comparisons.csv", paired_comparisons(records))
    write_csv(out / "batch_sensitivity.csv", [r for r in summaries if r["method"].startswith("td_") and r["split"] == "test"])
    verify_numerical_tables(out)
    write_json(out / "analysis_notes.json", {
        "uncertainty": "Sample SD and SE describe five training seeds on one fixed split with shared tuning.",
        "paired_intervals": "Descriptive two-sided t intervals, df=4; no multiplicity correction or independent-window claim.",
        "primary_batch": 256, "sensitivity_batch": 128,
        "selection": "One common configuration per dataset/backbone/method/fixed batch using mean validation MSE at 80,90,100 across five seeds.",
        "point_predictions": "Actual epoch/round-100 model; no test-residual correction or best-checkpoint substitution.",
        "theory_scope": "Neural networks, Adam, non-balanced application pairs and dependent inputs are empirical extensions; the manuscript's linear guarantees do not establish their performance.",
        "electricity_budget": "100 stochastic rounds of ceil(N/24) independent uniform source draws with replacement; this is not the exhaustive-training optimization trajectory.",
    })
    if figures_dir is not None:
        create_figures(out, figures_dir)
    return summaries


def verify_numerical_tables(table_dir):
    """Independently recompute mean/SD/SE using the standard library."""
    root = Path(table_dir)
    by_key = defaultdict(list)
    for row in read_csv(root / "per_seed_metrics.csv"):
        key = tuple(str(row[k]) for k in METRIC_KEYS)
        by_key[key].append(float(row["value"]))
    checked = 0
    for row in read_csv(root / "summary_metrics.csv"):
        key = tuple(str(row[k]) for k in METRIC_KEYS)
        values = by_key.pop(key)
        if int(row["n"]) != len(values):
            raise ValueError("Summary count does not match per-seed records")
        finite_count = sum(math.isfinite(v) for v in values)
        if int(row["finite_seeds"]) != finite_count:
            raise ValueError("Summary finite-seed count does not match per-seed records")
        if all(math.isfinite(v) for v in values):
            mean = statistics.mean(values)
            sd = statistics.stdev(values) if len(values) > 1 else math.nan
        else:
            # Independent IEEE arithmetic rules: any NaN or both signs of
            # infinity make the mean undefined. A single infinity sign is
            # retained. Seed variability with any nonfinite value is undefined.
            positive_inf = any(v == math.inf for v in values)
            negative_inf = any(v == -math.inf for v in values)
            mean = (math.nan if any(math.isnan(v) for v in values) or (positive_inf and negative_inf)
                    else math.inf if positive_inf else -math.inf)
            sd = math.nan
        expected = {"mean": mean, "sample_sd": sd, "standard_error": sd / math.sqrt(len(values))}
        for field, value in expected.items():
            saved = float(row[field]) if row[field] else math.nan
            if math.isnan(value) and math.isnan(saved):
                continue
            if not math.isclose(value, saved, rel_tol=1e-11, abs_tol=1e-12):
                raise ValueError(f"Independent recomputation mismatch in {key}: {field}")
        checked += 1
    if by_key:
        raise ValueError("A per-seed metric group was omitted from the summary")
    write_json(root / "verification.json", {"summary_groups_verified": checked,
               "method": "Independent standard-library mean/sample SD/SE from per_seed_metrics.csv",
               "nonfinite_values": "Preserved; never filtered to improve aggregate performance"})
    return checked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    select = commands.add_parser("select")
    select.add_argument("--manifest", default="configs/fulltd/manifest.json")
    select.add_argument("--results-dir", required=True)
    select.add_argument("--output-dir", required=True)
    select.add_argument("--dataset", default="all", choices=("all", "electricity", "metr_la", "fd001"))
    report = commands.add_parser("summarize")
    report.add_argument("--metrics", required=True)
    report.add_argument("--table-dir", required=True)
    report.add_argument("--figures-dir", required=True)
    args = parser.parse_args()
    if args.command == "select":
        select_candidates(args.manifest, args.results_dir, args.output_dir, args.dataset)
    else:
        analyze_results(args.metrics, args.table_dir, args.figures_dir)


if __name__ == "__main__":
    main()
