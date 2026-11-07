"""Unify completed FD001 Gamma runs with archived MSE/Gaussian-NLL baselines.

No optimizer, readout refit, Slurm, or test re-evaluation. Original baseline
scores come from the previously completed v7 FD001 experiment (B=256);
new Gamma variants use B=128. Preserve this distinction throughout.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
import hashlib
import io
import json
import math
import os
from pathlib import Path
import statistics
import zipfile

import numpy as np

from fd001_gamma.common import atomic_json, campaign_paths, file_lock, read_json, write_csv
from fd001_gamma.protocol import EVALUATIONS, SEEDS, VARIANTS

BASELINES = ("mse", "gaussian_nll")
METHODS = ("MSE (B256)", "Gaussian NLL (B256)",
           "TD typed (B128)", "TD raw26 (B128)", "TD representation (B128)")
GAMMA_NAMES = dict(zip(VARIANTS, METHODS[2:]))
BASELINE_NAMES = dict(zip(BASELINES, METHODS[:2]))
T95 = 2.7764451051977987


def locate_source(spec, results):
    """Prefer already imported baseline records over old HPC paths."""
    if spec:
        candidate = Path(spec).expanduser()
        if not candidate.exists():
            raise FileNotFoundError(f"Legacy baseline source does not exist: {candidate}")
        return candidate
    imported = results / "legacy_baselines"
    if (imported / "selections.json").is_file():
        return imported
    for candidate in (
        Path.home() / "scratch/fulltd/outputs_fd001",
        Path.home() / "scratch/fulltd/fd001_legacy_baselines_only.zip",
        Path.home() / "fd001_legacy_baselines_only.zip",
        Path.home() / "scratch/fulltd/fd001.zip",
    ):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Completed FD001 MSE/Gaussian-NLL outputs are not on this server. "
        "Copy the supplied fd001_legacy_baselines_only.zip to "
        "~/scratch/fulltd/ and rerun. No retraining is needed."
    )


def read_source(path):
    """Read a legacy outputs folder or its ZIP archive, without unpacking."""
    source = Path(path)
    names = ("selections.json", "metrics.csv", "validation_records.csv",
             "scores.csv", "candidate_accounting.csv")
    def load_file(name):
        if source.is_dir():
            return (source / name).read_bytes()
        with zipfile.ZipFile(source) as archive:
            file = "outputs_fd001/" + name
            if file not in archive.namelist() and name in archive.namelist():
                file = name
            return archive.read(file)
    raw = {name: load_file(name) for name in names}
    for optional in ("baseline_archive_provenance.json", "evaluation_manifest.json"):
        try:
            raw[optional] = load_file(optional)
        except (KeyError, FileNotFoundError):
            pass
    def table(name):
        return list(csv.DictReader(io.StringIO(raw[name].decode("utf-8"), newline="")))
    return dict(source=str(source), source_sha256={name: hashlib.sha256(value).hexdigest()
                                                  for name, value in raw.items()},
                selections=json.loads(raw["selections.json"]),
                metrics=table("metrics.csv"),
                validation=table("validation_records.csv"),
                scores=table("scores.csv"),
                accounting=table("candidate_accounting.csv"),
                archive_provenance=json.loads(raw["baseline_archive_provenance.json"])
                if "baseline_archive_provenance.json" in raw else None,
                evaluation_manifest=json.loads(raw["evaluation_manifest.json"])
                if "evaluation_manifest.json" in raw else None)


def validate_legacy(payload):
    """Recompute baseline selection from all 60 original candidates."""
    selection = payload["selections"]
    if selection.get("dataset_scope") != "fd001" or selection.get("selection_epochs") != [80, 90, 100]:
        raise ValueError("Legacy FD001 selection scope/epoch policy mismatch")
    groups = {g["method"]: g for g in selection["groups"] if g["method"] in BASELINES
              and g["dataset"] == "fd001" and g["backbone"] == "tcn"
              and int(g["batch_size"]) == 256}
    if set(groups) != set(BASELINES):
        raise ValueError("Both previously selected FD001 MSE and Gaussian-NLL groups are required")
    accounts, scores, validations = payload["accounting"], payload["scores"], payload["validation"]
    selected_ids = set()
    for method in BASELINES:
        group = groups[method]
        if group["state"] != "selected" or len(group["run_ids"]) != 5:
            raise ValueError(f"Baseline {method} was not five-seed validation selected")
        expected = {f"fd001_tcn_{method}_b256_h{group['configuration']}_s{s}" for s in SEEDS}
        if set(group["run_ids"]) != expected:
            raise ValueError(f"Unexpected baseline selected run IDs for {method}")
        selected_ids.update(expected)
        candidate = [r for r in accounts if r["method"] == method]
        if (len(candidate) != 30 or
            {(int(r["configuration"]), int(r["seed"])) for r in candidate} !=
                {(i, s) for i in range(6) for s in SEEDS} or
            any(r["state"] != "complete" or int(r["epochs_completed"]) != 100 for r in candidate)):
            raise ValueError(f"Baseline {method} must have 30 completed original candidates")
        score_rows = [r for r in scores if r["method"] == method]
        if len(score_rows) != 6:
            raise ValueError(f"Baseline {method} lacks its six hyperparameter scores")
        computed = {}
        for i in range(6):
            run_ids = {f"fd001_tcn_{method}_b256_h{i}_s{s}" for s in SEEDS}
            rs = [r for r in validations if r["run_id"] in run_ids]
            if (len(rs) != 50 or
                {(int(r["seed"]), int(r["epoch"])) for r in rs} !=
                    {(seed, epoch) for seed in SEEDS for epoch in EVALUATIONS}):
                raise ValueError(f"Baseline {method} configuration {i} lacks validation evidence")
            late = [float(r["mse"]) for r in rs if int(r["epoch"]) in (80, 90, 100)]
            computed[i] = statistics.mean(late)
            published = next((r for r in score_rows if int(r["configuration"]) == i), None)
            if published is None or not math.isclose(float(published["score"]), computed[i],
                                                      rel_tol=1e-9, abs_tol=1e-11):
                raise ValueError(f"Legacy {method} validation score does not reproduce")
        best = min(computed, key=lambda i: (computed[i], i))
        if best != int(group["configuration"]) or not math.isclose(
                float(group["score"]), computed[best], rel_tol=1e-9, abs_tol=1e-11):
            raise ValueError(f"Legacy {method} selection is inconsistent with its validation data")
    chosen = [r for r in payload["metrics"] if r["run_id"] in selected_ids]
    if (len(chosen) != 420 or any(r.get("variant") != "trained" for r in chosen) or
            len({r["run_id"] for r in chosen}) != 10):
        raise ValueError("Expected 420 already evaluated, non-refit baseline metric rows")
    if len({(r["run_id"], r["split"], r["units"], str(r["horizon"]), r["metric"])
            for r in chosen}) != len(chosen):
        raise ValueError("Duplicate baseline metric key")
    if any(r["dataset"] != "fd001" or r["backbone"] != "tcn"
           or r["method"] not in BASELINES or int(r["batch_size"]) != 256 for r in chosen):
        raise ValueError("Legacy metrics include a different experiment")
    if any(r["seed"] != r["run_id"].split("_s")[-1] for r in chosen):
        raise ValueError("Baseline seed identity mismatch")
    selected_validation = [r for r in validations if r["run_id"] in selected_ids]
    if len(selected_validation) != 100:
        raise ValueError("Exactly ten validation epochs per selected baseline seed are required")
    return groups, chosen, selected_validation


def archive_baselines(payload, results_dir):
    """Keep a baseline-only copy next to Gamma results, preserving original files."""
    groups, metrics, validation = validate_legacy(payload)
    destination = Path(results_dir) / "legacy_baselines"
    # Don't overwrite an existing import with different records.
    if (destination / "selections.json").exists():
        previous = read_source(destination)
        _, old_metrics, old_validation = validate_legacy(previous)
        def canonical(rows):
            return sorted(json.dumps(row, sort_keys=True) for row in rows)
        if canonical(old_metrics) != canonical(metrics) or canonical(old_validation) != canonical(validation):
            raise ValueError("Different baseline evidence already imported; refusing overwrite")
        return previous
    destination.mkdir(parents=True, exist_ok=True)
    # Baseline original tables remain untouched; copy *only* MSE/Gaussian-NLL,
    # excluding old TD and refit artifacts.
    atomic_json(destination / "selections.json", dict(
        dataset_scope="fd001", selection_epochs=[80, 90, 100],
        groups=[groups[name] for name in BASELINES]))
    write_csv(destination / "metrics.csv", metrics)
    write_csv(destination / "validation_records.csv",
              [r for r in payload["validation"] if r["method"] in BASELINES])
    write_csv(destination / "scores.csv",
              [r for r in payload["scores"] if r["method"] in BASELINES])
    write_csv(destination / "candidate_accounting.csv",
              [r for r in payload["accounting"] if r["method"] in BASELINES])
    old_source = payload["archive_provenance"] or {}
    fingerprint = (old_source.get("legacy_data_fingerprint") or
        ((payload["evaluation_manifest"] or {}).get("data_fingerprints", {})
         .get("fd001", {}).get("fingerprint")))
    atomic_json(destination / "import_provenance.json", dict(
        legacy_source=payload["source"], original_file_sha256=payload["source_sha256"],
        source_archive_sha256=old_source.get("source_archive_sha256"),
        original_checkpoint_sha256=old_source.get("selected_checkpoint_sha256"),
        legacy_data_fingerprint=fingerprint, imported_methods=list(BASELINES),
        refit_included=False, old_td_included=False,
        warning="Archived completed-model metrics, not freshly evaluated checkpoints"))
    return read_source(destination)


def gamma_records(gamma_dir):
    root = Path(gamma_dir)
    selected = read_json(root / "selections.json")
    choices = {r["gamma_variant"]: r for r in selected["per_variant"]}
    if set(choices) != set(VARIANTS) or any(len(r["run_ids"]) != 5 for r in choices.values()):
        raise ValueError("Gamma five-seed selection is incomplete; run Gamma analysis first")
    selected_ids = {rid for group in choices.values() for rid in group["run_ids"]}
    def load_csv(name):
        with (root / name).open(newline="", encoding="utf-8") as stream:
            return list(csv.DictReader(stream))
    metrics = load_csv("metrics.csv")
    validation = load_csv("validation_records.csv")
    if (len(metrics) == 0 or {r["run_id"] for r in metrics} != selected_ids or
            any(r["gamma_variant"] not in VARIANTS for r in metrics)):
        raise ValueError("Gamma metrics do not correspond to the five-seed selected runs")
    chosen_validation = [r for r in validation if r["run_id"] in selected_ids]
    if (len(chosen_validation) != 150 or
            len({(r["run_id"], int(r["epoch"])) for r in chosen_validation}) != 150):
        raise ValueError("Gamma selected validation curves must cover 15 seeds x ten epochs")
    return metrics, chosen_validation, selected


def normalize_metrics(gamma_rows, baseline_rows):
    all_rows = []
    for source, records in (("gamma_2026", gamma_rows), ("legacy_v7", baseline_rows)):
        for row in records:
            variant = row.get("gamma_variant")
            name = GAMMA_NAMES[variant] if source == "gamma_2026" else BASELINE_NAMES[row["method"]]
            if row.get("variant", "trained") != "trained":
                raise ValueError("Refitted/readout results must not enter FD001 comparisons")
            value = float(row["value"])
            if not math.isfinite(value):
                raise ValueError(f"Nonfinite metric in selected run {row['run_id']}")
            all_rows.append(dict(run_id=row["run_id"], method=name, source=source,
                                 training_method="td_full" if variant else row["method"],
                                 batch_size=128 if variant else 256,
                                 seed=int(row["seed"]), split=row["split"],
                                 units=row["units"], horizon=str(row["horizon"]),
                                 metric=row["metric"], value=value,
                                 metric_status=row.get("metric_status", "finite")))
    keys = ("method", "split", "units", "horizon", "metric")
    groups = defaultdict(list)
    for row in all_rows:
        groups[tuple(row[k] for k in keys)].append(row)
    summary = []
    for key, members in sorted(groups.items()):
        members.sort(key=lambda row: row["seed"])
        if [r["seed"] for r in members] != list(SEEDS):
            raise ValueError(f"Expected five distinct seeds in {key}")
        values = [r["value"] for r in members]
        mean = statistics.mean(values)
        sd = statistics.stdev(values)
        summary.append(dict(zip(keys, key), n=5, mean=mean, sample_sd=sd,
                            standard_error=sd / math.sqrt(5),
                            source=members[0]["source"],
                            batch_size=members[0]["batch_size"]))
    if {r["method"] for r in all_rows} != set(METHODS):
        raise ValueError("All three Gamma variants and two baselines are required")
    return all_rows, summary


def paired_differences(records):
    indexed = defaultdict(dict)
    for row in records:
        key = (row["split"], row["units"], row["horizon"], row["metric"])
        indexed[key].setdefault(row["method"], {})[row["seed"]] = row["value"]
    output = []
    for key, methods in sorted(indexed.items()):
        for gamma in METHODS[2:]:
            for baseline in METHODS[:2]:
                if gamma not in methods or baseline not in methods:
                    raise ValueError(f"Incomplete comparison {gamma}/{baseline}/{key}")
                differences = [methods[gamma][s] - methods[baseline][s] for s in SEEDS]
                baseline_mean = statistics.mean(methods[baseline][s] for s in SEEDS)
                mean = statistics.mean(differences)
                sd = statistics.stdev(differences)
                se = sd / math.sqrt(5)
                output.append(dict(zip(("split", "units", "horizon", "metric"), key),
                                   gamma_method=gamma, baseline_method=baseline,
                                   n=5, difference="Gamma minus baseline (negative favors Gamma)",
                                   mean_difference=mean, sample_sd=sd, standard_error=se,
                                   descriptive_t95_lower=mean - T95 * se,
                                   descriptive_t95_upper=mean + T95 * se,
                                   relative_improvement=(baseline_mean -
                                       statistics.mean(methods[gamma][s] for s in SEEDS)) /
                                       baseline_mean if baseline_mean else None))
    return output


def combined_validation(gamma_rows, legacy_rows):
    entries = []
    for source, records in (("gamma_2026", gamma_rows), ("legacy_v7", legacy_rows)):
        for row in records:
            variant = row.get("gamma_variant")
            method = GAMMA_NAMES[variant] if source == "gamma_2026" else BASELINE_NAMES[row["method"]]
            entries.append(dict(method=method, seed=int(row["seed"]),
                                epoch=int(row["epoch"]), mse=float(row["mse"]),
                                mae=float(row["mae"]), source=source))
    groups = defaultdict(list)
    for row in entries:
        groups[(row["method"], row["epoch"])].append(row)
    if set(x["method"] for x in entries) != set(METHODS) or len(groups) != 50:
        raise ValueError("Five methods x ten validation epochs are required")
    means = []
    for (method, epoch), rows in sorted(groups.items()):
        if sorted(r["seed"] for r in rows) != list(SEEDS):
            raise ValueError(f"Invalid five-seed validation group: {method} / {epoch}")
        for metric in ("mse", "mae"):
            vals = [float(r[metric]) for r in rows]
            means.append(dict(method=method, epoch=epoch, metric=metric,
                              mean=statistics.mean(vals),
                              standard_error=statistics.stdev(vals) / math.sqrt(5), n=5))
    return entries, means


def make_figures(output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = Path(output_dir)
    def rows(name):
        with (out / name).open(newline="", encoding="utf-8") as stream:
            return list(csv.DictReader(stream))
    summary = rows("summary_metrics.csv")
    validation = rows("validation_summary.csv")
    figures = out / "figures"
    figures.mkdir(exist_ok=True, parents=True)
    def save(fig, name, data):
        fig.tight_layout()
        for ext in ("pdf", "svg"):
            fig.savefig(figures / f"{name}.{ext}", bbox_inches="tight")
        plt.close(fig)
        write_csv(figures / f"{name}.csv", data)
    for units, metric in (("cycles", "rmse"), ("cycles", "mae"),
                          ("score", "rul_score_sum")):
        chosen = [next((r for r in summary if r["method"] == method and
                        r["split"] == "test" and r["units"] == units and
                        r["horizon"] == "all" and r["metric"] == metric), None)
                  for method in METHODS]
        if any(r is None for r in chosen):
            raise ValueError(f"Missing test metric for comparison: {units}/{metric}")
        fig, ax = plt.subplots(figsize=(9.6, 4.5))
        x = np.arange(len(METHODS))
        means = [float(row["mean"]) for row in chosen]
        errors = [float(row["standard_error"]) for row in chosen]
        ax.bar(x, means)
        ax.errorbar(x, means, yerr=errors, fmt="none", capsize=4)
        ax.set_xticks(x, METHODS, rotation=15, ha="right")
        ax.set_ylabel("Test " + ("asymmetric RUL score (sum)" if metric.startswith("rul_")
                                 else metric.upper() + " (cycles)"))
        ax.set_title("FD001: previous baselines (B256) vs new TD Gamma (B128)")
        save(fig, f"fd001_all_test_{metric}", chosen)
    for metric in ("mse", "mae"):
        fig, ax = plt.subplots(figsize=(8.0, 4.8))
        plot_rows = []
        for method in METHODS:
            series = sorted((r for r in validation if r["method"] == method and
                             r["metric"] == metric), key=lambda r: int(r["epoch"]))
            if len(series) != 10:
                raise ValueError(f"Missing validation epochs for {method}")
            xx = [int(r["epoch"]) for r in series]
            means = np.array([float(r["mean"]) for r in series])
            se = np.array([float(r["standard_error"]) for r in series])
            ax.plot(xx, means, marker="o", ms=3, label=method)
            ax.fill_between(xx, means - se, means + se, alpha=.10)
            plot_rows.extend(series)
        ax.set_xlabel("Completed full training epoch")
        ax.set_ylabel(f"Validation {metric.upper()} (standardized)")
        ax.legend(fontsize=8)
        save(fig, f"fd001_all_validation_{metric}", plot_rows)
    print("Combined FD001 figures:", figures)


def run(*, source=None, output=None, results=None, plots_only=False):
    data, gamma_results, gamma_out = campaign_paths()
    result_root = Path(results) if results else gamma_results
    output_root = Path(output) if output else gamma_out / "combined"
    if plots_only:
        for name in ("summary_metrics.csv", "validation_summary.csv"):
            if not (output_root / name).is_file():
                raise FileNotFoundError(f"Run complete unified FD001 analysis first: {output_root / name}")
        make_figures(output_root)
        return
    legacy = archive_baselines(read_source(locate_source(source, result_root)), result_root)
    groups, legacy_metrics, legacy_validation = validate_legacy(legacy)
    gamma_metrics, gamma_validation, gamma_selection = gamma_records(gamma_out)
    # Historical and new runs used the same FD001 processing protocol.
    # Verify this against the actual prepared FD001 data, not its folder name.
    provenance_path = result_root / "legacy_baselines" / "import_provenance.json"
    provenance = read_json(provenance_path)
    expected_fingerprint = provenance.get("legacy_data_fingerprint")
    if not expected_fingerprint:
        raise ValueError("Legacy baseline source is missing its FD001 data fingerprint; "
                         "use the supplied baseline-only ZIP or original evaluation_manifest.json")
    from fulltd.data import load_dataset
    actual = load_dataset("fd001", data, "train").metadata.get("fingerprint")
    if expected_fingerprint != actual:
        raise ValueError("Legacy MSE/NLL and Gamma dataset fingerprints differ; comparison blocked")
    metrics, summary = normalize_metrics(gamma_metrics, legacy_metrics)
    val_rows, val_summary = combined_validation(gamma_validation, legacy_validation)
    output_root.mkdir(parents=True, exist_ok=True)
    with file_lock(output_root / ".combined_analysis.lock"):
        write_csv(output_root / "metrics.csv", metrics)
        write_csv(output_root / "summary_metrics.csv", summary)
        write_csv(output_root / "paired_comparisons.csv", paired_differences(metrics))
        write_csv(output_root / "validation_records.csv", val_rows)
        write_csv(output_root / "validation_summary.csv", val_summary)
        atomic_json(output_root / "selections.json", dict(
            gamma=gamma_selection, historical_baselines=[groups[m] for m in BASELINES],
            selection_policy="Each experiment independently selected by five-seed validation MSE at epochs 80,90,100",
            comparison="FD001 TCN: MSE/NLL B256 vs three Gamma TD B128",
            no_old_td=True, no_refits=True))
        atomic_json(output_root / "provenance.json", dict(
            original_baseline_source=provenance,
            new_gamma_outputs=str(gamma_out),
            original_baseline_candidates=60, new_gamma_candidates=180,
            selected_model_seeds=25, trained_model_only=True,
            differences_are_descriptive=True,
            note="Two different training batch sizes; results are not a controlled same-batch comparison"))
        make_figures(output_root)
    print("Combined FD001 analysis complete (5 methods, 25 trained seed models):", output_root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("run", "plots"), nargs="?", default="run")
    parser.add_argument("--legacy", help="Old outputs_fd001 folder or ZIP archive, optional after import")
    parser.add_argument("--results-dir", type=Path, help="FD001 Gamma results root")
    parser.add_argument("--output-dir", type=Path, help="Combined output root")
    args = parser.parse_args()
    run(source=args.legacy, results=args.results_dir, output=args.output_dir,
        plots_only=args.mode == "plots")


if __name__ == "__main__":
    main()
