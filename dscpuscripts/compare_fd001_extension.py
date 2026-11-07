"""Compare four Gamma choices with rerun MSE/NLL baselines on the same AMD host.

Six methods, five seeds each, validation-selected HPs, no refit/retraining.
X axis is cumulative optimizer UPDATES (B256 baselines vs B128 Gamma).
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import math
from pathlib import Path
import statistics

from dscpuscripts.standard_runner import paths as baseline_paths
from dscpuscripts.fd001_report import read_outputs, selected_records, target_scale
from fd001_gamma.common import atomic_json, file_lock, read_json, write_csv, campaign_paths as original_gamma_paths
from fd001_gamma_ext.runtime import campaign_paths
from fd001_gamma_ext.analysis import VARIANTS

METHODS = ("MSE (B256)", "Gaussian NLL (B256)",
           "TD typed (B128)", "TD raw26 (B128)",
           "TD representation (B128)", "TD pairwise (B128)")
BASE_NAMES = {"mse": METHODS[0], "gaussian_nll": METHODS[1]}
GAMMA_NAMES = dict(zip(VARIANTS, METHODS[2:]))
METRICS = ("rmse", "mae", "rul_score_sum")


def csv_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def report(out):
    baseline_data, baseline_results, baseline_out = baseline_paths("fd001_baselines")
    _, extension_results, extension_out = campaign_paths()
    if not (extension_out / "selections.json").exists():
        raise FileNotFoundError("First run dscpuscripts/analyze_fd001_gamma_extension.sh")
    if not (baseline_out / "selections.json").exists():
        raise FileNotFoundError("First run dscpuscripts/analyze_standard.sh fd001_baselines")
    ext_selection = read_json(extension_out / "selections.json")
    choices = ext_selection["per_variant"]
    if {c["gamma_variant"] for c in choices} != set(VARIANTS):
        raise ValueError("All four Gamma variants must have been selected")
    baseline_payload = read_outputs(baseline_out)
    base_metrics, base_curve, base_selections, base_scale = selected_records(baseline_payload, "baseline")
    ext_metrics = csv_rows(extension_out / "metrics.csv")
    ext_valid = csv_rows(extension_out / "validation_rmse.csv")
    ext_scale = target_scale(ext_metrics)
    if not math.isclose(ext_scale, base_scale, rel_tol=1e-7):
        raise ValueError("Baseline and Gamma target scale disagree")
    # Strict same-data comparison: directly examine the saved training identities.
    _, original_gamma_results, _ = original_gamma_paths()
    gamma_ident = set()
    for choice in choices:
        root = original_gamma_results if choice["origin"] == "original" else extension_results
        for rid in choice["run_ids"]:
            status = read_json(root / rid / "status.json")
            gamma_ident.add((status["identity"]["data_sha256"], status["identity"]["validation_sha256"]))
    base_ident = set()
    for group in baseline_payload["selection"]["groups"]:
        if group["method"] not in BASE_NAMES:
            continue
        for rid in group["run_ids"]:
            status = read_json(baseline_results / rid / "status.json")
            base_ident.add((status["identity"]["data_sha256"], status["identity"]["validation_sha256"]))
    if len(gamma_ident) != 1 or base_ident != gamma_ident:
        raise ValueError("Baseline and Gamma processed training/validation identities differ")

    values = []
    for row in base_metrics:
        values.append(dict(method=BASE_NAMES[row["method"]], metric=row["metric"],
                           mean=row["mean"], sample_sd=row["sample_sd"],
                           standard_error=row["standard_error"], n=5))
    for variant in VARIANTS:
        selected_ids = set(next(c["run_ids"] for c in choices if c["gamma_variant"] == variant))
        for metric in METRICS:
            samples = [float(r["value"]) for r in ext_metrics if
                       r["run_id"] in selected_ids and r["split"] == "test" and
                       r["horizon"] == "all" and r["metric"] == metric and
                       r["units"] == ("score" if metric == "rul_score_sum" else "cycles")]
            if len(samples) != 5:
                raise ValueError(f"Missing five test seed metrics: {variant} {metric}")
            sd = statistics.stdev(samples)
            values.append(dict(method=GAMMA_NAMES[variant], metric=metric,
                               mean=statistics.mean(samples), sample_sd=sd,
                               standard_error=sd / math.sqrt(5), n=5))
    if len(values) != 6 * len(METRICS):
        raise ValueError("Six methods x three test metrics are required")

    curve = []
    for row in base_curve:
        method = row["method"]
        group = next(g for g in baseline_payload["selection"]["groups"]
                     if g["method"] == method and g["dataset"] == "fd001")
        updates = {int(r["optimizer_steps"]) for r in baseline_payload["validation"]
                   if r["run_id"] in group["run_ids"] and int(r["epoch"]) == row["epoch"]}
        if len(updates) != 1:
            raise ValueError(f"Baseline {method} has missing/mismatched optimizer updates")
        curve.append(dict(method=BASE_NAMES[method], optimizer_steps=updates.pop(),
                          epoch=row["epoch"], mean=row["mean"],
                          standard_error=row["standard_error"], n=5))
    for variant in VARIANTS:
        groups = defaultdict(list)
        for r in ext_valid:
            if r["gamma_variant"] == variant:
                groups[int(r["optimizer_steps"])].append(r)
        if len(groups) != 10:
            raise ValueError(f"Incomplete extension curve: {variant}")
        for updates, records in sorted(groups.items()):
            if sorted(int(r["seed"]) for r in records) != list(range(5)):
                raise ValueError("Extension curve missing seeds")
            rmses = [float(r["rmse_cycles"]) for r in records]
            curve.append(dict(method=GAMMA_NAMES[variant], optimizer_steps=updates,
                              epoch=int(records[0]["epoch"]), mean=statistics.mean(rmses),
                              standard_error=statistics.stdev(rmses) / math.sqrt(5), n=5))
    return values, curve, choices, base_selections, base_scale



def plot(out):
    """All FD001 comparison plots reuse the same validated rendering function."""
    from dscpuscripts.fd001_report import plot_learning_curve
    curves = csv_rows(Path(out) / "validation_rmse_updates.csv")
    plot_learning_curve(curves, Path(out) / "figures" /
                        "fd001_six_methods_validation_rmse_updates")


def run(mode):
    from dscpuscripts.result_paths import configure_environment
    configure_environment()
    from fd001_gamma.common import cpu_environment
    cpu_environment()
    _, _, gamma_out = campaign_paths()
    # Reporting is dataset-owned, not an extension- or machine-specific path.
    out = gamma_out.parent / "comparison"
    if mode == "plots":
        if not (out / "validation_rmse_updates.csv").is_file():
            raise FileNotFoundError("Run combined analysis first")
        plot(out)
        return
    table, curve, choices, baseline_selections, scale = report(out)
    with file_lock(out / ".comparison.lock"):
        write_csv(out / "test_performance.csv", table)
        write_csv(out / "validation_rmse_updates.csv", curve)
        write_csv(out / "selected_hyperparameters.csv", [
            dict(method=BASE_NAMES[g["method"]], alpha=g["alpha"], beta=g.get("beta"),
                 batch_size=256, selection_mse=g["validation_selection_mse"]) for g in baseline_selections
        ] + [
            dict(method=GAMMA_NAMES[g["gamma_variant"]], alpha=g["alpha"], beta=g["beta"],
                 batch_size=128, selection_mse=g["score"], origin=g["origin"]) for g in choices
        ])
        atomic_json(out / "analysis_notes.json", dict(target_scale=scale,
            selection="each method: mean validation MSE epochs 80/90/100 across five seeds",
            x_axis="true cumulative Adam optimizer updates, not epochs",
            final="only epoch-100 models, no refit",
            comparison_caveat="B256 baselines have half the optimizer steps per full epoch of B128 Gamma"))
        plot(out)
    print("Six-method FD001 comparison:", out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("run", "plots"), default="run", nargs="?")
    args = parser.parse_args()
    run(args.mode)


if __name__ == "__main__":
    main()
