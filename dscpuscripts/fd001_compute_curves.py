"""FD001 validation learning curves versus an *approximate* compute budget.

No per-checkpoint wall-clock timings were recorded by the FD001 campaign, so
this plots an explicitly labeled predictor forward/backward example-pass proxy,
NOT measured seconds. Its intent is to check how much work a method spends to
reach a given validation error, with all four methods forced to B256.

MSE / Gaussian NLL: 1 predictor example-pass per source.
TD typed / TD pairwise: 2 predictor example-passes (source and partner).
That proxy omits operator head, variance head, validation and checkpoint costs,
as well as hardware efficiency. It is not a measurement of wall-clock time.

Uses the original five-seed final-quarter validation-MSE HP rule within B256,
without looking at test metrics. Does not alter any candidate/training record.
"""
from __future__ import annotations

from collections import defaultdict
import math
from pathlib import Path
import statistics

from dscpuscripts.common import read_json
from dscpuscripts.fd001_analysis import audit_and_select
from dscpuscripts.fd001_protocol import METHODS, SEEDS
from dscpuscripts.result_paths import paths as dataset_paths


def processed_sources(n_sources: int, batch: int, optimizer_steps: int) -> int:
    """Exactly account for the smaller final source batch in every epoch."""
    if n_sources < 1 or batch < 1 or optimizer_steps < 0:
        raise ValueError("Positive sample and batch counts; nonnegative update count required")
    steps_per_epoch = math.ceil(n_sources / batch)
    complete, remaining = divmod(optimizer_steps, steps_per_epoch)
    return complete * n_sources + min(n_sources, remaining * batch)


def relative_compute(method: str, n_sources: int, batch: int, steps: int) -> float:
    """Normalize against the number of examples in MSE B256 100 epochs."""
    passes = 2 if method.startswith("td_") else 1
    return passes * processed_sources(n_sources, batch, steps) / (100 * n_sources)


def collect():
    """Audit real candidates and choose HPs by canonical validation rule in B256."""
    _, hp_scores = audit_and_select(dataset_paths("fd001")[1])
    _, results, output = dataset_paths("fd001")
    curves = {}
    chosen = []
    for method in METHODS:
        candidates = [row for row in hp_scores
                      if row["method"] == method and row["batch_size"] == 256]
        if not candidates:
            raise ValueError(f"No B256 candidates for {method}")
        best = min(candidates, key=lambda row: (
            row["score"], row["alpha"], row["beta"] or 0.0, row["configuration"]))
        if not math.isfinite(best["score"]) or len(best["run_ids"]) != len(SEEDS):
            raise ValueError(f"No five-seed finite B256 HP for {method}")
        grouped = defaultdict(list)
        expected_n = None
        for run_id in best["run_ids"]:
            folder = results / run_id
            status = read_json(folder / "status.json")
            n_sources = int(status["n_sources"])
            if expected_n is not None and expected_n != n_sources:
                raise ValueError("Inconsistent training-source counts across seeds")
            expected_n = n_sources
            curve = read_json(folder / "update_curve.json")
            for row in curve:
                step = int(row["optimizer_steps"])
                for metric in ("rmse_cycles", "rul_score_sum"):
                    value = float(row[metric])
                    grouped[(step, metric)].append(value)
        series = defaultdict(list)
        for (step, metric), values in sorted(grouped.items()):
            if len(values) != len(SEEDS):
                raise ValueError(f"Missing seed for {method}, update {step}")
            se = statistics.stdev(values) / math.sqrt(len(values))
            series[metric].append(dict(x=relative_compute(method, expected_n, 256, step),
                                       mean=statistics.mean(values), se=se))
        curves[method] = series
        chosen.append(dict(method=method, batch_size=256,
                           alpha=best["alpha"], beta=best["beta"],
                           selection_mse=best["score"], run_ids=best["run_ids"]))
    return curves, chosen, output


def plot(curves, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    figures = Path(output) / "figures" / "compute_proxy"
    figures.mkdir(parents=True, exist_ok=True)
    labels = {"mse":"MSE B256", "gaussian_nll":"Gaussian NLL B256",
              "td_typed":"TD typed B256", "td_pairwise":"TD pairwise B256"}
    for metric, ylabel in (("rmse_cycles", "Validation RMSE (cycles)"),
                           ("rul_score_sum", "Validation RUL score (sum)")):
        for view in ("zoom", "full"):
            fig, ax = plt.subplots(figsize=(9.4, 5.2))
            for method in METHODS:
                seq = curves[method][metric]
                x = np.asarray([a["x"] for a in seq], dtype=float)
                y = np.asarray([a["mean"] for a in seq], dtype=float)
                se = np.asarray([a["se"] for a in seq], dtype=float)
                ax.plot(x, y, lw=1.9, marker="o", markersize=3,
                        label=labels[method])
                if np.isfinite(y).all() and np.isfinite(se).all():
                    ax.fill_between(x, y-se, y+se, alpha=0.11)
            ax.set_xlim(0, 2.04)  # Fixed comparative budget, never hide early updates.
            ax.set_xlabel("Approximate predictor training compute (MSE B256 at 100 epochs = 1)")
            ax.set_ylabel(ylabel)
            if metric=="rmse_cycles" and view=="zoom":
                ax.set_ylim(25,40)
            if metric=="rul_score_sum":
                ax.set_yscale("symlog",linthresh=1000)
                if view=="zoom":
                    ax.set_ylim(3000,250000)
            ax.axvline(1,ls="--",lw=1,alpha=.55,label="B256 baseline 100-epoch compute")
            ax.grid(alpha=.25)
            ax.legend(loc="best",fontsize=8)
            ax.set_title("Proxy only — not measured CPU/GPU wall-clock time",fontsize=10)
            fig.tight_layout()
            name=f"fd001_b256_{metric}_vs_compute_proxy"
            if view=="full":
                name+="_full"
            for ext in ("pdf","png","svg"):
                fig.savefig(figures/f"{name}.{ext}",dpi=180,bbox_inches="tight")
            plt.close(fig)
    return figures


def main():
    from dscpuscripts.common import cpu_environment, write_csv
    cpu_environment()
    curves, choices, output = collect()
    figures = plot(curves, output)
    # Evidence, not a replacement for the original selection files.
    write_csv(figures / "b256_selected_hyperparameters.csv",choices)
    print("B256-only proxy curves saved:", figures)
    print("IMPORTANT: This is not measured wall-clock or total hardware compute.")


if __name__ == "__main__":
    main()
