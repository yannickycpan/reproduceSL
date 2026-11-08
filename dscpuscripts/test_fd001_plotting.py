"""Smoketest both late-training and full-range FD001 plots."""
import csv

import pytest

matplotlib = pytest.importorskip("matplotlib")

from dscpuscripts.fd001_analysis import METHODS, plot


def _csv(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def test_late_training_plots_keep_full_figure_companions(tmp_path):
    metric_names = ("rmse_cycles", "mae_cycles", "rul_score_sum")
    curves = []
    for method_index, method in enumerate(METHODS):
        for metric in metric_names:
            for step in (0, 500, 1000, 1500):
                huge_start = 1e48 if metric == "rul_score_sum" else 300.0
                late_value = (3000 + 200*method_index if metric == "rul_score_sum"
                              else 20 + 2*method_index)
                curves.append(dict(method=method, metric=metric,
                                   optimizer_steps=step, n=5,
                                   mean=huge_start if step == 0 else late_value + 500/step,
                                   sample_sd=1.0, standard_error=0.5,
                                   finite_seeds=5))
    _csv(tmp_path / "validation_learning_curves.csv", curves)
    _csv(tmp_path / "selected_hyperparameters.csv", [
        dict(method=m, alpha=0.001, beta="", batch_size=256,
             selection_mse=0.25, run_ids="[]") for m in METHODS
    ])
    _csv(tmp_path / "performance_summary.csv", [
        dict(method=m, split=split, metric=metric,
             units="score" if metric=="rul_score_sum" else "cycles",
             n=5, mean=1234 if metric=="rul_score_sum" else 30,
             sample_sd=1, standard_error=0.5, finite_seeds=5)
        for m in METHODS for split in ("validation","test")
        for metric in ("rmse","mae","rul_score_sum")
    ])
    plot(tmp_path)
    for metric in metric_names:
        for suffix in ("", "_full"):
            for ext in ("pdf", "svg"):
                path=tmp_path/"figures"/f"validation_{metric}{suffix}_vs_updates.{ext}"
                assert path.is_file() and path.stat().st_size>1000
