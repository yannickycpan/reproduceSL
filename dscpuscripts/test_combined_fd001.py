"""Synthetic integration tests for unified FD001 baseline + Gamma reporting.

No refit or model training. The old TD and readout-refit rows are excluded.
"""
import csv
import io
import json
from pathlib import Path
import zipfile

import pytest

from dscpuscripts import combined_fd001 as combined
from fd001_gamma.common import atomic_json, read_json, write_csv
from fd001_gamma.protocol import EVALUATIONS, SEEDS, VARIANTS


def old_fixture(tmp_path):
    source = tmp_path / "outputs_fd001"
    source.mkdir()
    selections, metrics, validation, scores, accounting = [], [], [], [], []
    for method in combined.BASELINES:
        run_ids = []
        for configuration in range(6):
            scores.append(dict(method=method, configuration=configuration,
                               score=.2 + .01 * configuration, alpha=.001))
            for seed in SEEDS:
                rid = f"fd001_tcn_{method}_b256_h{configuration}_s{seed}"
                accounting.append(dict(run_id=rid, method=method,
                                       configuration=configuration, seed=seed,
                                       state="complete", epochs_completed=100))
                for epoch in EVALUATIONS:
                    validation.append(dict(run_id=rid, method=method,
                                           seed=seed, epoch=epoch, mse=.2 + .01 * configuration,
                                           mae=.3 + .01 * configuration))
                if configuration == 0:
                    run_ids.append(rid)
                    for split in ("train", "validation", "test"):
                        for units, horizon, metric, v in (
                            ("cycles", "1", "rmse", 30.),
                            ("cycles", "all", "rmse", 30.),
                            ("cycles", "1", "mae", 20.),
                            ("cycles", "all", "mae", 20.),
                            ("score", "all", "rul_score_sum", 42.),
                            ("score", "all", "rul_score_mean", .4),
                            ("standardized", "1", "rmse", .3),
                            ("standardized", "all", "rmse", .3),
                            ("standardized", "1", "mae", .2),
                            ("standardized", "all", "mae", .2),
                            ("standardized", "1", "mse", .1),
                            ("standardized", "all", "mse", .1),
                            ("cycles", "1", "mse", 900.),
                            ("cycles", "all", "mse", 900.),
                        ):
                            metrics.append(dict(run_id=rid, method=method, batch_size=256,
                                                seed=seed, split=split, units=units,
                                                horizon=horizon, metric=metric, value=v,
                                                variant="trained", dataset="fd001", backbone="tcn"))
    for method in combined.BASELINES:
        selections.append(dict(method=method, dataset="fd001", backbone="tcn",
                               batch_size=256, state="selected", configuration=0,
                               score=.2, run_ids=[f"fd001_tcn_{method}_b256_h0_s{s}" for s in SEEDS]))
    atomic_json(source / "selections.json", dict(dataset_scope="fd001",
                 selection_epochs=[80, 90, 100], groups=selections))
    write_csv(source / "metrics.csv", metrics)
    write_csv(source / "validation_records.csv", validation)
    write_csv(source / "scores.csv", scores)
    write_csv(source / "candidate_accounting.csv", accounting)
    return source


def gamma_fixture(tmp_path):
    out = tmp_path / "gamma_out"
    out.mkdir()
    selections, metrics, validation = [], [], []
    for variant in VARIANTS:
        run_ids = []
        for seed in SEEDS:
            rid = f"gamma_{variant}_s{seed}"
            run_ids.append(rid)
            for epoch in EVALUATIONS:
                validation.append(dict(run_id=rid, gamma_variant=variant,
                                       seed=seed, epoch=epoch, mse=.2, mae=.3))
            for split in ("train", "validation", "test"):
                for units, horizon, metric, v in (
                    ("cycles", "1", "rmse", 30.),
                    ("cycles", "all", "rmse", 30.),
                    ("cycles", "1", "mae", 20.),
                    ("cycles", "all", "mae", 20.),
                    ("score", "all", "rul_score_sum", 42.),
                    ("score", "all", "rul_score_mean", .4),
                    ("standardized", "1", "rmse", .3),
                    ("standardized", "all", "rmse", .3),
                    ("standardized", "1", "mae", .2),
                    ("standardized", "all", "mae", .2),
                    ("standardized", "1", "mse", .1),
                    ("standardized", "all", "mse", .1),
                    ("cycles", "1", "mse", 900.),
                    ("cycles", "all", "mse", 900.),
                ):
                    metrics.append(dict(run_id=rid, method="td_full",
                                        gamma_variant=variant, variant="trained",
                                        batch_size=128, seed=seed, split=split,
                                        units=units, horizon=horizon,
                                        metric=metric, value=v))
        selections.append(dict(gamma_variant=variant, run_ids=run_ids))
    atomic_json(out / "selections.json", dict(per_variant=selections))
    write_csv(out / "metrics.csv", metrics)
    write_csv(out / "validation_records.csv", validation)
    return out


def test_archive_import_excludes_old_td_and_refits(tmp_path):
    source = old_fixture(tmp_path)
    with (source / "metrics.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    original = list(rows)
    rows.append(dict(original[0], method="td_full", variant="refit_td",
                     run_id="fd001_tcn_td_full_b256_h1_s0"))
    write_csv(source / "metrics.csv", rows)
    result_root = tmp_path / "new_results"
    payload = combined.archive_baselines(combined.read_source(source), result_root)
    groups, selected, selected_validation = combined.validate_legacy(payload)
    assert set(groups) == set(combined.BASELINES)
    assert len(selected) == 420 and len(selected_validation) == 100
    assert all(r["method"] != "td_full" and r["variant"] == "trained" for r in selected)
    assert len(list((result_root / "legacy_baselines").glob("*.csv"))) == 4
    assert len(list(source.glob("*.csv"))) == 4
    assert "td_full" in (source / "metrics.csv").read_text()  # source untouched
    # Re-importing identical evidence is idempotent.
    combined.archive_baselines(combined.read_source(source), result_root)


def test_selectors_and_legacy_scores_are_audited(tmp_path):
    source = old_fixture(tmp_path)
    payload = combined.read_source(source)
    combined.validate_legacy(payload)
    payload["scores"][0]["score"] = "0"
    with pytest.raises(ValueError, match="validation score does not reproduce"):
        combined.validate_legacy(payload)


def test_one_command_combines_five_algorithms_without_fitting(tmp_path, monkeypatch):
    source, out = old_fixture(tmp_path), gamma_fixture(tmp_path)
    results = tmp_path / "new_results"
    monkeypatch.setattr(combined, "campaign_paths",
                        lambda: (tmp_path / "data_not_needed", results, out))
    monkeypatch.setattr(combined, "make_figures", lambda root: None)
    combined.run(source=source)
    new = out / "combined"
    with (new / "summary_metrics.csv").open(newline="") as stream:
        groups = list(csv.DictReader(stream))
    assert {r["method"] for r in groups} == set(combined.METHODS)
    assert {(r["method"], int(r["batch_size"])) for r in groups} == {
        ("MSE (B256)", 256), ("Gaussian NLL (B256)", 256),
        ("TD typed (B128)", 128), ("TD raw26 (B128)", 128),
        ("TD representation (B128)", 128)}
    comparison = list(csv.DictReader((new / "paired_comparisons.csv").open(newline="")))
    assert len(comparison) > 0
    assert {r["baseline_method"] for r in comparison} == set(combined.METHODS[:2])
    assert {r["gamma_method"] for r in comparison} == set(combined.METHODS[2:])
    assert read_json(new / "provenance.json")["trained_model_only"] is True
    # Running again uses the new result-root archive, no old files required.
    combined.run()


def test_replot_requires_saved_tables_and_never_imports_baselines(tmp_path, monkeypatch):
    out = tmp_path / "combined"
    with pytest.raises(FileNotFoundError):
        combined.run(output=out, plots_only=True)
    out.mkdir(parents=True)
    write_csv(out / "summary_metrics.csv", [dict(method="fake", mean=1)])
    write_csv(out / "validation_summary.csv", [dict(method="fake", mean=1)])
    called = []
    monkeypatch.setattr(combined, "make_figures", lambda path: called.append(path))
    combined.run(output=out, plots_only=True)
    assert called == [out]
