"""Offline tests for FD001 Gamma status, read-only plots and no-refit workflow."""
import csv
import inspect
from pathlib import Path
import sys

import pytest

from dscpuscripts import status_fd001_gamma as status_report
from fd001_gamma.common import atomic_json, write_csv
from fd001_gamma.protocol import VARIANTS, digest, make_manifest
import fd001_gamma.analysis as analysis


def test_status_distinguishes_complete_numerical_failure_and_incomplete(tmp_path, monkeypatch, capsys):
    configs = make_manifest()[:4]
    monkeypatch.setattr(status_report, "make_manifest", lambda: configs)
    code = "scientific-hash-for-test"
    for config in configs[:3]:
        directory = tmp_path / config["run_id"]
        directory.mkdir()
        state = "complete" if config == configs[0] else (
            "numerical_failure" if config == configs[1] else "complete")
        status = dict(config=config, state=state,
                      identity=dict(config_sha256=digest(config), code_sha256=code),
                      epochs_completed=100, next_epoch=101)
        atomic_json(directory / "status.json", status)
        if config == configs[0]:
            for name in ("final.pt", "validation.json", "rounds.json"):
                (directory / name).touch()
    rows = status_report.inspect(tmp_path, code_hash=code)
    assert [r["state"] for r in rows] == [
        "complete", "numerical_failure", "incomplete_final", "missing"]
    assert status_report.report(rows) is False
    output = capsys.readouterr().out
    assert "1/4 complete" in output
    assert "1 numerical failures" in output
    assert "NOT READY" in output


def test_status_rejects_changed_scientific_identity(tmp_path, monkeypatch):
    config = make_manifest()[0]
    monkeypatch.setattr(status_report, "make_manifest", lambda: [config])
    directory = tmp_path / config["run_id"]
    directory.mkdir()
    atomic_json(directory / "status.json",
                dict(config=config, state="complete",
                     identity=dict(config_sha256=digest(config),
                                   code_sha256="old-code"),
                     epochs_completed=100, next_epoch=101))
    rows = status_report.inspect(tmp_path, code_hash="new-code")
    assert rows[0]["state"] == "incompatible"
    assert not status_report.report(rows, show=0)


def make_saved_csv_fixture(out):
    all_metrics, validation, gammas, per_variant = [], [], [], []
    for variant_i, variant in enumerate(VARIANTS):
        run_ids = []
        for seed in range(5):
            run_id = f"fixture_{variant}_{seed}"
            run_ids.append(run_id)
            all_metrics.append(dict(
                run_id=run_id, gamma_variant=variant, split="test",
                units="cycles", horizon="all", metric="rmse",
                seed=seed, value=10.0 + variant_i + 0.1 * seed))
            for epoch in (10, 20):
                validation.append(dict(
                    run_id=run_id, gamma_variant=variant, seed=seed,
                    epoch=epoch, mse=20.0 + variant_i + 0.1 * seed,
                    mae=4.0 + variant_i + 0.1 * seed))
                for kind in (1, 2):
                    gammas.append(dict(run_id=run_id, gamma_variant=variant,
                                       seed=seed, epoch=epoch, type=kind,
                                       mean=0.1 * kind + 0.05 * variant_i))
        per_variant.append(dict(gamma_variant=variant, state="selected",
                                run_ids=run_ids))
    atomic_json(out / "selections.json", dict(per_variant=per_variant))
    write_csv(out / "metrics.csv", all_metrics)
    write_csv(out / "gamma_records.csv", gammas)
    write_csv(out / "validation_records.csv", validation)


def test_saved_gamma_analysis_replots_without_checkpoints_or_refit(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    out = tmp_path / "outputs"
    out.mkdir()
    make_saved_csv_fixture(out)
    monkeypatch.setattr(analysis, "EVALUATIONS", (10, 20))
    monkeypatch.setattr(analysis, "campaign_paths",
                        lambda: (tmp_path / "nonexistent-data",
                                 tmp_path / "nonexistent-results", out))

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Plots-only mode must never open checkpoints, "
                             "rerun selection, or refit a model")

    monkeypatch.setattr(analysis, "select", forbidden)
    monkeypatch.setattr(analysis, "evaluate_selected", forbidden)
    monkeypatch.setattr(sys, "argv", ["fd001_gamma.analysis", "plots"])
    analysis.main()
    figures = out / "figures"
    for filename in ("primary_fd001_gamma_rmse.pdf",
                     "primary_fd001_gamma_rmse.svg",
                     "validation_fd001_gamma_mse.pdf",
                     "validation_fd001_gamma_mae.pdf",
                     "gamma_fd001_lag1.pdf",
                     "gamma_fd001_lag5.pdf"):
        assert (figures / filename).is_file(), filename
        assert (figures / filename).stat().st_size > 0
    assert not (out / "refits").exists()
    assert not (out / "endpoint_cache").exists()


def test_plots_only_rejects_missing_tables_instead_of_creating_zeros(tmp_path):
    with pytest.raises(FileNotFoundError, match="full Gamma analysis"):
        analysis.saved_figure_inputs(tmp_path)


def test_gamma_wrapper_is_direct_and_never_refits():
    root = Path(__file__).resolve().parents[1]
    source = (root / "dscpuscripts" / "analyze_fd001_gamma.sh").read_text()
    assert "sbatch" not in source and "squeue" not in source
    assert "fulltd.evaluate" not in source
    assert 'fd001_gamma.analysis run' in source
    assert 'fd001_gamma.analysis plots' in source
    evaluation = inspect.getsource(analysis.evaluate_selected)
    assert "refit" not in evaluation.lower()
    assert 'out / "endpoint_cache"' in evaluation
    assert 'if endpoints and split == "test":' in evaluation
    assert '"gamma_endpoint_metrics.json"' not in evaluation
