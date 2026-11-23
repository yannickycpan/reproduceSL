"""Regression check that new Combo test exports share the original folder safely."""
import csv
import json

import pytest

from dscpuscripts.fd001_combo_mse500_export import merge_original_and_combo_exports
from dscpuscripts.common import atomic_json, write_csv


def _fixture(out, label, n, metric_name, gamma_name, manifest_name):
    records = []
    gamma = []
    for i in range(n):
        run_id = f"{label}_run_{i:04d}"
        (out / "records" / (run_id + ".json")).touch()
        (out / "predictions" / (run_id + ".npz")).touch()
        records.append(dict(run_id=run_id, method="td_mse_combo" if label == "new" else "mse",
                            optimizer_steps=500, test_rmse_cycles=25, test_rul_score_sum=1234))
        gamma.append(dict(run_id=run_id, optimizer_steps=500, lag=1, mean=0.9))
    write_csv(out / metric_name, records)
    write_csv(out / gamma_name, gamma)
    atomic_json(out / manifest_name, dict(
        n_runs=n, n_checkpoints=n, n_engine_predictions=n*100,
        test_data_sha256="shared-test-dataset",
        validation_data_sha256="shared-validation-dataset"))


def test_combines_2350_runs_without_overwriting_original_csv(tmp_path):
    (tmp_path / "records").mkdir()
    (tmp_path / "predictions").mkdir()
    _fixture(tmp_path, "old", 1600, "checkpoint_metrics.csv",
             "checkpoint_gamma.csv", "export_manifest.json")
    _fixture(tmp_path, "new", 750, "combo_mse500_checkpoint_metrics.csv",
             "combo_mse500_checkpoint_gamma.csv", "combo_mse500_export_manifest.json")
    before_metrics = (tmp_path / "checkpoint_metrics.csv").read_bytes()
    before_manifest = (tmp_path / "export_manifest.json").read_bytes()
    counts = merge_original_and_combo_exports(tmp_path)
    assert counts == dict(n_runs=2350, n_checkpoints=2350)
    assert (tmp_path / "checkpoint_metrics.csv").read_bytes() == before_metrics
    assert (tmp_path / "export_manifest.json").read_bytes() == before_manifest
    with (tmp_path / "combined_checkpoint_metrics.csv").open(newline="") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 2350
    assert {r["experiment"] for r in rows} == {
        "original_lagscanv3", "combo_stage1_mse500"}
    assert len({r["run_id"] for r in rows}) == 2350
    assert json.loads((tmp_path / "combined_export_manifest.json").read_text())[
        "n_engine_predictions"] == 235000
    counts2 = merge_original_and_combo_exports(tmp_path)
    assert counts2 == counts


def test_refuses_to_merge_without_original_export(tmp_path):
    with pytest.raises(FileNotFoundError, match="Run both original and new"):
        merge_original_and_combo_exports(tmp_path)
