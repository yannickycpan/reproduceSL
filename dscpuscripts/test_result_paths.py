"""Regressions for dataset_results paths and shared FD001 reporting."""
from __future__ import annotations

from pathlib import Path
import csv

import pytest

from dscpuscripts import result_paths
from dscpuscripts import standard_runner
from dscpuscripts import fd001_report
from fd001_gamma.common import campaign_paths as original_gamma_paths
from fd001_gamma_ext.runtime import campaign_paths as extension_gamma_paths


@pytest.fixture
def dataset_layout(tmp_path, monkeypatch):
    monkeypatch.setenv("FULLTD_RESULTS_ROOT", str(tmp_path / "fulltd"))
    for name in (
        "FULLTD_FD001_ROOT", "FULLTD_ELECTRICITY_ROOT",
        "FULLTD_METR_LA_ROOT", "FULLTD_DATA_DIR",
        "FULLTD_FD001_GAMMA_RESULTS_DIR", "FULLTD_FD001_GAMMA_OUTPUT_DIR",
        "FULLTD_FD001_GAMMA_EXT_RESULTS_DIR", "FULLTD_FD001_GAMMA_EXT_OUTPUT_DIR",
    ):
        monkeypatch.delenv(name, raising=False)
    return tmp_path / "fulltd"


def test_every_fd001_method_shares_one_result_directory(dataset_layout):
    base = dataset_layout
    d0, results0, out0 = result_paths.paths("fd001")
    d1, results1, out1 = standard_runner.paths("fd001_baselines")
    assert d0 == d1 == base / "data"
    assert results0 == results1 == base / "fd001_results"
    assert out0 == base / "fd001_outputs"
    assert out1 == out0 / "baselines"
    result_paths.configure_environment()
    assert original_gamma_paths() == (d0, results0, out0)
    assert extension_gamma_paths() == (d0, results0, out0 / "extension")


def test_guard_prevents_retraining_runs_saved_in_previous_nested_layout(
        dataset_layout):
    from fd001_gamma.protocol import make_manifest
    config = make_manifest()[0]
    results = dataset_layout / "fd001_results"
    previous = results / "results"
    old_run = previous / config["run_id"]
    old_run.mkdir(parents=True)
    (old_run / "status.json").write_text('{"state":"complete"}')
    with pytest.raises(RuntimeError, match="existing run records"):
        result_paths.guard_unmigrated_runs(results, [config], [previous])
    current = results / config["run_id"]
    current.mkdir()
    (current / "status.json").write_text('{"state":"complete"}')
    result_paths.guard_unmigrated_runs(results, [config], [previous])


def test_dataset_specific_results_on_any_host(dataset_layout):
    for name in ("electricity", "metr_la"):
        data, results, outputs = standard_runner.paths(name)
        assert data == dataset_layout / "data"
        assert results == dataset_layout / (name + "_results")
        assert outputs == dataset_layout / (name + "_outputs")
    with pytest.raises(ValueError, match="Unknown dataset"):
        result_paths.paths("unknown")


def test_existing_flat_gamma_directories_are_detected_without_moving_them(
        dataset_layout, monkeypatch):
    from fd001_gamma.protocol import make_manifest
    from dscpuscripts.status_fd001_gamma import inspect
    from fd001_gamma.common import atomic_json
    from fd001_gamma.protocol import digest
    root = dataset_layout / "fd001_results"
    sample = make_manifest()[0]
    directory = root / sample["run_id"]
    directory.mkdir(parents=True)
    for name in ("final.pt", "rounds.json", "validation.json"):
        (directory / name).touch()
    atomic_json(directory / "status.json", dict(
        config=sample, state="complete", epochs_completed=100, next_epoch=101,
        identity=dict(config_sha256=digest(sample), code_sha256="fixture-science")))
    import dscpuscripts.status_fd001_gamma as status
    monkeypatch.setattr(status, "make_manifest", lambda: [sample])
    assert result_paths.paths("fd001")[1] == root
    assert inspect(root, code_hash="fixture-science")[0]["state"] == "complete"
    assert not (root / "results").exists()


def test_explicit_root_override_is_consistent(tmp_path, monkeypatch):
    monkeypatch.setenv("FULLTD_FD001_ROOT", str(tmp_path / "user-fd001"))
    d, results, out = result_paths.paths("fd001", "comparison")
    assert results == tmp_path / "user-fd001"
    assert out == tmp_path / "fd001_outputs" / "comparison"
    with pytest.raises(ValueError, match="Invalid output"):
        result_paths.paths("fd001", "../../unsafe")


def test_shared_learning_curve_plots_four_and_six_methods(tmp_path):
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt

    def synthetic(keys):
        records = []
        for index, k in enumerate(keys):
            for epoch in range(10, 101, 10):
                records.append(dict(method=k, epoch=epoch,
                                    optimizer_steps=epoch * (110 if index >= 2 else 55),
                                    mean=20 + index * .2 + epoch / 100,
                                    standard_error=.15))
        return records

    # The four-Gamma analysis passes canonical ids; the six-method
    # comparison passes display labels. Both use this same renderer.
    four = ("typed", "raw26", "representation", "pairwise")
    six = tuple(fd001_report.LABELS[k] for k in fd001_report.ORDER)
    for label, methods in (("four", four), ("six", six)):
        stem = tmp_path / label
        fd001_report.plot_learning_curve(synthetic(methods), stem)
        for ext in (".svg", ".pdf", ".png"):
            assert stem.with_suffix(ext).is_file()
            assert stem.with_suffix(ext).stat().st_size > 100
    plt.close("all")


def test_removed_duplicate_report_is_no_longer_imported():
    root = Path(__file__).resolve().parents[1]
    assert not (root / "dscpuscripts/combined_fd001.py").exists()
    assert not (root / "dscpuscripts/analyze_fd001_all.sh").exists()
    for path in ("dscpuscripts/fd001_report.py",
                 "dscpuscripts/compare_fd001_extension.py",
                 "dscpuscripts/standard_analysis.py"):
        source = (root / path).read_text()
        assert "import dscpuscripts.combined_fd001" not in source
        assert "from dscpuscripts.combined_fd001" not in source


def test_scientific_protocol_hashes_stay_separate():
    from fd001_gamma.common import scientific_hash as old_hash
    from fd001_gamma_ext.runtime import scientific_hash as new_hash
    assert len(old_hash()) == len(new_hash()) == 64
    assert old_hash() != new_hash()
