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
    assert results0 == results1 == base / "fd001_results" / "results"
    assert out0 == base / "fd001_results" / "outputs"
    assert out1 == out0 / "baselines"
    result_paths.configure_environment()
    assert original_gamma_paths() == (d0, results0, out0)
    assert extension_gamma_paths() == (d0, results0, out0 / "extension")


def test_dataset_specific_results_on_any_host(dataset_layout):
    for name in ("electricity", "metr_la"):
        data, results, outputs = standard_runner.paths(name)
        assert data == dataset_layout / "data"
        assert results == dataset_layout / (name + "_results") / "results"
        assert outputs == dataset_layout / (name + "_results") / "outputs"
    with pytest.raises(ValueError, match="Unknown dataset"):
        result_paths.paths("unknown")


def test_explicit_root_override_is_consistent(tmp_path, monkeypatch):
    monkeypatch.setenv("FULLTD_FD001_ROOT", str(tmp_path / "user-fd001"))
    d, results, out = result_paths.paths("fd001", "comparison")
    assert results == tmp_path / "user-fd001" / "results"
    assert out == tmp_path / "user-fd001" / "outputs" / "comparison"
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
