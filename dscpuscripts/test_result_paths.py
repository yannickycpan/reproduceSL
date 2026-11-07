"""Tests for the one shared results/outputs convention and safe migrations."""
from pathlib import Path

import pytest

from dscpuscripts import result_paths
from dscpuscripts import standard_runner
from dscpuscripts.fd001_protocol import make_manifest


@pytest.fixture
def roots(tmp_path, monkeypatch):
    monkeypatch.setenv("FULLTD_RESULTS_ROOT", str(tmp_path))
    for name in ("FULLTD_FD001_ROOT", "FULLTD_ELECTRICITY_ROOT",
                 "FULLTD_METR_LA_ROOT", "FULLTD_DATA_DIR"):
        monkeypatch.delenv(name, raising=False)
    return tmp_path


def test_fd001_one_results_and_one_sibling_outputs_directory(roots):
    data, results, out = result_paths.paths("fd001")
    assert (data, results, out) == (roots / "data",
                                   roots / "fd001_results",
                                   roots / "fd001_outputs")
    assert standard_runner.paths("fd001") == (data, results, out)
    assert not results.exists() and not out.exists()


def test_each_dataset_has_its_own_sibling_output_folder(roots):
    for name in ("electricity", "metr_la"):
        data, results, out = standard_runner.paths(name)
        assert data == roots / "data"
        assert results == roots / (name + "_results")
        assert out == roots / (name + "_outputs")
    with pytest.raises(ValueError, match="Unknown dataset"):
        result_paths.paths("unknown")


def test_explicit_override_and_safe_output_kind(roots, monkeypatch):
    monkeypatch.setenv("FULLTD_FD001_ROOT", str(roots / "special" / "engine_runs"))
    _, results, out = result_paths.paths("fd001", "comparison")
    assert results == roots / "special" / "engine_runs"
    assert out == roots / "special" / "fd001_outputs" / "comparison"
    with pytest.raises(ValueError, match="Invalid output"):
        result_paths.paths("fd001", "../../unsafe")


def test_guard_refuses_retraining_missing_run_checkpoint(roots):
    config = make_manifest()[0]
    dest = roots / "fd001_results"
    old = dest / "results" / config["run_id"]
    old.mkdir(parents=True)
    (old / "recovery.pt").touch()
    with pytest.raises(RuntimeError, match="existing run records"):
        result_paths.guard_unmigrated_runs(dest, [config], [old.parent])
    new = dest / config["run_id"]
    new.mkdir()
    (new / "recovery.pt").touch()
    result_paths.guard_unmigrated_runs(dest, [config], [old.parent])
