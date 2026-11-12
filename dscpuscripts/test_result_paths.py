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
        if name == "metr_la":
            from dscpuscripts.metr_la_sampling import SOURCE_FRACTION, fraction_tag
            stem = "metr_la_" + fraction_tag(SOURCE_FRACTION)
            assert results == roots / (stem + "_results")
            assert out == roots / (stem + "_outputs")
        else:
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


def test_metr_la_fraction_specific_directory_rejects_legacy_override(roots, monkeypatch):
    monkeypatch.setenv("FULLTD_METR_LA_ROOT", str(roots / "metr_la_results"))
    with pytest.raises(ValueError, match="legacy METR-LA"):
        result_paths.paths("metr_la")
    monkeypatch.delenv("FULLTD_METR_LA_ROOT")
    from dscpuscripts.metr_la_sampling import fraction_tag
    assert fraction_tag(.10) == "f10pct"
    assert fraction_tag(.25) == "f25pct"
    assert fraction_tag(1.0) == "f100pct"


@pytest.mark.parametrize("fraction,tag,cadence", [
    ("0.10", "f10pct", 100),
    ("0.25", "f25pct", 250),
    ("1.0", "f100pct", 500),
])
def test_fraction_controls_manifest_and_root_in_fresh_process(tmp_path, fraction, tag, cadence):
    import json
    import os
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[1]
    program = (
        "import json; "
        "from dscpuscripts.application_protocol import make_manifest; "
        "from dscpuscripts.result_paths import paths; "
        "rows=[r for r in make_manifest() if r['dataset']=='metr_la']; "
        "print(json.dumps(dict(count=len(rows), config=rows[0], "
        "results=str(paths('metr_la')[1]), outputs=str(paths('metr_la')[2]))))"
    )
    env = dict(os.environ)
    env.update(FULLTD_METR_LA_SOURCE_FRACTION=fraction,
               FULLTD_RESULTS_ROOT=str(tmp_path))
    env.pop("FULLTD_METR_LA_ROOT", None)
    result = subprocess.run([sys.executable, "-c", program], cwd=root, env=env,
                            check=True, capture_output=True, text=True)
    record = json.loads(result.stdout.strip())
    assert record["count"] == 525
    config = record["config"]
    assert config["source_fraction"] == float(fraction)
    assert config["learning_curve_updates"] == cadence
    assert config["run_id"].endswith("_" + tag)
    assert config["source_sampling"] == (
        "random_reshuffling" if float(fraction) == 1.0 else
        "iid_uniform_with_replacement")
    assert record["results"] == str(tmp_path / ("metr_la_" + tag + "_results"))
    assert record["outputs"] == str(tmp_path / ("metr_la_" + tag + "_outputs"))
