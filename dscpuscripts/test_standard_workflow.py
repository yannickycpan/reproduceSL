"""Regression tests for shared CPU dispatch without historical FD001 branches."""
import argparse
from collections import Counter
from pathlib import Path
import sys
from types import ModuleType

import pytest

from dscpuscripts import standard_runner as runner
from dscpuscripts.common import atomic_json
from dscpuscripts.fd001_protocol import make_manifest as fd001_manifest
from dscpuscripts.application_protocol import make_manifest as other_manifest, digest


@pytest.mark.parametrize("name,count", [
    ("fd001", 600), ("electricity", 700), ("metr_la", 525)])
def test_one_manifest_per_dataset(name, count):
    runs=runner.configurations(name)
    assert len(runs)==count
    assert len({r["run_id"] for r in runs})==count
    assert all(r["epochs"]==100 for r in runs)
    if name=="fd001":
        assert runs==fd001_manifest()
        assert set(r["method"] for r in runs)=={
            "mse","gaussian_nll","td_typed","td_pairwise"}
    else:
        assert runs==[r for r in other_manifest() if r["dataset"]==name]


def test_no_older_fd001_campaign_or_gamma_launcher():
    assert runner.CAMPAIGNS==("fd001","electricity","metr_la")
    for entry in ("fd001_baselines","fd001_v2b"):
        with pytest.raises(ValueError):
            runner.configurations(entry)
    root=Path(__file__).resolve().parents[1]
    # Local __pycache__ folders may remain after git removes tracked modules.
    for name in ("fd001_gamma", "fd001_gamma_ext"):
        assert not list((root / name).glob("*.py"))
    for name in ("dscpuscripts/fd001_v2_protocol.py",
                 "dscpuscripts/direct_runner.py"):
        assert not (root / name).is_file()


def test_run_classifier_preserves_identity_and_requires_real_outputs(tmp_path):
    config=fd001_manifest()[0]
    assert runner.classify(config,tmp_path,"code")=="missing"
    directory=tmp_path/config["run_id"]
    directory.mkdir()
    atomic_json(directory/"status.json",dict(
        config=config,identity=dict(config_sha256=digest(config),
                                    code_sha256="code"),
        state="complete",epochs_completed=100))
    assert runner.classify(config,tmp_path,"code")=="resume"
    for filename in ("final.pt","update_curve.json","rounds.json"):
        (directory/filename).touch()
    assert runner.classify(config,tmp_path,"code")=="complete"
    with pytest.raises(ValueError,match="Incompatible"):
        runner.classify(config,tmp_path,"other")
    status=__import__("json").loads((directory/"status.json").read_text())
    status["state"]="numerical_failure"
    atomic_json(directory/"status.json",status)
    assert runner.classify(config,tmp_path,"code")=="numerical_failure"


def test_unified_fd001_analyzer_has_no_training_or_refit():
    root=Path(__file__).resolve().parents[1]
    script=(root/"dscpuscripts/standard_analysis.py").read_text()
    report=(root/"dscpuscripts/fd001_analysis.py").read_text()
    assert "from dscpuscripts.fd001_analysis import run" in script
    for source in (script,report):
        assert "optimizer.step(" not in source
        assert "train_candidate(" not in source
        assert "refit_readouts" not in source
