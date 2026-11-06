"""Targeted tests for scheduler-free dispatch and safe cross-server imports."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from dscpuscripts import direct_runner, import_results
from fd001_gamma.common import atomic_json
from fd001_gamma.protocol import make_manifest, digest


@pytest.mark.parametrize(("spec", "total", "expected"), [
    ("0", 1, (0,)),
    ("0-3", 5, (0, 1, 2, 3)),
    ("4", 5, (4,)),
    ("0,2-3", 5, (0, 2, 3)),
    ("4,0,0", 5, (0, 4)),
])
def test_parse_shards(spec, total, expected):
    assert direct_runner.parse_shards(spec, total) == expected


@pytest.mark.parametrize(("spec", "total"), [
    ("", 5), ("foo", 5), ("0-5", 5), ("4-2", 5), ("-1", 5),
    ("0", 0), ("1,,2", 5),
])
def test_bad_shards_rejected(spec, total):
    with pytest.raises(ValueError):
        direct_runner.parse_shards(spec, total)


def test_two_server_shards_are_disjoint_complete_and_seed_balanced():
    all_ids = [c["run_id"] for c in make_manifest()]
    large = direct_runner.selected_configs(5, (0, 1, 2, 3))
    small = direct_runner.selected_configs(5, (4,))
    assert len(all_ids) == 180 and len(large) == 144 and len(small) == 36
    assert set(c["run_id"] for c in large).isdisjoint(c["run_id"] for c in small)
    assert set(c["run_id"] for c in large + small) == set(all_ids)
    assert set(c["seed"] for c in large) == {0, 1, 2, 3}
    assert set(c["seed"] for c in small) == {4}


def test_classify_preserves_existing_result_identity_and_completion(tmp_path):
    config = make_manifest()[0]
    root = tmp_path / "results"
    path = root / config["run_id"]
    assert direct_runner.classify(config, root, "hash") == "start_epoch_1"
    path.mkdir(parents=True)
    (path / "recovery.pt").touch()
    assert direct_runner.classify(config, root, "hash") == "resume_checkpoint"
    (path / "final.pt").touch()
    assert direct_runner.classify(config, root, "hash") == "repair_final"
    status = {"config": config, "identity": {"config_sha256": digest(config),
                                             "code_sha256": "hash"},
              "state": "complete", "epochs_completed": 100}
    atomic_json(path / "status.json", status)
    assert direct_runner.classify(config, root, "hash") == "repair_final"
    (path / "validation.json").touch()
    (path / "rounds.json").touch()
    assert direct_runner.classify(config, root, "hash") == "complete"
    status["state"] = "numerical_failure"
    atomic_json(path / "status.json", status)
    assert direct_runner.classify(config, root, "hash") == "numerical_failure"
    status["identity"]["code_sha256"] = "wrong"
    atomic_json(path / "status.json", status)
    with pytest.raises(RuntimeError, match="Incompatible"):
        direct_runner.classify(config, root, "hash")


def test_dry_run_no_scheduler_no_written_files(tmp_path, monkeypatch):
    results = tmp_path / "never-created-results"
    monkeypatch.setattr(direct_runner, "campaign_paths", lambda: (tmp_path / "data", results, tmp_path / "outputs"))
    monkeypatch.setattr(direct_runner, "scientific_hash", lambda: "fixed-code")
    args = argparse.Namespace(mode="dry-run", workers=4, shards=5, shard_indices="4",
                              microbatch=128, max_runs=0)
    assert direct_runner.run_campaign(args) == 0
    assert not results.exists()


def test_import_validation_rejects_mismatched_data_and_incomplete_runs(tmp_path):
    config = make_manifest()[0]
    p = tmp_path / config["run_id"]
    p.mkdir()
    identity = dict(config_sha256=digest(config), code_sha256="code",
                    data_sha256="train", validation_sha256="val")
    status = dict(config=config, protocol_id=config["protocol_id"], identity=identity,
                  state="complete", epochs_completed=100)
    atomic_json(p / "status.json", status)
    with pytest.raises(ValueError, match="missing evidence"):
        import_results.validate_incoming(p, config, "code", "train", "val")
    for file in ("final.pt", "validation.json", "rounds.json"):
        (p / file).write_bytes(b"fixture")
    assert import_results.validate_incoming(p, config, "code", "train", "val") == "complete"
    with pytest.raises(ValueError, match="Different scientific code or processed"):
        import_results.validate_incoming(p, config, "code", "DIFFERENT", "val")
    status["state"] = "running"
    atomic_json(p / "status.json", status)
    with pytest.raises(ValueError, match="Only terminal"):
        import_results.validate_incoming(p, config, "code", "train", "val")


def test_import_tree_digest_detects_content_conflicts(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "status.json").write_text('{"a": 1}')
    (second / "status.json").write_text('{"a": 1}')
    assert import_results.tree_digest(first) == import_results.tree_digest(second)
    (second / "status.json").write_text('{"a": 2}')
    assert import_results.tree_digest(first) != import_results.tree_digest(second)
