"""Single-server CPU dispatch, automatic worker sizing, checkpoint safety."""
import argparse
import os

import pytest

from dscpuscripts import direct_runner
from fd001_gamma.common import atomic_json
from fd001_gamma.protocol import make_manifest, digest


def test_all_five_seeds_and_thirty_six_configs_are_locally_assigned():
    runs = make_manifest()
    assert len(runs) == 180
    assert set(c["seed"] for c in runs) == set(range(5))
    assert len(set(c["run_id"] for c in runs)) == 180
    assert len(set((c["gamma_variant"], c["alpha"], c["beta"]) for c in runs)) == 36


@pytest.mark.parametrize(("idle", "expected"), [
    (1., 128),
    (.8, 128),
    (.5, 96),
    (.25, 48),
    (.1, 19),
    (0., 1),
])
def test_auto_workers_uses_idle_physical_cores_not_logical_threads(monkeypatch, idle, expected):
    monkeypatch.setattr(direct_runner, "available_physical_cores", lambda: 256)
    monkeypatch.setattr(direct_runner, "cpu_idle_fraction", lambda: idle)
    monkeypatch.setattr(os, "sched_getaffinity", lambda x: set(range(512)))
    assert direct_runner.choose_workers("auto") == (expected, 256, idle)


def test_explicit_worker_override_and_invalid_worker_rejected(monkeypatch):
    monkeypatch.setattr(direct_runner, "available_physical_cores", lambda: 256)
    monkeypatch.setattr(os, "sched_getaffinity", lambda x: set(range(512)))
    assert direct_runner.choose_workers("128") == (128, 256, None)
    with pytest.raises(ValueError, match="positive"):
        direct_runner.choose_workers("0")
    with pytest.raises(ValueError, match="positive integer"):
        direct_runner.choose_workers("invalid")
    with pytest.raises(ValueError, match="logical CPUs"):
        direct_runner.choose_workers("513")


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


def test_prepared_data_mismatch_blocks_checkpoint_reuse(tmp_path):
    config = make_manifest()[0]
    directory = tmp_path / config["run_id"]
    directory.mkdir(parents=True)
    atomic_json(directory / "status.json", dict(identity=dict(
        data_sha256="original-train", validation_sha256="original-val")))
    direct_runner.validate_prepared_data([config], tmp_path, "original-train", "original-val")
    with pytest.raises(RuntimeError, match="different FD001 processed data"):
        direct_runner.validate_prepared_data([config], tmp_path, "different-train", "original-val")


def test_dry_run_does_not_touch_data_or_start_workers(tmp_path, monkeypatch):
    results = tmp_path / "never-created-results"
    monkeypatch.setattr(direct_runner, "campaign_paths", lambda: (tmp_path / "data", results, tmp_path / "outputs"))
    monkeypatch.setattr(direct_runner, "scientific_hash", lambda: "fixed-code")
    monkeypatch.setattr(direct_runner, "choose_workers", lambda value: (96, 256, .5))
    monkeypatch.setattr(os, "getloadavg", lambda: (5., 5., 5.))
    args = argparse.Namespace(mode="dry-run", workers="auto", microbatch=128, max_runs=0)
    assert direct_runner.run_campaign(args) == 0
    assert not results.exists()


def test_one_server_runner_has_no_slurm_or_remote_sharding_interface():
    from pathlib import Path
    file = Path(direct_runner.__file__)
    script = file.read_text()
    assert "--shards" not in script and "--shard-indices" not in script
    assert '"sbatch"' not in script and '"squeue"' not in script
    assert '"srun"' not in script
