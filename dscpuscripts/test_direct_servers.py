"""Single-server CPU dispatch, automatic worker sizing, checkpoint safety."""
import argparse
import os

import pytest

from dscpuscripts import direct_runner
from fd001_gamma.common import atomic_json
from fd001_gamma.protocol import make_manifest, digest


@pytest.mark.parametrize(("active", "idle", "ramp", "expected"), [
    (0, 0., 8, 0),          # All cores occupied: queue remains pending.
    (0, .3, 8, 8),         # Do not immediately flood a shared server.
    (8, .3, 8, 16),        # More workers start as capacity is observed.
    (32, 0., 8, 32),      # High contention: do not kill already-running jobs.
    (120, 1., 8, 128),    # Never exceed user's explicit concurrency ceiling.
    (48, .1, 4, 52),      # User-configured ramp-up is respected.
])
def test_adaptive_dispatch_is_capacity_aware_and_nonpreemptive(active, idle, ramp, expected):
    assert direct_runner.adaptive_worker_target(128, active, 256, idle, ramp) == expected


@pytest.mark.parametrize(("max_workers", "active", "physical", "idle", "ramp"), [
    (0, 0, 256, .3, 8), (128, -1, 256, .3, 8),
    (128, 0, 0, .3, 8), (128, 0, 256, -0.1, 8),
    (128, 0, 256, 1.1, 8), (128, 0, 256, .3, 0),
    (128, 129, 256, .3, 8),
])
def test_bad_adaptive_sizing_inputs_are_rejected(max_workers, active, physical, idle, ramp):
    with pytest.raises(ValueError, match="Invalid adaptive"):
        direct_runner.adaptive_worker_target(max_workers, active, physical, idle, ramp)


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


def test_physical_core_idle_sampling_does_not_count_idle_smt_sibling_as_free_core(tmp_path, monkeypatch):
    # Core 0 has one fully busy sibling and one idle sibling. Core 1 has
    # two idle siblings. Available physical idle fraction should be 1/2,
    # not the misleading 3/4 logical-thread average.
    from pathlib import Path as RealPath
    for cpu, core in enumerate((0, 0, 1, 1)):
        directory = tmp_path / "topology" / f"cpu{cpu}" / "topology"
        directory.mkdir(parents=True)
        (directory / "physical_package_id").write_text("0")
        (directory / "core_id").write_text(str(core))
    stat = tmp_path / "procstat"
    stat.write_text(
        "cpu 0 0 0 400 0 0 0 0\n"
        + "".join(f"cpu{c} 0 0 0 100 0 0 0 0\n" for c in range(4))
        + "intr 0\n"
    )
    def after_sample(_):
        # CPU0 fully busy, CPU1 fully idle, CPUs2/3 fully idle.
        stat.write_text(
            "cpu 100 0 0 700 0 0 0 0\n"
            "cpu0 100 0 0 100 0 0 0 0\n"
            "cpu1 0 0 0 200 0 0 0 0\n"
            "cpu2 0 0 0 200 0 0 0 0\n"
            "cpu3 0 0 0 200 0 0 0 0\n"
            "intr 0\n"
        )
    original_path = RealPath
    def fake_path(value):
        name = str(value)
        if name == "/proc/stat":
            return stat
        if name.startswith("/sys/devices/system/cpu/"):
            return tmp_path / "topology" / name.removeprefix("/sys/devices/system/cpu/")
        return original_path(value)
    monkeypatch.setattr(direct_runner, "Path", fake_path)
    monkeypatch.setattr(direct_runner.os, "sched_getaffinity", lambda _: {0, 1, 2, 3})
    monkeypatch.setattr(direct_runner.time, "sleep", after_sample)
    assert direct_runner.cpu_idle_fraction() == pytest.approx(.5)


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
    args = argparse.Namespace(mode="dry-run", workers="auto", microbatch=128,
                              max_runs=0, adaptive=False, check_seconds=30., ramp_up=8)
    assert direct_runner.run_campaign(args) == 0
    assert not results.exists()


def test_adaptive_dry_run_displays_ceiling_and_initial_safe_ramp(tmp_path, monkeypatch, capsys):
    results = tmp_path / "no-files-created"
    monkeypatch.setattr(direct_runner, "campaign_paths",
                        lambda: (tmp_path / "data", results, tmp_path / "out"))
    monkeypatch.setattr(direct_runner, "scientific_hash", lambda: "fixture-code")
    monkeypatch.setattr(direct_runner, "choose_workers", lambda value: (128, 256, None))
    monkeypatch.setattr(direct_runner, "cpu_idle_fraction", lambda: .3)
    monkeypatch.setattr(os, "getloadavg", lambda: (220., 220., 220.))
    args = argparse.Namespace(mode="dry-run", workers="128", microbatch=128,
                              max_runs=0, adaptive=True, check_seconds=30., ramp_up=8)
    assert direct_runner.run_campaign(args) == 0
    out = capsys.readouterr().out
    assert "maximum 128" in out
    assert "initial launch budget 8" in out
    assert not results.exists()


def test_one_server_runner_has_no_slurm_or_remote_sharding_interface():
    from pathlib import Path
    file = Path(direct_runner.__file__)
    script = file.read_text()
    assert "--shards" not in script and "--shard-indices" not in script
    assert '"sbatch"' not in script and '"squeue"' not in script
    assert '"srun"' not in script
