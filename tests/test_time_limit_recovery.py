"""Only epoch-end states are durable; incomplete work is deterministically replayed."""
import json

import numpy as np
import pytest
import torch

from fulltd import train
from test_fulltd_training import SyntheticData, config_for, tiny_training


@pytest.mark.parametrize("stop_chunk", [1, 3])
def test_validation_warning_keeps_only_last_completed_epoch(tmp_path, monkeypatch, stop_chunk):
    config = config_for()
    fitting = SyntheticData(8)
    reference = train.train_candidate(config, fitting, SyntheticData(11, split="validation"),
                                      tmp_path / "reference", microbatch=4)
    requested = False
    saves = []
    original_save = train.atomic_save

    def record_save(path, state):
        saves.append((path.name, state["next_epoch"], state["cursor"], len(state["validation"])))
        original_save(path, state)
    monkeypatch.setattr(train, "atomic_save", record_save)

    class InterruptedValidation(SyntheticData):
        def get_batch(self, ids):
            nonlocal requested
            batch = super().get_batch(ids)
            if len(self.reads) == stop_chunk:
                requested = True
            return batch

    validation = InterruptedValidation(11, split="validation")
    partial = train.train_candidate(config, fitting, validation, tmp_path / "resume", microbatch=4,
                                   stop_requested=lambda: requested)
    assert partial["state"] == "checkpointed"
    completed = 9 if stop_chunk == 1 else 10
    assert partial["next_epoch"] == completed + 1 and partial["cursor"] == 0
    assert len(partial["validation"]) == completed // 10
    assert saves == [("recovery.pt", e + 1, 0, e // 10) for e in range(1, completed + 1)]
    status = json.loads((tmp_path / "resume" / config["run_id"] / "status.json").read_text())
    assert status["epochs_completed"] == completed and status["optimizer_steps"] == completed
    replay = SyntheticData(11, split="validation")
    resumed = train.train_candidate(config, fitting, replay, tmp_path / "resume", microbatch=4)
    assert resumed["validation"] == reference["validation"]
    assert resumed["counters"] == reference["counters"]
    for key, value in reference["objective"].items():
        torch.testing.assert_close(value, resumed["objective"][key], rtol=0, atol=0)
    for key, state in reference["optimizer"]["state"].items():
        for name, value in state.items():
            torch.testing.assert_close(value, resumed["optimizer"]["state"][key][name], rtol=0, atol=0)
    np.testing.assert_array_equal(np.concatenate(replay.reads[:3]), np.arange(11))


@pytest.mark.parametrize("dataset", ["electricity", "metr_la", "fd001"])
def test_exactly_one_checkpoint_per_completed_epoch_for_every_dataset(tmp_path, monkeypatch, dataset):
    config = config_for(dataset) if dataset != "metr_la" else next(
        c for c in train.load_manifest("configs/fulltd/manifest.json")
        if c["dataset"] == dataset and c["method"] == "td_full" and c["seed"] == 0)
    fitting, validation = SyntheticData(8, dataset), SyntheticData(11, dataset, split="validation")
    saves = []
    original_save = train.atomic_save

    def record_save(path, state):
        completed = state["next_epoch"] - 1
        assert state["cursor"] == 0
        assert len(state["round_records"]) == completed
        assert len(state["validation"]) == completed // 10
        saves.append((path.name, completed))
        original_save(path, state)
    monkeypatch.setattr(train, "atomic_save", record_save)
    train.train_candidate(config, fitting, validation, tmp_path, microbatch=4)
    assert saves == [("recovery.pt", e) for e in range(1, 100)] + [("final.pt", 100)]


def test_warning_before_first_epoch_writes_no_checkpoint(tmp_path):
    config = config_for()
    fitting, validation = SyntheticData(8), SyntheticData(4, split="validation")
    partial = train.train_candidate(config, fitting, validation, tmp_path, stop_requested=lambda: True)
    assert partial["state"] == "stopped_no_checkpoint"
    assert partial["next_epoch"] == 1 and partial["cursor"] == 0
    assert partial["counters"]["optimizer_steps"] == 0
    assert validation.reads == [] and fitting.reads == []
    assert not (tmp_path / config["run_id"] / "recovery.pt").exists()
    assert not (tmp_path / config["run_id"] / "final.pt").exists()


def test_negative_cli_index_rejected_before_loading_data(tmp_path, monkeypatch):
    monkeypatch.setattr(train, "load_manifest", lambda p: [config_for()])
    with pytest.raises(ValueError, match="Index must be"):
        train.run_index("manifest.json", -1, tmp_path / "missing-data", tmp_path)
