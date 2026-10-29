"""A scheduler allocation boundary must not change the scientific trajectory."""
import json

import numpy as np
import pytest
import torch

from fulltd import train
from test_fulltd_training import SyntheticData, config_for, tiny_training


@pytest.mark.parametrize("stop_chunk", [1, 3])
def test_resume_inside_or_at_end_of_validation_is_exact(tmp_path, stop_chunk):
    config = config_for()
    fitting = SyntheticData(8)
    reference = train.train_candidate(config, fitting, SyntheticData(11, split="validation"),
                                       tmp_path / "reference", microbatch=4, recovery_every=1)
    requested = False

    class InterruptedValidation(SyntheticData):
        def get_batch(self, ids):
            nonlocal requested
            batch = super().get_batch(ids)
            if len(self.reads) == stop_chunk:
                requested = True
            return batch

    validation = InterruptedValidation(11, split="validation")
    partial = train.train_candidate(config, fitting, validation, tmp_path / "resume", microbatch=4,
                                   recovery_every=1, stop_requested=lambda: requested)
    assert partial["state"] == "checkpointed"
    assert partial["next_epoch"] == 10
    assert partial["cursor"] == 8
    assert partial["validation"] == []  # no partial score can enter selection
    assert partial["validation_progress"]["cursor"] == min(4 * stop_chunk, 11)
    status = json.loads((tmp_path / "resume" / config["run_id"] / "status.json").read_text())
    assert status["validation_cursor"] == min(4 * stop_chunk, 11)
    remainder = SyntheticData(11, split="validation")
    resumed = train.train_candidate(config, fitting, remainder, tmp_path / "resume", microbatch=4, recovery_every=1)
    assert resumed["validation"] == reference["validation"]
    assert resumed["counters"] == reference["counters"]
    assert resumed["validation_progress"] is None
    for key, value in reference["objective"].items():
        torch.testing.assert_close(value, resumed["objective"][key], rtol=0, atol=0)
    for key, state in reference["optimizer"]["state"].items():
        for name, value in state.items():
            torch.testing.assert_close(value, resumed["optimizer"]["state"][key][name], rtol=0, atol=0)
    first_eval_reads = remainder.reads[:3-stop_chunk]
    if first_eval_reads:
        np.testing.assert_array_equal(np.concatenate(first_eval_reads), np.arange(4*stop_chunk, 11))


def test_warning_before_first_batch_saves_initial_state_without_update(tmp_path):
    config = config_for()
    fitting, validation = SyntheticData(8), SyntheticData(4, split="validation")
    partial = train.train_candidate(config, fitting, validation, tmp_path, stop_requested=lambda: True)
    assert partial["state"] == "checkpointed"
    assert partial["next_epoch"] == 1 and partial["cursor"] == 0
    assert partial["counters"]["optimizer_steps"] == 0
    assert partial["counters"]["source_draws"] == 0
    assert validation.reads == [] and fitting.reads == []
    assert (tmp_path / config["run_id"] / "recovery.pt").exists()


def test_negative_cli_index_rejected_before_loading_data(tmp_path, monkeypatch):
    monkeypatch.setattr(train, "load_manifest", lambda p: [config_for()])
    with pytest.raises(ValueError, match="Index must be"):
        train.run_index("manifest.json", -1, tmp_path / "missing-data", tmp_path)


