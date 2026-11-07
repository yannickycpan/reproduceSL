"""Focused regression tests for the fresh 270-run FD001 update-checkpoint sweep.

No real-data training; tests use a tiny deterministic model and synthetic
engine data to validate actual 500-update snapshots and exact mid-epoch resume.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from dscpuscripts import fd001_v2_protocol as protocol
from dscpuscripts import fd001_v2_train as train
from dscpuscripts import fd001_v2_analysis as analysis
from dscpuscripts import result_paths


class ToyFD001:
    def __init__(self, split, n=1300):
        self.dataset = "fd001"
        self.split = split
        self.entity_ids = np.zeros(n, dtype=np.int64)
        self.origins = np.arange(1, n+1, dtype=np.int64)
        self.metadata = {"fingerprint": "synthetic-fd001-for-checkpoint-test"}
        self.x = np.arange(n, dtype=np.float32).reshape(n,1,1) / 1000.
        self.y = (self.x[:,0,:] * 0.4 + 0.2).astype(np.float32)

    def __len__(self):
        return len(self.x)

    def get_batch(self, indices):
        return dict(x=self.x[indices], y=self.y[indices],
                    mask=np.ones_like(self.y[indices], dtype=bool))

    def inverse(self, values, indices):
        return values


class ToyPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1,1)

    def forward(self, x):
        return self.linear(x[:,0,:])


def test_v2_canonical_manifest_and_hp_grids():
    configs = protocol.make_manifest()
    assert len(configs) == 270
    assert Counter(c["method"] for c in configs) == {
        "mse": 35, "gaussian_nll": 35, "td_typed": 75, "td_pairwise": 125}
    assert set(protocol.METHODS) == {"mse","gaussian_nll","td_typed","td_pairwise"}
    assert protocol.BASELINE_ALPHAS == (.0002,.0003,.0005,.0007,.001,.003,.01)
    assert protocol.BETAS == (.00003,.0001,.0003,.001,.003)
    assert all(c["learning_curve_updates"] == protocol.EVAL_UPDATE_PERIOD
               and c["snapshot_updates"] == protocol.EVAL_UPDATE_PERIOD
               and c["selection_epochs"] == [80,90,100] for c in configs)
    assert len(set(c["run_id"] for c in configs)) == len(configs)


def test_checkpointed_training_restores_exact_mid_epoch_trajectory(
        tmp_path, monkeypatch):
    from agents.fulltd_mse import MSE
    config = next(c for c in protocol.make_manifest()
                  if c["method"] == "mse" and c["seed"] == 0)
    monkeypatch.setattr(train, "code_hash", lambda: "test-scientific-code-v2")
    monkeypatch.setattr(train, "build_objective", lambda config: MSE(ToyPredictor()))
    full, interrupted = tmp_path / "full", tmp_path / "interrupted"
    tr, va = ToyFD001("train"), ToyFD001("val", n=45)
    # Six Adam steps/epoch in this toy dataset: 600 total. Step 500 falls
    # mid-epoch, while step 533 interruption additionally tests cursor replay.
    first = train.train_candidate(config, tr, va, full, microbatch=256)
    paused = train.train_candidate(config, tr, va, interrupted,
                                   microbatch=256, stop_after_updates=533)
    assert first["state"] == "complete"
    assert paused["state"] == "checkpointed"
    assert paused["optimizer_steps"] == 533
    finished = train.train_candidate(config, tr, va, interrupted,
                                      microbatch=256)
    assert finished["state"] == "complete"
    assert finished["optimizer_steps"] == first["optimizer_steps"] == 600
    folder_a = full / config["run_id"]
    folder_b = interrupted / config["run_id"]
    for folder in (folder_a, folder_b):
        assert (folder / "checkpoints" / "step_0000000.pt").is_file()
        assert (folder / "checkpoints" / "step_0000500.pt").is_file()
        final = torch.load(folder / "final.pt", map_location="cpu", weights_only=False)
        assert [r["optimizer_steps"] for r in final["progress"]["curve"]] == [0,500]
        assert [r["epoch"] for r in final["progress"]["validation"]] == list(range(10,101,10))
    a = torch.load(folder_a / "final.pt", map_location="cpu", weights_only=False)
    b = torch.load(folder_b / "final.pt", map_location="cpu", weights_only=False)
    assert a["progress"]["validation"] == b["progress"]["validation"]
    assert a["progress"]["curve"] == b["progress"]["curve"]
    assert a["progress"]["rounds"] == b["progress"]["rounds"]
    for key in a["objective"]:
        torch.testing.assert_close(a["objective"][key], b["objective"][key],
                                   rtol=0, atol=0)


def test_v2_outputs_are_siblings_of_results(tmp_path, monkeypatch):
    monkeypatch.setenv("FULLTD_RESULTS_ROOT", str(tmp_path))
    monkeypatch.delenv("FULLTD_FD001_ROOT", raising=False)
    data, results, outputs = result_paths.paths("fd001")
    assert results == tmp_path / "fd001_results"
    assert outputs == tmp_path / "fd001_outputs"
    assert not results.exists() and not outputs.exists()


def test_v2_reporting_never_trains_or_refits():
    source = (Path(__file__).with_name("fd001_v2_analysis.py")).read_text()
    assert "optimizer.step(" not in source
    assert "refit_readouts" not in source
    assert "train_candidate(" not in source
    assert '("cycles", "rmse")' in source
    assert '("cycles", "mae")' in source
    assert '("score", "rul_score_sum")' in source
