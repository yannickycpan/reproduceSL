"""Regression tests for fresh 600-run FD001 sweep and 500-update recovery."""
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
    assert len(configs) == 600
    assert Counter(c["method"] for c in configs) == {
        "mse": 50, "gaussian_nll": 50, "td_typed": 250, "td_pairwise": 250}
    assert set(protocol.METHODS) == {"mse","gaussian_nll","td_typed","td_pairwise"}
    assert protocol.BASELINE_ALPHAS == (
        .0001, .0003, .0005, .0007, .0009, .001, .003, .005, .007, .01)
    assert protocol.TD_ALPHAS == (.00003, .0001, .0003, .001, .003)
    assert protocol.BATCH_SIZES["td_typed"] == (128, 256)
    assert protocol.BATCH_SIZES["td_pairwise"] == (128, 256)
    assert protocol.SELECTION_FRACTION == .25
    assert protocol.selection_update_steps(5500) == (4500, 5000, 5500)
    assert protocol.selection_update_steps(11000) == (
        8500, 9000, 9500, 10000, 10500, 11000)
    assert protocol.BETAS == (.00003,.0001,.0003,.001,.003)
    assert all(c["learning_curve_updates"] == protocol.EVAL_UPDATE_PERIOD
               and c["snapshot_updates"] == protocol.EVAL_UPDATE_PERIOD
               and c["selection_fraction"] == .25 for c in configs)
    assert all(c["run_id"].startswith("fd001_v2b_") for c in configs)
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
        assert "validation" not in final["progress"]
    a = torch.load(folder_a / "final.pt", map_location="cpu", weights_only=False)
    b = torch.load(folder_b / "final.pt", map_location="cpu", weights_only=False)
    assert a["progress"]["curve"] == b["progress"]["curve"]
    assert a["progress"]["rounds"] == b["progress"]["rounds"]
    for key in a["objective"]:
        torch.testing.assert_close(a["objective"][key], b["objective"][key],
                                   rtol=0, atol=0)


def test_v2_late_mse_selection_excludes_any_failed_seed(tmp_path, monkeypatch):
    import math
    from fd001_gamma.common import atomic_json
    from fulltd.sampling import sources_for_round, sequence_hash
    monkeypatch.setattr(analysis, "code_hash", lambda: "synthetic-v2")
    n = 1300
    root = tmp_path / "fd001_results"
    shas = {
        (seed, epoch): sequence_hash(sources_for_round("fd001", n, seed, epoch))
        for seed in range(5) for epoch in range(1, 101)
    }
    for config in protocol.make_manifest():
        directory = root / config["run_id"]
        directory.mkdir(parents=True)
        per_epoch = (n + config["batch_size"] - 1) // config["batch_size"]
        count = per_epoch * protocol.EPOCHS
        # The first pairwise HP has one genuinely divergent seed. Even if all
        # its remaining seeds are excellent it must receive infinite score.
        fail = (config["method"] == "td_pairwise" and config["batch_size"] == 128
                and config["configuration"] == 0 and config["seed"] == 0)
        state = "numerical_failure" if fail else "complete"
        atomic_json(directory / "status.json", dict(
            config=config, state=state, epochs_completed=100,
            next_epoch=101, n_sources=n, optimizer_steps=count,
            update_opportunities=count,
            initial_mean_sha256=f"same-init-s{config['seed']}",
            identity=dict(protocol_id=protocol.PROTOCOL_ID,
                          config_sha256=protocol.digest(config),
                          code_sha256="synthetic-v2",
                          data_sha256="same-training", validation_sha256="same-validation")))
        if fail:
            continue
        (directory / "final.pt").touch()
        score = 0.1 + .01 * config["configuration"]
        milestones = list(range(0, count + 1, protocol.EVAL_UPDATE_PERIOD))
        atomic_json(directory / "update_curve.json", [
            dict(epoch=min(protocol.EPOCHS, step//per_epoch+1),
                 optimizer_steps=step, mse_standardized=score)
            for step in milestones])
        atomic_json(directory / "rounds.json", [
            dict(epoch=e, source_draws=n,
                 source_sha256=shas[(config["seed"], e)],
                 optimizer_steps_cumulative=e*per_epoch)
            for e in range(1, 101)])
        snapshot_dir = directory / "checkpoints"
        snapshot_dir.mkdir()
        for step in milestones:
            (snapshot_dir / f"step_{step:07d}.pt").touch()
    selected, scores = analysis.audit_and_select(root)
    assert len(scores) == 120
    assert len(selected["selected"]) == 4
    assert selected["counts"] == {"complete":599, "numerical_failure":1}
    excluded = next(r for r in scores if
                    r["method"]=="td_pairwise" and r["configuration"] == 0
                    and r["batch_size"] == 128)
    assert math.isinf(excluded["score"])
    assert excluded["failed_seeds"] == [0]
    choice = next(r for r in selected["selected"] if r["method"]=="td_pairwise")
    assert (choice["batch_size"], choice["configuration"]) != (128, 0)


def test_pairwise_td_uses_same_update_checkpoint_and_selection_protocol(
        tmp_path, monkeypatch):
    from fd001_gamma_ext.objectives import PairwiseGammaTD

    class TinyPairwiseMean(nn.Module):
        feature_dim = 8
        output_dim = 1

        def __init__(self):
            super().__init__()
            self.encoding = nn.Linear(24, self.feature_dim)
            self.head = nn.Linear(self.feature_dim, 1)

        def features(self, x):
            return torch.tanh(self.encoding(x[:, :, -1]))

        def forward(self, x):
            return self.head(self.features(x))

    class PairwiseData(ToyFD001):
        def __init__(self, split, n):
            super().__init__(split, n)
            self.x = (np.arange(n * 24 * 30, dtype=np.float32)
                      .reshape(n, 24, 30) / 100000.)
            self.y = (self.x[:, :1, -1] * .4 + .2).astype(np.float32)

    config = next(c for c in protocol.make_manifest()
                  if c["method"] == "td_pairwise" and c["seed"] == 0)
    monkeypatch.setattr(train, "code_hash", lambda: "pairwise-v2-test-code")
    monkeypatch.setattr(train, "build_objective",
                        lambda _: PairwiseGammaTD(TinyPairwiseMean()))
    root = tmp_path / "pairwise"
    result = train.train_candidate(
        config, PairwiseData("train", 600), PairwiseData("val", 15),
        root, microbatch=128)
    assert result["state"] == "complete"
    assert result["optimizer_steps"] == 500
    directory = root / config["run_id"]
    assert (directory / "checkpoints" / "step_0000000.pt").exists()
    assert (directory / "checkpoints" / "step_0000500.pt").exists()
    checkpoint = torch.load(directory / "final.pt",
                            map_location="cpu", weights_only=False)
    assert [r["optimizer_steps"] for r in checkpoint["progress"]["curve"]] == [0,500]
    assert all("gamma_probe" in r for r in checkpoint["progress"]["curve"])
    assert "validation" not in checkpoint["progress"]


def test_v2_outputs_are_siblings_of_results(tmp_path, monkeypatch):
    monkeypatch.setenv("FULLTD_RESULTS_ROOT", str(tmp_path))
    monkeypatch.delenv("FULLTD_FD001_ROOT", raising=False)
    data, results, outputs = result_paths.paths("fd001")
    assert results == tmp_path / "fd001_results"
    assert outputs == tmp_path / "fd001_outputs"
    assert not results.exists() and not outputs.exists()


def test_matched_seed_comparison_does_not_omit_any_seed():
    records=[]
    for method_index,method in enumerate(protocol.METHODS):
        for seed in protocol.SEEDS:
            records.append(dict(method=method, seed=seed, split="test",
                                metric="rmse",units="cycles",
                                value=10.+method_index+seed*.1))
    paired=analysis.paired_differences(records)
    assert len(paired)==6
    first=next(r for r in paired
               if r["first"]=="mse" and r["second"]=="gaussian_nll")
    assert first["n"]==5
    assert first["mean_difference"]==pytest.approx(-1.)
    assert first["descriptive_t95_lower"]<=-1.<=first["descriptive_t95_upper"]


def test_v2_reporting_never_trains_or_refits():
    source = (Path(__file__).with_name("fd001_v2_analysis.py")).read_text()
    assert "optimizer.step(" not in source
    assert "refit_readouts" not in source
    assert "train_candidate(" not in source
    assert '("cycles", "rmse")' in source
    assert '("cycles", "mae")' in source
    assert '("score", "rul_score_sum")' in source
