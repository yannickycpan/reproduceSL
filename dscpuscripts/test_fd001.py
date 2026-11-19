"""Regression tests for fresh 1600-run FD001 lag sweep and 500-update recovery."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from dscpuscripts import fd001_protocol as protocol
from dscpuscripts import fd001_train as train
from dscpuscripts import fd001_analysis as analysis
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


def test_fd001_canonical_manifest_and_hp_grids():
    configs = protocol.make_manifest()
    assert len(configs) == 1600
    assert Counter(c["method"] for c in configs) == {
        "mse": 50, "gaussian_nll": 50, "td_typed": 750, "td_pairwise": 750}
    assert Counter((c["method"], c["batch_size"]) for c in configs) == {
        ("mse", 256): 50, ("gaussian_nll", 256): 50,
        ("td_typed", 128): 375, ("td_typed", 256): 375,
        ("td_pairwise", 128): 375, ("td_pairwise", 256): 375}
    assert set(protocol.METHODS) == {"mse","gaussian_nll","td_typed","td_pairwise"}
    assert protocol.BASELINE_ALPHAS == (
        .0001, .0003, .0005, .0007, .0009, .001, .003, .005, .007, .01)
    assert protocol.TD_ALPHAS == (.00003, .0001, .0003, .001, .003)
    assert protocol.BATCH_SIZES["td_typed"] == (128, 256)
    assert protocol.BATCH_SIZES["td_pairwise"] == (128, 256)
    assert analysis.SELECTION_FRACTION == .25
    assert analysis.selection_update_steps(5500) == (4500, 5000, 5500)
    assert analysis.selection_update_steps(11000) == (
        8500, 9000, 9500, 10000, 10500, 11000)
    assert protocol.BETAS == (.00003,.0001,.0003,.001,.003)
    assert all(c["learning_curve_updates"] == protocol.EVAL_UPDATE_PERIOD
               and c["snapshot_updates"] == protocol.EVAL_UPDATE_PERIOD
               and "selection_fraction" not in c and "selection" not in c
               for c in configs)
    assert all(c["run_id"].startswith("fd001_") for c in configs)
    assert len(set(c["run_id"] for c in configs)) == len(configs)


def test_checkpointed_training_restores_exact_mid_epoch_trajectory(
        tmp_path, monkeypatch):
    from agents.fulltd_mse import MSE
    config = next(c for c in protocol.make_manifest()
                  if c["method"] == "mse" and c["seed"] == 0)
    monkeypatch.setattr(train, "code_hash", lambda: "test-scientific-code")
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


def test_fd001_late_mse_selection_excludes_any_failed_seed(tmp_path, monkeypatch):
    import math
    from dscpuscripts.common import atomic_json
    from fulltd.sampling import sources_for_round, sequence_hash
    monkeypatch.setattr(analysis, "code_hash", lambda: "synthetic-current")
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
                and config["configuration"] == 0 and config["seed"] == 0
                and config["lags"] == [1, 5])
        state = "numerical_failure" if fail else "complete"
        atomic_json(directory / "status.json", dict(
            config=config, state=state, epochs_completed=100,
            next_epoch=101, n_sources=n, optimizer_steps=count,
            update_opportunities=count,
            initial_mean_sha256=f"same-init-s{config['seed']}",
            identity=dict(protocol_id=protocol.PROTOCOL_ID,
                          config_sha256=protocol.digest(config),
                          code_sha256="synthetic-current",
                          data_sha256="same-training", validation_sha256="same-validation")))
        if fail:
            continue
        (directory / "final.pt").touch()
        score = 0.1 + .01 * config["configuration"]
        milestones = list(range(0, count + 1, protocol.EVAL_UPDATE_PERIOD))
        tail = set(analysis.selection_update_steps(count))
        # Earlier 75% deliberately scores better. The selector must ignore
        # that earlier performance and use only the recorded final quarter.
        atomic_json(directory / "update_curve.json", [
            dict(epoch=min(protocol.EPOCHS, step//per_epoch+1),
                 optimizer_steps=step,
                 mse_standardized=score if step in tail else score*.1)
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
    assert len(scores) == 320
    assert len(selected["selected"]) == 8
    assert selected["counts"] == {"complete":1599, "numerical_failure":1}
    excluded = next(r for r in scores if
                    r["method"]=="td_pairwise" and r["lag_set"] == "1-5"
                    and r["configuration"] == 0 and r["batch_size"] == 128)
    assert math.isinf(excluded["score"])
    assert excluded["failed_seeds"] == [0]
    viable = next(r for r in scores if r["method"] == "mse"
                  and r["configuration"] == 0)
    assert viable["score"] == pytest.approx(0.1)  # NOT ~0.025 from early evaluations
    choice = next(r for r in selected["selected"] if r["method"]=="td_pairwise"
                  and r["lag_set"] == "1-5")
    assert choice["batch_size"] == 256
    assert choice["configuration"] == 0  # Same rate pair at B256 remains valid


def test_pairwise_td_uses_same_update_checkpoint_and_selection_protocol(
        tmp_path, monkeypatch):
    from dscpuscripts.fd001_objectives import PairwiseGammaTD

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


def test_fresh_campaign_status_requires_update_curve_not_old_epoch_validation(
        tmp_path):
    from dscpuscripts.standard_runner import classify
    from dscpuscripts.common import atomic_json
    from fulltd.protocol import digest

    config = next(c for c in protocol.make_manifest()
                  if c["method"] == "td_typed" and c["batch_size"] == 256)
    folder = tmp_path / config["run_id"]
    folder.mkdir()
    atomic_json(folder / "status.json", dict(
        config=config, state="complete", epochs_completed=100,
        identity=dict(config_sha256=digest(config), code_sha256="fixture-code")))
    (folder / "final.pt").touch()
    (folder / "rounds.json").touch()
    assert classify(config, tmp_path, "fixture-code") == "resume"
    (folder / "update_curve.json").touch()
    assert classify(config, tmp_path, "fixture-code") == "complete"
    assert not (folder / "validation.json").exists()


def test_fd001_outputs_are_siblings_of_results(tmp_path, monkeypatch):
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


def test_analysis_curve_nonfinite_checkpoint_and_json_agree(tmp_path):
    """Nonfinite diagnostics must not invalidate an otherwise matching run."""
    import inspect
    from dscpuscripts.common import atomic_json, read_json
    from dscpuscripts.fd001_train import json_safe

    checkpoint_curve = [dict(
        optimizer_steps=500, mse_standardized=1.2,
        rul_score_sum=float("inf"),
        gamma_probe=dict(types=[dict(mean=float("-inf"), std=float("nan"))]))]
    sidecar = tmp_path / "update_curve.json"
    atomic_json(sidecar, checkpoint_curve)
    recorded_curve = read_json(sidecar)

    assert checkpoint_curve != recorded_curve  # Raw floats vs JSON strings
    assert json_safe(checkpoint_curve) == recorded_curve
    # Check the real audit path, not just the serializer in isolation.
    assert 'json_safe(ckpt["progress"]["curve"])' in inspect.getsource(
        analysis.evaluate_selected)


def test_fd001_reporting_never_trains_or_refits():
    source = (Path(__file__).with_name("fd001_analysis.py")).read_text()
    assert "optimizer.step(" not in source
    assert "refit_readouts" not in source
    assert "train_candidate(" not in source
    assert '("cycles", "rmse")' in source
    assert '("cycles", "mae")' in source
    assert '("score", "rul_score_sum")' in source
