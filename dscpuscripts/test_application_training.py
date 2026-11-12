"""Current application integration: fixed protocol, exact restart and endpoints."""
from __future__ import annotations

from collections import Counter
import copy
import math

import numpy as np
import pytest
import torch
from torch import nn

from dscpuscripts import application_protocol as protocol
from dscpuscripts import application_train as train
from dscpuscripts.application_objectives import make_objective
from dscpuscripts.common import read_json


@pytest.fixture(autouse=True)
def one_thread():
    old, mkldnn = torch.get_num_threads(), torch.backends.mkldnn.enabled
    torch.set_num_threads(1)
    torch.backends.mkldnn.enabled = False
    yield
    torch.set_num_threads(old)
    torch.backends.mkldnn.enabled = mkldnn


class ToyData:
    def __init__(self, dataset, split, n=300, real=False):
        self.dataset, self.split = dataset, split
        self.entity_ids = np.zeros(n, dtype=np.int64)
        self.origins = np.arange(n, dtype=np.int64)
        if dataset == "electricity" and not real:
            midpoint = (n + 1) // 2
            self.entity_ids[midpoint:] = 1
            self.origins[midpoint:] -= midpoint
        self.metadata = dict(fingerprint=f"synthetic-{dataset}-{split}-{n}-{real}")
        rng = np.random.default_rng(715 if split == "train" else 716)
        if real and dataset == "electricity":
            self.x = rng.normal(size=(n, 1, 168)).astype(np.float32)
            self.y = rng.normal(size=(n, 24)).astype(np.float32)
        elif real:
            self.adjacency = np.eye(3, dtype=np.float32)
            self.x = rng.normal(size=(n, 2, 3, 12)).astype(np.float32)
            self.y = rng.normal(size=(n, 3, 12)).astype(np.float32)
        else:
            self.x = rng.normal(size=(n, 1, 2)).astype(np.float32)
            self.y = (self.x[:, 0, :] * .3 + .2).astype(np.float32)
        self.mask = np.ones_like(self.y, dtype=bool)

    def __len__(self):
        return len(self.x)

    def get_batch(self, indices):
        assert self.split != "test", "Candidate training must not read test targets"
        return dict(x=self.x[indices], y=self.y[indices], mask=self.mask[indices])

    def inverse(self, values, indices):
        return values * 2.0 + 3.0


class ToyMean(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_dim, self.output_dim = 4, 2
        self.hidden = nn.Linear(2, 4)
        self.head = nn.Linear(4, 2)

    def features(self, x):
        return torch.tanh(self.hidden(x[:, 0]))

    def forward(self, x):
        return self.head(self.features(x))


def config_for(dataset="metr_la", backbone="graph_wavenet", method="td_pairwise", batch_size=256):
    return copy.deepcopy(next(c for c in protocol.make_manifest() if
        (c["dataset"], c["backbone"], c["method"], c["batch_size"], c["seed"], c["configuration"]) ==
        (dataset, backbone, method, batch_size, 0, 0)))


def small_config(monkeypatch, **kwargs):
    config = config_for(**kwargs)
    config.update(epochs=2, learning_curve_updates=5, snapshot_updates=5)
    if config["dataset"] == "metr_la":
        # Existing checkpoint tests retain the original full-epoch fixture.
        # The 10%-sampling resume path is tested separately below.
        config["source_fraction"] = 1.0
        config["source_sampling"] = "random_reshuffling"
    # The production validator is tested separately; shortening only the
    # fixture keeps this a meaningful end-to-end restart test without a sweep.
    monkeypatch.setattr(train, "validate_config", lambda c: c)
    monkeypatch.setattr(train, "code_hash", lambda: "fixture-current-scientific-code")
    return config


def assert_tree_equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_tree_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_tree_equal(a, b)
    else:
        assert left == right


def test_application_manifest_matches_fd001_grids_without_legacy_methods():
    from dscpuscripts import fd001_protocol as fd001
    manifest = protocol.make_manifest()
    assert len(manifest) == len({c["run_id"] for c in manifest}) == 875
    assert Counter(c["dataset"] for c in manifest) == {"electricity": 350, "metr_la": 525}
    assert protocol.BASELINE_ALPHAS == fd001.BASELINE_ALPHAS
    assert protocol.TD_ALPHAS == fd001.TD_ALPHAS
    assert protocol.BETAS == (0.00003, 0.0001, 0.001, 0.003, 0.01)
    assert protocol.SEEDS == fd001.SEEDS and protocol.EPOCHS == fd001.EPOCHS == 100
    assert protocol.ELECTRICITY_ROUNDS == 240
    for config in manifest:
        assert config["batch_size"] == 256
        assert config["epochs"] == (240 if config["dataset"] == "electricity" else 100)
        assert not (config["dataset"] == "electricity" and config["backbone"] == "tcn")
        assert config["method"] not in ("td_full", "td_scalar", "dlinear_mse")
        assert config["backbone"] != "dlinear"
        assert "selection_epochs" not in config and "validation_epochs" not in config
        assert config["learning_curve_updates"] == (
            5000 if config["dataset"] == "electricity" else protocol.METR_LA_EVAL_PERIOD)
        if config["dataset"] == "metr_la":
            assert config["source_fraction"] == protocol.METR_LA_SOURCE_FRACTION
            assert config["source_sampling"] == (
                "random_reshuffling" if config["source_fraction"] == 1
                else "iid_uniform_with_replacement")
        assert config["optimizer"] == "adam" and config["weight_decay"] == 0
        assert config["gradient_clipping"] is None and config["learning_rate_schedule"] is None
    for dataset, backbone, methods in protocol.MATRIX:
        selected = [c for c in manifest if (c["dataset"], c["backbone"]) == (dataset, backbone)]
        counts = Counter(c["method"] for c in selected)
        assert counts == {method: 125 if method.startswith("td_") else 50 for method in methods}
    bad = dict(manifest[0], learning_curve_updates=500)
    with pytest.raises(ValueError, match="current"):
        protocol.validate_config(bad)
    with pytest.raises(ValueError, match="current"):
        protocol.validate_config(dict(manifest[0], method="td_full"))


def test_nondivisible_final_measurement_is_not_lost():
    assert protocol.evaluation_steps(257040, 5000) == tuple(
        [0, *range(5000, 257041, 5000), 257040])
    config = config_for("electricity", "patchtst")
    budget = protocol.update_budgets(config, 6578267)
    assert budget["draws_per_round"] == math.ceil(6578267 / 24)
    assert budget["total_updates"] == budget["matched_updates"] == 257040
    # ceil(N/24) draws adds less than one extra source per round.
    # Exactly ten *effective* passes is not generally an integer draw count.
    effective_passes = config["epochs"] * budget["draws_per_round"] / 6578267
    target_passes = config["epochs"] / config["source_divisor"]
    assert target_passes == 10
    assert target_passes <= effective_passes < target_passes + config["epochs"] / 6578267


def test_extended_round_sampler_reuses_original_law_and_is_deterministic():
    from fulltd.sampling import sources_for_round as frozen_source
    n = 10123
    for step in (1, 99, 100):
        np.testing.assert_array_equal(
            protocol.sources_for_application_round("electricity", n, 3, step),
            frozen_source("electricity", n, 3, step))
    for step in (101, 120, 240):
        a = protocol.sources_for_application_round("electricity", n, 3, step)
        b = protocol.sources_for_application_round("electricity", n, 3, step)
        assert len(a) == math.ceil(n / 24)
        np.testing.assert_array_equal(a, b)
        assert (a >= 0).all() and (a < n).all()
        assert not np.array_equal(a,
            protocol.sources_for_application_round("electricity", n, 4, step))
    with pytest.raises(ValueError, match="round"):
        protocol.sources_for_application_round("electricity", n, 3, 241)


@pytest.mark.parametrize("fraction", [0.1, 0.25, 1.0])
def test_metr_la_sampling_budget_reproducibility_and_exact_full_recovery(fraction):
    from fulltd.sampling import sources_for_round
    config = config_for()
    config["source_fraction"] = fraction
    n = 23967
    budget = protocol.update_budgets(config, n)
    assert budget["draws_per_round"] == math.ceil(n * fraction)
    assert budget["total_updates"] == 100 * math.ceil(math.ceil(n * fraction) / 256)
    draw = protocol.sources_for_application_round("metr_la", n, 2, 5, fraction)
    assert len(draw) == budget["draws_per_round"]
    assert (draw >= 0).all() and (draw < n).all()
    np.testing.assert_array_equal(
        draw, protocol.sources_for_application_round("metr_la", n, 2, 5, fraction))
    if fraction == 1:
        np.testing.assert_array_equal(draw, sources_for_round("metr_la", n, 2, 5))
    else:
        assert len(np.unique(draw)) < len(draw)


def test_fixed_electricity_validation_is_shared_and_rng_independent():
    data = ToyData("electricity", "val", n=29)
    np.random.seed(31)
    torch.manual_seed(32)
    np_state, torch_state = np.random.get_state(), torch.get_rng_state().clone()
    first, metadata = train.evaluation_validation_view(data)
    np.random.rand(80)
    second, other_metadata = train.evaluation_validation_view(data)
    assert metadata == other_metadata
    np.testing.assert_array_equal(first.indices, second.indices)
    assert metadata["sampled_size"] == math.ceil(15 / 5) + math.ceil(14 / 5)
    assert len(np.unique(first.indices)) == len(first)
    for row in metadata["per_customer"]:
        assert row["sampled_size"] == math.ceil(row["population_size"] / 5)
    torch.testing.assert_close(torch_state, torch.get_rng_state(), rtol=0, atol=0)
    # Restore the NumPy stream and check view construction makes no draws.
    np.random.set_state(np_state)
    train.evaluation_validation_view(data)
    after = np.random.get_state()
    assert np_state[0] == after[0] and np_state[2:] == after[2:]
    np.testing.assert_array_equal(np_state[1], after[1])
    ids = np.arange(len(first))
    np.testing.assert_array_equal(first.get_batch(ids)["x"], data.get_batch(first.indices)["x"])


@pytest.mark.parametrize("dataset,backbone,n", [("electricity", "patchtst", 6240), ("metr_la", "graph_wavenet", 300)])
@pytest.mark.parametrize("method", ["td_typed", "td_pairwise"])
def test_mid_round_resume_is_exact_and_retains_every_endpoint(tmp_path, monkeypatch, dataset, backbone, n, method):
    config = small_config(monkeypatch, dataset=dataset, backbone=backbone, method=method)
    monkeypatch.setattr(train, "build_objective", lambda c, data: make_objective(c["method"], ToyMean()))
    tr, va = ToyData(dataset, "train", n=n), ToyData(dataset, "val", n=35)
    full, resumed = tmp_path / "full", tmp_path / "resumed"
    done = train.train_candidate(config, tr, va, full, microbatch=19)
    paused = train.train_candidate(config, tr, va, resumed, microbatch=19, stop_after_updates=1)
    assert paused["state"] == "checkpointed" and paused["cursor"] == 256
    again = train.train_candidate(config, tr, va, resumed, microbatch=19)
    assert done["state"] == again["state"] == "complete"
    assert done["optimizer_steps"] == again["optimizer_steps"] == 4
    assert done["matched_updates"] == 4 and done["epochs_completed"] == 2
    a = torch.load(full / config["run_id"] / "final.pt", weights_only=False)
    b = torch.load(resumed / config["run_id"] / "final.pt", weights_only=False)
    for key in ("objective", "optimizer", "progress"):
        assert_tree_equal(a[key], b[key])
    for root in (full, resumed):
        directory = root / config["run_id"]
        assert [r["optimizer_steps"] for r in read_json(directory / "update_curve.json")] == [0, 4]
        for step in (0, 4):
            snapshot = torch.load(directory / "checkpoints" / f"step_{step:07d}.pt", weights_only=False)
            assert snapshot["progress"]["optimizer_steps"] == step
            assert snapshot["progress"]["curve"][-1]["optimizer_steps"] == step
            assert snapshot["progress"]["pending_evaluation"] is None
        assert len(read_json(directory / "rounds.json")) == 2


def test_ten_percent_metr_la_resume_matches_uninterrupted(tmp_path, monkeypatch):
    config = small_config(monkeypatch, method="td_pairwise")
    config.update(source_fraction=0.1, source_sampling="iid_uniform_with_replacement",
                  learning_curve_updates=1, snapshot_updates=1)
    monkeypatch.setattr(train, "build_objective",
                        lambda c, data: make_objective(c["method"], ToyMean()))
    tr, va = ToyData("metr_la", "train", n=300), ToyData("metr_la", "val", n=24)
    done = train.train_candidate(config, tr, va, tmp_path / "full", microbatch=16)
    paused = train.train_candidate(config, tr, va, tmp_path / "resumed",
                                   microbatch=16, stop_after_updates=1)
    assert paused["state"] == "checkpointed"
    resumed = train.train_candidate(config, tr, va, tmp_path / "resumed", microbatch=16)
    assert done["state"] == resumed["state"] == "complete"
    assert done["optimizer_steps"] == resumed["optimizer_steps"] == 2
    assert done["source_draws"] == 2 * math.ceil(len(tr) * 0.1)
    a = torch.load(tmp_path / "full" / config["run_id"] / "final.pt", weights_only=False)
    b = torch.load(tmp_path / "resumed" / config["run_id"] / "final.pt", weights_only=False)
    for key in ("objective", "optimizer", "progress"):
        assert_tree_equal(a[key], b[key])


def test_electricity_resumes_exactly_after_round_100(tmp_path, monkeypatch):
    config = small_config(monkeypatch, dataset="electricity", backbone="patchtst",
                          method="td_typed")
    config.update(epochs=102, learning_curve_updates=200, snapshot_updates=200)
    monkeypatch.setattr(train, "build_objective",
                        lambda c, data: make_objective(c["method"], ToyMean()))
    tr, va = ToyData("electricity", "train", n=48), ToyData("electricity", "val", n=30)
    normal, interrupted = tmp_path / "normal", tmp_path / "interrupted"
    completed = train.train_candidate(config, tr, va, normal, microbatch=2)
    paused = train.train_candidate(config, tr, va, interrupted, microbatch=2,
                                   stop_after_updates=101)
    assert paused["state"] == "checkpointed"
    assert paused["optimizer_steps"] == 101
    resumed = train.train_candidate(config, tr, va, interrupted, microbatch=2)
    assert completed["state"] == resumed["state"] == "complete"
    assert resumed["epochs_completed"] == 102
    assert resumed["optimizer_steps"] == 102
    left = torch.load(normal / config["run_id"] / "final.pt", weights_only=False)
    right = torch.load(interrupted / config["run_id"] / "final.pt", weights_only=False)
    for key in ("objective", "optimizer", "progress"):
        assert_tree_equal(left[key], right[key])


def test_interrupted_validation_replays_observation_before_next_update(tmp_path, monkeypatch):
    config = small_config(monkeypatch)
    monkeypatch.setattr(train, "build_objective", lambda c, data: make_objective(c["method"], ToyMean()))
    tr, va = ToyData("metr_la", "train"), ToyData("metr_la", "val", n=45)
    capture = train.capture_validation
    full = tmp_path / "full"
    train.train_candidate(config, tr, va, full, microbatch=19)
    calls = {"batches": 0}

    def interrupt_after_one_batch():
        calls["batches"] += 1
        return calls["batches"] > 1

    def interrupted_capture(objective, validation, record, batch, step, epoch, stop_requested=None):
        return capture(objective, validation, record, batch, step, epoch,
                       interrupt_after_one_batch if step == 4 else stop_requested)

    monkeypatch.setattr(train, "capture_validation", interrupted_capture)
    resumed = tmp_path / "resumed"
    paused = train.train_candidate(config, tr, va, resumed, microbatch=19)
    assert paused["state"] == "checkpointed" and paused["optimizer_steps"] == 4
    recovery = torch.load(resumed / config["run_id"] / "recovery.pt", weights_only=False)
    assert recovery["progress"]["pending_evaluation"] == 4
    assert [r["optimizer_steps"] for r in recovery["progress"]["curve"]] == [0]
    monkeypatch.setattr(train, "capture_validation", capture)
    train.train_candidate(config, tr, va, resumed, microbatch=19)
    a = torch.load(full / config["run_id"] / "final.pt", weights_only=False)
    b = torch.load(resumed / config["run_id"] / "final.pt", weights_only=False)
    for key in ("objective", "optimizer", "progress"):
        assert_tree_equal(a[key], b[key])


def test_empty_masks_do_not_take_adam_steps_or_hide_budget_difference(tmp_path, monkeypatch):
    config = small_config(monkeypatch)
    monkeypatch.setattr(train, "build_objective", lambda c, data: make_objective(c["method"], ToyMean()))
    tr, va = ToyData("metr_la", "train"), ToyData("metr_la", "val", n=20)
    tr.mask.fill(False)
    status = train.train_candidate(config, tr, va, tmp_path, microbatch=32)
    assert status["state"] == "complete"
    assert status["optimizer_steps"] == 0
    # n=300, B256: two attempted batches per epoch, for two epochs.
    expected_opportunities = config["epochs"] * math.ceil(len(tr) / config["batch_size"])
    assert expected_opportunities == 4
    assert status["empty_mask_batches"] == status["update_opportunities"] == status["total_updates"] == expected_opportunities
    ckpt = torch.load(tmp_path / config["run_id"] / "final.pt", weights_only=False)
    assert ckpt["optimizer"]["state"] == {}
    assert [r["optimizer_steps"] for r in ckpt["progress"]["curve"]] == [0]


def test_initial_numerical_failure_is_preserved_without_restart(tmp_path, monkeypatch):
    config = small_config(monkeypatch)
    monkeypatch.setattr(train, "build_objective", lambda c, data: make_objective(c["method"], ToyMean()))

    def fail(*args, **kwargs):
        raise FloatingPointError("fixture divergent predictions")

    monkeypatch.setattr(train, "capture_validation", fail)
    tr, va = ToyData("metr_la", "train"), ToyData("metr_la", "val", n=10)
    failed = train.train_candidate(config, tr, va, tmp_path)
    assert failed["state"] == "numerical_failure"
    assert failed["optimizer_steps"] == 0 and "divergent" in failed["error"]
    assert read_json(tmp_path / config["run_id"] / "update_curve.json") == []
    assert train.train_candidate(config, tr, va, tmp_path) == failed


@pytest.mark.parametrize("dataset,backbone,method", [
    ("electricity", "patchtst", "td_typed"),
    ("electricity", "patchtst", "td_pairwise"),
    ("electricity", "patchtst", "gaussian_nll"),
    ("electricity", "patchtst", "td_pairwise"),
    ("metr_la", "graph_wavenet", "gaussian_nll"),
    ("metr_la", "graph_wavenet", "td_pairwise"),
    ("metr_la", "graph_wavenet", "td_pairwise_mae"),
])
def test_real_backbones_train_and_save_usable_final_state(tmp_path, monkeypatch, dataset, backbone, method):
    config = small_config(monkeypatch, dataset=dataset, backbone=backbone, method=method,
                          batch_size=256)
    config["epochs"] = 1
    tr, va = ToyData(dataset, "train", n=4, real=True), ToyData(dataset, "val", n=4, real=True)
    status = train.train_candidate(config, tr, va, tmp_path, microbatch=2)
    assert status["state"] == "complete" and status["optimizer_steps"] == 1
    checkpoint = torch.load(tmp_path / config["run_id"] / "final.pt", weights_only=False)
    objective = train.build_objective(config, tr)
    objective.load_state_dict(checkpoint["objective"])
    prediction = objective.predict(torch.as_tensor(va.x))
    assert tuple(prediction.shape) == tuple(va.y.shape)
    assert torch.isfinite(prediction).all()
    assert [r["optimizer_steps"] for r in checkpoint["progress"]["curve"]] == [0, 1]
