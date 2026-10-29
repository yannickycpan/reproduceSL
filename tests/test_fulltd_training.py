"""End-to-end protocol bookkeeping with tiny synthetic, non-scientific data."""
import copy
import json
import math

import numpy as np
import pytest
import torch
from torch import nn

from agents.fulltd_objectives import make_objective
from fulltd import train
from fulltd.metrics import MetricSums, evaluate, rul_score, validation_mse
from fulltd.protocol import make_manifest


class SyntheticData:
    """Tiny explicit dataset; test-split reads are forbidden in tuning tests."""
    def __init__(self, n=259, dataset="fd001", split="train", masked=False):
        self.dataset, self.split = dataset, split
        self.entity_ids = np.zeros(n, dtype=np.int64)
        self.origins = np.arange(n, dtype=np.int64) + 30
        self.metadata = dict(dataset=dataset, split=split, synthetic=True, size=n, masked=masked)
        values = np.arange(n, dtype=np.float32) / max(n, 1)
        self.x = np.stack((values, 1 - values), axis=1)
        self.y = (0.7 * values + 0.2 * values**2)[:, None]
        self.mask = np.ones((n, 1), dtype=bool)
        if masked:
            self.mask[1::2] = False
        self.reads = []

    def __len__(self):
        return len(self.origins)

    def get_batch(self, indices):
        if self.split == "test":
            raise AssertionError("Test targets were accessed by tuning")
        indices = np.asarray(indices)
        self.reads.append(indices.copy())
        return {"x": self.x[indices].copy(), "y": self.y[indices].copy(), "mask": self.mask[indices].copy()}

    def inverse(self, values, indices):
        return np.asarray(values) * 7.0 + 4.0


class TinyMean(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_dim, self.output_dim = 4, 1
        self.hidden = nn.Linear(2, 4)
        self.head = nn.Linear(4, 1)

    def features(self, x):
        return torch.tanh(self.hidden(x))

    def forward(self, x):
        return self.head(self.features(x))


def config_for(dataset="fd001", method="td_full", batch_size=256, configuration=0):
    return next(copy.deepcopy(c) for c in make_manifest()
                if c["dataset"] == dataset and c["backbone"] == "tcn"
                and c["method"] == method and c["batch_size"] == batch_size
                and c["configuration"] == configuration and c["seed"] == 0)


@pytest.fixture(autouse=True)
def tiny_training(monkeypatch):
    old_threads = torch.get_num_threads()
    old_deterministic = torch.are_deterministic_algorithms_enabled()
    torch.set_num_threads(1)

    def factory(config, dataset):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(1234 + config["seed"])
            mean = TinyMean()
        return make_objective(config["method"], mean)

    monkeypatch.setattr(train, "make_objective_for", factory)
    # Root/source review may run concurrently with these tests. The fixture
    # itself is fixed, and checkpoint compatibility is tested separately.
    monkeypatch.setattr(train, "code_hash", lambda: "synthetic-training-fixture-v1")
    yield
    torch.set_num_threads(old_threads)
    torch.use_deterministic_algorithms(old_deterministic)


def test_actual_100_full_epochs_and_ten_complete_validations(tmp_path):
    fitting, validation = SyntheticData(), SyntheticData(11, split="validation")
    config = config_for()
    result = train.train_candidate(config, fitting, validation, tmp_path, microbatch=128)
    assert result["next_epoch"] == 101 and result["cursor"] == 0
    assert [v["epoch"] for v in result["validation"]] == list(range(10, 101, 10))
    assert result["counters"]["source_draws"] == 25900
    assert result["counters"]["partner_draws"] == 25900
    assert result["counters"]["optimizer_steps"] == 200  # final short batch retained
    assert len(result["round_records"]) == 100
    assert len(validation.reads) == 10
    assert all(np.array_equal(read, np.arange(11)) for read in validation.reads)
    loaded = train.make_objective_for(config, fitting)
    loaded.load_state_dict(result["objective"])
    # The recorded epoch-100 MSE must describe this actual final checkpoint.
    assert validation_mse(loaded.model, validation) == pytest.approx(result["validation"][-1]["mse"], abs=1e-12)
    status = json.loads((tmp_path / config["run_id"] / "status.json").read_text())
    assert status["state"] == "complete" and status["epochs_completed"] == 100
    assert status["effective_original_passes"] == 100


def test_mid_epoch_recovery_matches_uninterrupted_weights_optimizer_and_sequences(tmp_path, monkeypatch):
    monkeypatch.setattr(train.platform, "processor", lambda: "original-test-host")
    config = config_for()
    fitting, validation = SyntheticData(), SyntheticData(8, split="validation")
    complete = train.train_candidate(config, fitting, validation, tmp_path / "full", microbatch=128)
    stopped = train.train_candidate(config, fitting, validation, tmp_path / "resume", microbatch=128,
                                     stop_after_batches=3)
    assert stopped["next_epoch"] == 2 and stopped["cursor"] == 256
    monkeypatch.setattr(train.platform, "processor", lambda: "resumed-test-host")
    resumed = train.train_candidate(config, fitting, validation, tmp_path / "resume", microbatch=128)
    assert complete["counters"] == resumed["counters"]
    assert complete["validation"] == resumed["validation"]
    assert complete["initial_mean_sha256"] == resumed["initial_mean_sha256"]
    for key, parameter in complete["objective"].items():
        torch.testing.assert_close(parameter, resumed["objective"][key], rtol=0, atol=0)
    for key, state in complete["optimizer"]["state"].items():
        for name, value in state.items():
            torch.testing.assert_close(value, resumed["optimizer"]["state"][key][name], rtol=0, atol=0)
    for original, recovered in zip(complete["round_records"], resumed["round_records"]):
        for key in ("source_sha256", "partner_sha256", "type_counts", "optimizer_steps_cumulative"):
            assert original[key] == recovered[key]
    assert (tmp_path / "resume" / config["run_id"] / "final.pt").exists()
    assert resumed["hardware"]["accelerator"] == "original-test-host"
    assert [segment["hardware"]["accelerator"] for segment in resumed["execution_segments"]] == [
        "original-test-host", "resumed-test-host"]


def test_electricity_round_sequence_and_partners_shared_across_batch_and_config(tmp_path):
    fitting = SyntheticData(6260, dataset="electricity")  # M=261, exceeds both batches
    validation = SyntheticData(4, dataset="electricity", split="validation")
    first = train.train_candidate(config_for("electricity"), fitting, validation, tmp_path / "b256",
                                  stop_after_batches=2)
    second = train.train_candidate(config_for("electricity", batch_size=128, configuration=1), fitting,
                                   validation, tmp_path / "b128", stop_after_batches=3)
    # stop occurs just after the last batch, before round record finalization;
    # source/partner consumption itself must be identical, including final 5.
    assert first["counters"]["source_draws"] == second["counters"]["source_draws"] == 261
    assert first["counters"]["partner_draws"] == second["counters"]["partner_draws"] == 261
    # Each run reads source then partner for each logical batch. The first has
    # four reads, the second six; compare full ordered source/partner streams.
    np.testing.assert_array_equal(np.concatenate(fitting.reads[0:4:2]), np.concatenate(fitting.reads[4:10:2]))
    np.testing.assert_array_equal(np.concatenate(fitting.reads[1:4:2]), np.concatenate(fitting.reads[5:10:2]))
    assert first["counters"]["optimizer_steps"] == 2
    assert second["counters"]["optimizer_steps"] == 3
    assert first["initial_mean_sha256"] == second["initial_mean_sha256"]
    ordinary = train.train_candidate(config_for("electricity", method="mse"), fitting, validation,
                                     tmp_path / "mse", stop_after_batches=2)
    np.testing.assert_array_equal(np.concatenate(fitting.reads[0:4:2]), np.concatenate(fitting.reads[10:12]))
    assert first["initial_mean_sha256"] == ordinary["initial_mean_sha256"]


def test_zero_pair_mask_counts_sources_without_adam_step(tmp_path):
    fitting, validation = SyntheticData(8, masked=True), SyntheticData(4, split="validation")
    config = config_for()
    initial = train.make_objective_for(config, fitting).state_dict()
    result = train.train_candidate(config, fitting, validation, tmp_path)
    assert result["counters"] == {
        "optimizer_steps": 0, "source_draws": 800, "partner_draws": 800,
        "source_valid_count": 400, "pair_valid_count": 0,
        "empty_mask_batches": 100, "update_opportunities": 100, "oom_retries": 0,
    }
    assert len(result["validation"]) == 10
    for key, tensor in initial.items():
        torch.testing.assert_close(tensor, result["objective"][key], rtol=0, atol=0)
    assert result["optimizer"]["state"] == {}


def test_nonfinite_candidate_failure_is_retained_and_not_silently_restarted(tmp_path):
    fitting, validation = SyntheticData(8), SyntheticData(4, split="validation")
    fitting.y[:] = np.nan
    config = config_for()
    result = train.train_candidate(config, fitting, validation, tmp_path)
    assert result["state"] == "numerical_failure"
    assert validation.reads == []
    assert not (tmp_path / config["run_id"] / "final.pt").exists()
    failure = json.loads((tmp_path / config["run_id"] / "status.json").read_text())
    assert failure["state"] == "numerical_failure"
    assert failure["source_draws"] == failure["partner_draws"] == 8
    assert failure["source_valid_count"] == failure["pair_valid_count"] == 8
    assert failure["update_opportunities"] == 1
    assert failure["optimizer_steps"] == 0
    # Making labels finite cannot turn this directory into a fresh experiment.
    fitting.y[:] = 0
    before = len(fitting.reads)
    second = train.train_candidate(config, fitting, validation, tmp_path)
    assert second["state"] == "numerical_failure" and len(fitting.reads) == before


def test_nonfinite_post_adam_parameter_counts_the_actual_step(tmp_path, monkeypatch):
    fitting, validation = SyntheticData(8), SyntheticData(4, split="validation")
    config = config_for()
    original_step = torch.optim.Adam.step

    def nonfinite_step(optimizer, *args, **kwargs):
        result = original_step(optimizer, *args, **kwargs)
        with torch.no_grad():
            optimizer.param_groups[0]["params"][0].reshape(-1)[0] = float("inf")
        return result

    monkeypatch.setattr(torch.optim.Adam, "step", nonfinite_step)
    result = train.train_candidate(config, fitting, validation, tmp_path)
    assert result["state"] == "numerical_failure"
    assert result["error"] == "Nonfinite parameter"
    failure = json.loads((tmp_path / config["run_id"] / "status.json").read_text())
    assert failure["optimizer_steps"] == 1
    assert failure["update_opportunities"] == 1
    assert failure["source_draws"] == failure["partner_draws"] == 8
    assert not (tmp_path / config["run_id"] / "final.pt").exists()


def test_checkpoint_data_compatibility_rejects_changed_preprocessing(tmp_path):
    fitting, validation = SyntheticData(8), SyntheticData(4, split="validation")
    config = config_for()
    train.train_candidate(config, fitting, validation, tmp_path, stop_after_batches=1)
    validation.metadata["scaler_changed"] = True
    with pytest.raises(ValueError, match="Incompatible checkpoint"):
        train.train_candidate(config, fitting, validation, tmp_path)


def test_completed_checkpoint_repairs_missing_or_stale_sidecars_without_retraining(tmp_path, monkeypatch):
    fitting, validation = SyntheticData(8), SyntheticData(4, split="validation")
    config = config_for()
    completed = train.train_candidate(config, fitting, validation, tmp_path)
    run_dir = tmp_path / config["run_id"]
    durable = torch.load(run_dir / "final.pt", weights_only=False)
    # Simulate a crash after final.pt is durable but the final sidecars are not.
    (run_dir / "status.json").write_text(json.dumps({"state": "running", "epochs_completed": 99}))
    (run_dir / "validation.json").unlink()
    (run_dir / "rounds.json").unlink()
    monkeypatch.setattr(train.platform, "processor", lambda: "different-repair-host")
    before_fitting, before_validation = len(fitting.reads), len(validation.reads)
    recovered = train.train_candidate(config, fitting, validation, tmp_path)
    assert len(fitting.reads) == before_fitting and len(validation.reads) == before_validation
    assert recovered["counters"] == completed["counters"]
    for key, tensor in completed["objective"].items():
        torch.testing.assert_close(tensor, recovered["objective"][key], rtol=0, atol=0)
    repaired = json.loads((run_dir / "status.json").read_text())
    assert repaired["state"] == "complete" and repaired["epochs_completed"] == 100
    assert repaired["hardware"] == durable["hardware"]
    assert repaired["hardware"]["accelerator"] != "different-repair-host"
    assert repaired["runtime_seconds"] == durable["runtime_seconds"]
    assert repaired["code_revision"] == durable["code_revision"]
    assert repaired["peak_accelerator_memory_bytes"] == durable["peak_accelerator_memory_bytes"]
    assert repaired["execution_segments"] == durable["execution_segments"]
    assert json.loads((run_dir / "validation.json").read_text()) == completed["validation"]
    assert len(json.loads((run_dir / "rounds.json").read_text())) == 100


def test_metric_equal_entity_weighting_and_rmse_after_aggregation():
    sums = MetricSums(equal_entity=True)
    # Customer 1 has two origins, customer 2 only one: each customer has 1/2
    # weight, and RMSE is sqrt(weighted MSE), not mean(customer RMSE).
    prediction = np.array([[1.0, 2.0], [3.0, 4.0], [10.0, 20.0]])
    target = np.zeros_like(prediction)
    mask = np.ones_like(target, dtype=bool)
    entities = np.array([1, 1, 2])
    sums.add(prediction[:1], target[:1], mask[:1], entities[:1])
    sums.add(prediction[1:], target[1:], mask[1:], entities[1:])
    records = sums.records("standardized")
    metric = {(r["horizon"], r["metric"]): r for r in records}
    expected = (np.mean(prediction[:2]**2) + np.mean(prediction[2:]**2)) / 2
    assert metric[("all", "mse")]["value"] == expected
    assert metric[("all", "rmse")]["value"] == math.sqrt(expected)
    assert metric[("1", "mse")]["value"] == (5 + 100) / 2
    assert metric[("2", "mse")]["value"] == (10 + 400) / 2
    assert metric[("all", "mse")]["valid_count"] == 6


def test_metric_masked_nan_unscaling_and_rul_sign_convention():
    sums = MetricSums()
    sums.add([[1.0, float("nan")], [3.0, 4.0]], [[0.0, float("nan")], [1.0, 1.0]],
             [[True, False], [True, True]], [0, 0])
    records = sums.records("mph")
    value = next(r for r in records if r["horizon"] == "all" and r["metric"] == "mse")
    assert value["value"] == 14 / 3 and value["valid_count"] == 3
    assert rul_score([-13.0, 10.0, 0.0], [0.0, 0.0, 0.0]) == pytest.approx(2 * np.expm1(1))
    assert math.isinf(rul_score([1e9], [0]))
    data = SyntheticData(2, split="validation")
    model = nn.Linear(2, 1)
    nn.init.zeros_(model.weight)
    nn.init.zeros_(model.bias)
    metrics, _ = evaluate(model, data, batch_size=1)
    mse = {r["units"]: r["value"] for r in metrics if r["horizon"] == "all" and r["metric"] == "mse"}
    assert mse["cycles"] == pytest.approx(mse["standardized"] * 49)
