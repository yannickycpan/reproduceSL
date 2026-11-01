"""All-horizon validation reductions preserve both ordinary error metrics."""
import numpy as np
import pytest
import torch

from fulltd.metrics import MetricSums, validation_mse, validation_scores


class ScoreData:
    split = "val"

    def __init__(self, name, masked=False):
        self.dataset = name
        shape = {"electricity": (24,), "fd001": (1,), "metr_la": (3, 12)}[name]
        rng = np.random.default_rng(12345)
        self.predictions = rng.normal(size=(25, *shape)).astype(np.float32)
        self.predictions *= np.geomspace(1e-6, 1e6, 25).astype(np.float32).reshape(25, *(1 for _ in shape))
        self.targets = rng.normal(size=self.predictions.shape).astype(np.float32)
        self.mask = rng.uniform(size=self.predictions.shape) > .3 if masked else np.ones_like(self.predictions, bool)
        self.entity_ids = np.array([7, 2, 7, 2, 7, 3, 7, 7, 2, 3, 7, 7, 7, 7, 3, 7, 7, 7, 7, 7, 7, 7, 7, 2, 7])
        self.reads = []

    def __len__(self):
        return len(self.targets)

    def get_batch(self, ids):
        self.reads.append(np.asarray(ids).copy())
        return dict(x=self.predictions[ids], y=self.targets[ids], mask=self.mask[ids])


def reference_scores(data, batch_size):
    accumulator = MetricSums(equal_entity=data.dataset in ("electricity", "fd001"))
    for start in range(0, len(data), batch_size):
        ids = np.arange(start, min(start + batch_size, len(data)))
        accumulator.add(data.predictions[ids], data.targets[ids], data.mask[ids], data.entity_ids[ids])
    return {r["metric"]: r["value"] for r in accumulator.records("standardized")
            if r["horizon"] == "all" and r["metric"] in ("mse", "mae")}


@pytest.mark.parametrize("name", ["electricity", "fd001", "metr_la"])
@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("batch_size", [1, 3, 8, 64])
def test_scores_bitwise_match_previous_all_horizon_reductions(name, masked, batch_size):
    data = ScoreData(name, masked)
    expected = reference_scores(data, batch_size)
    actual = validation_scores(torch.nn.Identity(), data, batch_size)
    assert set(actual) == {"mse", "mae"}
    assert {key: value.hex() for key, value in actual.items()} == {key: value.hex() for key, value in expected.items()}
    assert validation_mse(torch.nn.Identity(), data, batch_size) == actual["mse"]


def test_validation_does_not_compute_per_horizon_records_or_rmse(monkeypatch):
    data = ScoreData("electricity")
    expected = reference_scores(data, 3)
    def forbidden(*args, **kwargs):
        raise AssertionError("Validation must not use per-horizon/RMSE reductions")
    monkeypatch.setattr(MetricSums, "add", forbidden)
    monkeypatch.setattr(MetricSums, "records", forbidden)
    monkeypatch.setattr(np, "sqrt", forbidden)
    assert validation_scores(torch.nn.Identity(), data, 3) == expected


def test_masks_ignore_nonfinite_unobserved_errors_and_empty_batches():
    data = ScoreData("metr_la", masked=True)
    data.mask[:3] = False
    data.predictions[~data.mask] = np.inf
    expected = reference_scores(data, 3)
    assert validation_scores(torch.nn.Identity(), data, 3) == expected
    data.mask[:] = False
    with pytest.raises(ValueError, match="No observed targets"):
        validation_scores(torch.nn.Identity(), data, 3)


def test_validation_stop_returns_no_partial_scores_and_restarts_at_first_batch():
    data = ScoreData("fd001")
    def stop_requested():
        return len(data.reads) == 2
    assert validation_scores(torch.nn.Identity(), data, 3, stop_requested=stop_requested) is None
    assert len(data.reads) == 2
    data.reads.clear()
    assert validation_scores(torch.nn.Identity(), data, 3) == reference_scores(data, 3)
    assert np.array_equal(data.reads[0], np.arange(3))
    assert validation_mse(torch.nn.Identity(), data, 3, stop_requested=lambda: True) is None


@pytest.mark.parametrize("nonfinite", [np.inf, -np.inf, np.nan])
def test_observed_nonfinite_prediction_remains_a_numerical_failure(nonfinite):
    data = ScoreData("fd001")
    data.predictions[0, 0] = nonfinite
    with pytest.raises(FloatingPointError, match="Nonfinite point predictions"):
        validation_scores(torch.nn.Identity(), data, 3)


def test_finite_predictions_with_overflowing_mse_are_rejected():
    data = ScoreData("fd001")
    class ExtremePrediction(torch.nn.Module):
        def forward(self, x):
            return torch.full_like(x, 1e200, dtype=torch.float64)
    with pytest.raises(FloatingPointError, match="Nonfinite validation MSE/MAE"):
        validation_scores(ExtremePrediction(), data, 3)


def test_equal_entity_weighting_retains_both_mse_and_mae():
    data = ScoreData("fd001")
    data.targets[:] = 0
    data.predictions[:] = data.entity_ids[:, None]
    scores = validation_scores(torch.nn.Identity(), data, 3)
    assert scores["mae"] == pytest.approx(np.mean([2., 3., 7.]))
    assert scores["mse"] == pytest.approx(np.mean([4., 9., 49.]))
    assert scores["mae"] != pytest.approx(np.abs(data.predictions).mean())
    assert scores["mse"] != pytest.approx(np.square(data.predictions).mean())
