"""Regression checks for endpoint caches and durable analysis evidence."""
import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import numpy as np
import torch

from fulltd import analysis
from fulltd import evaluate as endpoint
from fulltd.protocol import make_manifest, digest
from fulltd.train import data_identity


def datasets():
    return tuple(SimpleNamespace(split=split, metadata={"fingerprint": "fixed-data"})
                 for split in ("train", "val", "test"))


def config_and_group():
    config = next(c for c in make_manifest() if c["dataset"] == "fd001")
    group = dict(group_id="fd001/tcn/mse/256", dataset="fd001", backbone="tcn",
                 method="mse", batch_size=256, configuration=0, alpha=.0001, beta=None,
                 state="selected", score=.25,
                 run_ids=[config["run_id"].removesuffix("s0") + f"s{s}" for s in range(5)])
    return config, group


def test_endpoint_cache_is_independent_of_other_dataset_selections(monkeypatch):
    config, group = config_and_group()
    monkeypatch.setattr(endpoint, "evaluation_code_hash", lambda: "fixed-evaluation")
    train, val, test = datasets()
    standalone = dict(dataset_scope="fd001", groups=[group])
    combined = dict(dataset_scope="all", groups=[dict(dataset="electricity"), group])
    assert digest(standalone) != digest(combined)
    first = endpoint.endpoint_cache_identity(config, standalone["groups"][0], "checkpoint", train, val, test)
    second = endpoint.endpoint_cache_identity(config, combined["groups"][1], "checkpoint", train, val, test)
    assert first == second
    for changed in (dict(group, score=.3), dict(group, configuration=1), dict(group, run_ids=["changed"])):
        assert first != endpoint.endpoint_cache_identity(config, changed, "checkpoint", train, val, test)
    assert first != endpoint.endpoint_cache_identity(config, group, "other-checkpoint", train, val, test)
    test.metadata = {"fingerprint": "changed-data"}
    assert first != endpoint.endpoint_cache_identity(config, group, "checkpoint", train, val, test)


@pytest.mark.parametrize("kind", ["json", "csv"])
def test_interrupted_sidecar_publication_preserves_previous_evidence(tmp_path, monkeypatch, kind):
    path = tmp_path / f"records.{kind}"
    original = '{"old": true}\n' if kind == "json" else "value\r\nold\r\n"
    path.write_text(original, newline="")
    before = path.read_bytes()
    def interrupted_replace(source, destination):
        assert Path(source).read_bytes()
        assert Path(destination).read_bytes() == before
        raise OSError("simulated termination before atomic publication")
    monkeypatch.setattr(analysis.os, "replace", interrupted_replace)
    with pytest.raises(OSError, match="simulated"):
        if kind == "json":
            analysis.write_json(path, {"new": True})
        else:
            analysis.write_csv(path, [{"value": "new"}])
    assert path.read_bytes() == before
    assert list(tmp_path.glob("*.tmp")) == []


def test_failed_run_preserves_partial_validation_and_infinite_score(tmp_path):
    config, _ = config_and_group()
    directory = tmp_path / config["run_id"]
    directory.mkdir()
    analysis.write_json(directory / "status.json", dict(config=config, protocol_id=config["protocol_id"],
                                                       state="numerical_failure", epochs_completed=20))
    records = [dict(epoch=10, mse=1.5), dict(epoch=20, mse=1.2)]
    analysis.write_json(directory / "validation.json", records)
    score, retained, status = analysis._validate_candidate(config, tmp_path)
    assert score == float("inf")
    assert retained == records
    analysis.write_json(directory / "validation.json", [dict(epoch=20, mse=1.2)])
    with pytest.raises(ValueError, match="partial validation"):
        analysis._validate_candidate(config, tmp_path)


def test_final_checkpoint_must_match_selected_sidecars(monkeypatch):
    config, _ = config_and_group()
    train, val, _ = datasets()
    monkeypatch.setattr(endpoint, "code_hash", lambda: "fixed-code")
    identity = dict(code_sha256="fixed-code", data_sha256=data_identity(train),
                    validation_sha256=data_identity(val))
    checkpoint = dict(config=config, next_epoch=101, cursor=0, identity=identity,
                      counters={"optimizer_steps": 100}, initial_mean_sha256="initial")
    status = dict(config=config, state="complete", epochs_completed=100, identity=identity,
                  optimizer_steps=100, initial_mean_sha256="initial")
    endpoint.validate_final_checkpoint(checkpoint, config, status, train, val)
    for changed in (dict(status, optimizer_steps=99), dict(status, initial_mean_sha256="other"),
                    dict(status, state="running")):
        with pytest.raises(ValueError):
            endpoint.validate_final_checkpoint(checkpoint, config, changed, train, val)


class EndpointData:
    def __init__(self, name="electricity", split="test", nonfinite=False):
        self.dataset, self.split = name, split
        self.metadata = {"fingerprint": "endpoint-fixture"}
        horizon = 24 if name == "electricity" else 1
        self.x = np.ones((3, horizon), dtype=np.float32)
        if nonfinite:
            self.x[1, -1] = float("inf")
        self.entity_ids = np.arange(3)

    def __len__(self):
        return len(self.x)

    def get_batch(self, ids):
        return dict(x=self.x[ids], y=np.zeros_like(self.x[ids]), mask=np.ones_like(self.x[ids], dtype=bool))

    def inverse(self, values, ids):
        return values


def test_endpoint_nonfinite_horizon_preserves_other_horizons_and_all_seeds():
    failures = []
    rows, _ = endpoint.evaluate_endpoint(torch.nn.Identity(), EndpointData(nonfinite=True), 2, "cpu",
                                        failures, dict(run_id="fixture", split="test", variant="trained"))
    find = lambda h, m: next(r for r in rows if r["units"] == "standardized" and r["horizon"] == h and r["metric"] == m)
    assert find("1", "mse")["value"] == 1
    assert math.isinf(find("24", "mse")["value"])
    assert math.isinf(find("all", "mse")["value"])
    assert len(failures) == 1
    assert all(m["horizon"] in ("24", "all") for m in failures[0]["affected_metrics"])
    records = [dict(dataset="electricity", backbone="tcn", method="td_full", batch_size=256,
                    seed=s, split="test", variant="trained", **dict(find("24", "mse"), value=1 if s < 4 else math.inf))
               for s in range(5)]
    _, summaries = analysis.aggregate_metrics(records)
    assert summaries[0]["n"] == 5 and summaries[0]["finite_seeds"] == 4
    assert math.isinf(summaries[0]["mean"])


def test_endpoint_rul_overflow_keeps_finite_error_metrics():
    data = EndpointData("fd001")
    data.x[:] = 100000
    failures = []
    rows, engines = endpoint.evaluate_endpoint(torch.nn.Identity(), data, 2, "cpu", failures,
                                              dict(run_id="fixture", split="test", variant="trained"))
    assert all(math.isfinite(r["value"]) for r in rows if not r["metric"].startswith("rul_score"))
    assert all(math.isinf(r["value"]) for r in rows if r["metric"].startswith("rul_score"))
    assert all(m["metric"].startswith("rul_score") for m in failures[0]["affected_metrics"])
    assert len(engines) == 3


def test_endpoint_floating_point_exception_has_explicit_failure_rows(monkeypatch):
    def failed(*args, **kwargs):
        assert kwargs["allow_nonfinite"] is True
        raise FloatingPointError("fixture nonfinite predictions")
    monkeypatch.setattr(endpoint, "evaluate", failed)
    failures = []
    rows, engines = endpoint.evaluate_endpoint(None, EndpointData("fd001"), 2, "cpu", failures,
                                              dict(run_id="fixture", split="test", variant="trained"))
    assert engines == [] and len(rows) == 14
    assert all(math.isinf(row["value"]) and row["metric_status"] == "nonfinite_prediction" for row in rows)
    assert len(failures) == 1


def test_failed_refit_records_all_five_seeds_and_completes_endpoint_export(tmp_path, monkeypatch):
    from fulltd import refit
    configs = [r for r in make_manifest() if r["dataset"] == "fd001" and r["method"] == "td_full"
               and r["batch_size"] == 256 and r["configuration"] == 0]
    group = dict(dataset="fd001", state="selected", run_ids=[r["run_id"] for r in configs])
    selected = dict(dataset_scope="fd001", protocol_id=configs[0]["protocol_id"], groups=[group])
    results, tables = tmp_path / "runs", tmp_path / "tables"
    analysis.write_json(tables / "selections.json", selected)
    checkpoints = {}
    for config in configs:
        path = results / config["run_id"]
        analysis.write_json(path / "status.json", {})
        analysis.write_json(path / "validation.json", [])
        analysis.write_json(path / "rounds.json", [])
        (path / "final.pt").write_bytes(b"fixture weights")
        checkpoints[config["run_id"]] = dict(config=config, validation=[], round_records=[], objective={})
    class Objective:
        model, theta = torch.nn.Identity(), torch.zeros(2)
        def to(self, device):
            return self
        def load_state_dict(self, state):
            pass
    monkeypatch.setattr(endpoint, "load_manifest", lambda path: configs)
    monkeypatch.setattr(endpoint, "select_candidates", lambda *args, **kwargs: selected)
    monkeypatch.setattr(endpoint, "seed_runtime", lambda seed: None)
    monkeypatch.setattr(endpoint, "validate_final_checkpoint", lambda *args: None)
    monkeypatch.setattr(endpoint.torch, "load", lambda path, **kwargs: checkpoints[path.parent.name])
    monkeypatch.setattr(endpoint, "load_dataset", lambda name, root, split: EndpointData(name, split))
    monkeypatch.setattr(endpoint, "training_universe", lambda data, lags: (data, None, None))
    monkeypatch.setattr(endpoint, "make_objective_for", lambda *args: Objective())
    def failed_refit(*args, **kwargs):
        raise np.linalg.LinAlgError("fixture singular solve failure")
    monkeypatch.setattr(refit, "refit_readouts", failed_refit)
    rows = endpoint.evaluate_selected("manifest", results, tables, "data", "cpu", 2)
    _, summaries = analysis.aggregate_metrics(rows)
    refits = [r for r in summaries if r["variant"].startswith("refit_")]
    assert refits and all(r["n"] == 5 and r["finite_seeds"] == 0 and math.isinf(r["mean"]) for r in refits)
    assert all(math.isfinite(r["mean"]) for r in summaries if r["variant"] == "trained")
    assert (tables / "metrics.csv").is_file()
    for config in configs:
        failure = json.loads((results / config["run_id"] / "refits" / "failures.json").read_text())
        assert failure["stage"] == "readout_solve"
        cached = json.loads((results / config["run_id"] / "endpoint_metrics.json").read_text())
        assert cached["numerical_failures"]
        assert any(r["value"] == "Infinity" and r["metric_status"] == "readout_numerical_failure" for r in cached["rows"])


@pytest.mark.parametrize("values", [
    [1., 2., 3., 4., 5.],
    [1., 2., 3., 4., math.inf],
    [1., 2., 3., 4., -math.inf],
    [1., 2., 3., 4., math.nan],
    [1., 2., 3., -math.inf, math.inf],
])
@pytest.mark.parametrize("field", ["mean", "sample_sd", "standard_error", "finite_seeds"])
def test_independent_verifier_rejects_corrupted_finite_and_nonfinite_summaries(tmp_path, values, field):
    records = [dict(run_id=f"fixture{s}", dataset="fd001", backbone="tcn", method="mse", batch_size=256,
                    seed=s, split="test", units="cycles", horizon="all", metric="mse", value=value)
               for s, value in enumerate(values)]
    per_seed, summaries = analysis.aggregate_metrics(records)
    analysis.write_csv(tmp_path / "per_seed_metrics.csv", per_seed)
    analysis.write_csv(tmp_path / "summary_metrics.csv", summaries)
    assert analysis.verify_numerical_tables(tmp_path) == 1
    summaries[0][field] = -1 if field == "finite_seeds" else 0.
    analysis.write_csv(tmp_path / "summary_metrics.csv", summaries)
    with pytest.raises(ValueError, match="finite-seed count|Independent recomputation mismatch"):
        analysis.verify_numerical_tables(tmp_path)


def test_independent_verifier_distinguishes_nan_and_infinity(tmp_path):
    records = [dict(run_id=f"fixture{s}", dataset="fd001", backbone="tcn", method="mse", batch_size=256,
                    seed=s, split="test", units="cycles", horizon="all", metric="mse", value=value)
               for s, value in enumerate([1., 2., 3., 4., math.inf])]
    per_seed, summaries = analysis.aggregate_metrics(records)
    analysis.write_csv(tmp_path / "per_seed_metrics.csv", per_seed)
    for field, replacement in (("mean", math.nan), ("mean", -math.inf),
                               ("sample_sd", math.inf), ("standard_error", math.inf)):
        altered = copy.deepcopy(summaries)
        altered[0][field] = replacement
        analysis.write_csv(tmp_path / "summary_metrics.csv", altered)
        with pytest.raises(ValueError, match="Independent recomputation mismatch"):
            analysis.verify_numerical_tables(tmp_path)
