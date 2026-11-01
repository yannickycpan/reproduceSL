"""Stratified validation fairness, weighting, provenance and endpoint checks."""
import copy
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fulltd import analysis
from fulltd.evaluate import endpoint_cache_identity, evaluate_endpoint
from fulltd.metrics import validation_mse
from fulltd.protocol import EVALUATIONS, PROTOCOL_ID, digest, make_manifest
from fulltd.validation import (ELECTRICITY_POLICY, FULL_POLICY, ValidationView,
                               indices_hash, validation_view, validate_sampling_record)


class ValidationFixture:
    dataset, split = "electricity", "val"
    metadata = {"fixture": True, "validation_universe": "fixed"}

    def __init__(self, counts=(1, 6, 11), shuffle=True):
        entities = np.repeat(np.arange(len(counts)), counts)
        origins = np.concatenate([np.arange(n) for n in counts])
        order = np.random.default_rng(19).permutation(len(entities)) if shuffle else np.arange(len(entities))
        self.entity_ids, self.origins = entities[order], origins[order]
        self.batch_reads = []

    def __len__(self):
        return len(self.entity_ids)

    def get_batch(self, ids):
        self.batch_reads.append(np.asarray(ids).copy())
        x = np.repeat(((self.origins[ids] + 1) * (self.entity_ids[ids] + 1))[:, None], 24, axis=1).astype(np.float32)
        return dict(x=x, y=np.zeros_like(x), mask=np.ones_like(x, dtype=bool))

    def inverse(self, values, ids):
        return np.asarray(values) + np.asarray(ids)[:, None] * 1000

    def persistence(self, ids):
        return np.repeat(np.asarray(ids)[:, None], 24, axis=1)


def test_stratified_sample_counts_lazy_mapping_and_round100():
    data = ValidationFixture()
    view, record = validation_view(data, 100, "fixture")
    assert data.batch_reads == []
    assert isinstance(view, ValidationView) and view.parent is data
    assert len(view) == 1 + 2 + 3
    assert record["policy"] == ELECTRICITY_POLICY
    assert record["epoch"] == 100 and record["population_size"] == 18 and record["sampled_size"] == 6
    assert [r["sampled_size"] for r in record["per_customer"]] == [1, 2, 3]
    assert len(set(view.indices)) == len(view)
    assert np.all(view.indices[:-1] < view.indices[1:])
    assert record["indices_sha256"] == indices_hash(view.indices)
    validate_sampling_record("electricity", 100, record, "fixture")
    ids = np.array([0, len(view) - 1])
    assert np.array_equal(view.entity_ids[ids], data.entity_ids[view.indices[ids]])
    assert np.array_equal(view.origins[ids], data.origins[view.indices[ids]])
    assert np.array_equal(view.get_batch(ids)["x"], data.get_batch(view.indices[ids])["x"])
    assert np.array_equal(view.inverse(np.zeros((2, 24)), ids), data.inverse(np.zeros((2, 24)), view.indices[ids]))
    assert np.array_equal(view.persistence(ids), data.persistence(view.indices[ids]))


def test_samples_independent_per_candidate_reproducible_without_consuming_global_rng():
    data = ValidationFixture((60, 101, 122))
    random.seed(99)
    np.random.seed(99)
    torch.manual_seed(99)
    python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    hashes = set()
    for config in make_manifest():
        if config["dataset"] == "electricity":
            view, record = validation_view(data, 80, config["run_id"])
            repeated, repeated_record = validation_view(data, 80, config["run_id"])
            assert record == repeated_record
            assert np.array_equal(view.indices, repeated.indices)
            hashes.add(record["indices_sha256"])
    assert len(hashes) == sum(c["dataset"] == "electricity" for c in make_manifest())
    assert random.getstate() == python_state
    after = np.random.get_state()
    assert after[0] == numpy_state[0] and np.array_equal(after[1], numpy_state[1]) and after[2:] == numpy_state[2:]
    assert torch.equal(torch.get_rng_state(), torch_state)
    hashes = [validation_view(data, epoch, "fixture")[1]["indices_sha256"] for epoch in EVALUATIONS]
    assert len(set(hashes)) == len(EVALUATIONS)
    assert validation_view(data, 100, "fixture")[1] == validation_view(data, 100, "fixture")[1]
    with pytest.raises(ValueError, match="run_id"):
        validation_view(data, 100)


def test_uniform_within_customer_sampling_and_equal_customer_mse():
    # Many independent customer streams exercise inclusion of every origin,
    # including neighbors that a fixed stride would permanently omit.
    data = ValidationFixture((10,) * 1000, shuffle=False)
    view, _ = validation_view(data, 10, "fixture")
    frequencies = np.bincount(view.origins, minlength=10)
    assert len(view) == 2000
    assert np.all(np.abs(frequencies - 200) < 50)
    small = ValidationFixture((4, 12, 36))
    sampled, _ = validation_view(small, 90, "fixture")
    error = (sampled.origins + 1) * (sampled.entity_ids + 1)
    expected = np.mean([np.mean(error[sampled.entity_ids == customer] ** 2) for customer in np.unique(sampled.entity_ids)])
    assert validation_mse(torch.nn.Identity(), sampled, 2) == pytest.approx(expected)
    assert expected != pytest.approx(np.mean(error ** 2))


@pytest.mark.parametrize("name", ["metr_la", "fd001"])
def test_other_datasets_keep_complete_validation(name):
    data = ValidationFixture()
    data.dataset = name
    view, record = validation_view(data, 100)
    assert view is data
    assert record["policy"] == FULL_POLICY
    assert record["sampled_size"] == record["population_size"] == len(data)
    validate_sampling_record(name, 100, record)


def test_validation_stop_does_not_return_partial_metric_or_save_progress():
    data = ValidationFixture((20, 30))
    view, _ = validation_view(data, 100, "fixture")
    calls = 0
    def stop_requested():
        nonlocal calls
        calls += 1
        return calls == 2
    assert validation_mse(torch.nn.Identity(), view, 2, stop_requested=stop_requested) is None
    assert len(data.batch_reads) == 1
    assert np.array_equal(data.batch_reads[0], view.indices[:2])
    data.batch_reads.clear()
    result = validation_mse(torch.nn.Identity(), view, 2)
    assert np.array_equal(data.batch_reads[0], view.indices[:2])
    assert np.isfinite(result)


def _audit_fixture(directory, configs, data):
    for config in configs:
        dest = directory / config["run_id"]
        analysis.write_json(dest / "status.json", dict(
            state="numerical_failure", identity=dict(protocol_id=PROTOCOL_ID, config_sha256=digest(config),
                code_sha256="same-code", data_sha256="same-data", validation_sha256="same-validation"),
            initial_mean_sha256=f"{config['backbone']}-{config['seed']}", n_sources=9))
        record = validation_view(data, 10, config["run_id"])[1]
        analysis.write_json(dest / "validation.json", [dict(epoch=10, mse=1, validation_subset=record)])


def test_campaign_audit_checks_run_stream_identity_and_exact_counts(tmp_path):
    manifest = make_manifest()
    configs = [next(c for c in manifest if c["dataset"] == "electricity" and c["method"] == "mse"),
               next(c for c in manifest if c["dataset"] == "electricity" and c["backbone"] == "patchtst"
                    and c["method"] == "td_full" and c["batch_size"] == 128 and c["seed"] == 4)]
    data = ValidationFixture()
    _, record = validation_view(data, 10, configs[0]["run_id"])
    _audit_fixture(tmp_path, configs, data)
    assert analysis.audit_campaign(configs, tmp_path, "electricity")["runs"] == 2
    altered = copy.deepcopy(record)
    # Another run's otherwise valid sample may not be silently reused.
    analysis.write_json(tmp_path / configs[-1]["run_id"] / "validation.json", [dict(epoch=10, mse=1, validation_subset=altered)])
    with pytest.raises(ValueError, match="run_id"):
        analysis.audit_campaign(configs, tmp_path, "electricity")
    altered = copy.deepcopy(record)
    altered["per_customer"][1]["sampled_size"] += 1
    with pytest.raises(ValueError, match="ceil"):
        validate_sampling_record("electricity", 10, altered, configs[0]["run_id"])
    altered = copy.deepcopy(record)
    altered["sampled_size"] += 1
    with pytest.raises(ValueError, match="reconcile"):
        validate_sampling_record("electricity", 10, altered, configs[0]["run_id"])


def test_endpoint_cache_and_validation_rows_identify_round100_subset(monkeypatch):
    import fulltd.evaluate as endpoint
    data = ValidationFixture((20, 30))
    view, metadata = validation_view(data, 100, "fixture")
    metrics, _ = evaluate_endpoint(torch.nn.Identity(), view, 2, "cpu", [], {"split": "validation"})
    assert all(r["validation_epoch"] == 100 and r["validation_indices_sha256"] == metadata["indices_sha256"] for r in metrics)
    assert all(r["valid_count"] == len(view) * (24 if r["horizon"] == "all" else 1) for r in metrics)
    config = make_manifest()[0]
    train, val, test = (SimpleNamespace(split=s, metadata={"fixture": "fixed"}) for s in ("train", "val", "test"))
    monkeypatch.setattr(endpoint, "evaluation_code_hash", lambda: "evaluation")
    first = endpoint_cache_identity(config, {}, "checkpoint", train, val, test, metadata)
    changed = validation_view(data, 90, "fixture")[1]
    assert first != endpoint_cache_identity(config, {}, "checkpoint", train, val, test, changed)
    assert first == endpoint_cache_identity(config, {}, "checkpoint", train, val, test, validation_view(data, 100, "fixture")[1])


def test_selected_endpoint_and_refits_use_same_run100_sample_with_full_train_and_test(tmp_path, monkeypatch):
    from fulltd import evaluate as endpoint, refit
    configs = [c for c in make_manifest() if c["dataset"] == "electricity" and c["backbone"] == "tcn"
               and c["method"] == "td_full" and c["batch_size"] == 256 and c["configuration"] == 0][:2]
    selected = dict(dataset_scope="electricity", protocol_id=PROTOCOL_ID,
                    groups=[dict(dataset="electricity", state="selected", run_ids=[c["run_id"] for c in configs])])
    root, tables = tmp_path / "runs", tmp_path / "tables"
    analysis.write_json(tables / "selections.json", selected)
    full_validation = ValidationFixture((20, 30))
    checkpoints, opened, evaluated = {}, [], []
    for config in configs:
        directory = root / config["run_id"]
        validations = [dict(epoch=e, mse=1, validation_subset=validation_view(full_validation, e, config["run_id"])[1])
                       for e in EVALUATIONS]
        analysis.write_json(directory / "status.json", {})
        analysis.write_json(directory / "validation.json", validations)
        analysis.write_json(directory / "rounds.json", [])
        (directory / "final.pt").write_bytes(b"fixture final weights")
        checkpoints[config["run_id"]] = dict(config=config, validation=validations, round_records=[], objective={})
    class Objective:
        def __init__(self, config):
            self.model = torch.nn.Identity()
            self.model.run_id, self.model.variant = config["run_id"], "trained"
            self.theta = torch.zeros(2)
        def to(self, device):
            return self
        def load_state_dict(self, state):
            pass
    def load_data(name, path, split):
        opened.append(split)
        data = ValidationFixture((20, 30))
        data.split = split
        return data
    def refitted(base, fit):
        model = torch.nn.Identity()
        model.run_id, model.variant = base.run_id, fit
        return model
    def measure(model, data, *args, **kwargs):
        evaluated.append(dict(run_id="electricity_persistence" if model is None else model.run_id,
                              variant="persistence" if model is None else model.variant, split=data.split,
                              indices=getattr(data, "indices", np.arange(len(data))).copy()))
        return [dict(units="standardized", horizon="all", metric="mse", value=1, valid_count=len(data) * 24)], []
    monkeypatch.setattr(endpoint, "load_manifest", lambda path: configs)
    monkeypatch.setattr(endpoint, "select_candidates", lambda *args, **kwargs: selected)
    monkeypatch.setattr(endpoint, "seed_runtime", lambda seed: None)
    monkeypatch.setattr(endpoint, "validate_final_checkpoint", lambda *args: None)
    monkeypatch.setattr(endpoint.torch, "load", lambda path, **kwargs: checkpoints[path.parent.name])
    monkeypatch.setattr(endpoint, "load_dataset", load_data)
    monkeypatch.setattr(endpoint, "training_universe", lambda data, lags: (data, None, None))
    monkeypatch.setattr(endpoint, "make_objective_for", lambda config, data: Objective(config))
    monkeypatch.setattr(endpoint, "evaluate", measure)
    monkeypatch.setattr(refit, "refit_readouts", lambda *args: {"ols": "ols", "td": "td"})
    monkeypatch.setattr(refit, "save_readouts", lambda *args: None)
    monkeypatch.setattr(refit, "RefittedPredictor", refitted)
    rows = endpoint.evaluate_selected("manifest", root, tables, "data", "cpu", 2)
    for config in configs:
        expected = validation_view(full_validation, 100, config["run_id"])[0].indices
        paired = [r for r in evaluated if r["run_id"] == config["run_id"] and r["split"] == "val"]
        assert {r["variant"] for r in paired} == {"trained", "ols", "td"}
        assert all(np.array_equal(r["indices"], expected) for r in paired)
    assert all(len(r["indices"]) == 50 for r in evaluated if r["split"] in ("train", "test"))
    assert all(r["validation_epoch"] == 100 for r in rows if r["split"] == "validation")
    persisted = analysis.read_json(tables / "evaluation_manifest.json")["endpoint_validation_subsets"]
    assert set(persisted) == {*(c["run_id"] for c in configs), "electricity_persistence"}
    # A self-consistent checkpoint/sidecar forgery of the subset still fails
    # independent regeneration, before any held-out targets are reopened.
    bad_run = configs[0]["run_id"]
    checkpoints[bad_run]["validation"][0]["validation_subset"]["indices_sha256"] = "0" * 64
    analysis.write_json(root / bad_run / "validation.json", checkpoints[bad_run]["validation"])
    opened.clear()
    with pytest.raises(ValueError, match="does not reproduce"):
        endpoint.evaluate_selected("manifest", root, tables, "data", "cpu", 2)
    assert "test" not in opened
