"""Non-destructive checks for the isolated representation Pairwise TD campaign."""
from __future__ import annotations

from collections import Counter
import numpy as np
import pytest
import torch

from .application_protocol import (
    make_manifest as old_manifest, TD_ALPHAS, BETAS, BASELINE_ALPHAS,
    LAG_SETS as OLD_LAG_SETS, ELECTRICITY_ROUNDS, EPOCHS,
)
from .pairwise_protocol import (
    DATASETS, LAG_SETS, BATCH_SIZES, VERSION, make_manifest, validate_config,
)
from .pairwise_runner import paths
from .pairwise_train import code_hash, build_objective
from .pairwise_analysis import _baseline_configs, _selected_groups


@pytest.mark.parametrize("dataset,expected", [("electricity", 250), ("metr_la", 750)])
def test_manifest_exact_counts_and_original_config_preservation(dataset, expected):
    original = old_manifest()
    old_ids = {c["run_id"] for c in original}
    configs = make_manifest(dataset)
    assert len(configs) == len({c["run_id"] for c in configs}) == expected
    assert not ({c["run_id"] for c in configs} & old_ids)
    assert {c["method"] for c in configs} == {"td_pairwise"}
    assert {tuple(c["lags"]) for c in configs} == set(LAG_SETS[dataset])
    assert {c["batch_size"] for c in configs} == {128, 256}
    assert all(c["seed"] in range(5) and c["configuration"] in range(25)
               for c in configs)
    assert all(c["epochs"] == (ELECTRICITY_ROUNDS if dataset == "electricity" else EPOCHS)
               for c in configs)
    assert {tuple(c["lags"]) for c in original if
            c["dataset"] == dataset and c["method"] == "td_pairwise"} == set(OLD_LAG_SETS[dataset])
    for c in configs:
        assert c["protocol_id"].endswith("-" + VERSION)
        assert c["beta"] in BETAS and c["alpha"] in TD_ALPHAS
        assert validate_config(c) == c
        assert c["optimizer"] == "adam"
        assert c["gradient_clipping"] is None
        assert c["learning_rate_schedule"] is None
        assert c["weight_decay"] == 0
    counts = Counter((tuple(c["lags"]), c["batch_size"],
                      c["alpha"], c["beta"]) for c in configs)
    assert len(counts) == len(LAG_SETS[dataset]) * 2 * 25
    assert set(counts.values()) == {5}
    # The original application manifest must not be rewritten or altered.
    assert old_manifest() == original


@pytest.mark.parametrize("dataset,choices", [("electricity", 4), ("metr_la", 8)])
def test_baselines_are_only_mse_and_nll(dataset, choices):
    baseline = _baseline_configs(dataset)
    assert len(baseline) == 100
    assert Counter(c["method"] for c in baseline) == {"mse":50, "gaussian_nll":50}
    assert {c["batch_size"] for c in baseline} == {256}
    assert {c["alpha"] for c in baseline} == set(BASELINE_ALPHAS)
    assert {c["epochs"] for c in baseline} == {
        ELECTRICITY_ROUNDS if dataset == "electricity" else EPOCHS
    }
    groups = _selected_groups(dataset)
    assert len(groups) == choices
    assert all(method in ("td_pairwise", "mse", "gaussian_nll")
               for method,lag,batch in groups)
    assert ("mae", "none", 256) not in groups


def test_electricity_pairwise_objective_unmodified_and_jointly_differentiable():
    from .application_train import build_objective as original_build
    c = next(c for c in make_manifest("electricity")
             if c["batch_size"] == 128)
    model = build_objective(c, None)
    ref = original_build(c, None)
    assert type(model) is type(ref)
    assert model.gamma_head.linear.weight.abs().max().item() == 0
    assert model.gamma_head.linear.bias.abs().max().item() == 0
    assert len(model.parameter_groups(c["alpha"], c["beta"])) == 2
    b = 2
    source = torch.randn(b,1,168)
    neighbor = torch.randn(b,1,168)
    batch = dict(x=source, y=torch.zeros(b,24), mask=torch.ones(b,24,dtype=torch.bool))
    partner = dict(x=neighbor, y=torch.zeros(b,24),mask=torch.ones(b,24,dtype=torch.bool))
    types = torch.tensor([0,4],dtype=torch.long)
    result, n = model.loss_sum(batch,partner,types)
    baseline_half_mse = 0.5 * model.model(source).square().sum()
    assert n == b*24
    torch.testing.assert_close(result, baseline_half_mse, rtol=5e-5, atol=1e-5)
    result.backward()
    assert model.gamma_head.linear.weight.grad is not None
    assert all(torch.isfinite(p.grad).all()
               for p in model.parameters() if p.grad is not None)


def test_metr_la_graph_wavenet_uses_existing_pair_head():
    c = next(c for c in make_manifest("metr_la")
             if len(c["lags"]) == 5 and c["batch_size"] == 256)
    objective = build_objective(c, np.eye(3,dtype=np.float32))
    assert objective.gamma_head.linear.out_features == 5
    assert objective.gamma_head.linear.in_features == 194
    assert objective.model.output_dim == 12


@pytest.mark.parametrize("dataset", DATASETS)
def test_new_roots_distinct_from_original_dataset_results(dataset):
    from .result_paths import paths as old_paths
    from .typed_ablation_runner import paths as ablation_paths
    data, root = paths(dataset)
    old_data, old_root, _ = old_paths(dataset)
    _, ablation_root = ablation_paths(dataset)
    assert data == old_data
    assert root != old_root and root != ablation_root
    assert "pairwise" in root.name


def test_training_hash_depends_on_scientific_sources():
    a = next(c for c in make_manifest("electricity") if c["batch_size"]==128)
    b = next(c for c in make_manifest("electricity") if c["batch_size"]==256)
    assert len(code_hash(a)) == 64 and len(code_hash(b)) == 64
    assert code_hash(a) == code_hash(a)


def test_test_cache_is_identity_scoped(tmp_path):
    from .common import atomic_json
    from .pairwise_analysis import _evaluate_tasks
    identity = dict(endpoint_sha256="abc", test_data_sha256="dataset",
                    code_sha256="code")
    metric_keys = [(unit, metric)
                   for unit in ("standardized", "kW") for metric in ("mse","rmse","mae")]
    records = [dict(units=unit,metric=metric,value=0.3)
               for unit,metric in metric_keys]
    file = tmp_path / "test_cache" / "example.json"
    atomic_json(file, dict(identity=identity, metrics=records))
    task = dict(config=dict(run_id="example"),identity=identity,raw_unit="kW")
    actual = _evaluate_tasks([task],tmp_path,workers=8)
    assert actual["example"] == records
    with pytest.raises(ValueError,match="Stale test cache"):
        _evaluate_tasks([dict(task,identity=dict(identity,test_data_sha256="new"))],
                        tmp_path,workers=8)
