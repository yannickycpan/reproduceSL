"""Mechanism, frozen-protocol and old-result-isolation tests (no dataset downloads)."""
from __future__ import annotations

from copy import deepcopy
import math

import pytest
import torch
from torch import nn

from dscpuscripts.application_objectives import TypedGammaTD
from dscpuscripts.typed_ablation_objectives import (
    ApplicationSemiTypedTD, ApplicationNonnegativeTypedTD)
from dscpuscripts.typed_ablation_protocol import make_manifest, validate_config


class ToyMean(nn.Module):
    feature_dim = 3
    output_dim = 2

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 2, bias=False, dtype=torch.float64)
        with torch.no_grad():
            self.linear.weight.copy_(torch.tensor([[.4, -.2, .6], [-.3, .8, .1]], dtype=torch.float64))

    def forward(self, x):
        return self.linear(x)


def pair_data():
    x = torch.tensor([[.4, -.5, .2], [.1, .3, -.7], [-.3, .8, .9]], dtype=torch.float64)
    y = torch.tensor([[1., -.2], [.4, .8], [.2, -.5]], dtype=torch.float64)
    z = torch.tensor([[-.1, .7, .3], [.5, -.2, .6], [.8, .1, -.3]], dtype=torch.float64)
    t = torch.tensor([[.2, -.7], [-.5, .1], [.9, .2]], dtype=torch.float64)
    return (dict(x=x, y=y, mask=torch.ones_like(y, dtype=torch.bool)),
            dict(x=z, y=t, mask=torch.ones_like(t, dtype=torch.bool)),
            torch.tensor([0, 1, 0]))


def grads(model):
    return [p.grad.detach().clone() for p in model.parameters()]


def test_semi_removes_only_neighbor_predictor_gradient():
    src, nbr, types = pair_data()
    full = TypedGammaTD(ToyMean()).double()
    semi = ApplicationSemiTypedTD(deepcopy(full.model)).double()
    with torch.no_grad():
        full.operator.copy_(torch.tensor([.4, -.3], dtype=torch.float64))
        semi.operator.copy_(full.operator)
    lf, n = full.loss_sum(src, nbr, types)
    ls, n2 = semi.loss_sum(src, nbr, types)
    assert n == n2 == 6
    torch.testing.assert_close(lf, ls, rtol=0, atol=0)
    (lf / n).backward()
    (ls / n).backward()
    torch.testing.assert_close(full.operator.grad, semi.operator.grad, rtol=1e-13, atol=1e-13)
    assert any(not torch.allclose(a, b) for a, b in zip(grads(full.model), grads(semi.model)))
    # The reference source-only gradient comes from the same TD residual,
    # not from a separate objective or stopped operator gradient.
    reference = ToyMean()
    gamma = semi.operator[types]
    with torch.no_grad():
        residual_j = nbr["y"] - reference(nbr["x"])
    residual_i = src["y"] - reference(src["x"])
    delta = residual_i - gamma[:, None] * residual_j
    (delta.square().sum() / (2 * n)).backward()
    for a, b in zip(grads(semi.model), grads(reference)):
        torch.testing.assert_close(a, b, rtol=1e-13, atol=1e-13)


def test_nonnegative_full_gradient_and_trainable_zero_raw_init():
    src, nbr, types = pair_data()
    model = ApplicationNonnegativeTypedTD(ToyMean()).double()
    torch.testing.assert_close(model.operator, torch.zeros_like(model.operator))
    gamma = model.gamma_values(src["x"], nbr["x"], types)
    assert bool((gamma > 0).all())
    torch.testing.assert_close(gamma, torch.full_like(gamma, 1e-3 * math.log(2.0)),
                               rtol=1e-12, atol=1e-12)
    loss, count = model.loss_sum(src, nbr, types)
    (loss / count).backward()
    assert bool(torch.isfinite(model.operator.grad).all())
    assert bool(torch.any(model.operator.grad != 0))  # unlike ReLU(0)
    assert any(bool(torch.any(p.grad != 0)) for p in model.model.parameters())
    # Numerical differentiation independently verifies the chain rule.
    raw_value = model.operator.detach().clone()
    with torch.no_grad():
        model.operator[0] += 1e-6
        plus = model.loss_sum(src, nbr, types)[0].item() / count
        model.operator[0] -= 2e-6
        minus = model.loss_sum(src, nbr, types)[0].item() / count
        model.operator.copy_(raw_value)
    assert (plus - minus) / (2e-6) == pytest.approx(model.operator.grad[0].item(), rel=1e-6, abs=1e-7)
    with torch.no_grad():
        model.operator[0] = -1.0
        model.operator[1] = 1.0
    assert bool((model.gamma_values(src["x"], nbr["x"], types) >= 0).all())


@pytest.mark.parametrize("dataset,lag_sets", [
    ("fd001", ((1,5),(1,3,5),(1,3,5,10,15))),
    ("electricity", ((1,24),(1,12,24),(1,12,24,36,48))),
    ("metr_la", ((1,3),(1,3,6),(1,3,6,12,24)))])
def test_add_only_manifest_keeps_original_matching_settings(dataset, lag_sets):
    configs = make_manifest(dataset)
    assert len(configs) == len({c["run_id"] for c in configs}) == 750
    assert {c["method"] for c in configs} == {"td_semi_typed", "td_nonnegative_typed"}
    assert {tuple(c["lags"]) for c in configs} == set(lag_sets)
    from collections import Counter
    assert Counter((c["method"], tuple(c["lags"])) for c in configs) == {
        (method, lag): 125 for method in ("td_semi_typed", "td_nonnegative_typed")
        for lag in lag_sets}
    assert {c["batch_size"] for c in configs} == {256}
    assert {c["seed"] for c in configs} == set(range(5))
    assert all(c["gradient_clipping"] is None and c["learning_rate_schedule"] is None
               and c["weight_decay"] == 0 for c in configs)
    for c in configs:
        assert validate_config(c) == c
    with pytest.raises(ValueError):
        validate_config(dict(configs[0], method="td_typed"))
    with pytest.raises(ValueError):
        validate_config(dict(configs[0], alpha=99.))


def test_isolated_trainer_has_same_unmodified_mean_initialization():
    from dscpuscripts.typed_ablation_train import build_objective, original_trainer, original_config
    for dataset in ("fd001", "electricity"):
        c = make_manifest(dataset)[0]
        old = original_trainer(dataset)
        original = old.build_objective(original_config(c)) if dataset == "fd001" else old.build_objective(original_config(c), None)
        new = build_objective(c)
        assert set(original.model.state_dict()) == set(new.model.state_dict())
        for k, a in original.model.state_dict().items():
            torch.testing.assert_close(a, new.model.state_dict()[k], rtol=0, atol=0)


def test_new_trainer_does_not_patch_old_module_globals(tmp_path, monkeypatch):
    from dscpuscripts import typed_ablation_train as ablation
    for dataset in ("fd001", "electricity"):
        original = ablation.original_trainer(dataset)
        # Probe the production wrapper without doing any optimizer updates.
        if dataset == "fd001":
            def fake_candidate(config, train_data, val_data, results_dir, **kwargs):
                return validate_config(config), build_objective(config), code_hash(), PROTOCOL_ID
        else:
            def fake_candidate(config, train_data, val_data, results_dir, **kwargs):
                return validate_config(config), build_objective(config, None), code_hash(config), config["protocol_id"]
        # fake_candidate's global namespace is substituted privately, while
        # original module functions remain exactly the same objects.
        originals = (getattr(original, "build_objective"), getattr(original, "code_hash"))
        monkeypatch.setattr(original, "train_candidate", fake_candidate)
        config = make_manifest(dataset)[0]
        returned, objective, new_hash, protocol_id = ablation.train_candidate(config, None, None, tmp_path)
        assert returned == config
        assert objective.gamma_variant == "semi_typed"
        assert new_hash == ablation.code_hash(config)
        assert protocol_id == config["protocol_id"]
        assert (original.build_objective, original.code_hash) == originals


def test_fd001_typed_semigrad_and_nonnegative_are_trainable():
    from dscpuscripts.typed_ablation_objectives import FD001SemiTypedTD, FD001NonnegativeTypedTD

    class SmallRUL(nn.Module):
        output_dim = 1
        feature_dim = 4
        def __init__(self):
            super().__init__()
            self.head = nn.Linear(24 * 30, 1, dtype=torch.float64)
        def forward(self, x):
            return self.head(x.reshape(len(x), -1))

    x = torch.linspace(-1, 1, steps=3 * 24 * 30, dtype=torch.float64).reshape(3, 24, 30)
    y = torch.tensor([[.1], [.2], [-.3]], dtype=torch.float64)
    src = dict(x=x, y=y, mask=torch.ones_like(y, dtype=torch.bool))
    nbr = dict(x=x.flip(0), y=y.flip(0), mask=torch.ones_like(y, dtype=torch.bool))
    kinds = torch.tensor([0, 1, 0])
    for cls in (FD001SemiTypedTD, FD001NonnegativeTypedTD):
        obj = cls(SmallRUL(), num_types=2).double()
        with torch.no_grad():
            if cls is FD001SemiTypedTD:
                obj.operator.copy_(torch.tensor([.2, -.1], dtype=torch.float64))
        loss, count = obj.loss_sum(src, nbr, kinds)
        assert count == 3
        assert torch.isfinite(loss)
        (loss / count).backward()
        assert torch.isfinite(obj.operator.grad).all()
        assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in obj.model.parameters())


def test_validation_selection_is_locked_and_never_uses_test_data(tmp_path, monkeypatch):
    from dscpuscripts import typed_ablation_analysis as analysis
    from dscpuscripts import standard_runner
    from dscpuscripts.common import atomic_json
    dataset = "fd001"
    monkeypatch.setattr(analysis, "paths", lambda _: (tmp_path / "data", tmp_path))
    monkeypatch.setattr(analysis, "code_hash", lambda _: "verified-new-science")
    monkeypatch.setattr(analysis, "file_hash", lambda _: "verified-checkpoint-hash")
    # This fixture tests selection locking independently of full checkpoint auditing.
    monkeypatch.setattr(analysis, "_verify_completed_record", lambda *args: None)
    monkeypatch.setattr(standard_runner, "classify", lambda c, root, hash: "complete")
    configs = make_manifest(dataset)
    for c in configs:
        folder = tmp_path / c["run_id"]
        folder.mkdir()
        atomic_json(folder / "status.json", dict(config=c, identity=dict(code_sha256="verified-new-science")))
        # One strictly better validation setting (h=2) for both variants.
        score = 0.1 if c["configuration"] == 2 else 1.0 + c["configuration"]
        atomic_json(folder / "update_curve.json", [
            dict(optimizer_steps=0, mse_standardized=0.0),
            dict(optimizer_steps=10, mse_standardized=score),
            dict(optimizer_steps=20, mse_standardized=score),
            dict(optimizer_steps=30, mse_standardized=score),
            dict(optimizer_steps=40, mse_standardized=score)])
    selected = analysis.select(dataset)
    assert {r["configuration"] for r in selected["selected"]} == {2}
    assert len(selected["selected"]) == 6
    assert len({(r["method"], r["lag_set"]) for r in selected["selected"]}) == 6
    assert len({rid for r in selected["selected"] for rid in r["run_ids"]}) == 30
    assert all(len(r["run_ids"]) == 5 for r in selected["selected"])
    assert (tmp_path / "typed_ablation_outputs" / "selection.json").is_file()
    # A changed validation score must not silently replace the locked plan.
    affected = next(c for c in configs if c["configuration"] == 2)
    atomic_json(tmp_path / affected["run_id"] / "update_curve.json", [
        dict(optimizer_steps=0, mse_standardized=0.0),
        dict(optimizer_steps=10, mse_standardized=999.0)])
    with pytest.raises(RuntimeError, match="Previous validation selection differs"):
        analysis.select(dataset)


def test_ablation_selection_rejects_incomplete_validation_history(tmp_path):
    from dscpuscripts.typed_ablation_analysis import _verify_completed_record, _digest
    cfg = next(c for c in make_manifest("fd001") if c["batch_size"] == 256)
    n = 1200
    expected = math.ceil(n / 256) * cfg["epochs"]
    folder = tmp_path / cfg["run_id"]
    folder.mkdir()
    (folder / "final.pt").write_bytes(b"test")
    status = dict(state="complete", n_sources=n, epochs_completed=cfg["epochs"],
                  next_epoch=cfg["epochs"] + 1, optimizer_steps=expected,
                  update_opportunities=expected,
                  identity=dict(protocol_id=cfg["protocol_id"],
                                config_sha256=_digest(cfg)))
    with pytest.raises(ValueError, match="validation history"):
        _verify_completed_record(cfg, folder, status,
                                 [dict(optimizer_steps=0, mse_standardized=0.)])


def test_combined_selection_reuses_original_b256_only(tmp_path, monkeypatch):
    from dscpuscripts import typed_ablation_comparison as comparison
    from dscpuscripts.fd001_protocol import make_manifest as original_manifest
    originals = [c for c in original_manifest()
                 if c["method"] in ("mse", "td_typed") and c["batch_size"] == 256]
    monkeypatch.setattr(comparison, "_original_manifest", lambda _: originals)
    monkeypatch.setattr(comparison, "_audit_reference",
                        lambda c,root,shared,sources: float(c["alpha"]) + (
                            float(c["beta"]) if c["beta"] is not None else 0.))
    monkeypatch.setattr(comparison, "file_hash", lambda p: "reference-checkpoint")
    choices = comparison._select_old("fd001", tmp_path)
    assert {(c["method"], c["lag_set"]) for c in choices} == {
        ("mse", "none"), ("td_typed", "1-5"),
        ("td_typed", "1-3-5"), ("td_typed", "1-3-5-10-15")}
    assert all(len(c["run_ids"]) == 5 for c in choices)
    assert all("_b256_" in run_id for c in choices for run_id in c["run_ids"])
