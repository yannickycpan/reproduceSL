"""Focused checks of the mathematical objective and fixed compact predictors."""
import copy

import numpy as np
import pytest
import torch
from torch import nn

from dscpuscripts.application_objectives import make_objective
from fulltd.models import (GraphWaveNet, PatchTST, TemporalResidual, build_model,
                           parameter_count, row_normalized_adjacency)


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    old_mkldnn = torch.backends.mkldnn.enabled
    torch.set_num_threads(1)
    # This Windows test host's oneDNN convolution JIT raises native access
    # violations. Use the ordinary CPU implementation for correctness tests.
    # Scientific GPU execution is unaffected.
    torch.backends.mkldnn.enabled = False
    yield
    torch.set_num_threads(old)
    torch.backends.mkldnn.enabled = old_mkldnn


def batch(x, y, mask=None):
    return {"x": x, "y": y,
            "mask": torch.ones_like(y, dtype=torch.bool) if mask is None else mask}


class TinyMean(nn.Module):
    feature_dim, output_dim = 3, 2

    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(2, self.feature_dim)
        self.head = nn.Linear(self.feature_dim, self.output_dim)

    def features(self, x):
        return torch.tanh(self.encoder(x))

    def forward(self, x):
        return self.head(self.features(x))


def tiny():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(12)
        return TinyMean().double()


def set_gamma_nonzero(objective):
    with torch.no_grad():
        if hasattr(objective, "operator"):
            objective.operator.copy_(torch.tensor([0.25, -0.4], dtype=torch.double))
        else:
            objective.gamma_head.linear.weight.fill_(0.02)
            objective.gamma_head.linear.bias.copy_(torch.tensor([0.25, -0.4], dtype=torch.double))


def operator_parameters(objective):
    return ([objective.operator] if hasattr(objective, "operator")
            else list(objective.gamma_head.parameters()))


def examples():
    source = batch(torch.tensor([[0.3, -0.4], [1.0, 0.2], [0.2, 0.7]], dtype=torch.double),
                   torch.tensor([[1.1, 0.7], [0.4, 1.3], [0.2, 0.6]], dtype=torch.double))
    partner = batch(source["x"] * 0.7 + 0.2, source["y"] * 0.8 + 0.1)
    return source, partner, torch.tensor([0, 1, 0])


@pytest.mark.parametrize("method", ["td_typed", "td_pairwise", "td_pairwise_mae"])
def test_td_autograd_matches_finite_differences(method):
    source, partner, types = examples()
    objective = make_objective(method, tiny()).double()
    set_gamma_nonzero(objective)
    total, count = objective.loss_sum(source, partner, types)
    (total / count).backward()
    # Check every nonlinear predictor parameter and both operator coordinates.
    for parameter in objective.parameters():
        analytic = parameter.grad.detach().clone().reshape(-1)
        for index in range(parameter.numel()):
            with torch.no_grad():
                flat = parameter.view(-1)
                original = flat[index].item()
                flat[index] = original + 1e-6
            plus = objective.loss_sum(source, partner, types)[0].item() / count
            with torch.no_grad():
                flat[index] = original - 1e-6
            minus = objective.loss_sum(source, partner, types)[0].item() / count
            with torch.no_grad():
                flat[index] = original
            assert analytic[index].item() == pytest.approx((plus - minus) / 2e-6, abs=2e-8, rel=2e-6)


@pytest.mark.parametrize("method", ["td_typed", "td_pairwise", "td_pairwise_mae"])
def test_partner_prediction_path_is_not_detached(method):
    source, partner, types = examples()
    objective = make_objective(method, tiny()).double()
    set_gamma_nonzero(objective)
    partner["x"] = partner["x"].clone().requires_grad_()
    total, count = objective.loss_sum(source, partner, types)
    gradient, = torch.autograd.grad(total / count, partner["x"])
    assert torch.isfinite(gradient).all() and torch.count_nonzero(gradient) > 0


@pytest.mark.parametrize("method", ["td_typed", "td_pairwise", "td_pairwise_mae"])
def test_zero_gamma_mean_gradient_matches_baseline_but_operator_moves(method):
    source, partner, types = examples()
    mean = tiny()
    baseline_method = "mae" if method == "td_pairwise_mae" else "mse"
    baseline = make_objective(baseline_method, copy.deepcopy(mean)).double()
    td = make_objective(method, copy.deepcopy(mean)).double()
    assert all(torch.count_nonzero(p) == 0 for p in operator_parameters(td))
    ordinary, count = baseline.loss_sum(source)
    temporal, pair_count = td.loss_sum(source, partner, types)
    assert count == pair_count
    torch.testing.assert_close(ordinary, temporal, rtol=0, atol=0)
    (ordinary / count).backward()
    (temporal / pair_count).backward()
    for ordinary_parameter, td_parameter in zip(baseline.model.parameters(), td.model.parameters()):
        torch.testing.assert_close(ordinary_parameter.grad, td_parameter.grad, rtol=0, atol=0)
    assert all(p.grad is not None and torch.count_nonzero(p.grad) > 0
               for p in operator_parameters(td))


@pytest.mark.parametrize("method", ["td_typed", "td_pairwise", "td_pairwise_mae"])
def test_td_masks_use_intersection_and_zero_valid_has_zero_gradient(method):
    source, partner, types = examples()
    source["mask"][0, 0] = False
    source["y"][0, 0] = float("nan")
    partner["mask"][1, :] = False
    partner["y"][1, :] = float("nan")
    objective = make_objective(method, tiny()).double()
    total, count = objective.loss_sum(source, partner, types)
    assert count == 3
    assert make_objective("mse", tiny()).loss_sum(source)[1] == 5
    assert torch.isfinite(total)
    total.backward()
    assert all(torch.isfinite(p.grad).all() for p in objective.parameters())
    partner["mask"].fill_(False)
    objective.zero_grad()
    total, count = objective.loss_sum(source, partner, types)
    assert count == 0 and total.item() == 0
    total.backward()
    assert all(torch.count_nonzero(p.grad) == 0 for p in objective.parameters())


@pytest.mark.parametrize("dataset,backbone,shape,output,count", [
    ("electricity", "tcn", (2, 1, 168), (2, 24), 137816),
    ("fd001", "tcn", (2, 24, 30), (2, 1), 92801),
    ("electricity", "patchtst", (2, 1, 168), (2, 24), 100056),
    ("metrla", "graphwavenet", (2, 2, 207, 12), (2, 207, 12), 134200),
])
def test_shapes_counts_and_finite_backward(dataset, backbone, shape, output, count):
    model = build_model(dataset, backbone, 3, adjacency=np.eye(207))
    prediction = model(torch.randn(shape))
    assert prediction.shape == output
    assert parameter_count(model) == count
    prediction.square().mean().backward()
    # Every registered parameter participates in point prediction.
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert not any(isinstance(m, (nn.Dropout, nn.BatchNorm1d, nn.BatchNorm2d)) for m in model.modules())


@pytest.mark.parametrize("backbone", ["tcn", "patchtst"])
def test_initial_mean_identical_across_objectives_and_independent_of_global_rng(backbone):
    mse = make_objective("mse", build_model("electricity", backbone, 2))
    torch.randn(85)
    td = make_objective("td_typed", build_model("electricity", backbone, 2))
    pair = make_objective("td_pairwise", build_model("electricity", backbone, 2))
    nll = make_objective("gaussian_nll", build_model("electricity", backbone, 2))
    for key, value in mse.model.state_dict().items():
        assert torch.equal(value, td.model.state_dict()[key])
        assert torch.equal(value, pair.model.state_dict()[key])
        assert torch.equal(value, nll.model.state_dict()[key])
    other = build_model("electricity", backbone, 3)
    assert any(not torch.equal(value, other.state_dict()[key]) for key, value in mse.model.state_dict().items())


def test_transformer_layers_are_independent_and_residual_attention_has_gradient():
    model = build_model("electricity", "patchtst", 2)
    assert not torch.equal(model.layers[0].query.weight, model.layers[1].query.weight)
    features = torch.randn(2, 20, 64)
    first, logits = model.layers[0](features, None)
    logits.retain_grad()
    second, _ = model.layers[1](first, logits)
    second.square().sum().backward()
    assert logits.grad is not None and logits.grad.abs().sum() > 0


def test_tcn_block_causality():
    layer = TemporalResidual(2, 64, dilation=4)
    first = torch.randn(2, 2, 30)
    second = first.clone()
    second[..., 16:] += 50
    torch.testing.assert_close(layer(first)[..., :16], layer(second)[..., :16], rtol=0, atol=0)


def test_gaussian_initial_variance_one_positivity_and_shared_feature_gradient():
    model = build_model("fd001", "tcn", 0).double()
    objective = make_objective("gaussian_nll", model).double()
    source = batch(torch.randn(3, 24, 30, dtype=torch.double), torch.randn(3, 1, dtype=torch.double))
    hidden = model.features(source["x"])
    variance = objective.variance(hidden)
    torch.testing.assert_close(variance, torch.ones_like(variance), atol=1e-7, rtol=0)
    assert torch.count_nonzero(objective.variance_head.weight) == 0
    with torch.no_grad():
        objective.variance_head.weight.fill_(0.01)
    total, count = objective.loss_sum(source)
    total.backward()
    assert model.blocks[0].first.weight.grad.abs().sum() > 0
    assert objective.variance_head.weight.grad.abs().sum() > 0
    with torch.no_grad():
        objective.variance_head.bias.fill_(-1e6)
    assert (objective.variance(hidden) >= 1e-6).all()


@pytest.mark.parametrize("method", ["td_typed", "td_pairwise", "td_pairwise_mae"])
def test_microbatch_gradient_and_adam_update_match_global_logical_batch(method):
    source, partner, types = examples()
    partner["mask"][1, 0] = False  # deliberately unequal valid counts
    large = make_objective(method, tiny()).double()
    set_gamma_nonzero(large)
    small = copy.deepcopy(large)
    optimizers = [torch.optim.Adam(o.parameter_groups(3e-4, 3e-3), betas=(0.9, 0.999),
                                   eps=1e-8, weight_decay=0, amsgrad=False)
                  for o in (large, small)]
    total, count = large.loss_sum(source, partner, types)
    (total / count).backward()
    for lower, upper in [(0, 2), (2, 3)]:
        chunk = {k: v[lower:upper] for k, v in source.items()}
        other_chunk = {k: v[lower:upper] for k, v in partner.items()}
        loss, _ = small.loss_sum(chunk, other_chunk, types[lower:upper])
        (loss / count).backward()
    for a, b in zip(large.parameters(), small.parameters()):
        torch.testing.assert_close(a.grad, b.grad, rtol=1e-12, atol=1e-12)
    for optimizer in optimizers:
        optimizer.step()
        assert [g["lr"] for g in optimizer.param_groups] == [3e-4, 3e-3]
        assert all(g["weight_decay"] == 0 and not g["amsgrad"] for g in optimizer.param_groups)
    for a, b in zip(large.parameters(), small.parameters()):
        torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-12)


def test_graph_adjacency_direction_and_normalization():
    raw = np.array([[0.0, 2.0, 0.0], [1.0, 0.0, 3.0], [0.0, 0.0, 0.0]])
    normalized = row_normalized_adjacency(raw)
    torch.testing.assert_close(normalized.sum(1), torch.tensor([1.0, 1.0, 0.0]))
    model = build_model("metrla", "graphwavenet", 4, adjacency=raw)
    torch.testing.assert_close(model.forward_support, normalized)
    torch.testing.assert_close(model.backward_support, row_normalized_adjacency(raw.T))
    assert model.receptive_field == 13


@pytest.mark.parametrize("method", ["td_typed", "td_pairwise", "td_pairwise_mae"])
def test_graph_shared_adaptive_support_microbatch_gradient_is_exact(method):
    torch.manual_seed(123)
    source = batch(torch.randn(3, 2, 4, 12, dtype=torch.double),
                   torch.randn(3, 4, 12, dtype=torch.double))
    partner = batch(source["x"] * 0.7, source["y"] * 0.8)
    partner["mask"][1, 0, :5] = False
    types = torch.tensor([0, 1, 0])
    full = make_objective(method, build_model("metrla", "graphwavenet", 1, np.eye(4))).double()
    set_gamma_nonzero(full)
    chunks = copy.deepcopy(full)
    loss, valid = full.loss_sum(source, partner, types)
    (loss / valid).backward()
    for index in range(3):
        part = {k: v[index:index+1] for k, v in source.items()}
        other = {k: v[index:index+1] for k, v in partner.items()}
        subtotal, _ = chunks.loss_sum(part, other, types[index:index+1])
        (subtotal / valid).backward()
    for (name, a), (_, b) in zip(full.named_parameters(), chunks.named_parameters()):
        assert a.grad is not None, name
        torch.testing.assert_close(a.grad, b.grad, rtol=1e-9, atol=1e-10, msg=name)
    assert full.model.node_left.grad.abs().sum() > 0
    assert full.model.node_right.grad.abs().sum() > 0


def test_graph_nodewise_linear_head_equals_reference_1x1_convolution():
    model = build_model("metrla", "graphwavenet", 0, np.eye(3)).double()
    hidden = torch.randn(2, 3, 192, dtype=torch.double)
    convolution = nn.Conv2d(192, 12, 1).double()
    with torch.no_grad():
        convolution.weight.copy_(model.head.weight[:, :, None, None])
        convolution.bias.copy_(model.head.bias)
    result = convolution(hidden.transpose(1, 2).unsqueeze(-1))[:, :, :, 0].transpose(1, 2)
    torch.testing.assert_close(model.head(hidden), result, atol=1e-14, rtol=1e-14)


def test_graph_pairwise_head_preserves_sensor_coordinates_when_flattening():
    """Equal pooled means must still permit different global pair coefficients."""
    model = build_model("metr_la", "graph_wavenet", 0, np.eye(3)).double()
    objective = make_objective("td_pairwise", model).double()
    width = model.feature_dim
    assert objective.gamma_head.feature_dim == 3 * width
    source = torch.zeros(2, 3, width, dtype=torch.double)
    source[0, 0, 4] = 2.
    source[1, 1, 4] = 2.
    neighbor = torch.zeros_like(source)
    torch.testing.assert_close(source.mean(1)[0], source.mean(1)[1])
    # Select one sensor's absolute difference coordinate; a pooled adaptation
    # would erase the distinction between these two otherwise equal windows.
    with torch.no_grad():
        objective.gamma_head.linear.weight[0, 1 + width + 4] = .3
        objective.gamma_head.linear.bias[0] = .1
    x = torch.zeros(2, 2, 3, 12, dtype=torch.double)
    gamma = objective.gamma_values(x, x, torch.zeros(2, dtype=torch.long),
                                   hidden=source, neighbor_hidden=neighbor)
    torch.testing.assert_close(gamma, torch.tensor([.1, .7], dtype=torch.double))
    assert gamma.shape == (2,)


@pytest.mark.parametrize("dataset,backbone,x_shape,y_shape", [
    ("electricity", "tcn", (2, 1, 168), (2, 24)),
    ("electricity", "patchtst", (2, 1, 168), (2, 24)),
    ("metr_la", "graph_wavenet", (2, 2, 4, 12), (2, 4, 12)),
])
def test_gaussian_nll_matches_each_application_backbone(dataset, backbone, x_shape, y_shape):
    model = build_model(dataset, backbone, 0, adjacency=np.eye(4)).double()
    objective = make_objective("gaussian_nll", model).double()
    x = torch.randn(x_shape, dtype=torch.double)
    y = torch.randn(y_shape, dtype=torch.double)
    source = batch(x, y)
    variance = objective.variance(model.features(x))
    assert variance.shape == y.shape
    torch.testing.assert_close(variance, torch.ones_like(variance), rtol=0, atol=1e-7)
    loss, valid = objective.loss_sum(source)
    assert valid == y.numel() and torch.isfinite(loss)
    (loss / valid).backward()
    assert objective.variance_head.weight.grad.abs().sum() > 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in model.parameters())


@pytest.mark.parametrize("method", ["td_full", "td_scalar"])
def test_retired_td_objectives_are_not_runnable_via_frozen_factory(method):
    from agents.fulltd_objectives import make_objective as frozen_factory
    with pytest.raises(ValueError, match="(?i)(retired|current|replaced|removed)"):
        frozen_factory(method, tiny())
    # Importing the preserved compatibility names must leave the two FD001
    # baseline construction paths usable with the same deterministic predictor.
    for baseline in ("mse", "gaussian_nll"):
        model = tiny()
        objective = frozen_factory(baseline, model)
        assert objective.model is model
