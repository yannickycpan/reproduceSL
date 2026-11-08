"""FD001's typed and pairwise TD mathematics for vector forecasting targets.

The predictor is identical across methods. Graph WaveNet features are flattened
only when forming the pairwise coefficient; its prediction head stays nodewise.
One coefficient per temporal pair is broadcast over every sensor and horizon.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from agents.fulltd_objectives import Objective, masked_residual
from dscpuscripts.fd001_objectives import PairwiseGammaHead


def check_pairs(source, neighbor, types):
    if source.ndim < 2 or source.shape != neighbor.shape or not len(source):
        raise ValueError("TD requires equally shaped, nonempty source/partner inputs")
    if (types is None or types.shape != (len(source),) or
            types.dtype not in (torch.int32, torch.int64) or
            bool(torch.any((types < 0) | (types > 1)))):
        raise ValueError("Expected integer lag types 0 and 1 for each pair")


def _rates(alpha, beta):
    if beta is None or not all(math.isfinite(v) and v > 0 for v in (alpha, beta)):
        raise ValueError("Both predictor and operator rates must be positive and finite")


def pair_delta(prediction, other_prediction, batch, partner, gamma):
    source, source_mask = masked_residual(prediction, batch)
    neighbor, neighbor_mask = masked_residual(other_prediction, partner)
    if source.shape != neighbor.shape:
        raise ValueError("Pair targets must have matching sensors and forecast horizons")
    mask = source_mask & neighbor_mask
    coefficients = gamma.reshape((-1,) + (1,) * (source.ndim - 1))
    delta = torch.where(mask, source - coefficients * neighbor, torch.zeros_like(source))
    return delta, int(mask.sum().item())


class TypedGammaTD(Objective):
    is_td = True
    gamma_variant = "typed"

    def __init__(self, model):
        super().__init__(model)
        self.operator = nn.Parameter(torch.zeros(2))

    def parameter_groups(self, alpha, beta=None):
        _rates(alpha, beta)
        return [{"params": list(self.model.parameters()), "lr": alpha},
                {"params": [self.operator], "lr": beta}]

    def gamma_values(self, x, other, types, *, hidden=None, neighbor_hidden=None):
        check_pairs(x, other, types)
        return self.operator[types.to(device=self.operator.device, dtype=torch.long)]

    def loss_sum(self, batch, partner=None, types=None):
        if partner is None:
            raise ValueError("TD requires a sampled partner")
        x, other = batch["x"], partner["x"]
        gamma = self.gamma_values(x, other, types)
        delta, count = pair_delta(self.model(x), self.model(other), batch, partner, gamma)
        return .5 * delta.square().sum(), count


class PairwiseGammaTD(Objective):
    is_td = True
    gamma_variant = "pairwise"

    def __init__(self, model, *, absolute=False, feature_dim=None):
        super().__init__(model)
        if feature_dim is None:
            feature_dim = model.feature_dim
            # Graph WaveNet supplies [batch, sensors, feature_dim]. Sensor order
            # is the fixed, recorded adjacency order used by the mean predictor.
            if hasattr(model, "node_left"):
                feature_dim *= int(model.node_left.shape[0])
        if not isinstance(feature_dim, int) or feature_dim < 1:
            raise ValueError("A positive flattened representation dimension is required")
        self.gamma_head = PairwiseGammaHead(feature_dim)
        self.absolute = bool(absolute)

    def parameter_groups(self, alpha, beta=None):
        _rates(alpha, beta)
        return [{"params": list(self.model.parameters()), "lr": alpha},
                {"params": list(self.gamma_head.parameters()), "lr": beta}]

    def gamma_values(self, x, other, types, *, hidden=None, neighbor_hidden=None):
        check_pairs(x, other, types)
        hi = self.model.features(x) if hidden is None else hidden
        hj = self.model.features(other) if neighbor_hidden is None else neighbor_hidden
        return self.gamma_head(hi.flatten(start_dim=1), hj.flatten(start_dim=1), types)

    def loss_sum(self, batch, partner=None, types=None):
        if partner is None:
            raise ValueError("TD requires a sampled partner")
        x, other = batch["x"], partner["x"]
        check_pairs(x, other, types)
        hi, hj = self.model.features(x), self.model.features(other)
        gamma = self.gamma_values(x, other, types, hidden=hi, neighbor_hidden=hj)
        delta, count = pair_delta(self.model.head(hi), self.model.head(hj),
                                  batch, partner, gamma)
        loss = delta.abs().sum() if self.absolute else .5 * delta.square().sum()
        return loss, count


def make_objective(method, model):
    """Only the current four principal methods and METR-LA MAE comparators."""
    from agents.fulltd_mse import MSE
    from agents.fulltd_gaussian_nll import GaussianNLL
    from agents.fulltd_mae import MAE
    if method == "td_typed":
        return TypedGammaTD(model)
    if method in ("td_pairwise", "td_pairwise_mae"):
        return PairwiseGammaTD(model, absolute=method == "td_pairwise_mae")
    classes = {"mse": MSE, "gaussian_nll": GaussianNLL, "mae": MAE}
    if method not in classes:
        raise ValueError(f"Unsupported current application objective: {method}")
    return classes[method](model)
