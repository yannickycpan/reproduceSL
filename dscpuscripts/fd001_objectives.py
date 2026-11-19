"""Current FD001 TD operator objectives: typed and pairwise only.

Both use the original full-gradient pair-residual loss. No source-only head,
raw26 head, MAE training objective, target network, or stop-gradient path.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from agents.fulltd_objectives import Objective, masked_residual


def check_pairs(x, partner, types, num_types):
    if x.ndim != 3 or x.shape[1:] != (24, 30) or partner.shape != x.shape:
        raise ValueError("FD001 requires matching [B,24,30] source/partner tensors")
    if (types is None or types.shape != (len(x),) or
            types.dtype not in (torch.int32, torch.int64) or
            bool(torch.any((types < 0) | (types >= num_types)))):
        raise ValueError("Expected zero-based integer lag types for each pair")


class TypedGammaTD(Objective):
    is_td = True
    gamma_variant = "typed"

    def __init__(self, model, num_types=2):
        super().__init__(model)
        if model.output_dim != 1 or num_types < 1:
            raise ValueError("FD001 TD requires scalar RUL and positive lag type count")
        self.operator = nn.Parameter(torch.zeros(num_types))

    def parameter_groups(self, alpha, beta=None):
        if beta is None or not all(math.isfinite(v) and v > 0 for v in (alpha, beta)):
            raise ValueError("Both predictor and operator rates must be positive and finite")
        return [{"params": list(self.model.parameters()), "lr": alpha},
                {"params": [self.operator], "lr": beta}]

    def gamma_values(self, x, other, types, *, hidden=None, neighbor_hidden=None):
        check_pairs(x, other, types, self.operator.numel())
        return self.operator[types.to(device=self.operator.device, dtype=torch.long)]

    def loss_sum(self, batch, partner=None, types=None):
        if partner is None:
            raise ValueError("TD requires a sampled partner")
        x, other = batch["x"], partner["x"]
        gamma = self.gamma_values(x, other, types)
        source, source_mask = masked_residual(self.model(x), batch)
        neighbor, neighbor_mask = masked_residual(self.model(other), partner)
        delta = torch.where(source_mask & neighbor_mask,
                            source - gamma[:, None] * neighbor,
                            torch.zeros_like(source))
        return .5 * delta.square().sum(), int((source_mask & neighbor_mask).sum().item())


class PairwiseGammaHead(nn.Module):
    """Gamma_l = a_l^T [<h_i,h_j>/d, |h_i-h_j|, ||h_i-h_j||/sqrt(d)] + b_l."""

    def __init__(self, feature_dim: int, num_types: int = 2):
        super().__init__()
        if feature_dim < 1 or num_types < 1:
            raise ValueError("Positive feature dimension and lag type count required")
        self.feature_dim = int(feature_dim)
        self.num_types = int(num_types)
        # Construct without perturbing the caller's training RNG state.
        with torch.random.fork_rng(devices=[]):
            self.linear = nn.Linear(self.feature_dim + 2, self.num_types, bias=True)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, source, neighbor, types):
        if (source.ndim not in (2, 3) or source.shape != neighbor.shape or
                source.shape[-1] != self.feature_dim):
            raise ValueError("Expected matching [B,D] or [B,sensors,D] representations")
        if (types is None or types.shape != (len(source),) or
                types.dtype not in (torch.int32, torch.int64) or
                bool(torch.any((types < 0) | (types >= self.num_types)))):
            raise ValueError("Invalid pairwise TD lag type")
        diff = source - neighbor
        if source.ndim == 3:
            # METR-LA: retain magnitude before averaging across sensors.
            dimension = float(source.shape[1] * self.feature_dim)
            absolute_difference = diff.abs().mean(dim=1)
            inner_product = (source * neighbor).sum(dim=(1, 2))[:, None] / dimension
            distance = torch.linalg.vector_norm(
                diff.flatten(start_dim=1), dim=-1, keepdim=True) / math.sqrt(dimension)
        else:
            dimension = float(self.feature_dim)
            absolute_difference = diff.abs()
            inner_product = (source * neighbor).sum(-1, keepdim=True) / dimension
            distance = torch.linalg.vector_norm(diff, dim=-1, keepdim=True) / math.sqrt(dimension)
        pair = torch.cat((inner_product, absolute_difference, distance), dim=-1)
        return self.linear(pair).gather(
            1, types.to(device=source.device, dtype=torch.long)[:, None]).squeeze(-1)


class PairwiseGammaTD(Objective):
    is_td = True
    gamma_variant = "pairwise"

    def __init__(self, model, num_types=2):
        super().__init__(model)
        if not isinstance(model.feature_dim, int) or model.feature_dim < 1:
            raise ValueError("Model must supply a positive feature_dim")
        if model.output_dim != 1:
            raise ValueError("FD001 TD predicts one scalar remaining lifetime")
        self.gamma_head = PairwiseGammaHead(model.feature_dim, num_types=num_types)

    def parameter_groups(self, alpha, beta=None):
        if beta is None or not all(math.isfinite(v) and v > 0 for v in (alpha, beta)):
            raise ValueError("Both predictor and operator rates must be positive and finite")
        return [{"params": list(self.model.parameters()), "lr": alpha},
                {"params": list(self.gamma_head.parameters()), "lr": beta}]

    def gamma_values(self, x, other, types, *, hidden=None, neighbor_hidden=None):
        check_pairs(x, other, types, self.gamma_head.num_types)
        hi = self.model.features(x) if hidden is None else hidden
        hj = self.model.features(other) if neighbor_hidden is None else neighbor_hidden
        return self.gamma_head(hi, hj, types)

    def loss_sum(self, batch, partner=None, types=None):
        if partner is None:
            raise ValueError("TD requires a sampled partner")
        x, other = batch["x"], partner["x"]
        check_pairs(x, other, types, self.gamma_head.num_types)
        hi, hj = self.model.features(x), self.model.features(other)
        pred_i, pred_j = self.model.head(hi), self.model.head(hj)
        gamma = self.gamma_head(hi, hj, types)
        source, mask_i = masked_residual(pred_i, batch)
        neighbor, mask_j = masked_residual(pred_j, partner)
        mask = mask_i & mask_j
        delta = torch.where(mask, source - gamma[:, None] * neighbor,
                            torch.zeros_like(source))
        return .5 * delta.square().sum(), int(mask.sum().item())
