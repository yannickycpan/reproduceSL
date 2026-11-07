"""Dimension-independent pairwise Gamma head and unchanged old source-only Gamma.

The pairwise head explicitly accepts [B,d] hidden vectors from any backbone;
d is read from the representation, not fixed at 64. Both hidden vectors
participate in the full gradient. Gamma is signed and unconstrained.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from agents.fulltd_objectives import Objective, masked_residual
from fd001_gamma.objectives import GammaTD
from fulltd.models import build_model
from .protocol import LAGS


class PairwiseGammaHead(nn.Module):
    """Gamma_l = a_l^T [<h_i,h_j>/d, |h_i-h_j|, ||h_i-h_j||/sqrt(d)] + b_l."""

    def __init__(self, feature_dim: int, num_types: int = 2):
        super().__init__()
        if feature_dim <= 0 or num_types <= 0:
            raise ValueError("Positive feature dimension and type count required")
        self.feature_dim = int(feature_dim)
        self.num_types = int(num_types)
        # The dimensions of a_l and b_l are inferred from d and num_types.
        with torch.random.fork_rng(devices=[]):
            self.linear = nn.Linear(self.feature_dim + 2, self.num_types, bias=True)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def pair_features(self, source: torch.Tensor, neighbor: torch.Tensor) -> torch.Tensor:
        if source.shape != neighbor.shape or source.ndim != 2 or source.shape[-1] != self.feature_dim:
            raise ValueError("Expected matching [batch, feature_dim] hidden vectors")
        diff = source - neighbor
        dimension = float(self.feature_dim)
        similarity = (source * neighbor).sum(dim=-1, keepdim=True) / dimension
        absolute_difference = diff.abs()
        distance = torch.linalg.vector_norm(diff, ord=2, dim=-1, keepdim=True) / math.sqrt(dimension)
        return torch.cat((similarity, absolute_difference, distance), dim=-1)

    def forward(self, source, neighbor, types):
        if types is None or types.ndim != 1 or len(types) != len(source):
            raise ValueError("Exactly one lag type per pair is required")
        if types.dtype not in (torch.int32, torch.int64) or torch.any(
                (types < 0) | (types >= self.num_types)):
            raise ValueError("Invalid lag type index")
        features = self.pair_features(source, neighbor)
        coefficients = self.linear(features)
        return coefficients.gather(1, types.to(device=source.device, dtype=torch.long)[:, None]).squeeze(-1)


class PairwiseGammaTD(Objective):
    is_td = True
    gamma_variant = "pairwise"

    def __init__(self, model):
        super().__init__(model)
        if not isinstance(model.feature_dim, int) or model.feature_dim <= 0:
            raise ValueError("Backbone must expose a positive vector feature_dim")
        self.gamma_head = PairwiseGammaHead(model.feature_dim, num_types=len(LAGS))

    def operator_parameters(self):
        return list(self.gamma_head.parameters())

    def parameter_groups(self, alpha, beta=None):
        if beta is None or not all(math.isfinite(v) and v > 0 for v in (alpha, beta)):
            raise ValueError("Both positive finite learning rates are required")
        return [{"params": list(self.model.parameters()), "lr": alpha},
                {"params": self.operator_parameters(), "lr": beta}]

    def gamma_values(self, x, other, types, *, hidden=None, neighbor_hidden=None):
        source_features = self.model.features(x) if hidden is None else hidden
        neighbor_features = self.model.features(other) if neighbor_hidden is None else neighbor_hidden
        return self.gamma_head(source_features, neighbor_features, types)

    def loss_sum(self, batch, partner=None, types=None):
        if partner is None:
            raise ValueError("Full-gradient TD requires a sampled partner")
        x, other = batch["x"], partner["x"]
        # The two backbone passes are shared by the prediction and Gamma heads.
        # Neither h_i nor h_j is detached; gradients reach BOTH hidden vectors.
        hi, hj = self.model.features(x), self.model.features(other)
        pred_i, pred_j = self.model.head(hi), self.model.head(hj)
        gamma = self.gamma_head(hi, hj, types)
        residual_i, mask_i = masked_residual(pred_i, batch)
        residual_j, mask_j = masked_residual(pred_j, partner)
        mask = mask_i & mask_j
        delta = torch.where(mask, residual_i - gamma[:, None] * residual_j,
                            torch.zeros_like(residual_i))
        return .5 * delta.square().sum(), int(mask.sum().item())

    def operator_state(self):
        return {"head": {k: v.detach().cpu().tolist()
                         for k, v in self.gamma_head.state_dict().items()},
                "feature_dim": self.gamma_head.feature_dim}


def make_objective(config):
    model = build_model("fd001", "tcn", config["seed"])
    if config["gamma_variant"] == "representation":
        # EXACT old implementation, with additional alpha values only.
        return GammaTD(model, "representation")
    if config["gamma_variant"] == "pairwise":
        return PairwiseGammaTD(model)
    raise ValueError("Unexpected extension Gamma variant")
