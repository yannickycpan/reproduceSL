"""Only the requested three full-gradient TD parameterizations.

Raw means observable input features, using the EXISTING training-only channel
standardization. raw26 = [one_hot(lag type), x_i[:, -1] - x_j[:, -1]].
The representation head uses source h_i only, with both outputs trained jointly.
"""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F

from agents.fulltd_objectives import Objective, masked_residual
from fulltd.models import build_model
from .protocol import VARIANTS


class GammaTD(Objective):
    is_td = True

    def __init__(self, model: nn.Module, variant: str):
        super().__init__(model)
        if variant not in VARIANTS:
            raise ValueError(f"Unknown Gamma variant: {variant}")
        if model.feature_dim != 64 or model.output_dim != 1:
            raise ValueError("This sweep requires the unchanged FD001 TCN (64 features, scalar output)")
        self.gamma_variant = variant
        if variant == "typed":
            self.operator = nn.Parameter(torch.zeros(2))
        elif variant == "raw26":
            # The two exhaustive type indicators supply intercepts: no extra bias.
            self.operator = nn.Parameter(torch.zeros(26))
        else:
            # Restore the caller's RNG stream; auxiliary head creation must not
            # change predictor initialization or subsequent random sampling.
            with torch.random.fork_rng(devices=[]):
                self.gamma_head = nn.Linear(64, 2)
            nn.init.zeros_(self.gamma_head.weight)
            nn.init.zeros_(self.gamma_head.bias)

    def operator_parameters(self):
        return list(self.gamma_head.parameters()) if self.gamma_variant == "representation" else [self.operator]

    def parameter_groups(self, alpha, beta=None):
        if beta is None or not all(math.isfinite(v) and v > 0 for v in (alpha, beta)):
            raise ValueError("Both predictor alpha and Gamma beta must be positive and finite")
        return [{"params": list(self.model.parameters()), "lr": alpha},
                {"params": self.operator_parameters(), "lr": beta}]

    @staticmethod
    def check_inputs(x, other, types):
        if x.ndim != 3 or x.shape[1:] != (24, 30) or other.shape != x.shape:
            raise ValueError("FD001 inputs must both have shape [B,24,30]")
        if types is None or types.shape != (len(x),):
            raise ValueError("Expected one lag type per source")
        if types.dtype not in (torch.int32, torch.int64) or ((types < 0) | (types > 1)).any():
            raise ValueError("Lag types must be integer 0 (one cycle) or 1 (five cycles)")

    def raw_features(self, x, other, types):
        self.check_inputs(x, other, types)
        return torch.cat((F.one_hot(types.long(), 2).to(dtype=x.dtype, device=x.device),
                          x[:, :, -1] - other[:, :, -1]), dim=1)

    def gamma_values(self, x, other, types, *, hidden=None):
        self.check_inputs(x, other, types)
        types = types.to(device=x.device, dtype=torch.long)
        if self.gamma_variant == "typed":
            return self.operator[types]
        if self.gamma_variant == "raw26":
            return self.raw_features(x, other, types) @ self.operator
        h = self.model.features(x) if hidden is None else hidden
        # No detach: the full derivative includes the path through Gamma(h_i).
        return self.gamma_head(h).gather(1, types[:, None]).squeeze(1)

    def loss_sum(self, batch, partner=None, types=None):
        if partner is None:
            raise ValueError("Full-gradient TD needs a sampled training partner")
        x, other = batch["x"], partner["x"]
        self.check_inputs(x, other, types)
        if self.gamma_variant == "representation":
            hidden = self.model.features(x)
            source_prediction = self.model.head(hidden)
            gamma = self.gamma_values(x, other, types, hidden=hidden)
        else:
            source_prediction = self.model(x)
            gamma = self.gamma_values(x, other, types)
        residual, source_mask = masked_residual(source_prediction, batch)
        other_residual, partner_mask = masked_residual(self.model(other), partner)
        mask = source_mask & partner_mask
        delta = torch.where(mask, residual - gamma[:, None] * other_residual,
                            torch.zeros_like(residual))
        return 0.5 * delta.square().sum(), int(mask.sum().item())

    def operator_state(self):
        if self.gamma_variant == "representation":
            return {key: value.detach().cpu().tolist() for key, value in self.gamma_head.state_dict().items()}
        return {"weights": self.operator.detach().cpu().tolist()}


def make_objective(config):
    model = build_model("fd001", "tcn", config["seed"])
    return GammaTD(model, config["gamma_variant"])
