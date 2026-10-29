"""Input-dependent Gaussian NLL with a matched common mean predictor."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from agents.fulltd_objectives import Objective, masked_residual


class GaussianNLL(Objective):
    variance_floor = 1e-6

    def __init__(self, model):
        super().__init__(model)
        # Construction cannot perturb the caller's predictor/random stream.
        # Zero weights and constant bias need no random auxiliary draw.
        with torch.random.fork_rng(devices=[]):
            self.variance_head = nn.Linear(model.feature_dim, model.output_dim)
        nn.init.zeros_(self.variance_head.weight)
        nn.init.constant_(self.variance_head.bias, math.log(math.expm1(1.0 - self.variance_floor)))

    def variance(self, hidden):
        return self.variance_floor + F.softplus(self.variance_head(hidden))

    def loss_sum(self, batch, partner=None, types=None):
        hidden = self.model.features(batch["x"])
        mean = self.model.head(hidden)
        residual, mask = masked_residual(mean, batch)
        variance = self.variance(hidden)
        scalar_loss = 0.5 * (residual.square() / variance + variance.log())
        return torch.where(mask, scalar_loss, torch.zeros_like(scalar_loss)).sum(), int(mask.sum().item())
