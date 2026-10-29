"""Ordinary half-MSE, using all valid source targets."""
from agents.fulltd_objectives import Objective, masked_residual


class MSE(Objective):
    def loss_sum(self, batch, partner=None, types=None):
        residual, mask = masked_residual(self.predict(batch["x"]), batch)
        return 0.5 * residual.square().sum(), int(mask.sum().item())
