"""Traffic MAE baseline using all observed source targets."""
from agents.fulltd_objectives import Objective, masked_residual


class MAE(Objective):
    def loss_sum(self, batch, partner=None, types=None):
        residual, mask = masked_residual(self.predict(batch["x"]), batch)
        return residual.abs().sum(), int(mask.sum().item())
