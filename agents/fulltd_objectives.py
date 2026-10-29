"""Small objective interface, separate from the repository's legacy agents."""
from __future__ import annotations

import torch
from torch import nn


class Objective(nn.Module):
    is_td = False

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        # Point inference always consists only of the common mean network.
        return self.model(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.predict(x)

    def parameter_groups(self, alpha: float, beta: float | None = None):
        return [{"params": list(self.parameters()), "lr": alpha}]


def masked_residual(prediction, batch):
    if prediction.shape != batch["y"].shape:
        raise ValueError(f"Prediction/target shape mismatch: {prediction.shape} vs {batch['y'].shape}")
    mask = batch["mask"].bool()
    if mask.shape != prediction.shape:
        raise ValueError("Target mask shape must equal prediction shape")
    # Mask BEFORE squaring; masked NaN labels cannot poison gradients.
    residual = torch.where(mask, batch["y"] - prediction, torch.zeros_like(prediction))
    return residual, mask


class TDObjective(Objective):
    is_td = True

    def __init__(self, model: nn.Module, coordinates: int):
        super().__init__(model)
        self.theta = nn.Parameter(torch.zeros(coordinates))

    def parameter_groups(self, alpha: float, beta: float | None = None):
        if beta is None:
            raise ValueError("TD requires its prescribed operator learning rate")
        return [{"params": list(self.model.parameters()), "lr": alpha},
                {"params": [self.theta], "lr": beta}]

    def loss_sum(self, batch, partner=None, types=None):
        if partner is None or types is None:
            raise ValueError("TD requires one partner and temporal type per source")
        residual, source_mask = masked_residual(self.predict(batch["x"]), batch)
        other_residual, partner_mask = masked_residual(self.predict(partner["x"]), partner)
        if residual.shape != other_residual.shape:
            raise ValueError("Partners must match source sensors and horizons")
        types = types.to(device=residual.device, dtype=torch.long)
        if types.shape != (residual.shape[0],) or ((types < 0) | (types > 1)).any():
            raise ValueError("Temporal types must be a length-B vector of zero-based 0/1 labels")
        coefficients = self.theta[types] if self.theta.numel() == 2 else self.theta[0].expand_as(types)
        coefficients = coefficients.reshape((-1,) + (1,) * (residual.ndim - 1))
        mask = source_mask & partner_mask
        # Both predictions remain in the graph. Neither residuals nor logits
        # are detached, and no averaging of partner residuals precedes squaring.
        delta = torch.where(mask, residual - coefficients * other_residual,
                            torch.zeros_like(residual))
        return 0.5 * delta.square().sum(), int(mask.sum().item())


def make_objective(method: str, model: nn.Module) -> Objective:
    from agents.fulltd_mse import MSE
    from agents.fulltd_td_full import TDFull
    from agents.fulltd_td_scalar import TDScalar
    from agents.fulltd_gaussian_nll import GaussianNLL
    from agents.fulltd_mae import MAE
    from agents.fulltd_dlinear_mse import DLinearMSE
    key = method.lower().replace("-", "_")
    classes = {"mse": MSE, "td_full": TDFull, "td_scalar": TDScalar,
               "gaussian_nll": GaussianNLL, "mae": MAE, "dlinear_mse": DLinearMSE}
    if key not in classes:
        raise ValueError(f"Unprescribed objective: {method}")
    return classes[key](model)
