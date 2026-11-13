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

def make_objective(method: str, model: nn.Module) -> Objective:
    """Construct supported baseline objectives; the retired TD modes are removed."""
    from agents.fulltd_mse import MSE
    from agents.fulltd_gaussian_nll import GaussianNLL
    from agents.fulltd_mae import MAE

    key = method.lower().replace("-", "_")
    classes = {"mse": MSE, "gaussian_nll": GaussianNLL, "mae": MAE}
    if key not in classes:
        raise ValueError(f"Removed or unsupported legacy objective: {method}")
    return classes[key](model)
