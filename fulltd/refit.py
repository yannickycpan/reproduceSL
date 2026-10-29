"""Frozen-representation, frozen-operator linear readout diagnostics.

Only selected primary TD-full Electricity-TCN and FD001-TCN models are eligible.
All fitting windows contribute with weight 1/N. TD enumerates the exact local
conditional pair law, including its asymmetric boundary probabilities. Neither
readout changes the reported trained model. No ridge or neural retraining occurs.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .analysis import write_json

# This is a numerical rank convention for the Gram matrix, not a regularizer.
# It is fixed before inspecting outcomes and recorded with every solve.
GRAM_RCOND = 1e-12


@dataclass
class ReadoutFit:
    coefficients: np.ndarray  # [hidden dimension + bias, output horizon]
    diagnostics: dict


def solve_normal_equations(gram, rhs, *, n_sources, n_terms, label):
    """Unregularized rank-revealing minimum-norm solve in float64.

The small streamed Gram matrix is symmetrized only to remove roundoff. A singular
direction is omitted by the stated relative singular-value tolerance; no positive
diagonal or penalty is added. Both the normal-equation residual and conditioning
are disclosed so the diagnostic cannot disguise an ill-conditioned result.
"""
    gram = np.asarray(gram, dtype=np.float64)
    rhs = np.asarray(rhs, dtype=np.float64)
    if gram.ndim != 2 or gram.shape[0] != gram.shape[1] or rhs.shape[0] != gram.shape[0]:
        raise ValueError("Incompatible normal-equation dimensions")
    if not np.isfinite(gram).all() or not np.isfinite(rhs).all():
        raise FloatingPointError("Nonfinite frozen-feature normal equations")
    gram = (gram + gram.T) * .5
    coefficients, _, rank, singular_values = np.linalg.lstsq(gram, rhs, rcond=GRAM_RCOND)
    residual = gram @ coefficients - rhs
    rhs_norm = float(np.linalg.norm(rhs))
    norm_residual = float(np.linalg.norm(residual))
    largest = float(singular_values[0]) if len(singular_values) else 0.0
    retained_smallest = float(singular_values[rank - 1]) if rank else 0.0
    condition = largest / retained_smallest if rank and retained_smallest > 0 else float("inf")
    diagnostics = {
        "label": label, "n_sources": int(n_sources), "enumerated_terms": int(n_terms),
        "dimension_including_bias": int(gram.shape[0]), "output_dimension": int(rhs.shape[1]),
        "rank": int(rank), "minimum_norm": True, "regularization": "none",
        "solver": "numpy.linalg.lstsq on streamed float64 Gram equations",
        "gram_relative_rank_tolerance": GRAM_RCOND,
        "gram_absolute_singular_cutoff": largest * GRAM_RCOND,
        "gram_singular_values": singular_values.tolist(),
        "retained_gram_condition_number": condition,
        "retained_design_condition_number": float(np.sqrt(condition)),
        "full_design_condition_number": float(np.sqrt(condition)) if rank == gram.shape[0] else float("inf"),
        "normal_equation_residual_norm": norm_residual,
        "relative_normal_equation_residual": norm_residual / rhs_norm if rhs_norm else norm_residual,
        "source_weight": "1/N; TD edge weight P(j|i)/N",
        "bias_coordinate": "1 for OLS; 1-Gamma_ij for TD",
    }
    return ReadoutFit(coefficients, diagnostics)


def _features(model, batch, device):
    x = torch.as_tensor(batch["x"], dtype=torch.float32, device=device)
    with torch.inference_mode():
        hidden = model.features(x).detach().to(device="cpu", dtype=torch.float64).numpy()
    if hidden.ndim != 2:
        raise ValueError("Readout diagnostic is restricted to a vector TCN representation")
    return np.column_stack((hidden, np.ones(len(hidden), dtype=np.float64)))


def _targets(batch):
    y = np.asarray(batch["y"], dtype=np.float64)
    if y.ndim == 1:
        y = y[:, None]
    if y.ndim != 2 or not np.asarray(batch["mask"], dtype=bool).all():
        raise ValueError("This readout diagnostic requires complete Electricity/FD001 scalar targets")
    if not np.isfinite(y).all():
        raise FloatingPointError("Nonfinite targets in readout diagnostic")
    return y


def refit_readouts(model, dataset, pair_law, theta, device="cpu", batch_size=1024):
    """Stream all sources and exact pair probabilities without storing features."""
    if batch_size <= 0 or len(dataset) == 0:
        raise ValueError("A positive batch and nonempty fitting dataset are required")
    theta = np.asarray(theta, dtype=np.float64).reshape(-1)
    if theta.shape != (2,) or not np.isfinite(theta).all():
        raise ValueError("Only a finite frozen TD-full two-coordinate operator is permitted")
    previous_training = model.training
    model.eval()
    n = len(dataset)
    ols_gram = ols_rhs = td_gram = td_rhs = None
    edge_count, probability_mass = 0, 0.0
    try:
        for start in range(0, n, batch_size):
            indices = np.arange(start, min(start + batch_size, n), dtype=np.int64)
            source_batch = dataset.get_batch(indices)
            source_features = _features(model, source_batch, device)
            source_targets = _targets(source_batch)
            if ols_gram is None:
                d, h = source_features.shape[1], source_targets.shape[1]
                ols_gram = np.zeros((d, d), dtype=np.float64)
                ols_rhs = np.zeros((d, h), dtype=np.float64)
                td_gram, td_rhs = np.zeros_like(ols_gram), np.zeros_like(ols_rhs)
            ols_gram += source_features.T @ source_features / n
            ols_rhs += source_features.T @ source_targets / n
            partners, types, probabilities = pair_law.enumerate(indices)
            partners = np.asarray(partners, dtype=np.int64)
            types = np.asarray(types, dtype=np.int64)
            probabilities = np.asarray(probabilities, dtype=np.float64)
            if partners.shape != types.shape or partners.shape != probabilities.shape or partners.shape[0] != len(indices):
                raise ValueError("Incorrect partner-enumeration shapes")
            if (probabilities < 0).any() or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-12, rtol=0):
                raise ValueError("The exact conditional pair law must sum to one for each source")
            source_row, edge_slot = np.nonzero(probabilities > 0)
            valid_partners = partners[source_row, edge_slot]
            valid_types = types[source_row, edge_slot]
            weights = probabilities[source_row, edge_slot] / n
            if (valid_types < 0).any() or (valid_types > 1).any():
                raise ValueError("Pair types must use zero-based operator coordinates 0 and 1")
            if (valid_partners < 0).any() or (valid_partners >= n).any():
                raise ValueError("Partner outside the full fitting-source universe")
            # Bound feature extraction memory even when every source has four edges.
            for edge_start in range(0, len(valid_partners), batch_size):
                edge_stop = min(edge_start + batch_size, len(valid_partners))
                sl = slice(edge_start, edge_stop)
                partner_batch = dataset.get_batch(valid_partners[sl])
                partner_features = _features(model, partner_batch, device)
                partner_targets = _targets(partner_batch)
                gamma = theta[valid_types[sl]][:, None]
                z = source_features[source_row[sl]] - gamma * partner_features
                y = source_targets[source_row[sl]] - gamma * partner_targets
                w = weights[sl, None]
                td_gram += z.T @ (w * z)
                td_rhs += z.T @ (w * y)
            edge_count += len(valid_partners)
            probability_mass += float(weights.sum())
    finally:
        model.train(previous_training)
    if not np.isclose(probability_mass, 1, rtol=0, atol=1e-10):
        raise ValueError("Enumerated pair law did not preserve uniform source mass")
    result = {
        "ols": solve_normal_equations(ols_gram, ols_rhs, n_sources=n, n_terms=n, label="frozen_representation_ols"),
        "td": solve_normal_equations(td_gram, td_rhs, n_sources=n, n_terms=edge_count, label="frozen_representation_fixed_theta_td"),
    }
    for fit in result.values():
        fit.diagnostics.update(theta_frozen=theta.tolist(), enumerated_pair_probability_mass=probability_mass,
                               physical_feature_batch_size=batch_size)
    return result


class RefittedPredictor(nn.Module):
    """Read-only wrapper: use the original frozen representation with a new head."""
    def __init__(self, base_model, fit):
        super().__init__()
        self.base_model = base_model
        coefficients = torch.as_tensor(fit.coefficients, dtype=torch.float64)
        self.register_buffer("coefficients", coefficients)

    def forward(self, x):
        hidden = self.base_model.features(x).to(dtype=torch.float64)
        return hidden @ self.coefficients[:-1] + self.coefficients[-1]


def save_readouts(output_dir, fits, config, checkpoint_sha256):
    if config["dataset"] not in ("electricity", "fd001") or config["backbone"] != "tcn" or config["method"] != "td_full" or config["batch_size"] != 256:
        raise ValueError("Readout refitting is restricted to selected primary TD-full TCN models")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    np.savez(output / "readouts.npz", **{name: fit.coefficients for name, fit in fits.items()})
    write_json(output / "diagnostics.json", {
        "config": config, "source_final_checkpoint_sha256": checkpoint_sha256,
        "fits": {name: fit.diagnostics for name, fit in fits.items()},
        "scope": "Conditional readout diagnostic; original actual final model remains the headline predictor.",
    })
