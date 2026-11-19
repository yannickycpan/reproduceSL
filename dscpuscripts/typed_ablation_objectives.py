"""Typed-only TD mechanism ablations; the existing full-TD objectives are untouched.

The semi-gradient experiment stops the *neighbor prediction* gradient, not
learning of the typed operator. Nonnegative full TD uses a smooth, fixed
softplus map of the original zero-initialized raw operator parameters.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F

from dscpuscripts.application_objectives import pair_delta
from dscpuscripts.application_objectives import TypedGammaTD as ApplicationTypedTD
from dscpuscripts.fd001_objectives import TypedGammaTD as FD001TypedTD


class _SemiGradientTyped:
    gamma_variant = "semi_typed"

    def loss_sum(self, batch, partner=None, types=None):
        if partner is None:
            raise ValueError("TD requires a sampled partner")
        gamma = self.gamma_values(batch["x"], partner["x"], types)
        source_prediction = self.model(batch["x"])
        # Semi-gradient: the partner is a fixed TD target for predictor updates.
        # Gamma is independent of representations in the *typed* formulation;
        # its parameter gradient is retained by placing it outside no_grad.
        with torch.no_grad():
            partner_prediction = self.model(partner["x"])
        delta, valid = pair_delta(source_prediction, partner_prediction, batch, partner, gamma)
        return 0.5 * delta.square().sum(), valid


class _NonnegativeFullTyped:
    gamma_variant = "nonnegative_full_typed"
    # Not tuned. At raw 0, Gamma = 0.000693... and its derivative is 0.5;
    # unlike ReLU(raw), the operator can learn immediately from initialization.
    softplus_scale = 1e-3

    def gamma_values(self, x, other, types, *, hidden=None, neighbor_hidden=None):
        raw = super().gamma_values(x, other, types, hidden=hidden, neighbor_hidden=neighbor_hidden)
        return self.softplus_scale * F.softplus(raw / self.softplus_scale)


class FD001SemiTypedTD(_SemiGradientTyped, FD001TypedTD):
    pass


class FD001NonnegativeTypedTD(_NonnegativeFullTyped, FD001TypedTD):
    pass


class ApplicationSemiTypedTD(_SemiGradientTyped, ApplicationTypedTD):
    pass


class ApplicationNonnegativeTypedTD(_NonnegativeFullTyped, ApplicationTypedTD):
    pass


METHOD_CLASSES = {
    "fd001": {
        "td_semi_typed": FD001SemiTypedTD,
        "td_nonnegative_typed": FD001NonnegativeTypedTD,
    },
    "electricity": {
        "td_semi_typed": ApplicationSemiTypedTD,
        "td_nonnegative_typed": ApplicationNonnegativeTypedTD,
    },
    "metr_la": {
        "td_semi_typed": ApplicationSemiTypedTD,
        "td_nonnegative_typed": ApplicationNonnegativeTypedTD,
    },
}


def from_original_typed(config, old_objective):
    """Reuse precisely the existing mean, initialization and parameter groups."""
    method = config["method"]
    dataset = config["dataset"]
    new_cls = METHOD_CLASSES[dataset][method]
    return new_cls(old_objective.model, num_types=len(config["lags"]))
