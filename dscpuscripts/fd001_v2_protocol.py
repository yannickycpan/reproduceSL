"""Fresh FD001 four-method sweep, revision v2b: 600 independent runs.

This new configuration namespace never overwrites old v2 or Gamma checkpoints.
The only modifications to the original training procedure are the requested
validation and full-checkpoint cadence; optimizer, objective, sampling, data,
backbones, and number of training epochs remain fixed.
"""
from __future__ import annotations

import math
import os

from fulltd.protocol import digest

PROTOCOL_ID = "fd001-four-methods-update-v2b"
SEEDS = tuple(range(5))
EPOCHS = 100
SELECTION_FRACTION = 0.25
# This is an immutable per-run scientific setting. Do not change it after
# launching a campaign; every config and checkpoint identity includes it.
EVAL_UPDATE_PERIOD = int(os.environ.get("FULLTD_FD001_EVAL_EVERY", "500"))
if EVAL_UPDATE_PERIOD < 1:
    raise ValueError("FULLTD_FD001_EVAL_EVERY must be positive")

BETAS = (0.00003, 0.0001, 0.0003, 0.001, 0.003)
BASELINE_ALPHAS = (0.0001, 0.0003, 0.0005, 0.0007, 0.0009,
                   0.001, 0.003, 0.005, 0.007, 0.01)
TD_ALPHAS = (0.00003, 0.0001, 0.0003, 0.001, 0.003)
METHODS = ("mse", "gaussian_nll", "td_typed", "td_pairwise")
ALPHAS = dict(mse=BASELINE_ALPHAS, gaussian_nll=BASELINE_ALPHAS,
              td_typed=TD_ALPHAS, td_pairwise=TD_ALPHAS)
BATCH_SIZES = dict(mse=(256,), gaussian_nll=(256,),
                   td_typed=(128, 256), td_pairwise=(128, 256))


def selection_update_steps(total_updates, period=EVAL_UPDATE_PERIOD):
    """Last 25% of post-start 0/period/2period/... validation checkpoints.

    Update zero is the initial diagnostic, never part of HP selection.
    Selection is computed from actually scheduled observations, not interpolated
    epochs or unevaluated final states.
    """
    if type(total_updates) is not int or total_updates < 1 or period < 1:
        raise ValueError("Expected positive integer update count and period")
    measured = list(range(period, total_updates + 1, period))
    if not measured:
        raise ValueError("Campaign must contain at least one post-start evaluation")
    n = math.ceil(SELECTION_FRACTION * len(measured))
    return tuple(measured[-n:])


def make_manifest():
    runs = []
    for method in METHODS:
        rate_pairs = [(alpha, beta) for alpha in ALPHAS[method]
                      for beta in (BETAS if method.startswith("td_") else (None,))]
        for batch_size in BATCH_SIZES[method]:
            for setting, (alpha, beta) in enumerate(rate_pairs):
                for seed in SEEDS:
                    runs.append(dict(
                        run_id=f"fd001_v2b_{method}_b{batch_size}_h{setting}_s{seed}",
                        protocol_id=PROTOCOL_ID, dataset="fd001", backbone="tcn",
                        method=method, alpha=alpha, beta=beta,
                        batch_size=batch_size, configuration=setting, seed=seed,
                        epochs=EPOCHS, source_sampling="random_reshuffling",
                        optimizer="adam", adam_betas=[0.9, 0.999], adam_eps=1e-8,
                        weight_decay=0, gradient_clipping=None, learning_rate_schedule=None,
                        selection_fraction=SELECTION_FRACTION,
                        selection="last_quarter_of_positive_update_evaluations",
                        learning_curve_updates=EVAL_UPDATE_PERIOD,
                        snapshot_updates=EVAL_UPDATE_PERIOD,
                        validation_sampling="full", lags=[1, 5],
                        checkpoint_policy="update_snapshots_and_resumable_cursor"))
    assert len(runs) == 600
    assert len({c["run_id"] for c in runs}) == 600
    return runs


def validate_config(config):
    expected = {c["run_id"]: c for c in make_manifest()}
    if (not isinstance(config, dict) or config.get("run_id") not in expected
            or expected[config["run_id"]] != config):
        raise ValueError("Configuration is not in the frozen fresh FD001 600-run sweep")
    return config
