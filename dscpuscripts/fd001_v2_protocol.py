"""Frozen fresh FD001 four-method sweep: 270 runs, version 2.

Old completed runs remain untouched. Validation/optimizer schedules are
independent: MSE-based HP selection at epochs 80/90/100, learning curves and
retained checkpoints at step 0 and every 500 optimizer updates.
"""
from __future__ import annotations

import os

from fulltd.protocol import digest

PROTOCOL_ID = "fd001-four-methods-update-v2"
SEEDS = tuple(range(5))
EPOCHS = 100
SELECT_EPOCHS = (80, 90, 100)
EPOCH_VALIDATIONS = tuple(range(10, 101, 10))
# Optional future experiment setting. It is retained inside every run config
# and therefore in the checkpoint identity; changing it cannot reuse old runs.
EVAL_UPDATE_PERIOD = int(os.environ.get("FULLTD_FD001_EVAL_EVERY", "500"))
if EVAL_UPDATE_PERIOD < 1:
    raise ValueError("FULLTD_FD001_EVAL_EVERY must be a positive integer")
BETAS = (0.00003, 0.0001, 0.0003, 0.001, 0.003)
BASELINE_ALPHAS = (0.0002, 0.0003, 0.0005, 0.0007, 0.001, 0.003, 0.01)
TYPED_ALPHAS = (0.0003, 0.001, 0.003)
PAIRWISE_ALPHAS = (0.00003, 0.0001, 0.0003, 0.001, 0.003)
METHODS = ("mse", "gaussian_nll", "td_typed", "td_pairwise")
ALPHAS = dict(mse=BASELINE_ALPHAS, gaussian_nll=BASELINE_ALPHAS,
              td_typed=TYPED_ALPHAS, td_pairwise=PAIRWISE_ALPHAS)
BATCH_SIZE = dict(mse=256, gaussian_nll=256, td_typed=128, td_pairwise=128)


def make_manifest():
    runs = []
    for method in METHODS:
        rates = [(alpha, beta) for alpha in ALPHAS[method]
                 for beta in (BETAS if method.startswith("td_") else (None,))]
        for setting, (alpha, beta) in enumerate(rates):
            for seed in SEEDS:
                runs.append(dict(
                    run_id=f"fd001_v2_{method}_b{BATCH_SIZE[method]}_h{setting}_s{seed}",
                    protocol_id=PROTOCOL_ID, dataset="fd001", backbone="tcn",
                    method=method, alpha=alpha, beta=beta,
                    batch_size=BATCH_SIZE[method], configuration=setting, seed=seed,
                    epochs=EPOCHS, source_sampling="random_reshuffling",
                    optimizer="adam", adam_betas=[0.9, 0.999], adam_eps=1e-8,
                    weight_decay=0, gradient_clipping=None, learning_rate_schedule=None,
                    selection_epochs=list(SELECT_EPOCHS),
                    validation_epochs=list(EPOCH_VALIDATIONS),
                    learning_curve_updates=EVAL_UPDATE_PERIOD,
                    snapshot_updates=EVAL_UPDATE_PERIOD,
                    validation_sampling="full", lags=[1, 5],
                    checkpoint_policy="update_snapshots_and_resumable_cursor"))
    assert len(runs) == 270
    assert len({c["run_id"] for c in runs}) == 270
    return runs


def validate_config(config):
    expected = {c["run_id"]: c for c in make_manifest()}
    if (not isinstance(config, dict) or config.get("run_id") not in expected
            or expected[config["run_id"]] != config):
        raise ValueError("Configuration is not in frozen FD001 v2 270-run sweep")
    return config
