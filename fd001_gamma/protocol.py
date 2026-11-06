"""The three categorical Gamma choices and the fixed 3 x 4 rate grid."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from . import PROTOCOL_ID

VARIANTS = ("typed", "raw26", "representation")
ALPHAS = (3e-4, 1e-3, 3e-3)
BETAS = (3e-5, 1e-4, 3e-4, 1e-3)
SEEDS = tuple(range(5))
EVALUATIONS = tuple(range(10, 101, 10))
SELECTION_EPOCHS = (80, 90, 100)
LAGS = (1, 5)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def make_manifest():
    runs = []
    for variant in VARIANTS:
        for ai, alpha in enumerate(ALPHAS):
            for bi, beta in enumerate(BETAS):
                for seed in SEEDS:
                    runs.append(dict(
                        run_id=f"fd001_gamma_v1_{variant}_b128_a{ai}_b{bi}_s{seed}",
                        protocol_id=PROTOCOL_ID, dataset="fd001", backbone="tcn", method="td_full",
                        gamma_variant=variant, alpha=alpha, beta=beta,
                        configuration=ai * len(BETAS) + bi, seed=seed, batch_size=128,
                        epochs=100, training_unit="full_epoch", source_sampling="random_reshuffling",
                        lags=list(LAGS), gamma_initialization="zero", gamma_output="unrestricted_signed",
                        optimizer="adam", adam_betas=[0.9, 0.999], adam_eps=1e-8,
                        weight_decay=0, gradient_clipping=None, learning_rate_schedule=None,
                        checkpoint_policy="epoch_end_only", validation_sampling="full",
                        validation_epochs=list(EVALUATIONS), selection_epochs=list(SELECTION_EPOCHS),
                        selection="five_seed_pooled_late_validation_mse"))
    return runs


def validate_config(config):
    expected = {r["run_id"]: r for r in make_manifest()}
    if not isinstance(config, dict) or config.get("run_id") not in expected or config != expected[config["run_id"]]:
        raise ValueError("Configuration differs from the frozen FD001 Gamma sweep")
    return config


def validate_manifest(runs, *, complete=False):
    if not isinstance(runs, list) or not runs:
        raise ValueError("Manifest must be a nonempty list")
    for run in runs:
        validate_config(run)
    ids = [r["run_id"] for r in runs]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate run IDs")
    if complete and set(ids) != {r["run_id"] for r in make_manifest()}:
        raise ValueError("Joint selection requires all 180 canonical runs")
    return runs


def load_manifest(path, *, complete=False):
    return validate_manifest(json.loads(Path(path).read_text()), complete=complete)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    path = Path(args.output)
    payload = make_manifest()
    if path.exists() and json.loads(path.read_text()) != payload:
        raise ValueError(f"Refusing to overwrite a different manifest: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"{len(payload)} runs: 3 variants x 12 rate pairs x 5 seeds; logical batch 128")


if __name__ == "__main__":
    main()
