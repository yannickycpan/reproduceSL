"""Exactly 140 additional FD001 runs; original 180 configs are never modified.

Old representation: 2 extra predictor rates x 4 operator rates x 5 seeds = 40.
New pairwise head: 5 predictor rates x 4 operator rates x 5 seeds = 100.
All other training, data, sampling, and validation settings match Gamma v1.
"""
from __future__ import annotations

from fd001_gamma.protocol import (BETAS, EVALUATIONS, LAGS, SEEDS,
                                  SELECTION_EPOCHS, digest)
from fd001_gamma.protocol import make_manifest as original_manifest
from . import PROTOCOL_ID

ADDITIONAL_ALPHAS = (0.00003, 0.0001)
PAIRWISE_ALPHAS = (0.00003, 0.0001, 0.0003, 0.001, 0.003)
VARIANTS = ("representation", "pairwise")


def make_manifest():
    original = original_manifest()[0]
    template = {k: v for k, v in original.items()
                if k not in ("run_id", "protocol_id", "gamma_variant",
                             "alpha", "beta", "configuration", "seed")}
    runs = []
    for variant, alphas in (("representation", ADDITIONAL_ALPHAS),
                            ("pairwise", PAIRWISE_ALPHAS)):
        for alpha_i, alpha in enumerate(alphas):
            for beta_i, beta in enumerate(BETAS):
                for seed in SEEDS:
                    runs.append(dict(template,
                        run_id=f"fd001_gamma_ext_v1_{variant}_b128_a{alpha_i}_b{beta_i}_s{seed}",
                        protocol_id=PROTOCOL_ID,
                        gamma_variant=variant,
                        alpha=alpha, beta=beta,
                        configuration=alpha_i * len(BETAS) + beta_i, seed=seed))
    assert len(runs) == 140
    assert len({r["run_id"] for r in runs}) == len(runs)
    assert not ({r["run_id"] for r in runs} &
                {r["run_id"] for r in original_manifest()})
    return runs


def validate_config(config):
    expected = {r["run_id"]: r for r in make_manifest()}
    if not isinstance(config, dict) or config.get("run_id") not in expected:
        raise ValueError("Not part of the frozen 140-run FD001 extension manifest")
    if config != expected[config["run_id"]]:
        raise ValueError("Extension config differs from frozen protocol")
    return config
