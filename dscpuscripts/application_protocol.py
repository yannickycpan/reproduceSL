"""Current Electricity/METR-LA sweeps, aligned with the frozen FD001 design.

Applications now use B256-only TD and a revised operator grid. FD001 remains
frozen. Electricity's large source population necessitates:
100 rounds of ceil(N/24) draws, and a shared 20% validation view every 5,000
updates. Hyperparameter selection is a reporting decision, as it is in FD001.
This module deliberately does not import FD001's environment-dependent cadence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

PROTOCOL_ID = "applications-fd001-aligned-v2-b256-beta"
SEEDS = tuple(range(5))
EPOCHS = 100
BETAS = (0.00003, 0.0001, 0.001, 0.003, 0.01)
BASELINE_ALPHAS = (0.0001, 0.0003, 0.0005, 0.0007, 0.0009,
                   0.001, 0.003, 0.005, 0.007, 0.01)
TD_ALPHAS = (0.00003, 0.0001, 0.0003, 0.001, 0.003)
EVAL_UPDATE_PERIOD = {"electricity": 5000, "metr_la": 500}
LAGS = {"electricity": (1, 24), "metr_la": (1, 3)}
DATASET_SCOPES = ("all", "electricity", "metr_la")
PRINCIPAL_METHODS = ("mse", "gaussian_nll", "td_typed", "td_pairwise")
MATRIX = (
    ("electricity", "tcn", PRINCIPAL_METHODS),
    ("electricity", "patchtst", PRINCIPAL_METHODS),
    ("metr_la", "graph_wavenet", PRINCIPAL_METHODS + ("mae", "td_pairwise_mae")),
)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def datasets_for_scope(scope):
    if scope == "all":
        return ("electricity", "metr_la")
    if scope in ("electricity", "metr_la"):
        return (scope,)
    raise ValueError(f"Unknown application dataset scope: {scope}")


def make_manifest():
    runs = []
    for dataset, backbone, methods in MATRIX:
        for method in methods:
            td = method.startswith("td_")
            rates = [(a, b) for a in (TD_ALPHAS if td else BASELINE_ALPHAS)
                     for b in (BETAS if td else (None,))]
            for batch_size in (256,):
                for setting, (alpha, beta) in enumerate(rates):
                    for seed in SEEDS:
                        runs.append(dict(
                            run_id=f"{dataset}_{backbone}_{method}_b{batch_size}_h{setting}_s{seed}",
                            protocol_id=PROTOCOL_ID, dataset=dataset, backbone=backbone,
                            method=method, alpha=alpha, beta=beta, batch_size=batch_size,
                            configuration=setting, seed=seed, epochs=EPOCHS,
                            training_unit="stochastic_round" if dataset == "electricity" else "full_epoch",
                            source_sampling="iid_uniform_with_replacement" if dataset == "electricity" else "random_reshuffling",
                            source_divisor=24 if dataset == "electricity" else 1,
                            optimizer="adam", adam_betas=[0.9, 0.999], adam_eps=1e-8,
                            weight_decay=0, gradient_clipping=None, learning_rate_schedule=None,
                            learning_curve_updates=EVAL_UPDATE_PERIOD[dataset],
                            snapshot_updates=EVAL_UPDATE_PERIOD[dataset],
                            validation_sampling="fixed_customer_stratified_without_replacement" if dataset == "electricity" else "full",
                            validation_fraction=0.2 if dataset == "electricity" else 1.0,
                            lags=list(LAGS[dataset]),
                            checkpoint_policy="update_snapshots_and_resumable_cursor"))
    if len(runs) != 1225 or len({r["run_id"] for r in runs}) != len(runs):
        raise AssertionError("Expected 700 Electricity and 525 METR-LA candidates")
    return runs


def validate_config(config):
    expected = {r["run_id"]: r for r in make_manifest()}
    if (not isinstance(config, dict) or config.get("run_id") not in expected
            or config != expected[config["run_id"]]):
        raise ValueError("Configuration is not in the current FD001-aligned application sweep")
    return config


def load_manifest(path):
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(records, list) or any(not isinstance(r, dict) for r in records):
        raise ValueError("Manifest must be a list of explicit configurations")
    if len({r.get("run_id") for r in records}) != len(records):
        raise ValueError("Manifest run IDs must be unique")
    for record in records:
        validate_config(record)
    return records


def update_budgets(config, n_sources):
    """Nominal logical-batch budgets; actual Adam steps are recorded separately."""
    if not isinstance(n_sources, int) or n_sources < 1:
        raise ValueError("A positive integer source population is required")
    draws = math.ceil(n_sources / config["source_divisor"])
    total = config["epochs"] * math.ceil(draws / config["batch_size"])
    matched = config["epochs"] * math.ceil(draws / 256)
    return dict(draws_per_round=draws, updates_per_epoch=math.ceil(draws / config["batch_size"]),
                total_updates=total, matched_updates=matched)


def evaluation_steps(total_updates, eval_period, matched_updates=None):
    """Include the initial state and exact endpoints, even when off the grid."""
    if type(total_updates) is not int or total_updates < 1 or type(eval_period) is not int or eval_period < 1:
        raise ValueError("Positive integer update budget and evaluation period required")
    steps = {0, total_updates, *range(eval_period, total_updates + 1, eval_period)}
    if matched_updates is not None:
        if type(matched_updates) is not int or not 0 < matched_updates <= total_updates:
            raise ValueError("Matched-update endpoint must lie inside the training budget")
        steps.add(matched_updates)
    return tuple(sorted(steps))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="configs/fulltd")
    args = parser.parse_args()
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    manifest = make_manifest()
    for name in ("manifest", "electricity", "metr_la"):
        subset = manifest if name == "manifest" else [r for r in manifest if r["dataset"] == name]
        (destination / (name + ".json")).write_text(json.dumps(subset, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(manifest)} current Electricity/METR-LA candidates")


if __name__ == "__main__":
    main()
