"""Dataset-scoped, explicit Electricity and METR-LA experimental manifests.

FD001 is independent and unchanged. The METR-LA source fraction defaults to
10%, with 100 stochastic rounds. A 100% fraction recovers the exact original
full-permutation training setting. Electricity retains its prior protocol.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from . import electricity_sampling, metr_la_sampling

SEEDS = tuple(range(5))
EPOCHS = 100
ELECTRICITY_ROUNDS = electricity_sampling.ELECTRICITY_ROUNDS
METR_LA_SOURCE_FRACTION = metr_la_sampling.SOURCE_FRACTION
METR_LA_EVAL_PERIOD = metr_la_sampling.evaluation_period(METR_LA_SOURCE_FRACTION)
PROTOCOL_IDS = {
    "electricity": "applications-fd001-aligned-v3-electricity-10passes",
    "metr_la": "metr-la-configurable-fraction-v1",
}
# Kept as an imported compatibility alias for legacy reporting tests; scientific
# checkpoints always use the dataset-specific config["protocol_id"].
PROTOCOL_ID = PROTOCOL_IDS["metr_la"]
BETAS = (0.00003, 0.0001, 0.001, 0.003, 0.01)
BASELINE_ALPHAS = (0.0001, 0.0003, 0.0005, 0.0007, 0.0009,
                   0.001, 0.003, 0.005, 0.007, 0.01)
TD_ALPHAS = (0.00003, 0.0001, 0.0003, 0.001, 0.003)
EVAL_UPDATE_PERIOD = {"electricity": 5000, "metr_la": METR_LA_EVAL_PERIOD}
LAGS = {"electricity": (1, 24), "metr_la": (1, 3)}
PRINCIPAL_METHODS = ("mse", "gaussian_nll", "td_typed", "td_pairwise")
MATRIX = (
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
        sampled = dataset == "metr_la" and METR_LA_SOURCE_FRACTION < 1
        for method in methods:
            td = method.startswith("td_")
            rates = [(a, b) for a in (TD_ALPHAS if td else BASELINE_ALPHAS)
                     for b in (BETAS if td else (None,))]
            for setting, (alpha, beta) in enumerate(rates):
                for seed in SEEDS:
                    name = f"{dataset}_{backbone}_{method}_b256_h{setting}_s{seed}"
                    if dataset == "metr_la":
                        name += "_" + metr_la_sampling.fraction_tag(METR_LA_SOURCE_FRACTION)
                    config = dict(
                        run_id=name, protocol_id=PROTOCOL_IDS[dataset],
                        dataset=dataset, backbone=backbone, method=method,
                        alpha=alpha, beta=beta, batch_size=256,
                        configuration=setting, seed=seed,
                        epochs=ELECTRICITY_ROUNDS if dataset == "electricity" else EPOCHS,
                        training_unit="stochastic_round" if (dataset == "electricity" or sampled)
                                      else "full_epoch",
                        source_sampling=("iid_uniform_with_replacement" if (dataset == "electricity" or sampled)
                                         else "random_reshuffling"),
                        source_divisor=24 if dataset == "electricity" else 1,
                        optimizer="adam", adam_betas=[0.9, 0.999], adam_eps=1e-8,
                        weight_decay=0, gradient_clipping=None, learning_rate_schedule=None,
                        learning_curve_updates=EVAL_UPDATE_PERIOD[dataset],
                        snapshot_updates=EVAL_UPDATE_PERIOD[dataset],
                        validation_sampling=("fixed_customer_stratified_without_replacement"
                                             if dataset == "electricity" else "full"),
                        validation_fraction=0.2 if dataset == "electricity" else 1.0,
                        lags=list(LAGS[dataset]),
                        checkpoint_policy="update_snapshots_and_resumable_cursor")
                    if dataset == "metr_la":
                        config["source_fraction"] = METR_LA_SOURCE_FRACTION
                    runs.append(config)
    if len(runs) != 875 or len({r["run_id"] for r in runs}) != 875:
        raise AssertionError("Expected 350 Electricity and 525 METR-LA runs")
    return runs


def validate_config(config):
    expected = {r["run_id"]: r for r in make_manifest()}
    if (not isinstance(config, dict) or config.get("run_id") not in expected
            or config != expected[config["run_id"]]):
        raise ValueError("Configuration is not in the prescribed application sweep")
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


def sources_for_application_round(dataset, n_sources, seed, round_number, fraction=None):
    """Only dispatch; actual source laws are isolated in dataset-specific modules."""
    if dataset == "electricity":
        return electricity_sampling.sources_for_round(n_sources, seed, round_number)
    if dataset == "metr_la":
        f = METR_LA_SOURCE_FRACTION if fraction is None else fraction
        return metr_la_sampling.sources_for_round(n_sources, seed, round_number, f)
    raise ValueError("Unknown application dataset")


def update_budgets(config, n_sources):
    """Logical-batch update budgets with dataset-specific source draws."""
    if config["dataset"] == "electricity":
        draws = electricity_sampling.draws_per_round(n_sources)
    elif config["dataset"] == "metr_la":
        draws = metr_la_sampling.draws_per_round(n_sources, config["source_fraction"])
    else:
        raise ValueError("Unknown application dataset")
    per_round = math.ceil(draws / config["batch_size"])
    matched = config["epochs"] * math.ceil(draws / 256)
    return dict(draws_per_round=draws, updates_per_epoch=per_round,
                total_updates=config["epochs"] * per_round, matched_updates=matched)


def evaluation_steps(total_updates, eval_period, matched_updates=None):
    """Include update 0 and final budget, even when an endpoint is off-grid."""
    if type(total_updates) is not int or total_updates < 1 or type(eval_period) is not int or eval_period < 1:
        raise ValueError("Positive update budget and evaluation period required")
    steps = {0, total_updates, *range(eval_period, total_updates + 1, eval_period)}
    if matched_updates is not None:
        if type(matched_updates) is not int or not 0 < matched_updates <= total_updates:
            raise ValueError("Matched-update endpoint must lie inside training budget")
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
        (destination / (name + ".json")).write_text(
            json.dumps(subset, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(manifest)} application candidates")


if __name__ == "__main__":
    main()
