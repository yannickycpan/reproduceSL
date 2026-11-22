"""Independent representation-based Pairwise TD neural experiments.

Uses the existing application Pairwise objective/trainer without modifications.
Baselines remain in the original application result roots.
"""
from __future__ import annotations

from functools import lru_cache

DATASETS = ("electricity", "metr_la")
METHOD = "td_pairwise"
BATCH_SIZES = (128, 256)
VERSION = "pairwise-nn-v1"
LAG_SETS = {
    "electricity": ((1, 12, 24, 36, 48),),
    "metr_la": ((1, 3), (1, 3, 6), (1, 3, 6, 12, 24)),
}


@lru_cache(maxsize=2)
def make_manifest(dataset):
    if dataset not in DATASETS:
        raise ValueError("Unsupported pairwise dataset")
    from .application_protocol import make_manifest as original_manifest
    originals = [c for c in original_manifest()
                 if c["dataset"] == dataset and c["method"] == METHOD
                 and c["batch_size"] == 256
                 and tuple(c["lags"]) in LAG_SETS[dataset]]
    expected_originals = 125 * len(LAG_SETS[dataset])
    if len(originals) != expected_originals:
        raise ValueError("Original 25-setting x 5-seed Pairwise grid is incomplete")
    result = []
    for original in originals:
        for batch_size in BATCH_SIZES:
            c = dict(original)
            c["batch_size"] = batch_size
            c["protocol_id"] = f"{original['protocol_id']}-{VERSION}"
            c["run_id"] = (original["run_id"].replace("_b256_", f"_b{batch_size}_")
                           + "_pairwisennv1")
            if c["run_id"] == original["run_id"]:
                raise AssertionError("Must not overwrite historical Pairwise runs")
            result.append(c)
    expected = expected_originals * len(BATCH_SIZES)
    if len(result) != expected or len({c["run_id"] for c in result}) != expected:
        raise AssertionError("Pairwise candidate count or identifier collision")
    for c in result:
        if (c["method"] != METHOD or c["batch_size"] not in BATCH_SIZES
                or tuple(c["lags"]) not in LAG_SETS[dataset]
                or c["seed"] not in range(5) or c["configuration"] not in range(25)):
            raise ValueError("Unexpected Pairwise configuration")
    return tuple(result)


@lru_cache(maxsize=2)
def _by_id(dataset):
    return {c["run_id"]: c for c in make_manifest(dataset)}


def validate_config(config):
    if not isinstance(config, dict) or config.get("dataset") not in DATASETS:
        raise ValueError("Unsupported Pairwise configuration")
    if _by_id(config["dataset"]).get(config.get("run_id")) != config:
        raise ValueError("Configuration does not match the frozen Pairwise scan")
    return config
