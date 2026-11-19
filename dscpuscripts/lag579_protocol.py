"""Prescribed 5/7/9 settings; only 7/9 require new training runs."""
from __future__ import annotations

from functools import lru_cache

DATASETS = ("fd001", "electricity")
METHODS = ("td_typed", "td_semi_typed", "td_nonnegative_typed")
VERSION = "lag579-v1"
LAG_SETS = {
    "fd001": (
        (1, 3, 5, 10, 15),
        (1, 3, 5, 10, 15, 20, 30),
        (1, 3, 5, 10, 15, 20, 30, 40, 60),
    ),
    "electricity": (
        (1, 12, 24, 36, 48),
        (1, 12, 24, 36, 48, 72, 96),
        (1, 12, 24, 36, 48, 72, 96, 120, 168),
    ),
}


@lru_cache(maxsize=2)
def _five_lag_reference(dataset):
    if dataset == "fd001":
        from .fd001_protocol import make_manifest
    elif dataset == "electricity":
        from .application_protocol import make_manifest
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    original = [
        cfg for cfg in make_manifest()
        if cfg["dataset"] == dataset
        and cfg["method"] == "td_typed"
        and cfg["batch_size"] == 256
        and tuple(cfg["lags"]) == LAG_SETS[dataset][0]
    ]
    if len(original) != 125:
        raise AssertionError(f"Expected 125 original B256 five-lag settings: {dataset}")
    if len({(c["configuration"], c["seed"]) for c in original}) != 125:
        raise AssertionError("Original five-lag grid is not complete")
    return tuple(original)


def make_manifest(dataset):
    if dataset not in DATASETS:
        raise ValueError(f"Unsupported dataset: {dataset}")
    result = []
    for original in _five_lag_reference(dataset):
        prefix, marker, _ = original["run_id"].rpartition("_lags")
        if not marker or "_td_typed_" not in prefix:
            raise AssertionError("Unexpected reference run identifier")
        for lags in LAG_SETS[dataset][1:]:
            for method in METHODS:
                config = dict(original)
                config["method"] = method
                config["lags"] = list(lags)
                config["protocol_id"] = f"{original['protocol_id']}-{VERSION}"
                config["run_id"] = (
                    prefix.replace("_td_typed_", f"_{method}_")
                    + "_lags" + "-".join(str(x) for x in lags)
                    + "_lag579v1"
                )
                result.append(config)
    if len(result) != 750 or len({c["run_id"] for c in result}) != 750:
        raise AssertionError("Expected 750 unique new candidates per dataset")
    return result


@lru_cache(maxsize=2)
def _by_id(dataset):
    return {c["run_id"]: c for c in make_manifest(dataset)}


def validate_config(config):
    if not isinstance(config, dict) or config.get("dataset") not in DATASETS:
        raise ValueError("Unknown configuration")
    expected = _by_id(config["dataset"]).get(config.get("run_id"))
    if expected != config:
        raise ValueError("Configuration does not match the frozen manifest")
    return config
