"""Add-only, existing 2/3/5-lag, B256 typed-TD mechanism comparisons.

Existing MSE and full typed TD runs are comparison references, NOT rerun here.
All original manifest files, protocol IDs and scientific source files are frozen.
"""
from __future__ import annotations

from functools import lru_cache

METHODS = ("td_semi_typed", "td_nonnegative_typed")
DATASETS = ("fd001", "electricity", "metr_la")
VERSION = "typed-mechanism-ablation-v2-lag-matched"


@lru_cache(maxsize=3)
def _frozen_configs(dataset):
    if dataset == "fd001":
        from dscpuscripts.fd001_protocol import make_manifest, LAG_SETS
        original = make_manifest()
        lag_sets = {tuple(lags) for lags in LAG_SETS}
    elif dataset in ("electricity", "metr_la"):
        from dscpuscripts.application_protocol import make_manifest, LAG_SETS
        original = make_manifest()
        lag_sets = {tuple(lags) for lags in LAG_SETS[dataset]}
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    configs = tuple(c for c in original if c["dataset"] == dataset
                    and c["method"] == "td_typed" and c["batch_size"] == 256
                    and tuple(c["lags"]) in lag_sets)
    if len(configs) != 375:
        raise AssertionError(f"Expected exactly 375 original B256 typed configurations for {dataset}")
    if {tuple(c["lags"]) for c in configs} != lag_sets:
        raise AssertionError(f"Missing an original typed TD lag set for {dataset}")
    return configs


def make_manifest(dataset):
    configs = []
    for old in _frozen_configs(dataset):
        for method in METHODS:
            config = dict(old)
            config["method"] = method
            config["protocol_id"] = f"{old['protocol_id']}-{VERSION}"
            config["run_id"] = old["run_id"].replace("_td_typed_", f"_{method}_") + "_ablationv2"
            if config["run_id"] == old["run_id"]:
                raise AssertionError("Ablation run identifier must differ from the original")
            configs.append(config)
    if len(configs) != 750 or len({c["run_id"] for c in configs}) != 750:
        raise AssertionError("Expected 750 distinct new candidate runs per dataset")
    return configs


@lru_cache(maxsize=3)
def _manifest_by_id(dataset):
    return {c["run_id"]: c for c in make_manifest(dataset)}


def validate_config(config):
    if not isinstance(config, dict) or config.get("dataset") not in DATASETS:
        raise ValueError("Unsupported typed-ablation configuration")
    reference = _manifest_by_id(config["dataset"]).get(config.get("run_id"))
    if reference is None or reference != config:
        raise ValueError("Configuration is not in the immutable typed-ablation manifest")
    return config
