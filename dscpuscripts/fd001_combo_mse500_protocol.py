"""Additional FD001 Combo manifest, on fullTD, preserving all legacy runs.

Stage 1 is independently retrained MSE at alpha=0.007 for exactly 500 Adam
updates (partway through its tenth epoch on official FD001). Later Gamma and
TD stages scan the exact original beta/alpha/batch/lag/seed settings.
"""
from __future__ import annotations
from dscpuscripts.fd001_protocol import (
    make_manifest as original_manifest, BASELINE_ALPHAS,
)
PROTOCOL_ID = "fd001-extra-combo-mse007-500updates-v1"
MSE_STAGE_ALPHA = 0.007
MSE_STAGE_UPDATES = 500
RUN_SUFFIX = "_mseh8_mse500v1"


def make_manifest():
    if BASELINE_ALPHAS[8] != MSE_STAGE_ALPHA:
        raise ValueError("MSE alpha index 8 changed")
    rows = []
    for orig in original_manifest():
        if orig["method"] != "td_mse_combo":
            continue
        cfg = dict(orig)
        cfg.update(protocol_id=PROTOCOL_ID, mse_alpha=MSE_STAGE_ALPHA,
                   mse_max_updates=MSE_STAGE_UPDATES,
                   mse_stop_rule="fixed_optimizer_updates")
        del cfg["mse_epochs"]
        cfg["run_id"] = (
            f"fd001_td_mse_combo_b{cfg['batch_size']}_h{cfg['configuration']}"
            f"_s{cfg['seed']}_lags{'-'.join(map(str, cfg['lags']))}{RUN_SUFFIX}")
        rows.append(cfg)
    if len(rows) != 750 or len(set(x["run_id"] for x in rows)) != 750:
        raise AssertionError("Expected exactly 750 distinct new Combo runs")
    return rows


def validate_config(config):
    candidates = {x["run_id"]: x for x in make_manifest()}
    if (not isinstance(config, dict)
            or candidates.get(config.get("run_id")) != config):
        raise ValueError("Not in new FD001 Combo MSE-500 experiment manifest")
    return config
