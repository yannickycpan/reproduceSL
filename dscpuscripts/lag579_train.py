"""Use the unchanged dataset training functions with an isolated configuration namespace."""
from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
from types import FunctionType

from .lag579_protocol import validate_config
from .typed_ablation_train import original_trainer, original_config, from_original_typed

_SCIENTIFIC_DEPENDENCIES = (
    "dscpuscripts/lag579_protocol.py",
    "dscpuscripts/lag579_train.py",
    "dscpuscripts/typed_ablation_train.py",
    "dscpuscripts/typed_ablation_objectives.py",
    "dscpuscripts/application_objectives.py",
)


# The FD001 five/seven/nine-lag models were completed before the
# Electricity-only manifest change. Keep their exact recorded code identities
# ONLY if all previously hashed scientific files are byte-for-byte unchanged.
_FD001_LEGACY_CODES = {
    "td_typed": "afe696da123d0cb443e93d5bb4cbea483c80d12e9a0209e48a4f29bbfb0a0e33",
    "td_semi_typed": "45bc20a2918263763b1774ccc2fd2e6351f6aa0f735e6bdde33444ffb4763722",
    "td_nonnegative_typed": "d28ba41bb85c3c9c98a8c3634711ca564720b86bb02da2b9c322ac94170dedba",
}
_FD001_FROZEN_SOURCES = {
    "dscpuscripts/fd001_protocol.py": "c282216d27d4c74d10829a490bc22890972b21f5",
    "dscpuscripts/fd001_train.py": "c9152bfa9d69bb1329b1f4e6eb322d7fd99a15bf",
    "dscpuscripts/fd001_objectives.py": "a9441639989327e012dfb5ab5772aec77ad131c7",
    "dscpuscripts/common.py": "ee54413a29b2e27b4cc45befb0566962e258df40",
    "agents/fulltd_objectives.py": "1017a3409ec2f387961b99371e6e94e4b5ebfc1e",
    "agents/fulltd_mse.py": "1827c00fb1633cd010282d5a11f222b7a77d38b1",
    "agents/fulltd_gaussian_nll.py": "4d0c081f87f5066e200cf00504b461bd45f0b98e",
    "fulltd/protocol.py": "6160aa0f9fd1ccdc972cf61b2021c376e664b1ad",
    "fulltd/train.py": "41a5dff57aea41a08da6b9667918850e0eb2d443",
    "fulltd/data.py": "ae2efb47b170c0066194b72ed1362718824378c8",
    "fulltd/metrics.py": "0b8bf318f3cebc62d9dc9b0e4b385e332a3da78a",
    "fulltd/models.py": "31d6de173dd2313c58d63b275dce6b1034f955d8",
    "fulltd/sampling.py": "23b3e958af72bdb7e0140eb5e98ed14bdeaa7dc2",
    "dscpuscripts/typed_ablation_train.py": "2431adedf9d9906686bf81a6da924a0503f13e47",
    "dscpuscripts/typed_ablation_objectives.py": "03d9722a76ee4ff77e7026fcfc913f3a138da4b0",
    "dscpuscripts/application_objectives.py": "017ecc14a0a69ec89d5733656f243a9767d70f7d",
}

@lru_cache(maxsize=1)
def _check_fd001_compatibility():
    root = Path(__file__).resolve().parents[1]

    def git_blob_sha(content):
        return hashlib.sha1(b"blob " + str(len(content)).encode("ascii") + b"\0" + content).hexdigest()

    for relative, expected in _FD001_FROZEN_SOURCES.items():
        source = (root / relative).read_bytes().replace(b"\r\n", b"\n")
        if git_blob_sha(source) != expected:
            raise RuntimeError(f"FD001 scientific source changed: {relative}")

    # Reconstruct ONLY the historical Electricity-only settings text. The
    # original protocol's Git blob must still match exactly, proving its FD001
    # configuration/algorithm portion has not changed.
    protocol = (root / "dscpuscripts/lag579_protocol.py").read_text(encoding="utf-8").replace("\r\n", "\n")
    patch5 = "    \"electricity\": (\n        (1, 12, 24, 36, 48),\n    ),"
    previous = "    \"electricity\": (\n        (1, 12, 24, 36, 48),\n        (1, 12, 24, 36, 48, 72, 96),\n        (1, 12, 24, 36, 48, 72, 96, 120, 168),\n    ),"
    patchloop = '        for lags in (LAG_SETS[dataset][1:] if dataset == "fd001" else LAG_SETS[dataset]):'
    patchcount = "    expected_count = 750 if dataset == \"fd001\" else 375\n    if len(result) != expected_count or len({c[\"run_id\"] for c in result}) != expected_count:\n        raise AssertionError(f\"Expected {expected_count} distinct candidates for {dataset}\")"
    if (protocol.count(patch5) != 1 or protocol.count(patchloop) != 1
            or protocol.count(patchcount) != 1):
        raise RuntimeError("FD001 protocol compatibility cannot be verified")
    historical = protocol.replace(patch5, previous).replace(
        patchloop, '        for lags in LAG_SETS[dataset][1:]:').replace(
        patchcount, "    if len(result) != 750 or len({c[\"run_id\"] for c in result}) != 750:\n        raise AssertionError(\"Expected 750 unique new candidates per dataset\")")
    if git_blob_sha(historical.encode("utf-8")) != "284265a693295c20a8f92a5c61f3b2edf62482a6":
        raise RuntimeError("FD001 protocol changed; refusing stale checkpoint identity")


def code_hash(config):
    if config["dataset"] == "fd001":
        _check_fd001_compatibility()
        return _FD001_LEGACY_CODES[config["method"]]
    trainer = original_trainer(config["dataset"])
    base = (trainer.code_hash() if config["dataset"] == "fd001"
            else trainer.code_hash(original_config(config)))
    h = hashlib.sha256()
    h.update(b"lag579-v1\0")
    h.update(base.encode("ascii"))
    h.update(config["method"].encode("ascii"))
    root = Path(__file__).resolve().parents[1]
    for relative in _SCIENTIFIC_DEPENDENCIES:
        h.update(relative.encode("ascii") + b"\0")
        h.update((root / relative).read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()


def build_objective(config, dataset=None):
    trainer = original_trainer(config["dataset"])
    old = (trainer.build_objective(original_config(config)) if config["dataset"] == "fd001"
           else trainer.build_objective(original_config(config), dataset))
    if config["method"] == "td_typed":
        return old
    return from_original_typed(config, old)


def train_candidate(config, train_data, val_data, results_dir, **kwargs):
    validate_config(config)
    trainer = original_trainer(config["dataset"])
    old = trainer.train_candidate
    private_globals = dict(old.__globals__)
    private_globals["validate_config"] = validate_config
    private_globals["build_objective"] = build_objective
    if config["dataset"] == "fd001":
        private_globals["PROTOCOL_ID"] = config["protocol_id"]
        private_globals["code_hash"] = lambda: code_hash(config)
    else:
        private_globals["code_hash"] = code_hash
    isolated = FunctionType(old.__code__, private_globals, old.__name__,
                            old.__defaults__, old.__closure__)
    isolated.__kwdefaults__ = old.__kwdefaults__
    return isolated(config, train_data, val_data, results_dir, **kwargs)
