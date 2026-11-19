"""Use the unchanged dataset training functions with an isolated configuration namespace."""
from __future__ import annotations

import hashlib
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


def code_hash(config):
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
