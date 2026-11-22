"""Isolate Pairwise run identities while reusing the original trainer bytecode."""
from __future__ import annotations

import hashlib
from pathlib import Path
from types import FunctionType

from .pairwise_protocol import validate_config

def _original_trainer():
    # Delay torch import until the CPU worker has set thread env variables.
    from . import application_train
    return application_train


_SCIENTIFIC_DEPENDENCIES = (
    "dscpuscripts/pairwise_protocol.py",
    "dscpuscripts/pairwise_train.py",
)


def code_hash(config):
    validate_config(config)
    h = hashlib.sha256()
    h.update(b"pairwise-neural-v1\0")
    h.update(_original_trainer().code_hash(config).encode("ascii"))
    root = Path(__file__).resolve().parents[1]
    for relative in _SCIENTIFIC_DEPENDENCIES:
        h.update(relative.encode("ascii") + b"\0")
        h.update((root / relative).read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()


def build_objective(config, dataset):
    validate_config(config)
    return _original_trainer().build_objective(config, dataset)


def train_candidate(config, train_data, val_data, results_root, **kwargs):
    validate_config(config)
    old = _original_trainer().train_candidate
    private = dict(old.__globals__)
    private["validate_config"] = validate_config
    private["code_hash"] = code_hash
    isolated = FunctionType(old.__code__, private, old.__name__,
                            old.__defaults__, old.__closure__)
    isolated.__kwdefaults__ = old.__kwdefaults__
    return isolated(config, train_data, val_data, results_root, **kwargs)
