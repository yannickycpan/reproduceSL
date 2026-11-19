"""Call the original FD001/application training loop with new typed objectives.

The original training functions, module globals, checkpoints, and hashes are
never altered. A private function-global namespace supplies the new objective,
config validator and run-specific scientific hash for new run IDs only.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from types import FunctionType

from .typed_ablation_objectives import from_original_typed
from .typed_ablation_protocol import validate_config


_SCIENTIFIC_ADDITIONS = (
    "dscpuscripts/typed_ablation_objectives.py",
    "dscpuscripts/typed_ablation_protocol.py",
    "dscpuscripts/typed_ablation_train.py",
    "dscpuscripts/application_objectives.py",
)


def original_trainer(dataset):
    if dataset == "fd001":
        from . import fd001_train
        return fd001_train
    if dataset in ("electricity", "metr_la"):
        from . import application_train
        return application_train
    raise ValueError(f"Unsupported dataset: {dataset}")


def original_config(config):
    c = dict(config)
    c["method"] = "td_typed"
    return c


def code_hash(config):
    """Bind new runs to original science plus all new scientific dependencies."""
    original = original_trainer(config["dataset"])
    base = (original.code_hash() if config["dataset"] == "fd001"
            else original.code_hash(original_config(config)))
    h = hashlib.sha256()
    h.update(b"typed-ablation-v2\0")
    h.update(base.encode("ascii"))
    h.update(config["method"].encode("ascii"))
    root = Path(__file__).resolve().parents[1]
    for path in _SCIENTIFIC_ADDITIONS:
        h.update(path.encode("ascii") + b"\0")
        h.update((root / path).read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()


def build_objective(config, dataset=None):
    original = original_trainer(config["dataset"])
    old = (original.build_objective(original_config(config)) if config["dataset"] == "fd001"
           else original.build_objective(original_config(config), dataset))
    new = from_original_typed(config, old)
    # No random initialization or alteration of the shared predictor.
    return new


def train_candidate(config, train_data, val_data, results_dir, **kwargs):
    """Use original training bytecode with isolated, explicitly scoped factories."""
    validate_config(config)
    original = original_trainer(config["dataset"])
    target = original.train_candidate
    env = dict(target.__globals__)
    env["validate_config"] = validate_config
    env["build_objective"] = build_objective
    if config["dataset"] == "fd001":
        # The old FD001 function uses its module-level PROTOCOL_ID, unlike applications.
        # Bind the new ID only in this private namespace; historical trainers stay intact.
        env["PROTOCOL_ID"] = config["protocol_id"]
        env["code_hash"] = lambda: code_hash(config)
    else:
        env["code_hash"] = code_hash
    isolated = FunctionType(target.__code__, env, target.__name__,
                            target.__defaults__, target.__closure__)
    isolated.__kwdefaults__ = target.__kwdefaults__
    return isolated(config, train_data, val_data, results_dir, **kwargs)
