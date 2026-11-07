"""Isolated extension context: use the original epoch trainer without editing it.

The original 180-run scientific hash and checkpoints must remain unchanged.
Only extension workers temporarily bind the new frozen protocol, objective
factory, and hash to the EXISTING identical FD001 training machinery.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import os
from pathlib import Path

from fd001_gamma.common import campaign_paths as original_paths
from fd001_gamma.common import scientific_hash as original_scientific_hash
from . import PROTOCOL_ID
from .objectives import make_objective
from .protocol import make_manifest, validate_config


def campaign_paths():
    data, _, _ = original_paths()
    base = Path.home() / "scratch" / "fulltd" / "fd001_gamma_extension_v1"
    results = Path(os.environ.get("FULLTD_FD001_GAMMA_EXT_RESULTS_DIR",
                                  base / "results")).expanduser().resolve()
    output = Path(os.environ.get("FULLTD_FD001_GAMMA_EXT_OUTPUT_DIR",
                                 base / "outputs")).expanduser().resolve()
    return data, results, output


def scientific_hash():
    root = Path(__file__).resolve().parents[1]
    h = hashlib.sha256()
    h.update(b"original-fd001-gamma-science:")
    h.update(original_scientific_hash().encode("ascii"))
    for name in ("fd001_gamma_ext/__init__.py",
                 "fd001_gamma_ext/protocol.py",
                 "fd001_gamma_ext/objectives.py",
                 "fd001_gamma_ext/runtime.py"):
        h.update(name.encode("utf-8"))
        h.update((root / name).read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()


@contextmanager
def bind_original_training_engine():
    """Apply extension bindings for one worker only, restoring them afterward."""
    import fd001_gamma.train as trainer
    fields = dict(PROTOCOL_ID=PROTOCOL_ID, validate_config=validate_config,
                  make_objective=make_objective, scientific_hash=scientific_hash)
    previous = {name: getattr(trainer, name) for name in fields}
    try:
        for name, value in fields.items():
            setattr(trainer, name, value)
        yield trainer
    finally:
        for name, value in previous.items():
            setattr(trainer, name, value)


def worker(args):
    from fd001_gamma.common import cpu_environment
    cpu_environment()
    os.environ["PYTHONUNBUFFERED"] = "1"
    if scientific_hash() != args.expected_code_hash:
        raise ValueError("Extension scientific implementation changed since dispatch")
    configs = {c["run_id"]: c for c in make_manifest()}
    config = configs.get(args.run_id)
    if config is None:
        raise ValueError(f"Unknown extension run ID: {args.run_id}")
    import signal
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    stopping = False

    def request_stop(signum, frame):
        nonlocal stopping
        stopping = True

    previous_signals = {}
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        previous = signal.getsignal(sig)
        if sig == signal.SIGHUP and previous == signal.SIG_IGN:
            continue
        previous_signals[sig] = signal.signal(sig, request_stop)
    try:
        with bind_original_training_engine() as trainer:
            result = trainer.run_config(config, args.data_dir, args.results_dir, device="cpu",
                                        microbatch=args.microbatch, stop_requested=lambda: stopping)
    finally:
        for sig, handler in previous_signals.items():
            signal.signal(sig, handler)
    state = result.get("state")
    print(f"EXTENSION WORKER: {config['run_id']} -> {state}", flush=True)
    if state in ("complete", "numerical_failure"):
        return 0
    if stopping and state in ("checkpointed", "stopped_no_checkpoint"):
        return 75
    raise RuntimeError(f"Unexpected extension worker outcome: {state}")
