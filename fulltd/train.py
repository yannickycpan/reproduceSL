"""Shared deterministic training utilities for current TD experiments.

FD001 and application training are implemented in dscpuscripts/;
this module intentionally has no standalone legacy experiment runner.
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import hashlib
import random

import numpy as np
import torch


def state_hash(state):
    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        h.update(key.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def tensor_batch(batch, start, stop, device):
    return {key: torch.as_tensor(np.asarray(batch[key][start:stop]), device=device,
                                dtype=torch.bool if key == "mask" else torch.float32)
            for key in ("x", "y", "mask")}


def logical_backward(objective, batch, partner, types, device, microbatch):
    """Accumulate sums / GLOBAL valid count, including unequal final chunks."""
    mask = batch["mask"] if partner is None else batch["mask"] & partner["mask"]
    count = int(np.sum(mask))
    if count == 0:
        return 0., 0
    total = 0.
    for start in range(0, len(batch["x"]), microbatch):
        stop = min(start + microbatch, len(batch["x"]))
        source_t = tensor_batch(batch, start, stop, device)
        partner_t = None if partner is None else tensor_batch(partner, start, stop, device)
        type_t = None if types is None else torch.as_tensor(types[start:stop], dtype=torch.long, device=device)
        loss_sum, valid = objective.loss_sum(source_t, partner_t, type_t)
        if not torch.isfinite(loss_sum):
            raise FloatingPointError("Nonfinite loss")
        if valid:
            (loss_sum / count).backward()
        total += float(loss_sum.detach())
    return total, count


def seed_runtime(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
