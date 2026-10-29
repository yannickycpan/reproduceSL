"""Compute-node preprocessing, exact budget accounting and throwaway throughput."""
import argparse
import math
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from .data import load_dataset
from .protocol import LAGS, make_manifest
from .sampling import rng_for, sources_for_round, training_universe
from .train import atomic_json, logical_backward, make_objective_for, seed_runtime, finite_gradients


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="all", choices=("all", *LAGS))
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("No usable GPU in allocated job; candidate arrays must not start")
    from . import data as preparation
    report = dict(hardware=dict(torch=torch.__version__, cuda=torch.version.cuda,
                               device=torch.cuda.get_device_name() if args.device == "cuda" else "cpu"),
                  storage_free_bytes=shutil.disk_usage(args.data_dir).free, datasets={}, throughput=[])
    for name in LAGS if args.dataset == "all" else [args.dataset]:
        metadata_path = Path(args.data_dir) / "processed" / name / "metadata.json"
        if not metadata_path.exists():
            getattr(preparation, "prepare_" + name)(args.data_dir)
        preparation.verify_dataset(name, args.data_dir)
        train, law, excluded = training_universe(load_dataset(name, args.data_dir, "train"), LAGS[name])
        n = len(train)
        draws = math.ceil(n / 24) if name == "electricity" else n
        report["datasets"][name] = dict(n=n, draws_per_round_or_epoch=draws, total_source_draws=100*draws,
            effective_original_passes=100*draws/n, update_opportunities={str(b):100*math.ceil(draws/b) for b in (128,256)},
            isolated_excluded=excluded, data_fingerprint=train.metadata.get("fingerprint"))
        # Only execution throughput, never validation/test or rate selection.
        configs = [c for c in make_manifest() if c["dataset"] == name and c["seed"] == 0 and c["configuration"] == 0
                   and c["batch_size"] == 256 and c["method"] in ("td_full", "dlinear_mse")]
        for config in configs:
            seed_runtime(0)
            obj = make_objective_for(config, train).to(args.device)
            ids = sources_for_round(name, n, 0, 1)[:min(256, draws)]
            partners, types = law.sample(ids, rng_for(name, 0, 1, 22))
            batch, partner = train.get_batch(ids), train.get_batch(partners)
            td = config["method"] == "td_full"
            optimizer = torch.optim.Adam(obj.parameter_groups(config["alpha"], config["beta"]),
                                         betas=(.9,.999),eps=1e-8,weight_decay=0,amsgrad=False)
            if args.device == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(3):
                optimizer.zero_grad(set_to_none=True)
                logical_backward(obj, batch, partner if td else None, types if td else None, args.device, 32)
                if not finite_gradients(obj):
                    raise FloatingPointError("Preflight gradient is nonfinite")
                optimizer.step()
                if not all(bool(torch.isfinite(p).all()) for p in obj.parameters()):
                    raise FloatingPointError("Preflight parameter is nonfinite")
            if args.device == "cuda":
                torch.cuda.synchronize()
            report["throughput"].append(dict(dataset=name,backbone=config["backbone"],method=config["method"],
                source_windows=len(ids),microbatch=32,throwaway_steps=3,seconds=time.perf_counter()-start))
            del obj, optimizer
        atomic_json(args.output, report)
    print(f"Preflight passed; wrote {args.output}. No scientific candidate was trained.")


if __name__ == "__main__":
    main()
