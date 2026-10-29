"""Open test targets only after every selection decision in a dataset is fixed."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .analysis import select_candidates, write_csv, write_json
from .data import load_dataset, sha256
from .metrics import evaluate
from .protocol import LAGS, digest, load_manifest
from .sampling import training_universe
from .train import make_objective_for, code_hash, seed_runtime, data_identity


def evaluate_selected(manifest_path, results_dir, selection_dir, data_dir, device="cpu", batch_size=128):
    manifest = load_manifest(manifest_path)
    by_id = {r["run_id"]: r for r in manifest}
    root, out = Path(results_dir), Path(selection_dir)
    locked = json.loads((out / "selections.json").read_text())
    # Recompute the same one selector from all validation evidence, never from test.
    checked = select_candidates(manifest, root, out, dataset=locked["dataset_scope"], write_outputs=False)
    if digest(checked) != digest(locked):
        raise ValueError("Selection evidence changed; review the regenerated selection before test evaluation")
    rows, datasets = [], {}
    checkpoint_hashes = {}
    for group in checked["groups"]:
        if group["state"] != "selected":
            continue
        name = group["dataset"]
        if name not in datasets:
            train, law, _ = training_universe(load_dataset(name, data_dir, "train"), LAGS[name])
            datasets[name] = (train, load_dataset(name, data_dir, "val"), None, law)
        train, val, test, law = datasets[name]
        for run_id in group["run_ids"]:
            config = by_id[run_id]
            seed_runtime(config["seed"])
            checkpoint_path = root / run_id / "final.pt"
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            if checkpoint["config"] != config or checkpoint["next_epoch"] != 101 or checkpoint["cursor"] != 0:
                raise ValueError("Evaluation requires the actual compatible epoch/round-100 model")
            identity = checkpoint["identity"]
            if (identity["code_sha256"] != code_hash() or identity["data_sha256"] != data_identity(train)
                    or identity["validation_sha256"] != data_identity(val)):
                raise ValueError("Code or processed data changed since fitting; preserve prior run and audit compatibility")
            if test is None:
                test = load_dataset(name, data_dir, "test")
                datasets[name] = (train, val, test, law)
            objective = make_objective_for(config, train).to(device)
            objective.load_state_dict(checkpoint["objective"])
            checksum = sha256(checkpoint_path)
            checkpoint_hashes[run_id] = checksum
            metadata = {key: config[key] for key in ("run_id", "dataset", "backbone", "method", "batch_size", "seed", "alpha", "beta")}
            result_file = root / run_id / "endpoint_metrics.json"
            if result_file.exists():
                saved = json.loads(result_file.read_text())
                if saved["checkpoint_sha256"] != checksum or saved["selection_sha256"] != digest(checked):
                    raise ValueError("Endpoint cache belongs to different checkpoint/selection")
                rows.extend(saved["rows"])
                continue
            run_rows = []
            for split, data in (("train", train), ("validation", val), ("test", test)):
                metrics, endpoints = evaluate(objective.model, data, batch_size, device)
                run_rows.extend(dict(**metadata, split=split, variant="trained", **metric) for metric in metrics)
                if endpoints:
                    write_csv(root / run_id / "test_engine_predictions.csv", endpoints)
            if name in ("electricity", "fd001") and config["backbone"] == "tcn" and config["method"] == "td_full" and config["batch_size"] == 256:
                from .refit import refit_readouts, RefittedPredictor, save_readouts
                fits = refit_readouts(objective.model, train, law, objective.theta.detach().cpu().numpy(), device, batch_size)
                save_readouts(root / run_id / "refits", fits, config, checksum)
                for label, fit in fits.items():
                    predictor = RefittedPredictor(objective.model, fit).to(device)
                    for split, data in (("validation", val), ("test", test)):
                        metrics, endpoints = evaluate(predictor, data, batch_size, device)
                        run_rows.extend(dict(**metadata, split=split, variant="refit_" + label, **metric) for metric in metrics)
                        if endpoints:
                            write_csv(root / run_id / "refits" / f"{label}_test_engines.csv", endpoints)
            write_json(result_file, dict(checkpoint_sha256=checksum, selection_sha256=digest(checked), rows=run_rows))
            rows.extend(run_rows)
    # Include deterministic persistence even when all trained methods fail.
    scope = checked["dataset_scope"]
    names = ("electricity", "metr_la", "fd001") if scope == "all" else (scope,)
    for name in names:
        if name == "fd001":
            continue
        for split in ("val", "test"):
            data = load_dataset(name, data_dir, split)
            metrics, _ = evaluate(None, data, batch_size, device, persistence=True)
            rows.extend(dict(run_id=name + "_persistence", dataset=name, backbone="persistence",
                             method="daily_persistence" if name == "electricity" else "last_observation",
                             batch_size=256, seed=-1, alpha=None, beta=None,
                             split="validation" if split == "val" else split, variant="trained", **metric) for metric in metrics)
    write_csv(out / "metrics.csv", rows)
    write_json(out / "evaluation_manifest.json", dict(selection_sha256=digest(checked), checkpoint_hashes=checkpoint_hashes,
               protocol_id=checked["protocol_id"], data_fingerprints={k: v[0].metadata for k, v in datasets.items()},
               point_prediction="actual final mean network only", refit_scope="primary TD-full TCN Electricity/FD001 only"))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="configs/fulltd/manifest.json")
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--selection-dir", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    evaluate_selected(args.manifest, args.results_dir, args.selection_dir, args.data_dir, args.device, args.batch_size)


if __name__ == "__main__":
    main()
