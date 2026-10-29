"""Open test targets only after every selection decision in a dataset is fixed."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from .analysis import select_candidates, write_csv, write_json
from .data import load_dataset, sha256
from .metrics import evaluate
from .protocol import LAGS, digest, load_manifest
from .sampling import training_universe
from .train import make_objective_for, code_hash, seed_runtime, data_identity

EVALUATION_VERSION = "td-v7-endpoints-v2"


def evaluation_code_hash():
    """Keep cached predictions bound to their numerical evaluation code."""
    h = hashlib.sha256()
    for name in ("evaluate.py", "metrics.py", "refit.py"):
        path = Path(__file__).with_name(name)
        h.update(name.encode())
        h.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()


def endpoint_cache_identity(config, selected_group, checkpoint_sha256, train, val, test):
    # A later all-dataset analysis must reuse a dataset's already selected
    # models. Unrelated selections and output-directory scopes are immaterial.
    return dict(version=EVALUATION_VERSION, config_sha256=digest(config),
                selected_group_sha256=digest(selected_group),
                checkpoint_sha256=checkpoint_sha256,
                training_data_sha256=data_identity(train), validation_data_sha256=data_identity(val),
                test_data_sha256=data_identity(test), evaluation_code_sha256=evaluation_code_hash())


def validate_final_checkpoint(checkpoint, config, status, train, val):
    """The selected sidecars and loaded final weights must describe one run."""
    if checkpoint["config"] != config or checkpoint["next_epoch"] != 101 or checkpoint["cursor"] != 0:
        raise ValueError("Evaluation requires the actual compatible epoch/round-100 model")
    identity = checkpoint["identity"]
    if (identity["code_sha256"] != code_hash() or identity["data_sha256"] != data_identity(train)
            or identity["validation_sha256"] != data_identity(val)):
        raise ValueError("Code or processed data changed since fitting; preserve prior run and audit compatibility")
    if (status.get("state") != "complete" or status.get("identity") != identity
            or status.get("config") != config or status.get("epochs_completed") != 100):
        raise ValueError("Final checkpoint and completed selection status disagree")
    if checkpoint.get("initial_mean_sha256") != status.get("initial_mean_sha256"):
        raise ValueError("Final checkpoint and initial-state identity disagree")
    for key, value in checkpoint["counters"].items():
        if status.get(key) != value:
            raise ValueError(f"Final checkpoint and status counter disagree: {key}")


def failure_metrics(dataset, reason, status="nonfinite_prediction"):
    """Explicit nonfinite failure markers, never substitutes for measured errors."""
    horizon = {"electricity": 24, "metr_la": 12, "fd001": 1}[dataset.dataset]
    raw_units = {"electricity": "kW", "metr_la": "mph", "fd001": "cycles"}[dataset.dataset]
    rows = []
    for units in ("standardized", raw_units):
        for h in [*(str(i) for i in range(1, horizon + 1)), "all"]:
            for metric in ("mae", "mse", "rmse"):
                rows.append(dict(units=units, horizon=h, metric=metric, value=math.inf,
                                 valid_count=None, entity_count=None, aggregation="unavailable_due_to_failure",
                                 metric_status=status, failure_reason=reason))
    if dataset.dataset == "fd001":
        for metric in ("rul_score_sum", "rul_score_mean"):
            rows.append(dict(units="score", horizon="all", metric=metric, value=math.inf,
                             valid_count=None, aggregation="unavailable_due_to_failure",
                             metric_status=status, failure_reason=reason))
    return rows


def evaluate_endpoint(model, dataset, batch_size, device, failures, context, persistence=False):
    """Retain numerical endpoint failures and continue independent evaluations."""
    try:
        metrics, endpoints = evaluate(model, dataset, batch_size, device,
                                      persistence=persistence, allow_nonfinite=True)
    except FloatingPointError as error:
        metrics, endpoints = failure_metrics(dataset, str(error)), []
    affected = []
    for metric in metrics:
        if not math.isfinite(float(metric["value"])):
            if metric["metric"].startswith("rul_score"):
                metric.setdefault("metric_status", "nonfinite_rul_score")
            else:
                metric.setdefault("metric_status", "nonfinite_metric")
            affected.append({key: metric.get(key) for key in
                             ("units", "horizon", "metric", "metric_status", "failure_reason")})
        else:
            metric.setdefault("metric_status", "finite")
    if affected:
        failures.append(dict(context, stage="point_evaluation", affected_metrics=affected,
                             note="Nonfinite values or explicit failure markers are retained in the five-seed tables."))
    return metrics, endpoints


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
            status = json.loads((root / run_id / "status.json").read_text())
            validate_final_checkpoint(checkpoint, config, status, train, val)
            if checkpoint["validation"] != json.loads((root / run_id / "validation.json").read_text()):
                raise ValueError("Final checkpoint validation records differ from the records used for selection")
            if checkpoint["round_records"] != json.loads((root / run_id / "rounds.json").read_text()):
                raise ValueError("Final checkpoint source-round records differ from the campaign audit")
            if test is None:
                test = load_dataset(name, data_dir, "test")
                datasets[name] = (train, val, test, law)
            objective = make_objective_for(config, train).to(device)
            objective.load_state_dict(checkpoint["objective"])
            checksum = sha256(checkpoint_path)
            checkpoint_hashes[run_id] = checksum
            cache_identity = endpoint_cache_identity(config, group, checksum, train, val, test)
            metadata = {key: config[key] for key in ("run_id", "dataset", "backbone", "method", "batch_size", "seed", "alpha", "beta")}
            result_file = root / run_id / "endpoint_metrics.json"
            if result_file.exists():
                saved = json.loads(result_file.read_text())
                if saved.get("identity") != cache_identity:
                    raise ValueError("Endpoint cache belongs to different checkpoint/group/data/evaluation code; preserve it separately before reevaluating")
                rows.extend(saved["rows"])
                continue
            run_rows, failures = [], []
            for split, data in (("train", train), ("validation", val), ("test", test)):
                metrics, endpoints = evaluate_endpoint(objective.model, data, batch_size, device, failures,
                                                       dict(run_id=run_id, split=split, variant="trained"))
                run_rows.extend(dict(**metadata, split=split, variant="trained", **metric) for metric in metrics)
                if endpoints:
                    write_csv(root / run_id / "test_engine_predictions.csv", endpoints)
            if name in ("electricity", "fd001") and config["backbone"] == "tcn" and config["method"] == "td_full" and config["batch_size"] == 256:
                from .refit import refit_readouts, RefittedPredictor, save_readouts
                try:
                    fits = refit_readouts(objective.model, train, law, objective.theta.detach().cpu().numpy(), device, batch_size)
                except (FloatingPointError, np.linalg.LinAlgError) as error:
                    fits = {}
                    reason = f"{type(error).__name__}: {error}"
                    failures.append(dict(run_id=run_id, stage="readout_solve", variants=["refit_ols", "refit_td"],
                                         reason=reason, note="No finite result is substituted for the failed diagnostic."))
                    write_json(root / run_id / "refits" / "failures.json", failures[-1])
                    for label in ("ols", "td"):
                        for split, data in (("validation", val), ("test", test)):
                            run_rows.extend(dict(**metadata, split=split, variant="refit_" + label, **metric)
                                            for metric in failure_metrics(data, reason, "readout_numerical_failure"))
                if fits:
                    save_readouts(root / run_id / "refits", fits, config, checksum)
                for label, fit in fits.items():
                    predictor = RefittedPredictor(objective.model, fit).to(device)
                    for split, data in (("validation", val), ("test", test)):
                        metrics, endpoints = evaluate_endpoint(predictor, data, batch_size, device, failures,
                                                               dict(run_id=run_id, split=split, variant="refit_" + label))
                        run_rows.extend(dict(**metadata, split=split, variant="refit_" + label, **metric) for metric in metrics)
                        if endpoints:
                            write_csv(root / run_id / "refits" / f"{label}_test_engines.csv", endpoints)
            write_json(root / run_id / "evaluation_failures.json", failures)
            write_json(result_file, dict(identity=cache_identity, rows=run_rows, numerical_failures=failures))
            rows.extend(run_rows)
    # Include deterministic persistence even when all trained methods fail.
    scope = checked["dataset_scope"]
    names = ("electricity", "metr_la", "fd001") if scope == "all" else (scope,)
    for name in names:
        if name == "fd001":
            continue
        for split in ("val", "test"):
            data = load_dataset(name, data_dir, split)
            failures = []
            metrics, _ = evaluate_endpoint(None, data, batch_size, device, failures,
                                           dict(run_id=name + "_persistence", split=split, variant="trained"), persistence=True)
            write_json(out / f"{name}_persistence_{split}_failures.json", failures)
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
