"""Audit both Gamma campaigns, select across old+new HPs, evaluate epoch-100 models.

Original 180-run protocol is unchanged. Rep(alpha=0.00003, 0.0001) extends
its original alpha grid; pairwise gets the complete five-alpha x four-beta grid.
All selection uses five-seed validation MSE at epochs 80, 90, 100, never test.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
import csv
import math
from pathlib import Path
import statistics

from fd001_gamma.common import (atomic_json, campaign_paths as original_paths, cpu_environment,
                                data_identity, file_hash, file_lock, read_json, write_csv)
from fd001_gamma.protocol import (make_manifest as original_manifest,
                                  VARIANTS as ORIGINAL_VARIANTS,
                                  EVALUATIONS, SEEDS, SELECTION_EPOCHS, digest, LAGS)
from . import PROTOCOL_ID
from .protocol import make_manifest as extension_manifest
from .runtime import (campaign_paths as extension_paths,
                      scientific_hash as extension_hash, bind_original_training_engine)

VARIANTS = (*ORIGINAL_VARIANTS, "pairwise")
SCORE_EPOCHS = frozenset(SELECTION_EPOCHS)


@contextmanager
def extension_audit_bindings():
    """Use the original rigorous Gamma source/seed/budget audit on extension data."""
    import fd001_gamma.analysis as analysis
    original = {k: getattr(analysis, k) for k in
                ("make_manifest", "scientific_hash", "PROTOCOL_ID", "VARIANTS")}
    try:
        analysis.make_manifest = extension_manifest
        analysis.scientific_hash = extension_hash
        analysis.PROTOCOL_ID = PROTOCOL_ID
        analysis.VARIANTS = ("representation", "pairwise")
        yield analysis
    finally:
        for name, value in original.items():
            setattr(analysis, name, value)


def audit_and_select(original_results, extension_results):
    import fd001_gamma.analysis as old_analysis
    old_audit = old_analysis.select(original_results)
    with extension_audit_bindings() as ext_analysis:
        ext_audit = ext_analysis.select(extension_results)

    # Cross-campaign checks preserve matched training sources, processed data,
    # validation data, and same per-seed initial predictor. Both independent
    # audits already verify every candidate and deterministic source stream.
    first = {}
    candidates = []
    for origin, root, manifest in (("original", Path(original_results), original_manifest()),
                                    ("extension", Path(extension_results), extension_manifest())):
        for config in manifest:
            path = root / config["run_id"]
            status = read_json(path / "status.json")
            identity = status["identity"]
            for key in ("data_sha256", "validation_sha256"):
                if key in first and first[key] != identity[key]:
                    raise ValueError(f"Original/extension FD001 data mismatch: {key}")
                first[key] = identity[key]
            init_key = ("initial_mean", config["seed"])
            initial_mean = status["initial_mean_sha256"]
            if init_key in first and first[init_key] != initial_mean:
                raise ValueError(f"Original/extension initial mean mismatch for seed {config['seed']}")
            first[init_key] = initial_mean
            if status["state"] == "complete":
                val = read_json(path / "validation.json")
                score = statistics.mean(float(r["mse"]) for r in val if r["epoch"] in SCORE_EPOCHS)
                if len([r for r in val if r["epoch"] in SCORE_EPOCHS]) != 3:
                    raise ValueError("Missing late validation scores")
            else:
                score = math.inf
            candidates.append(dict(origin=origin, gamma_variant=config["gamma_variant"],
                                   alpha=config["alpha"], beta=config["beta"], seed=config["seed"],
                                   run_id=config["run_id"], score=score, state=status["state"]))
    by_setting = defaultdict(list)
    for row in candidates:
        by_setting[(row["gamma_variant"], row["alpha"], row["beta"])].append(row)
    scores = []
    for (variant, alpha, beta), rows in by_setting.items():
        rows.sort(key=lambda r: r["seed"])
        if [r["seed"] for r in rows] != list(SEEDS):
            raise ValueError("One Gamma configuration does not have all five seeds")
        score = statistics.mean(r["score"] for r in rows) if all(
            math.isfinite(r["score"]) for r in rows) else math.inf
        scores.append(dict(gamma_variant=variant, alpha=alpha, beta=beta, score=score,
                           origin=rows[0]["origin"], run_ids=[r["run_id"] for r in rows],
                           failed_seeds=[r["seed"] for r in rows if r["state"] != "complete"]))
    choices = []
    for variant in VARIANTS:
        possible = [r for r in scores if r["gamma_variant"] == variant]
        expected = 20 if variant in ("representation", "pairwise") else 12
        if len(possible) != expected:
            raise ValueError(f"{variant}: expected {expected} HP combinations, got {len(possible)}")
        best = min(possible, key=lambda r: (r["score"], r["alpha"], r["beta"]))
        if not math.isfinite(best["score"]):
            raise ValueError(f"No successful five-seed configuration for {variant}")
        choices.append(best)
    selection = dict(protocol_id=PROTOCOL_ID,
                     original_audit_sha256=digest(old_audit),
                     extension_audit_sha256=digest(ext_audit),
                     selection_epochs=list(SELECTION_EPOCHS),
                     rule="five-seed mean validation MSE at epochs 80,90,100 (all rates jointly per variant)",
                     per_variant=choices,
                     grid=dict(original_representation_alphas=[.0003, .001, .003],
                               added_alphas=[.00003, .0001],
                               pairwise_alphas=[.00003, .0001, .0003, .001, .003]))
    return selection, scores, first


def load_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def evaluate_extension_selected(selection, original_results, extension_results, output_dir, data_dir):
    import torch
    from fulltd.data import load_dataset
    from fulltd.metrics import evaluate
    from fulltd.sampling import training_universe
    from .objectives import make_objective
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    old_out = original_paths()[2]
    old_rows = load_csv(old_out / "metrics.csv")
    old_by_id = defaultdict(list)
    for row in old_rows:
        old_by_id[row["run_id"]].append(row)
    existing = {c["run_id"]: c for c in original_manifest()}
    extended = {c["run_id"]: c for c in extension_manifest()}
    train, _, _ = training_universe(load_dataset("fd001", data_dir, "train"), LAGS)
    validation = load_dataset("fd001", data_dir, "val")
    test = None
    rows = []
    gamma = []
    for chosen in selection["per_variant"]:
        for run_id in chosen["run_ids"]:
            origin = chosen["origin"]
            root = Path(original_results if origin == "original" else extension_results)
            config = (existing if origin == "original" else extended)[run_id]
            ckpt_dir = root / run_id
            val_records = read_json(ckpt_dir / "validation.json")
            for record in val_records:
                for item in record["gamma_probe"]["types"]:
                    gamma.append(dict(run_id=run_id, gamma_variant=config["gamma_variant"],
                                      seed=config["seed"], epoch=record["epoch"], **item))
            if origin == "original":
                if not old_by_id[run_id]:
                    raise ValueError(f"Original selected Gamma metrics missing for {run_id}; "
                                     "run dscpuscripts/analyze_fd001_gamma.sh first")
                rows.extend(old_by_id[run_id])
                continue
            checkpoint_file = ckpt_dir / "final.pt"
            checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
            with bind_original_training_engine() as engine:
                identity = engine.checkpoint_identity(config, train, validation)
                engine.validate_checkpoint(checkpoint, config, identity, final=True)
            status = read_json(ckpt_dir / "status.json")
            if (status["state"] != "complete" or status["identity"] != identity
                    or checkpoint["validation"] != val_records
                    or checkpoint["round_records"] != read_json(ckpt_dir / "rounds.json")
                    or any(status.get(k) != v for k, v in checkpoint["counters"].items())):
                raise ValueError("Extension checkpoint and status/sidecars do not match")
            if test is None:
                test = load_dataset("fd001", data_dir, "test")
            cache_identity = dict(selection_sha256=digest(selection), checkpoint_sha256=file_hash(checkpoint_file),
                                  test_data_sha256=data_identity(test), code_sha256=extension_hash(),
                                  analysis_sha256=file_hash(__file__))
            cache = Path(output_dir) / "endpoint_cache" / (run_id + "_" + digest(cache_identity)[:16] + ".json")
            if cache.exists():
                saved = read_json(cache)
                if saved["identity"] != cache_identity:
                    raise ValueError("Endpoint cache identity mismatch")
                rows.extend(saved["rows"])
                continue
            objective = make_objective(config)
            objective.load_state_dict(checkpoint["objective"])
            result = []
            for split, data in (("train", train), ("validation", validation), ("test", test)):
                metrics, endpoints = evaluate(objective.model, data, batch_size=128, device="cpu",
                                              allow_nonfinite=True)
                result.extend(dict(run_id=run_id, gamma_variant=config["gamma_variant"],
                                   seed=config["seed"], alpha=config["alpha"], beta=config["beta"],
                                   dataset="fd001", backbone="tcn", method="td_full",
                                   batch_size=128, variant="trained", split=split, **metric)
                              for metric in metrics)
                if split == "test" and endpoints:
                    write_csv(Path(output_dir) / "predictions" / run_id / "test_engine_predictions.csv", endpoints)
            atomic_json(cache, dict(identity=cache_identity, rows=result))
            rows.extend(result)
            print("Evaluated new epoch-100 checkpoint:", run_id, flush=True)
    if len({r["run_id"] for r in rows}) != 20:
        raise ValueError("Expected exactly twenty selected trained seed models")
    return rows, gamma


def save_tables(selection, scores, rows, gamma, output):
    from fd001_gamma.analysis import aggregate, paired_comparisons
    from dscpuscripts.fd001_report import target_scale
    out = Path(output)
    configs = {c["run_id"]: c for c in (*original_manifest(), *extension_manifest())}
    validation = []
    for chosen in selection["per_variant"]:
        root = Path(original_paths()[1] if chosen["origin"] == "original"
                    else extension_paths()[1])
        for run_id in chosen["run_ids"]:
            c = configs[run_id]
            for rec in read_json(root / run_id / "validation.json"):
                validation.append(dict(run_id=run_id, gamma_variant=c["gamma_variant"], seed=c["seed"],
                                       alpha=c["alpha"], beta=c["beta"], epoch=rec["epoch"],
                                       optimizer_steps=rec["optimizer_steps"], mse=rec["mse"], mae=rec["mae"]))
    if len(validation) != 200:
        raise ValueError("Selected validation curves must cover four variants x five seeds x ten epochs")
    summaries = aggregate(rows)
    write_csv(out / "metrics.csv", rows)
    write_csv(out / "summary_metrics.csv", summaries)
    write_csv(out / "paired_comparisons.csv", paired_comparisons(rows))
    write_csv(out / "gamma_records.csv", gamma)
    write_csv(out / "validation_records.csv", validation)
    write_csv(out / "scores.csv", scores)
    scale = target_scale(rows)
    rmse = []
    for rec in validation:
        rmse.append(dict(**rec, rmse_cycles=math.sqrt(float(rec["mse"])) * scale))
    write_csv(out / "validation_rmse.csv", rmse)
    atomic_json(out / "evaluation_notes.json", dict(target_scale=scale, validation_x_axis="optimizer_steps",
                selection="MSE at epochs 80/90/100", testing="real epoch-100 models, no refit"))
    return rmse



def figures(output_dir):
    """Use the single FD001 optimizer-update learning-curve plotting utility."""
    from dscpuscripts.fd001_report import plot_learning_curve
    out = Path(output_dir)
    rows = load_csv(out / "validation_rmse.csv")
    destination = out / "figures"
    destination.mkdir(exist_ok=True, parents=True)
    plot_rows = []
    for variant in VARIANTS:
        groups = defaultdict(list)
        for row in rows:
            if row["gamma_variant"] == variant:
                groups[int(row["optimizer_steps"])].append(row)
        if len(groups) != len(EVALUATIONS):
            raise ValueError(f"Expected all ten validation checkpoints: {variant}")
        for updates, members in sorted(groups.items()):
            if sorted(int(m["seed"]) for m in members) != list(SEEDS):
                raise ValueError(f"Five seed validation records required: {variant}")
            epochs = {int(m["epoch"]) for m in members}
            if len(epochs) != 1:
                raise ValueError("Mismatched epoch for a validation update count")
            values = [float(m["rmse_cycles"]) for m in members]
            mean = statistics.mean(values)
            se = statistics.stdev(values) / math.sqrt(len(SEEDS))
            plot_rows.append(dict(method=variant, gamma_variant=variant,
                                  optimizer_steps=updates, epoch=epochs.pop(),
                                  mean=mean, standard_error=se, n=len(SEEDS)))
        write_csv(destination / f"validation_rmse_td_{variant}.csv",
                  [r for r in plot_rows if r["gamma_variant"] == variant])
    plot_learning_curve(plot_rows, destination / "fd001_gamma_four_learning_curves")

def run(mode):
    cpu_environment()
    original_data, original_results, _ = original_paths()
    _, ext_results, ext_out = extension_paths()
    if mode == "plots":
        if not (ext_out / "validation_rmse.csv").is_file():
            raise FileNotFoundError("Run extension analysis before plots-only")
        figures(ext_out)
        print("Extension Gamma plots from saved tables:", ext_out / "figures")
        return
    selection, scores, shared = audit_and_select(original_results, ext_results)
    if mode == "check":
        print("Extension ready: 180 original + 140 new candidates audited.")
        for choice in selection["per_variant"]:
            print(choice["gamma_variant"], choice["alpha"], choice["beta"],
                  choice["origin"], "validation MSE =", choice["score"])
        return
    with file_lock(ext_out / ".analysis.lock"):
        # The selection is locked before the test split is loaded.
        atomic_json(ext_out / "selections.json", selection)
        rows, gamma = evaluate_extension_selected(selection, original_results, ext_results,
                                                  ext_out, original_data)
        save_tables(selection, scores, rows, gamma, ext_out)
        figures(ext_out)
    print("Four Gamma methods analyzed without refits:", ext_out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", nargs="?", choices=("run", "check", "plots"), default="run")
    args = parser.parse_args()
    run(args.mode)


if __name__ == "__main__":
    main()
