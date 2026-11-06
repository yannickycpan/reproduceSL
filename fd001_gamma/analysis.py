"""Validation-only joint selection over Gamma variant and rates, then mean evaluation.

All 180 candidates must be terminal. Each categorical choice also retains its
validation-selected configuration for diagnostics. Test scores never choose a
variant, learning rate, seed, or epoch. Legacy results are not read or rewritten.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import itertools
import math
from pathlib import Path
import statistics

import numpy as np

from . import PROTOCOL_ID
from .common import (atomic_json, campaign_paths, cpu_environment, data_identity,
                     file_hash, file_lock, read_json, scientific_hash, write_csv)
from .protocol import EVALUATIONS, SELECTION_EPOCHS, SEEDS, VARIANTS, digest, make_manifest


def select(results_dir, output_dir=None):
    """Audit every candidate before producing any locked selection."""
    from fulltd.sampling import sources_for_round, sequence_hash
    results_dir = Path(results_dir)
    manifest = make_manifest()
    code = scientific_hash()
    by_configuration, evidence, candidate_rows = defaultdict(list), [], []
    validation_rows, shared, source_hashes = [], {}, {}

    def match(key, value):
        if value is None or (key in shared and shared[key] != value):
            raise ValueError(f"Inconsistent or missing campaign identity: {key}")
        shared[key] = value

    for config in manifest:
        directory = results_dir / config["run_id"]
        if not (directory / "status.json").is_file():
            raise ValueError(f"Missing candidate: {config['run_id']}")
        status = read_json(directory / "status.json")
        if status.get("config") != config or status.get("protocol_id") != PROTOCOL_ID:
            raise ValueError(f"Configuration/protocol mismatch: {config['run_id']}")
        identity = status.get("identity", {})
        if (identity.get("code_sha256") != code or identity.get("config_sha256") != digest(config)
                or identity.get("protocol_id") != PROTOCOL_ID):
            raise ValueError(f"Scientific identity mismatch: {config['run_id']}")
        for key in ("data_sha256", "validation_sha256"):
            match(key, identity.get(key))
        match((config["seed"], "initial_mean"), status.get("initial_mean_sha256"))
        state = status.get("state")
        if state not in ("complete", "numerical_failure"):
            raise ValueError(f"Operationally incomplete: {config['run_id']} ({state})")
        records = read_json(directory / "validation.json") if (directory / "validation.json").exists() else []
        rounds = read_json(directory / "rounds.json") if (directory / "rounds.json").exists() else []
        n = status.get("n_sources")
        if type(n) is not int or n < 1:
            raise ValueError("Invalid training universe size")
        match("n_sources", n)
        if [r["epoch"] for r in rounds] != list(range(1, len(rounds) + 1)) or len(rounds) > 100:
            raise ValueError("Invalid epoch sequence")
        if [r["epoch"] for r in records] != list(EVALUATIONS[:len(records)]) or len(records) > 10:
            raise ValueError("Invalid validation sequence")
        for record in records:
            if any(not math.isfinite(float(record[k])) or float(record[k]) < 0 for k in ("mse", "mae")):
                raise ValueError("Invalid finite validation evidence")
            if record.get("validation_sampling") != "full":
                raise ValueError("Validation must cover all prescribed FD001 prefixes")
            match("validation_population_size", record.get("validation_population_size"))
            validation_rows.append(dict(run_id=config["run_id"], gamma_variant=config["gamma_variant"],
                                        alpha=config["alpha"], beta=config["beta"], seed=config["seed"],
                                        epoch=record["epoch"], mse=record["mse"], mae=record["mae"],
                                        optimizer_steps=record["optimizer_steps"]))
        previous_steps = 0
        for record in rounds:
            key = (config["seed"], record["epoch"])
            if key not in source_hashes:
                source_hashes[key] = sequence_hash(sources_for_round("fd001", n, *key))
            if record["source_sha256"] != source_hashes[key] or record["source_draws"] != n:
                raise ValueError("Training-source sequence/budget mismatch")
            match(("partners", *key), record["partner_sha256"])
            if len(record["type_counts"]) != 2 or sum(record["type_counts"]) != n:
                raise ValueError("Each TD source must have exactly one typed partner")
            step = record["optimizer_steps_cumulative"]
            if step - previous_steps != math.ceil(n / 128):
                raise ValueError("Incorrect number of FD001 Adam updates per epoch")
            previous_steps = step
        if state == "complete":
            if (status.get("epochs_completed") != 100 or status.get("next_epoch") != 101
                    or not (directory / "final.pt").is_file() or len(records) != 10 or len(rounds) != 100):
                raise ValueError("Completed candidate lacks its final checkpoint or evidence")
            if (status.get("source_draws") != 100 * n or status.get("partner_draws") != 100 * n
                    or status.get("optimizer_steps") != 100 * math.ceil(n / 128)
                    or status.get("update_opportunities") != status["optimizer_steps"]
                    or status.get("empty_mask_batches") != 0):
                raise ValueError("Completed candidate has an incorrect training budget")
            score = statistics.mean(float(r["mse"]) for r in records if r["epoch"] in SELECTION_EPOCHS)
        else:
            score = math.inf
        by_configuration[(config["gamma_variant"], config["configuration"])].append((config, score, state))
        candidate_rows.append(dict(**config, state=state, late_validation_mse=score,
                                   runtime_seconds=status.get("runtime_seconds"),
                                   initial_mean_sha256=status["initial_mean_sha256"]))
        evidence.append(dict(config=config, status=status, validation=records, rounds=rounds))
    scores = []
    for (variant, configuration), rows in sorted(by_configuration.items()):
        rows.sort(key=lambda row: row[0]["seed"])
        if [r[0]["seed"] for r in rows] != list(SEEDS):
            raise ValueError("Selection requires all five seeds")
        score = statistics.mean(r[1] for r in rows) if all(math.isfinite(r[1]) for r in rows) else math.inf
        config = rows[0][0]
        scores.append(dict(gamma_variant=variant, configuration=configuration, alpha=config["alpha"], beta=config["beta"],
                           score=score, failed_seeds=[r[0]["seed"] for r in rows if r[2] == "numerical_failure"],
                           run_ids=[r[0]["run_id"] for r in rows]))
    order = lambda r: (r["score"], r["alpha"], r["beta"], VARIANTS.index(r["gamma_variant"]))
    winners = [min((r for r in scores if r["gamma_variant"] == variant), key=order) for variant in VARIANTS]
    for winner in winners:
        winner["state"] = "selected" if math.isfinite(winner["score"]) else "all_configurations_failed"
        if winner["state"] != "selected":
            winner["run_ids"] = []
    best = min(winners, key=order)
    primary = dict(best) if best["state"] == "selected" else None
    selection = dict(protocol_id=PROTOCOL_ID, manifest_sha256=digest(manifest), evidence_sha256=digest(evidence),
                     selection_epochs=list(SELECTION_EPOCHS), selection_rule="mean validation MSE over 5 seeds x epochs 80/90/100",
                     categorical_hyperparameter="gamma_variant", primary_selection=primary, per_variant=winners)
    # Strict JSON writer uses string markers for nonfinite configuration scores.
    # Canonicalize this object before return so a loaded lock compares identically.
    for row in winners:
        if not math.isfinite(row["score"]):
            row["score"] = "Infinity"
    if output_dir is not None:
        out = Path(output_dir)
        atomic_json(out / "selections.json", selection)
        write_csv(out / "scores.csv", scores)
        write_csv(out / "candidate_accounting.csv", candidate_rows)
        write_csv(out / "validation_records.csv", validation_rows)
        selected_ids = {rid for w in winners for rid in w["run_ids"]}
        write_csv(out / "selected_runs.csv", [dict(r, primary_selected=primary is not None and r["run_id"] in primary["run_ids"])
                                               for r in candidate_rows if r["run_id"] in selected_ids])
        atomic_json(out / "campaign_audit.json", dict(candidates=180, complete=sum(r["state"] == "complete" for r in candidate_rows),
                    numerical_failure=sum(r["state"] == "numerical_failure" for r in candidate_rows),
                    code_sha256=code, source_sequences_verified=len(source_hashes)))
    return selection


def aggregate(records):
    groups = defaultdict(list)
    keys = ("gamma_variant", "split", "units", "horizon", "metric")
    for row in records:
        groups[tuple(str(row[k]) for k in keys)].append(row)
    summary = []
    for key, rows in sorted(groups.items()):
        rows.sort(key=lambda r: int(r["seed"]))
        if [int(r["seed"]) for r in rows] != list(SEEDS):
            raise ValueError("Every summary group must retain all five seeds")
        v = np.asarray([float(r["value"]) for r in rows])
        with np.errstate(invalid="ignore", over="ignore"):
            mean, sd = float(v.mean()), float(v.std(ddof=1))
        # Independent standard-library recomputation for finite groups.
        if np.isfinite(v).all():
            if not math.isclose(mean, statistics.mean(v.tolist()), abs_tol=1e-12, rel_tol=1e-12):
                raise ValueError("Independent summary mean verification failed")
            if not math.isclose(sd, statistics.stdev(v.tolist()), abs_tol=1e-12, rel_tol=1e-12):
                raise ValueError("Independent summary SD verification failed")
        summary.append(dict(dataset="fd001", backbone="tcn", method="td_full", batch_size=128, variant="trained",
                            **dict(zip(keys, key)), n=5, mean=mean, sample_sd=sd, standard_error=sd / math.sqrt(5),
                            finite_seeds=int(np.isfinite(v).sum()), uncertainty="five training seeds"))
    return summary


def paired_comparisons(records):
    indexed = defaultdict(dict)
    for row in records:
        key = tuple(str(row[k]) for k in ("split", "units", "horizon", "metric"))
        indexed[key].setdefault(row["gamma_variant"], {})[int(row["seed"])] = float(row["value"])
    output = []
    for key, variants in sorted(indexed.items()):
        for first, second in itertools.combinations(sorted(variants), 2):
            if any(sorted(variants[v]) != list(SEEDS) for v in (first, second)):
                raise ValueError("Missing seed in paired comparison")
            with np.errstate(invalid="ignore", over="ignore"):
                d = np.array([variants[first][s] - variants[second][s] for s in SEEDS])
                mean, sd = float(d.mean()), float(d.std(ddof=1))
            se = sd / math.sqrt(5)
            output.append(dict(zip(("split", "units", "horizon", "metric"), key),
                               first=first, second=second, difference="first minus second", n=5,
                               mean_difference=mean, sample_sd_difference=sd, standard_error_difference=se,
                               descriptive_t95_lower=mean - 2.7764451051977987 * se,
                               descriptive_t95_upper=mean + 2.7764451051977987 * se))
    return output


def evaluate_selected(results_dir, output_dir, data_dir):
    import torch
    from fulltd.data import load_dataset
    from fulltd.metrics import evaluate
    from fulltd.sampling import training_universe
    from .objectives import make_objective
    from .protocol import LAGS
    from .train import checkpoint_identity, validate_checkpoint
    root, out = Path(results_dir), Path(output_dir)
    locked = read_json(out / "selections.json")
    if digest(select(root)) != digest(locked):
        raise ValueError("Selection evidence changed; test evaluation refused")
    configs = {c["run_id"]: c for c in make_manifest()}
    train, _, _ = training_universe(load_dataset("fd001", data_dir, "train"), LAGS)
    val = load_dataset("fd001", data_dir, "val")
    test, rows, gamma_rows = None, [], []
    analysis_hash = file_hash(__file__)
    for winner in locked["per_variant"]:
        for rid in winner["run_ids"]:
            config = configs[rid]
            directory = root / rid
            checkpoint = torch.load(directory / "final.pt", map_location="cpu", weights_only=False)
            identity = checkpoint_identity(config, train, val)
            validate_checkpoint(checkpoint, config, identity, final=True)
            status = read_json(directory / "status.json")
            if (status["state"] != "complete" or status["identity"] != identity
                    or any(status.get(k) != v for k, v in checkpoint["counters"].items())
                    or checkpoint["validation"] != read_json(directory / "validation.json")
                    or checkpoint["round_records"] != read_json(directory / "rounds.json")):
                raise ValueError("Final checkpoint, status and selection sidecars disagree")
            for validation in checkpoint["validation"]:
                for t in validation["gamma_probe"]["types"]:
                    gamma_rows.append(dict(run_id=rid, gamma_variant=config["gamma_variant"], seed=config["seed"],
                                           epoch=validation["epoch"], **t))
            if test is None:  # Only after validation-only selection and checkpoint checks.
                test = load_dataset("fd001", data_dir, "test")
            cache_identity = dict(selection_sha256=digest(locked), scientific_identity=identity,
                                  final_sha256=file_hash(directory / "final.pt"), test_data_sha256=data_identity(test),
                                  analysis_sha256=analysis_hash)
            cache = directory / "gamma_endpoint_metrics.json"
            if cache.exists():
                saved = read_json(cache)
                if saved["identity"] != cache_identity:
                    raise ValueError(f"Incompatible endpoint cache preserved: {cache}")
                rows.extend(saved["rows"])
                continue
            objective = make_objective(config)
            objective.load_state_dict(checkpoint["objective"])
            run_rows = []
            for split, data in (("train", train), ("validation", val), ("test", test)):
                metrics, endpoints = evaluate(objective.model, data, batch_size=128, device="cpu", allow_nonfinite=True)
                for metric in metrics:
                    run_rows.append(dict(run_id=rid, gamma_variant=config["gamma_variant"], seed=config["seed"],
                                         alpha=config["alpha"], beta=config["beta"], dataset="fd001", backbone="tcn",
                                         method="td_full", batch_size=128, variant="trained", split=split, **metric))
                if endpoints:
                    write_csv(directory / "test_engine_predictions.csv", endpoints)
            atomic_json(cache, dict(identity=cache_identity, rows=run_rows))
            rows.extend(run_rows)
    write_csv(out / "metrics.csv", rows)
    write_csv(out / "per_seed_metrics.csv", rows)
    write_csv(out / "summary_metrics.csv", aggregate(rows))
    write_csv(out / "paired_comparisons.csv", paired_comparisons(rows))
    write_csv(out / "gamma_records.csv", gamma_rows)
    atomic_json(out / "verification.json", dict(summary_groups_verified=len(aggregate(rows)),
                all_seed_values_retained=True, selection_uses_test=False,
                point_prediction="actual epoch-100 mean network only; Gamma never used at inference"))
    return rows, gamma_rows


def figures(output_dir, rows, gamma_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = Path(output_dir)
    destination = out / "figures"
    destination.mkdir(parents=True, exist_ok=True)
    summary = aggregate(rows)
    selected = read_json(out / "selections.json")

    def save(fig, stem, data):
        fig.tight_layout()
        for suffix in ("pdf", "svg"):
            fig.savefig(destination / f"{stem}.{suffix}", bbox_inches="tight")
        plt.close(fig)
        write_csv(destination / f"{stem}.csv", data)

    for units, metric in (("cycles", "rmse"), ("cycles", "mae"), ("score", "rul_score_sum")):
        data = [r for r in summary if r["split"] == "test" and r["units"] == units
                and r["metric"] == metric and r["horizon"] == "all"]
        if not data:
            continue
        fig, ax = plt.subplots(figsize=(7, 4))
        for i, row in enumerate(data):
            value, se = row["mean"], row["standard_error"]
            if math.isfinite(value):
                ax.bar(i, value)
                if math.isfinite(se):
                    ax.errorbar(i, value, yerr=se, fmt="none", capsize=3)
            else:
                ax.text(i, 0, "nonfinite", rotation=90)
        ax.set_xticks(range(len(data)), [r["gamma_variant"] for r in data])
        ax.set_ylabel(f"{metric} ({units}); mean +/- 1 SE")
        save(fig, f"primary_fd001_gamma_{metric}", data)
    import csv
    with (out / "validation_records.csv").open(newline="") as stream:
        validation = list(csv.DictReader(stream))
    for metric in ("mse", "mae"):
        fig, ax = plt.subplots(figsize=(7, 4))
        data = []
        for winner in selected["per_variant"]:
            if not winner["run_ids"]:
                continue
            means, errors = [], []
            for epoch in EVALUATIONS:
                v = [float(r[metric]) for r in validation if r["run_id"] in winner["run_ids"] and int(r["epoch"]) == epoch]
                if len(v) != 5:
                    raise ValueError("Selected validation curve is missing seeds")
                row = dict(gamma_variant=winner["gamma_variant"], epoch=epoch, metric=metric,
                           mean=statistics.mean(v), standard_error=statistics.stdev(v) / math.sqrt(5), n=5)
                data.append(row)
                means.append(row["mean"])
                errors.append(row["standard_error"])
            ax.errorbar(EVALUATIONS, means, yerr=errors, marker="o", capsize=2, label=winner["gamma_variant"])
        ax.set_xlabel("Completed full training epoch")
        ax.set_ylabel(f"Validation {metric.upper()} (standardized)")
        if data:
            ax.legend()
        save(fig, f"validation_fd001_gamma_{metric}", data)
    for kind, lag in enumerate((1, 5), 1):
        fig, ax = plt.subplots(figsize=(7, 4))
        data = []
        for winner in selected["per_variant"]:
            if not winner["run_ids"]:
                continue
            means, errors = [], []
            for epoch in EVALUATIONS:
                v = [float(r["mean"]) for r in gamma_rows if r["run_id"] in winner["run_ids"]
                     and r["epoch"] == epoch and r["type"] == kind]
                if len(v) != 5:
                    raise ValueError("Gamma trajectory is missing seeds")
                with np.errstate(invalid="ignore", over="ignore"):
                    mean, sd = float(np.mean(v)), float(np.std(v, ddof=1))
                row = dict(gamma_variant=winner["gamma_variant"], epoch=epoch, lag=lag,
                           mean=mean, standard_error=sd / math.sqrt(5), n=5)
                data.append(row)
                means.append(row["mean"])
                errors.append(row["standard_error"])
            ax.errorbar(EVALUATIONS, means, yerr=errors, marker="o", capsize=2, label=winner["gamma_variant"])
        ax.set_xlabel("Completed full training epoch")
        ax.set_ylabel(f"Mean realized Gamma, lag {lag}; fixed training probe")
        if data:
            ax.legend()
        save(fig, f"gamma_fd001_lag{lag}", data)


def main():
    cpu_environment()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "run"))
    args = parser.parse_args()
    data, results, out = campaign_paths()
    if args.command == "check":
        selected = select(results)
        print("READY: all 180 candidates terminal and audited.")
        print("Primary validation-selected Gamma configuration:", selected["primary_selection"])
        return
    import torch
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    with file_lock(out / ".analysis.lock"):
        select(results, out)
        rows, gamma = evaluate_selected(results, out, data)
        figures(out, rows, gamma)
    print("Analysis complete:", out)


if __name__ == "__main__":
    main()
