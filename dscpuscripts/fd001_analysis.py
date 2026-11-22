"""Unified FD001 comparison and TD diagnostics over three nested TD lag sets.

Select each method by five-seed mean standardized validation MSE over the last
25% of actual positive-update evaluations. The zero-update baseline is plotted
but excluded from selection. Test labels are never used for HP decisions.
"""
from __future__ import annotations

import argparse
from collections import defaultdict, Counter
import csv
import math
from pathlib import Path
import statistics

from dscpuscripts.common import atomic_json, file_lock, read_json, write_csv, file_hash, data_identity
from dscpuscripts.result_paths import paths as dataset_paths
from .fd001_protocol import (make_manifest, METHODS, SEEDS,
                              EPOCHS, EVAL_UPDATE_PERIOD, PROTOCOL_ID, digest)
from .fd001_train import code_hash, build_objective, json_safe

LABELS = dict(mse="MSE", gaussian_nll="Gaussian NLL",
              td_pairwise="TD pairwise",
              td_mse_combo="TD-MSE combo")


def lag_key(config):
    return "-".join(map(str, config["lags"])) if config["method"].startswith("td_") else "none"


def method_key(record):
    lag = record.get("lag_set", "none")
    return record["method"] if lag == "none" else f"{record['method']}@{lag}"


OUT_METRICS = (("cycles", "rmse"), ("cycles", "mae"), ("score", "rul_score_sum"))


SELECTION_FRACTION = 0.25  # Analysis-only, never a training-protocol parameter.


def selection_update_steps(total_updates, period=EVAL_UPDATE_PERIOD):
    """Final ceil(25%) of actual positive-update evaluation checkpoints."""
    if type(total_updates) is not int or total_updates < 1 or period < 1:
        raise ValueError("Expected a positive update budget and evaluation period")
    measured = list(range(period, total_updates + 1, period))
    if not measured:
        raise ValueError("No positive-update evaluations available for selection")
    return tuple(measured[-math.ceil(SELECTION_FRACTION * len(measured)):])


def load_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def expected_steps(n_sources, batch_size):
    return math.ceil(n_sources / batch_size) * EPOCHS


def audit_and_select(results):
    from fulltd.sampling import sources_for_round, sequence_hash
    candidates = make_manifest()
    expected_code = code_hash()
    seen_data, checked_sources, choices, scores = {}, {}, [], []
    by_hp = defaultdict(list)
    counts = Counter()
    for config in candidates:
        run_id = config["run_id"]
        directory = Path(results) / run_id
        status_path = directory / "status.json"
        if not status_path.exists():
            raise ValueError(f"Missing FD001 run status: {run_id}")
        status = read_json(status_path)
        if status.get("config") != config or status.get("identity", {}).get("code_sha256") != expected_code:
            raise ValueError(f"Scientific run/code mismatch: {run_id}")
        identity = status["identity"]
        if identity.get("protocol_id") != PROTOCOL_ID or identity.get("config_sha256") != digest(config):
            raise ValueError(f"Protocol identity mismatch: {run_id}")
        for key in ("data_sha256", "validation_sha256"):
            if key in seen_data and seen_data[key] != identity[key]:
                raise ValueError(f"Mixed training or validation data across methods: {run_id}")
            seen_data[key] = identity[key]
        initial_key = ("initial_mean", config["seed"])
        initial = status["initial_mean_sha256"]
        if initial_key in seen_data and seen_data[initial_key] != initial:
            raise ValueError(f"Different predictor initialization across methods, seed {config['seed']}")
        seen_data[initial_key] = initial
        state = status["state"]
        if state not in ("complete", "numerical_failure"):
            raise ValueError(f"Operationally unfinished candidate: {run_id} ({state})")
        counts[state] += 1
        score = math.inf
        if state == "complete":
            if not (directory / "final.pt").is_file():
                raise ValueError(f"Missing final checkpoint: {run_id}")
            combo = config["method"] == "td_mse_combo"
            n = status["n_sources"]
            if type(n) is not int or n < 1:
                raise ValueError("Invalid source universe size")
            per_epoch = math.ceil(n / config["batch_size"])
            if combo:
                phase_epochs = status.get("phase_epochs", {})
                gamma_epochs = phase_epochs.get("gamma")
                mse_epochs = phase_epochs.get("mse")
                predictor_epochs = phase_epochs.get("predictor")
                if (status.get("phase") != "complete"
                        or mse_epochs != EPOCHS or config["mse_epochs"] != EPOCHS
                        or predictor_epochs != EPOCHS or config["predictor_max_epochs"] != EPOCHS
                        or type(gamma_epochs) is not int
                        or not 1 <= gamma_epochs <= config["gamma_max_epochs"]
                        or status["epochs_completed"] != EPOCHS
                        or status["next_epoch"] != EPOCHS + 1
                        or status.get("stop_reasons", {}).get("gamma") not in (
                            "training_loss_plateau", "max_epochs")
                        or status.get("stop_reasons", {}).get("predictor") != "max_epochs"):
                    raise ValueError(f"Invalid three-stage completion evidence: {run_id}")
                for phase, count, batch in (
                        ("mse", EPOCHS, 256),
                        ("gamma", gamma_epochs, config["batch_size"]),
                        ("predictor", EPOCHS, config["batch_size"])):
                    rounds_file = ("rounds.json" if phase == "predictor"
                                   else f"{phase}_rounds.json")
                    records = read_json(directory / rounds_file)
                    updates_per_epoch = math.ceil(n / batch)
                    if ([x["epoch"] for x in records] != list(range(1, count + 1))
                            or any(x["source_draws"] != n or
                                   x["optimizer_steps"] != x["epoch"] * updates_per_epoch
                                   for x in records)
                            or status.get("phase_steps", {}).get(phase) != count * updates_per_epoch):
                        raise ValueError(f"Bad {phase} epoch/step accounting: {run_id}")
                    for row in records:
                        key = (config["seed"], row["epoch"])
                        if key not in checked_sources:
                            checked_sources[key] = sequence_hash(
                                sources_for_round("fd001", n, *key))
                        if row["source_sha256"] != checked_sources[key]:
                            raise ValueError(f"Stage source stream mismatch: {run_id}/{phase}")
                        if phase == "mse" and row["partner_sha256"] is not None:
                            raise ValueError("Stage 1 cannot draw partners")
                        if phase != "mse" and not isinstance(row["partner_sha256"], str):
                            raise ValueError("Missing TD partner stream digest")
                timing = status["timing"]
                required_times = [
                    f"{phase}_{suffix}" for phase in ("mse", "gamma", "predictor")
                    for suffix in ("update_seconds", "wall_seconds", "mean_update_seconds")]
                required_times += ["attributed_total_seconds"]
                if (status.get("stage1_model_sha256") is None
                        or len(status["stage1_model_sha256"]) != 64
                        or not all(timing.get(k) is not None and
                                   math.isfinite(float(timing[k])) and float(timing[k]) >= 0
                                   for k in required_times)):
                    raise ValueError(f"Missing stage timings or stage-1 identity: {run_id}")
                total = per_epoch * EPOCHS
            else:
                if (int(status["epochs_completed"]) != EPOCHS
                        or int(status["next_epoch"]) != EPOCHS + 1):
                    raise ValueError(f"Incomplete final epoch: {run_id}")
                total = per_epoch * EPOCHS
            if status["optimizer_steps"] != total or status["update_opportunities"] != total:
                raise ValueError(f"Incorrect optimizer budget: {run_id}")
            curve = read_json(directory / "update_curve.json")
            checkpoints = list(range(0, total + 1, EVAL_UPDATE_PERIOD))
            if [r["optimizer_steps"] for r in curve] != checkpoints:
                raise ValueError(f"Missing update-zero or periodic validation measurements: {run_id}")
            for record in curve:
                mse = float(record["mse_standardized"])
                if not math.isfinite(mse) or mse < 0:
                    raise ValueError(f"Nonfinite periodic validation MSE: {run_id}")
            for step in checkpoints:
                if not (directory / "checkpoints" / f"step_{step:07d}.pt").is_file():
                    raise ValueError(f"Missing retained step-{step} checkpoint: {run_id}")
            rounds = read_json(directory / "rounds.json")
            epoch_count = EPOCHS
            if [r["epoch"] for r in rounds] != list(range(1, epoch_count + 1)):
                raise ValueError(f"Incomplete training epoch records: {run_id}")
            for record in rounds:
                epoch = record["epoch"]
                key = (config["seed"], epoch)
                if key not in checked_sources:
                    checked_sources[key] = sequence_hash(
                        sources_for_round("fd001", n, *key))
                if record["source_sha256"] != checked_sources[key]:
                    raise ValueError(f"Source stream changed: {run_id}")
                if combo:
                    if record["optimizer_steps"] != epoch * per_epoch:
                        raise ValueError(f"Sequential third-stage optimizer mismatch: {run_id}")
                elif (record["source_draws"] != n or
                      record["optimizer_steps_cumulative"] != epoch * per_epoch):
                    raise ValueError(f"Original source or optimizer count mismatch: {run_id}")
            eligible_updates = selection_update_steps(total)
            score_lookup = {r["optimizer_steps"]: float(r["mse_standardized"])
                            for r in curve}
            if not eligible_updates or any(step not in score_lookup for step in eligible_updates):
                raise ValueError(f"Missing final-quarter validation evidence: {run_id}")
            score = statistics.mean(score_lookup[step] for step in eligible_updates)
        by_hp[(config["method"], lag_key(config), config["batch_size"], config["configuration"])].append(
            dict(config=config, state=state, score=score))
    for (method, lag_set, batch_size, setting), members in sorted(by_hp.items()):
        members.sort(key=lambda m: m["config"]["seed"])
        if [m["config"]["seed"] for m in members] != list(SEEDS):
            raise ValueError(f"Missing one of five seeds: {method}, B{batch_size}, h{setting}")
        chosen_config = members[0]["config"]
        pooled = statistics.mean(m["score"] for m in members) if all(
            math.isfinite(m["score"]) for m in members) else math.inf
        scores.append(dict(method=method, lag_set=lag_set, batch_size=batch_size, configuration=setting,
                           alpha=chosen_config["alpha"], beta=chosen_config["beta"],
                           mse_alpha=chosen_config.get("mse_alpha"),
                           score=pooled, failed_seeds=[m["config"]["seed"] for m in members
                                                      if m["state"] == "numerical_failure"],
                           run_ids=[m["config"]["run_id"] for m in members]))
    for method in METHODS:
        lag_sets = dict.fromkeys(lag_key(config) for config in candidates
                                 if config["method"] == method)
        for lag_set in lag_sets:
            options = [row for row in scores
                       if row["method"] == method and row["lag_set"] == lag_set]
            winning = min(options, key=lambda row: (
                row["score"], row["alpha"], row["beta"] or 0, row["batch_size"]))
            if not math.isfinite(winning["score"]):
                raise ValueError(f"No finite five-seed configuration for {method}/{lag_set}")
            choices.append(winning)
    selected = dict(protocol_id=PROTOCOL_ID, selection_fraction=SELECTION_FRACTION,
                    criterion="five-seed mean standardized validation MSE over the final "
                              f"ceil(25%) of post-start {EVAL_UPDATE_PERIOD}-update evaluations; "
                              "batch selected jointly within each method and lag set",
                    code_sha256=expected_code, manifest_sha256=digest(candidates),
                    data_sha256=seen_data["data_sha256"],
                    validation_sha256=seen_data["validation_sha256"],
                    selected=choices, counts=dict(counts))
    return selected, scores


def evaluate_selected(selection, root, data_dir, output):
    import torch
    from fulltd.data import load_dataset
    from fulltd.sampling import training_universe
    from fulltd.metrics import evaluate
    from fulltd.train import seed_runtime
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    train, _, _ = training_universe(load_dataset("fd001", data_dir, "train"), (1,5))
    val = load_dataset("fd001", data_dir, "val")
    if data_identity(train) != selection["data_sha256"] or data_identity(val) != selection["validation_sha256"]:
        raise ValueError("Prepared FD001 data changed since training")
    manifest = {c["run_id"]: c for c in make_manifest()}
    test = None
    per_seed, learning, gamma_rows = [], [], []
    for choice in selection["selected"]:
        for run_id in choice["run_ids"]:
            config = manifest[run_id]
            directory = Path(root) / run_id
            ckpt_path = directory / "final.pt"
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            status = read_json(directory / "status.json")
            if (ckpt["identity"] != status["identity"]
                    or ckpt["config"] != config or status["state"] != "complete"
                    or (ckpt["progress"].get("phase") != "complete"
                        if config["method"] == "td_mse_combo"
                        else ckpt["progress"]["epoch"] != EPOCHS+1)
                    or json_safe(ckpt["progress"]["curve"]) != read_json(directory / "update_curve.json")):
                raise ValueError(f"Selected checkpoint sidecars disagree: {run_id}")
            for record in ckpt["progress"]["curve"]:
                for metric in ("rmse_cycles", "mae_cycles", "rul_score_sum"):
                    learning.append(dict(method=config["method"], lag_set=lag_key(config), seed=config["seed"],
                                         alpha=config["alpha"], beta=config["beta"],
                                         batch_size=config["batch_size"],
                                         epoch=record["epoch"], optimizer_steps=record["optimizer_steps"],
                                         metric=metric, value=record[metric]))
                if config["method"].startswith("td_"):
                    for typ in record["gamma_probe"]["types"]:
                        gamma_rows.append(dict(method=config["method"], lag_set=lag_key(config), seed=config["seed"],
                                  batch_size=config["batch_size"],
                                  optimizer_steps=record["optimizer_steps"], **typ))
            if test is None:  # Selection complete BEFORE held-out targets are opened.
                test = load_dataset("fd001", data_dir, "test")
            key = dict(selection_sha256=digest(selection), final_sha256=file_hash(ckpt_path),
                       code_sha256=selection["code_sha256"], test_sha256=data_identity(test))
            cache = Path(output) / ".cache" / (run_id+"_"+digest(key)[:16]+".json")
            if cache.is_file():
                cached = read_json(cache)
                if cached["identity"] != key:
                    raise ValueError("Endpoint cache/checkpoint mismatch")
                per_seed.extend(cached["rows"])
                continue
            seed_runtime(config["seed"])
            objective = build_objective(config)
            objective.load_state_dict(ckpt["objective"])
            record_rows = []
            for split, dataset in (("validation",val),("test",test)):
                measurements, endpoints = evaluate(objective.model,dataset,batch_size=128,
                                                  device="cpu",allow_nonfinite=True)
                for units, metric in OUT_METRICS:
                    matched = [r for r in measurements if r["units"] == units and r["metric"] == metric
                               and str(r["horizon"]) == "all"]
                    if len(matched) != 1:
                        raise ValueError(f"Missing output metric: {split} {units}/{metric}")
                    record_rows.append(dict(run_id=run_id,method=config["method"],
                       lag_set=lag_key(config), seed=config["seed"], batch_size=config["batch_size"],
                       split=split, metric=metric,
                       units=units, value=matched[0]["value"]))
            atomic_json(cache,dict(identity=key,rows=record_rows))
            per_seed.extend(record_rows)
    return per_seed, learning, gamma_rows


def summarize(rows,keys,value="value"):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(row)
    summary=[]
    for key,records in sorted(groups.items()):
        records.sort(key=lambda r:int(r["seed"]))
        if [int(r["seed"]) for r in records] != list(SEEDS):
            raise ValueError(f"All selected methods must retain five seeds: {key}")
        values = [float(r[value]) for r in records]
        if all(math.isfinite(v) for v in values):
            mean,sd=statistics.mean(values),statistics.stdev(values)
        else:
            mean,sd=(math.inf if any(math.isinf(v) for v in values) else math.nan),math.nan
        summary.append(dict(zip(keys,key),n=len(values),mean=mean,sample_sd=sd,
                            standard_error=sd/math.sqrt(len(values)),
                            finite_seeds=sum(math.isfinite(v) for v in values)))
    return summary



def paired_differences(rows):
    """Descriptive matched-seed differences, never used for HP selection."""
    from itertools import combinations
    indexed=defaultdict(dict)
    for row in rows:
        key=(row["split"], row["metric"], row["units"])
        method=method_key(row)
        seed=int(row["seed"])
        if seed in indexed[key].setdefault(method,{}):
            raise ValueError("Duplicate selected seed metric")
        indexed[key][method][seed]=float(row["value"])
    output=[]
    # Five training seeds imply df=4; report descriptive intervals only.
    t95_df4=2.7764451051977987
    for (split,metric,units), methods in sorted(indexed.items()):
        methods_ordered = sorted(methods, key=lambda key:
                                 (list(METHODS).index(key.split("@")[0]), key))
        for first,second in combinations(methods_ordered,2):
            if (first not in methods or second not in methods or
                    sorted(methods[first])!=list(SEEDS) or
                    sorted(methods[second])!=list(SEEDS)):
                raise ValueError("Missing paired seed metrics")
            differences=[methods[first][seed]-methods[second][seed] for seed in SEEDS]
            if all(math.isfinite(x) for x in differences):
                mean=statistics.mean(differences)
                sd=statistics.stdev(differences)
                se=sd/math.sqrt(len(SEEDS))
                lo,hi=mean-t95_df4*se,mean+t95_df4*se
            else:
                mean,sd,se,lo,hi=math.nan,math.nan,math.nan,math.nan,math.nan
            output.append(dict(split=split,metric=metric,units=units,
                               first=first,second=second,
                               difference="first minus second (lower favors first)",
                               n=5,mean_difference=mean,
                               sample_sd=sd,standard_error=se,
                               descriptive_t95_lower=lo,descriptive_t95_upper=hi))
    return output


def plot(output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    out=Path(output)
    figs=out/"figures"
    figs.mkdir(parents=True,exist_ok=True)
    curves=load_csv(out/"validation_learning_curves.csv")
    tables=load_csv(out/"performance_summary.csv")
    selected=load_csv(out/"selected_hyperparameters.csv")
    display={method_key(row): f"{LABELS[row['method']]} (B{row['batch_size']}"
             + (f", lags {{{row['lag_set'].replace('-', ',')}}}" if row.get("lag_set", "none") != "none" else "")
             + ")" for row in selected}
    if len(display) != len(selected):
        raise ValueError("Selected TD lag variants must be distinct")
    plotted_methods = list(display)
    # The x-axis always includes the entire training history, including update 0.
    # Limit only the main figures' y-axis: early initialization outliers should
    # not make later RMSE/MAE/RUL differences visually indistinguishable.
    # Each main figure has an unzoomed '_full' companion for auditability.
    y_zoom = {
        "rmse_cycles": (25.0, 40.0),
        "mae_cycles": (18.0, 35.0),
        "rul_score_sum": (5_000.0, 250_000.0),
    }  # Edit these bounds to adjust the vertical range without truncating x.
    for metric,y_label in (("rmse_cycles","Validation RMSE (cycles)"),
                           ("mae_cycles","Validation MAE (cycles)"),
                           ("rul_score_sum","Validation summed RUL score")):
        by_method={}
        for method in plotted_methods:
            rows=sorted((r for r in curves if method_key(r)==method and r["metric"]==metric),
                        key=lambda r:int(r["optimizer_steps"]))
            if not rows or int(rows[0]["optimizer_steps"]) != 0:
                raise ValueError(f"Missing measured update-zero evaluation: {metric}/{method}")
            by_method[method]=rows
        for view in ("zoom","full"):
            fig,ax=plt.subplots(figsize=(9,4.8))
            for method in plotted_methods:
                rows=by_method[method]  # Never drop early updates from the x-axis.
                x=np.array([int(r["optimizer_steps"]) for r in rows])
                y=np.array([float(r["mean"]) for r in rows])
                se=np.array([float(r["standard_error"]) for r in rows])
                ax.plot(x,y,marker="o",ms=2.5,label=display[method])
                if np.isfinite(y).all() and np.isfinite(se).all():
                    ax.fill_between(x,y-se,y+se,alpha=.12)
            ax.set_xlabel("Cumulative Adam optimizer updates")
            ax.set_ylabel(y_label)
            if metric=="rul_score_sum":
                ax.set_yscale("symlog",linthresh=1000)
            if view=="zoom":
                ax.set_ylim(*y_zoom[metric])  # Zoom ONLY y, not x.
                ax.set_title("Y-axis zoom (all training updates retained)",fontsize=10)
            ax.legend(fontsize=8)
            ax.grid(alpha=.2)
            fig.tight_layout()
            stem="validation_"+metric+("_full" if view=="full" else "")+"_vs_updates"
            for ext in ("pdf","svg"):
                fig.savefig(figs/f"{stem}.{ext}",bbox_inches="tight")
            plt.close(fig)
    for metric in ("rmse","mae","rul_score_sum"):
        fig,ax=plt.subplots(figsize=(9,4.5))
        for k,method in enumerate(plotted_methods):
            v=[r for r in tables if method_key(r)==method and r["metric"]==metric]
            if len(v)!=2: raise ValueError("Missing validation or test result")
            for split,dx in (("validation",-.2),("test",.2)):
                record=next(r for r in v if r["split"]==split)
                value=float(record["mean"])
                ax.bar(k+dx,value,width=.38,label=split if k==0 else None)
        ax.set_xticks(range(len(plotted_methods)),[display[k] for k in plotted_methods],rotation=25,ha="right")
        if metric=="rul_score_sum":ax.set_yscale("symlog",linthresh=1000)
        ax.set_ylabel(metric+" (cycles)" if metric!="rul_score_sum" else "Summed RUL score")
        ax.legend()
        fig.tight_layout()
        for ext in ("pdf","svg"):
            fig.savefig(figs/f"validation_test_{metric}.{ext}",bbox_inches="tight")
        plt.close(fig)
    td=out/"td_diagnostics"/"gamma_summary.csv"
    if td.is_file():
        records=load_csv(td)
        fig,ax=plt.subplots(figsize=(9,4.8))
        for method in (m for m in plotted_methods if m.startswith("td_")):
            lag_values = sorted({int(r["lag"]) for r in records if method_key(r)==method})
            for lag in lag_values:
                values=sorted((r for r in records if method_key(r)==method and int(r["lag"])==lag),
                              key=lambda r:int(r["optimizer_steps"]))
                if values:
                    ax.plot([int(r["optimizer_steps"]) for r in values],
                            [float(r["mean"]) for r in values],
                            label=display[method]+f" lag {lag}")
        ax.set_xlabel("Cumulative Adam optimizer updates")
        ax.set_ylabel("Realized Gamma on fixed training probe")
        ax.legend()
        ax.grid(alpha=.2)
        fig.tight_layout()
        for ext in ("pdf","svg"):
            fig.savefig(figs/f"td_operator_gamma_vs_updates.{ext}",bbox_inches="tight")
        plt.close(fig)


def selected_timing(selection, results):
    """Per-seed wall and optimizer-update time; no test labels or new fits."""
    output = []
    for choice in selection["selected"]:
        for run_id in choice["run_ids"]:
            config = next(c for c in make_manifest() if c["run_id"] == run_id)
            status = read_json(Path(results) / run_id / "status.json")
            common = dict(method=config["method"], lag_set=lag_key(config),
                          batch_size=config["batch_size"], seed=config["seed"])
            if config["method"] == "td_mse_combo":
                timing = status["timing"]
                values = [
                    ("mse", "wall_seconds", timing["mse_wall_seconds"]),
                    ("mse", "mean_update_seconds", timing["mse_mean_update_seconds"]),
                    ("gamma", "wall_seconds", timing["gamma_wall_seconds"]),
                    ("gamma", "mean_update_seconds", timing["gamma_mean_update_seconds"]),
                    ("predictor", "wall_seconds", timing["predictor_wall_seconds"]),
                    ("predictor", "mean_update_seconds", timing["predictor_mean_update_seconds"]),
                    ("total", "wall_seconds", timing["attributed_total_seconds"]),
                ]
            else:
                values = [("training", "wall_seconds", status["runtime_seconds"]),
                          ("training", "mean_update_seconds", status.get("mean_update_seconds"))]
            for stage, metric, value in values:
                if value is not None and not math.isfinite(float(value)):
                    raise ValueError(f"Nonfinite timing: {run_id}/{stage}/{metric}")
                output.append(dict(**common, phase=stage, metric=metric, value=value))
    return output


def run(mode):
    data,root,out=dataset_paths("fd001")
    if mode=="plots":
        plot(out)
        return
    selected,scores=audit_and_select(root)
    if mode=="check":
        print("READY:", selected["counts"], f"{len(make_manifest())} candidates audited")
        return
    with file_lock(out/".analysis.lock"):
        atomic_json(out/"selections.json",selected)
        write_csv(out/"hp_scores.csv",scores)
        rows,curve,td=evaluate_selected(selected,root,data,out)
        write_csv(out/"per_seed_metrics.csv",rows)
        write_csv(out/"performance_summary.csv",summarize(rows,("method","lag_set","split","metric","units")))
        write_csv(out/"paired_comparisons.csv",paired_differences(rows))
        write_csv(out/"validation_curve_per_seed.csv",curve)
        write_csv(out/"validation_learning_curves.csv",summarize(curve,("method","lag_set","metric","optimizer_steps")))
        times = selected_timing(selected, root)
        write_csv(out/"timing_per_seed.csv", times)
        comparable = [r for r in times if r["value"] is not None]
        write_csv(out/"timing_summary.csv",
                  summarize(comparable, ("method", "lag_set", "batch_size",
                                         "phase", "metric")))
        write_csv(out/"selected_hyperparameters.csv",[
           dict(method=r["method"],lag_set=r["lag_set"],alpha=r["alpha"],beta=r["beta"],
                batch_size=r["batch_size"], mse_alpha=r.get("mse_alpha"),
                selection_mse=r["score"],
                run_ids=r["run_ids"])
           for r in selected["selected"]])
        write_csv(out/"td_diagnostics"/"gamma_per_seed.csv",td)
        write_csv(out/"td_diagnostics"/"gamma_summary.csv",
                  summarize(td,("method","lag_set","lag","type","optimizer_steps"),value="mean"))
        atomic_json(out/"analysis_notes.json", dict(
            selection=f"last 25% of actual positive {EVAL_UPDATE_PERIOD}-update validation "
                      "evaluations, five-seed pooled standardized MSE; "
                      "batch selected jointly, no test tuning",
            trained_models="100 epochs for MSE/NLL/Pairwise and combo MSE/predictor; combo Gamma up to 100 epochs with training-loss early stopping",
            metrics="RMSE/MAE cycles and summed RUL score on validation/test",
            curves="measured before training and every 500 Adam updates; no interpolation",
            timing="wall-clock includes checkpoint/validation overhead; per-update compute uses forward/backward/optimizer only; combo trains MSE afresh and its wall time is counted",
            caveat="TD batch 128/256 selected using validation; 128 receives more optimizer "
                   "updates in 100 epochs than baseline B256"))
        plot(out)
    print("FD001 four-method nested-lag combo analysis complete:",out)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",nargs="?",default="run",choices=("check","run","plots"))
    args=p.parse_args()
    run(args.mode)


if __name__=="__main__":
    main()
