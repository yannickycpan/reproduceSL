"""Single FD001 sweep analysis: four methods, one shared fd001_outputs directory.

Each method chooses its HP solely by five-seed mean validation MSE at epochs
80/90/100; test is opened only after all four choices are locked. All figures
use measured optimizer steps (including update zero), not interpolation.
"""
from __future__ import annotations

import argparse
from collections import defaultdict, Counter
import csv
import hashlib
import math
from pathlib import Path
import statistics

from fd001_gamma.common import atomic_json, file_lock, read_json, write_csv, file_hash, data_identity
from dscpuscripts.result_paths import paths as dataset_paths
from .fd001_v2_protocol import (make_manifest, METHODS, SEEDS, SELECT_EPOCHS,
                                 EPOCH_VALIDATIONS, EPOCHS, EVAL_UPDATE_PERIOD,
                                 PROTOCOL_ID, digest)
from .fd001_v2_train import code_hash, build_objective

LABELS = dict(mse="MSE (B256)", gaussian_nll="Gaussian NLL (B256)",
              td_typed="TD typed (B128)", td_pairwise="TD pairwise (B128)")
OUT_METRICS = (("cycles", "rmse"), ("cycles", "mae"), ("score", "rul_score_sum"))


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
            raise ValueError(f"Missing FD001 v2 run status: {run_id}")
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
            if int(status["epochs_completed"]) != EPOCHS or int(status["next_epoch"]) != EPOCHS+1:
                raise ValueError(f"Incomplete final epoch: {run_id}")
            n = status["n_sources"]
            if type(n) is not int or n < 1:
                raise ValueError("Invalid source universe size")
            total = expected_steps(n, config["batch_size"])
            if status["optimizer_steps"] != total or status["update_opportunities"] != total:
                raise ValueError(f"Incorrect optimizer budget: {run_id}")
            records = read_json(directory / "validation.json")
            if [r["epoch"] for r in records] != list(EPOCH_VALIDATIONS):
                raise ValueError(f"Incomplete epoch-based selection records: {run_id}")
            for r in records:
                if not math.isfinite(float(r["mse"])) or float(r["mse"]) < 0:
                    raise ValueError(f"Nonfinite selection MSE: {run_id}")
                if r["optimizer_steps"] != r["epoch"] * math.ceil(n/config["batch_size"]):
                    raise ValueError(f"Incorrect epoch selection optimizer step count: {run_id}")
            curve = read_json(directory / "update_curve.json")
            checkpoints = list(range(0, total+1, EVAL_UPDATE_PERIOD))
            if [r["optimizer_steps"] for r in curve] != checkpoints:
                raise ValueError(f"Missing update-zero or periodic validation measurements: {run_id}")
            for step in checkpoints:
                if not (directory / "checkpoints" / f"step_{step:07d}.pt").is_file():
                    raise ValueError(f"Missing retained step-{step} checkpoint: {run_id}")
            rounds = read_json(directory / "rounds.json")
            if [r["epoch"] for r in rounds] != list(range(1,EPOCHS+1)):
                raise ValueError(f"Incomplete training epoch records: {run_id}")
            for round_record in rounds:
                key = (config["seed"], round_record["epoch"])
                if key not in checked_sources:
                    checked_sources[key] = sequence_hash(sources_for_round("fd001", n, *key))
                if round_record["source_sha256"] != checked_sources[key] or round_record["source_draws"] != n:
                    raise ValueError(f"Training source hash/budget mismatch: {run_id}")
                if round_record["optimizer_steps_cumulative"] != round_record["epoch"]*math.ceil(n/config["batch_size"]):
                    raise ValueError(f"Training optimizer count mismatch: {run_id}")
            late = [float(r["mse"]) for r in records if r["epoch"] in SELECT_EPOCHS]
            if len(late) != len(SELECT_EPOCHS):
                raise ValueError(f"Missing late validation window: {run_id}")
            score = statistics.mean(late)
        by_hp[(config["method"], config["configuration"])].append(
            dict(config=config, state=state, score=score))
    for (method, setting), members in sorted(by_hp.items()):
        members.sort(key=lambda m: m["config"]["seed"])
        if [m["config"]["seed"] for m in members] != list(SEEDS):
            raise ValueError(f"Missing one of five seeds: {method}, h{setting}")
        chosen_config = members[0]["config"]
        pooled = statistics.mean(m["score"] for m in members) if all(
            math.isfinite(m["score"]) for m in members) else math.inf
        scores.append(dict(method=method, configuration=setting,
                           alpha=chosen_config["alpha"], beta=chosen_config["beta"],
                           score=pooled, failed_seeds=[m["config"]["seed"] for m in members
                                                      if m["state"] == "numerical_failure"],
                           run_ids=[m["config"]["run_id"] for m in members]))
    for method in METHODS:
        options = [r for r in scores if r["method"] == method]
        winning = min(options, key=lambda r: (r["score"],r["alpha"],r["beta"] or 0))
        if not math.isfinite(winning["score"]):
            raise ValueError(f"No finite five-seed configuration for {method}; no test report produced")
        choices.append(winning)
    selected = dict(protocol_id=PROTOCOL_ID, selection_epochs=list(SELECT_EPOCHS),
                    criterion="five-seed mean validation MSE at full epochs 80,90,100",
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
                    or ckpt["progress"]["epoch"] != EPOCHS+1
                    or ckpt["progress"]["validation"] != read_json(directory / "validation.json")
                    or ckpt["progress"]["curve"] != read_json(directory / "update_curve.json")):
                raise ValueError(f"Selected checkpoint sidecars disagree: {run_id}")
            for record in ckpt["progress"]["curve"]:
                for metric in ("rmse_cycles", "mae_cycles", "rul_score_sum"):
                    learning.append(dict(method=config["method"], seed=config["seed"],
                                         alpha=config["alpha"], beta=config["beta"],
                                         epoch=record["epoch"], optimizer_steps=record["optimizer_steps"],
                                         metric=metric, value=record[metric]))
                if config["method"].startswith("td_"):
                    for typ in record["gamma_probe"]["types"]:
                        gamma_rows.append(dict(method=config["method"], seed=config["seed"],
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
                       seed=config["seed"], split=split, metric=metric,
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
    for metric,y_label in (("rmse_cycles","Validation RMSE (cycles)"),
                           ("mae_cycles","Validation MAE (cycles)"),
                           ("rul_score_sum","Validation summed RUL score")):
        fig,ax=plt.subplots(figsize=(9,4.8))
        for method in METHODS:
            rows=sorted((r for r in curves if r["method"]==method and r["metric"]==metric),
                        key=lambda r:int(r["optimizer_steps"]))
            if not rows or int(rows[0]["optimizer_steps"]) != 0:
                raise ValueError("Learning curve must contain measured update-zero evaluations")
            x=np.array([int(r["optimizer_steps"]) for r in rows])
            y=np.array([float(r["mean"]) for r in rows])
            se=np.array([float(r["standard_error"]) for r in rows])
            ax.plot(x,y,marker="o",ms=2.5,label=LABELS[method])
            if np.isfinite(y).all() and np.isfinite(se).all():
                ax.fill_between(x,y-se,y+se,alpha=.1)
        ax.set_xlabel("Cumulative Adam optimizer updates")
        ax.set_ylabel(y_label)
        if metric == "rul_score_sum":
            ax.set_yscale("symlog",linthresh=1000)
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
        fig.tight_layout()
        stem="validation_"+metric+"_vs_updates"
        for ext in ("pdf","svg"):
            fig.savefig(figs/f"{stem}.{ext}",bbox_inches="tight")
        plt.close(fig)
    for metric in ("rmse","mae","rul_score_sum"):
        fig,ax=plt.subplots(figsize=(9,4.5))
        for k,method in enumerate(METHODS):
            v=[r for r in tables if r["method"]==method and r["metric"]==metric]
            if len(v)!=2: raise ValueError("Missing validation or test result")
            for split,dx in (("validation",-.2),("test",.2)):
                record=next(r for r in v if r["split"]==split)
                value=float(record["mean"])
                ax.bar(k+dx,value,width=.38,label=split if k==0 else None)
        ax.set_xticks(range(len(METHODS)),[LABELS[k] for k in METHODS],rotation=15,ha="right")
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
        for method in ("td_typed","td_pairwise"):
            for typ in (1,2):
                values=sorted((r for r in records if r["method"]==method and int(r["type"])==typ),
                              key=lambda r:int(r["optimizer_steps"]))
                if values:
                    ax.plot([int(r["optimizer_steps"]) for r in values],
                            [float(r["mean"]) for r in values],
                            label=method+f" type {typ}")
        ax.set_xlabel("Cumulative Adam optimizer updates")
        ax.set_ylabel("Realized Gamma on fixed training probe")
        ax.legend()
        ax.grid(alpha=.2)
        fig.tight_layout()
        for ext in ("pdf","svg"):
            fig.savefig(figs/f"td_operator_gamma_vs_updates.{ext}",bbox_inches="tight")
        plt.close(fig)


def run(mode):
    data,root,out=dataset_paths("fd001")
    if mode=="plots":
        plot(out)
        return
    selected,scores=audit_and_select(root)
    if mode=="check":
        print("READY:", selected["counts"], "270 candidates audited")
        return
    with file_lock(out/".analysis.lock"):
        atomic_json(out/"selections.json",selected)
        write_csv(out/"hp_scores.csv",scores)
        rows,curve,td=evaluate_selected(selected,root,data,out)
        write_csv(out/"per_seed_metrics.csv",rows)
        write_csv(out/"performance_summary.csv",summarize(rows,("method","split","metric","units")))
        write_csv(out/"validation_curve_per_seed.csv",curve)
        write_csv(out/"validation_learning_curves.csv",summarize(curve,("method","metric","optimizer_steps")))
        write_csv(out/"selected_hyperparameters.csv",[
           dict(method=r["method"],alpha=r["alpha"],beta=r["beta"],
                selection_mse=r["score"],run_ids=r["run_ids"])
           for r in selected["selected"]])
        write_csv(out/"td_diagnostics"/"gamma_per_seed.csv",td)
        write_csv(out/"td_diagnostics"/"gamma_summary.csv",
                  summarize(td,("method","type","optimizer_steps"),value="mean"))
        atomic_json(out/"analysis_notes.json", dict(
            selection="five-seed validation MSE epochs 80/90/100, no test tuning",
            trained_models="epoch-100 final checkpoints, no refit",
            metrics="RMSE/MAE cycles and summed RUL score on validation/test",
            curves="measured before training and every 500 Adam updates; no interpolation",
            caveat="TD B128 vs baseline B256: unequal numbers of optimizer updates after 100 epochs"))
        plot(out)
    print("FD001 four-method analysis complete:",out)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",nargs="?",default="run",choices=("check","run","plots"))
    args=p.parse_args()
    run(args.mode)


if __name__=="__main__":
    main()
