"""Read-only Gamma diagnostics for selected FD001 typed-TD models.

Usage:
    bash dscpuscripts/analyze_typed_ablation.sh fd001 --gamma

Reads saved checkpoints, gamma probes and existing test CSV; does not train,
select hyperparameters, or modify existing model/validation/test records.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

TAU = 1e-3  # frozen nonnegative typed TD parameterization
METHODS = ("td_typed", "td_semi_typed", "td_nonnegative_typed")


def read_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    if not rows:
        raise ValueError("Refuse to write an empty diagnostic")
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def file_sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def gamma_and_derivative(method, raw):
    if method != "td_nonnegative_typed":
        return raw, 1.0
    scaled = raw / TAU
    if scaled >= 0:
        return (raw + TAU * math.log1p(math.exp(-scaled)),
                1 / (1 + math.exp(-scaled)))
    expv = math.exp(scaled)
    return TAU * math.log1p(expv), expv / (1 + expv)


def analyze(old_root, new_root):
    import torch

    old_root, new_root = Path(old_root), Path(new_root)
    output = new_root / "typed_ablation_outputs"
    old = read_json(output / "four_method_selection.json")
    new = read_json(output / "selection.json")
    if old.get("dataset") != "fd001" or new.get("dataset") != "fd001":
        raise ValueError("This diagnostic is restricted to FD001")
    choices = [(entry, old_root) for entry in old["old_selections"]
               if entry["method"] == "td_typed"]
    choices += [(entry, new_root) for entry in new["selected"]
                if entry["method"] in METHODS[1:]]
    if len(choices) != 9 or {c["method"] for c, _ in choices} != set(METHODS):
        raise ValueError("Expected three methods and three lag sets each")

    test = {}
    testfile = output / "four_method_per_seed.csv"
    if testfile.exists():
        for row in read_csv(testfile):
            if (row["method"] in METHODS and row["metric"] == "rmse"
                    and row["units"] == "cycles"):
                key = row["method"], row["lag_set"], int(row["seed"])
                if key in test:
                    raise ValueError("Duplicate stored test RMSE")
                test[key] = float(row["value"])

    final_rows, trace_rows, fit_rows = [], [], []
    for choice, root in choices:
        method, lag_set = choice["method"], choice["lag_set"]
        if len(choice["run_ids"]) != 5:
            raise ValueError(f"Expected five seeds: {method}/{lag_set}")
        encountered = set()
        for run_id in choice["run_ids"]:
            directory = root / run_id
            status = read_json(directory / "status.json")
            config = status["config"]
            seed = int(config["seed"])
            encountered.add(seed)
            if (config["method"] != method or status["state"] != "complete"
                    or config["batch_size"] != 256
                    or "-".join(map(str, config["lags"])) != lag_set):
                raise ValueError(f"Wrong or unfinished checkpoint: {run_id}")
            checkpoint_path = directory / "final.pt"
            if file_sha(checkpoint_path) != choice["endpoint_sha256"][run_id]:
                raise ValueError(f"Selected checkpoint changed: {run_id}")
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if (checkpoint["config"] != config
                    or checkpoint["identity"] != status["identity"]):
                raise ValueError(f"Checkpoint identity mismatch: {run_id}")
            if "operator" not in checkpoint["objective"]:
                raise ValueError(f"Missing typed operator in {run_id}")
            raw = checkpoint["objective"]["operator"].detach().cpu().double().reshape(-1).tolist()
            del checkpoint
            lags = list(config["lags"])
            if len(raw) != len(lags) or any(not math.isfinite(z) for z in raw):
                raise ValueError(f"Nonfinite/mismatched raw Gamma: {run_id}")
            curve = read_json(directory / "update_curve.json")
            positives = [r for r in curve if int(r["optimizer_steps"]) > 0]
            if not positives:
                raise ValueError(f"Missing observed validation history: {run_id}")
            last_quarter = positives[-math.ceil(.25 * len(positives)):]
            rounds = read_json(directory / "rounds.json")
            if len(rounds) != config["epochs"]:
                raise ValueError(f"Incomplete training rounds: {run_id}")
            fit_rows.append(dict(method=method, lag_set=lag_set, seed=seed,
                                 run_id=run_id, alpha=config["alpha"], beta=config["beta"],
                                 final_epoch_sampled_pair_half_mse=rounds[-1]["sampled_half_pair_mse"],
                                 best_observed_validation_mse=min(float(x["mse_standardized"]) for x in positives),
                                 last_observed_validation_mse=curve[-1]["mse_standardized"],
                                 final_quarter_validation_mse=mean(float(x["mse_standardized"])
                                                                   for x in last_quarter),
                                 test_rmse_cycles=test.get((method, lag_set, seed), "")))
            for index, (lag, value) in enumerate(zip(lags, raw), start=1):
                gamma, derivative = gamma_and_derivative(method, float(value))
                final_rows.append(dict(method=method, lag_set=lag_set, seed=seed,
                                       run_id=run_id, lag=int(lag), lag_index=index,
                                       raw_gamma=float(value), effective_gamma=gamma,
                                       derivative_factor=derivative, negative=int(gamma < 0),
                                       near_zero=int(abs(gamma) < 1e-5),
                                       saturated_1e_3=int(method == "td_nonnegative_typed"
                                                         and derivative < 1e-3),
                                       saturated_1e_6=int(method == "td_nonnegative_typed"
                                                         and derivative < 1e-6)))
            for record in curve:
                probe = record.get("gamma_probe", {})
                entries = probe.get("types", [])
                if len(entries) != len(lags):
                    raise ValueError(f"Missing typed Gamma probe in {run_id}")
                for index, entry in enumerate(entries, start=1):
                    if entry["type"] != index or entry["lag"] != lags[index-1]:
                        raise ValueError(f"Gamma probe lag mismatch: {run_id}")
                    trace_rows.append(dict(method=method, lag_set=lag_set, seed=seed,
                                           run_id=run_id, lag=int(entry["lag"]),
                                           optimizer_steps=int(record["optimizer_steps"]),
                                           validation_mse=record["mse_standardized"],
                                           gamma_mean=entry["mean"], gamma_std=entry["std"],
                                           gamma_min=entry["minimum"], gamma_max=entry["maximum"],
                                           negative_fraction=entry["negative_fraction"],
                                           pair_count=entry["pair_count"]))
        if encountered != set(range(5)):
            raise ValueError(f"Missing/duplicate seed: {method}/{lag_set}")

    grouped = defaultdict(list)
    for row in final_rows:
        grouped[row["method"],row["lag_set"],row["lag"]].append(row)
    summaries = []
    for (method, lag_set, lag), rows in sorted(grouped.items()):
        if sorted(r["seed"] for r in rows) != list(range(5)):
            raise ValueError("Five-seed Gamma summary incomplete")
        values = [r["effective_gamma"] for r in rows]
        summaries.append(dict(method=method, lag_set=lag_set, lag=lag, n=5,
                              gamma_mean=mean(values), gamma_sd=stdev(values),
                              mean_abs_gamma=mean(abs(v) for v in values),
                              negative_seeds=sum(r["negative"] for r in rows),
                              near_zero_seeds=sum(r["near_zero"] for r in rows),
                              saturation_1e_3_seeds=sum(r["saturated_1e_3"] for r in rows),
                              saturation_1e_6_seeds=sum(r["saturated_1e_6"] for r in rows),
                              mean_derivative_factor=mean(r["derivative_factor"] for r in rows)))
    target=output / "gamma_mechanism"
    target.mkdir(parents=True, exist_ok=True)
    for filename, rows in (("selected_final_gamma.csv",final_rows),
                           ("selected_gamma_trajectory.csv",trace_rows),
                           ("selected_gamma_summary.csv",summaries),
                           ("selected_optimization_diagnostics.csv",fit_rows)):
        write_csv(target / filename, rows)
    notes=dict(scope="FD001 three typed methods, three lag sets, five selected seeds",
               tau=TAU, interpretation="Gamma is one coefficient per lag, not per pair",
               negative="For typed Gamma, per-lag negative_fraction is zero or one",
               saturation="Derivative factor=1 for signed methods, sigmoid(raw/tau) for nonnegative;"
                          " <1e-3 and <1e-6 mark weak sensitivity, not proof of optimization failure",
               fit="Last-epoch sampled pair loss is NOT final supervised training MSE",
               validation="Last stored checkpoint may precede the exact final optimizer step",
               privacy="Uses saved test summary only, does not open test examples",
               safety="Read-only old/new checkpoints. New diagnostic CSVs in gamma_mechanism/")
    (target/"definitions.json").write_text(json.dumps(notes,indent=2)+"\n",encoding="utf-8")
    print(f"Gamma mechanism saved at: {target}")
    print(f"Final Gamma coordinates: {len(final_rows)}, recorded probes: {len(trace_rows)},"
          f" lag summaries: {len(summaries)}, selected runs: {len(fit_rows)}")
    for row in summaries:
        if row["lag_set"]=="1-3-5-10-15":
            print(f"{row['method']} lag={row['lag']}: meanGamma={row['gamma_mean']:.5g},"
                  f" negative seeds={row['negative_seeds']}/5,"
                  f" saturated (<1e-3)={row['saturation_1e_3_seeds']}/5")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("dataset", choices=("fd001",))
    args=p.parse_args()
    base=Path(os.environ.get("FULLTD_RESULTS_ROOT", Path.home()/"scratch/fulltd"))
    old_root=Path(os.environ.get("FULLTD_FD001_ROOT",base/"fd001_results"))
    new_root=Path(os.environ.get("FULLTD_TYPED_ABLATION_FD001_ROOT",
                                 base/"typed_ablation_fd001_results"))
    analyze(old_root, new_root)


if __name__=="__main__":
    main()
