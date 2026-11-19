"""FD001 B256 *test* learning curves versus normalized training-compute proxy.

Only the original {1,5} TD lags enter this existing B256 compute comparison.
The primary HP selection remains FIVE-SEED, FINAL-QUARTER VALIDATION MSE,
separately within the fixed batch-256 group. We lock these choices BEFORE
loading test data. The selected models' saved step snapshots at 0, 500, ...
5,500 updates are then evaluated on test solely for descriptive learning
curves. Test metrics never choose HPs, snapshots, or stopping points.

The old archive contains validation measurements at every 500 updates but test
measurements only at final checkpoints. Therefore generating a test *curve*
requires inference using the existing saved step checkpoints, NOT retraining.

No per-checkpoint wall-clock times were recorded. The horizontal axis is
a transparent training-compute proxy, NOT measured seconds:
  MSE, Gaussian NLL: one predictor source example per training source.
  TD typed, TD pairwise: two predictor examples, source + partner.
It omits per-model head operations, overhead and differing hardware efficiency.
"""
from __future__ import annotations

from collections import defaultdict
import math
from pathlib import Path

from dscpuscripts.common import (atomic_json, cpu_environment, data_identity,
                                  file_hash, file_lock, read_json, write_csv)
from dscpuscripts.fd001_analysis import audit_and_select, summarize
from dscpuscripts.fd001_protocol import (EVAL_UPDATE_PERIOD, METHODS, SEEDS,
                                          make_manifest)
from dscpuscripts.fd001_train import build_objective, code_hash, json_safe
from dscpuscripts.result_paths import paths as dataset_paths
from fulltd.protocol import digest

REVISION = "fd001-test-compute-proxy-v1"
METRICS = (("rmse_cycles", "cycles", "rmse"),
           ("rul_score_sum", "score", "rul_score_sum"))


def processed_sources(n_sources: int, batch: int, optimizer_steps: int) -> int:
    """Count training examples including the shorter final batch each epoch."""
    if n_sources < 1 or batch < 1 or optimizer_steps < 0:
        raise ValueError("Expected positive sources/batch and nonnegative updates")
    steps_per_epoch = math.ceil(n_sources / batch)
    complete, remaining = divmod(optimizer_steps, steps_per_epoch)
    return complete * n_sources + min(n_sources, remaining * batch)


def relative_compute(method: str, n_sources: int, batch: int, steps: int) -> float:
    """MSE B256 for 100 complete passes has normalized budget 1."""
    if method not in METHODS:
        raise ValueError("Unknown FD001 method")
    multiplier = 2 if method.startswith("td_") else 1
    return multiplier * processed_sources(n_sources, batch, steps) / (100 * n_sources)


def validation_only_plan(results: Path):
    """Audit the campaign and choose the B256 HP of each method WITHOUT test."""
    canonical, scores = audit_and_select(results)
    if canonical["counts"].get("complete") != len(make_manifest()):
        raise ValueError("Test curves require all prescribed FD001 candidates complete")
    chosen = []
    for method in METHODS:
        choices = [row for row in scores
                   if row["method"] == method and row["lag_set"] == ("1-5" if method.startswith("td_") else "none")
                    and row["batch_size"] == 256]
        if not choices:
            raise ValueError(f"Missing B256 HP grid for {method}")
        best = min(choices, key=lambda row: (
            row["score"], row["alpha"], row["beta"] or 0.0,
            row["configuration"]))
        if not math.isfinite(best["score"]) or len(best["run_ids"]) != len(SEEDS):
            raise ValueError(f"No finite five-seed B256 setting for {method}")
        # Enforce the same measured update grid for all seeds and B256 methods.
        n_sources = None
        steps = None
        for run_id in best["run_ids"]:
            folder = results / run_id
            status = read_json(folder / "status.json")
            current_n = int(status["n_sources"])
            current_steps = [int(r["optimizer_steps"])
                             for r in read_json(folder / "update_curve.json")]
            expected = list(range(0, int(status["optimizer_steps"])+1,
                                  EVAL_UPDATE_PERIOD))
            if current_steps != expected or len(expected) < 2:
                raise ValueError(f"Missing saved measurements at {run_id}")
            if n_sources is not None and current_n != n_sources:
                raise ValueError("Inconsistent source universe")
            if steps is not None and current_steps != steps:
                raise ValueError("Mismatched B256 checkpoint cadence")
            n_sources, steps = current_n, current_steps
            for step in steps:
                snapshot = folder / "checkpoints" / f"step_{step:07d}.pt"
                if not snapshot.is_file():
                    raise FileNotFoundError(f"Missing selected snapshot: {snapshot}")
        chosen.append(dict(method=method, batch_size=256,
                           alpha=best["alpha"], beta=best["beta"],
                           configuration=best["configuration"],
                           selection_mse=best["score"], run_ids=best["run_ids"],
                           n_sources=n_sources, optimizer_steps=steps))
    return dict(revision=REVISION, protocol_id=canonical["protocol_id"],
                selection_rule="Five-seed mean validation MSE at final ceil(25%) "
                               "positive-update measurements, restricted to fixed B256",
                scientific_code_sha256=code_hash(),
                validation_sha256=canonical["validation_sha256"],
                selections=chosen)


def _test_metrics(model, dataset):
    from fulltd.metrics import evaluate
    metrics, _ = evaluate(model, dataset, batch_size=128, device="cpu",
                          allow_nonfinite=True)
    by_key = {(row["units"], str(row["horizon"]), row["metric"]): row["value"]
              for row in metrics}
    output = {}
    for name, units, metric in METRICS:
        key = (units, "all", metric)
        if key not in by_key:
            raise ValueError(f"Missing test metric: {key}")
        output[name] = float(by_key[key])
    return output


def evaluate_test_curve(plan, results: Path, output: Path, test):
    """Read snapshots and cache test predictions; no training or test selection."""
    import torch
    from fulltd.train import seed_runtime
    manifest = {config["run_id"]: config for config in make_manifest()}
    test_sha = data_identity(test)
    rows = []
    for selected in plan["selections"]:
        for run_id in selected["run_ids"]:
            config = manifest[run_id]
            folder = results / run_id
            status = read_json(folder / "status.json")
            observations = read_json(folder / "update_curve.json")
            seed_runtime(config["seed"])
            objective = build_objective(config)
            for step in selected["optimizer_steps"]:
                path = folder / "checkpoints" / f"step_{step:07d}.pt"
                identity = dict(revision=REVISION, plan_sha256=digest(plan),
                                checkpoint_sha256=file_hash(path), test_sha256=test_sha)
                cache = output / "test_checkpoint_cache" / (
                    run_id + "_" + str(step) + "_" + digest(identity)[:16] + ".json")
                if cache.is_file():
                    saved = read_json(cache)
                    if saved["identity"] != identity:
                        raise ValueError(f"Stale test-curve cache: {cache}")
                    rows.extend(saved["rows"])
                    continue
                # All inputs are locally generated trusted checkpoints, never
                # deserialize untrusted model files.
                ckpt = torch.load(path, map_location="cpu", weights_only=False)
                if (ckpt.get("config") != config
                        or ckpt.get("identity") != status.get("identity")
                        or int(ckpt["progress"]["optimizer_steps"]) != step):
                    raise ValueError(f"Checkpoint identity/step mismatch: {path}")
                snapshot_curve = json_safe(ckpt["progress"]["curve"])
                if (snapshot_curve != observations[:len(snapshot_curve)]
                        or snapshot_curve[-1]["optimizer_steps"] != step):
                    raise ValueError(f"Checkpoint validation record mismatch: {path}")
                objective.load_state_dict(ckpt["objective"])
                values = _test_metrics(objective.model, test)
                one = [
                    dict(method=config["method"], batch_size=256,
                         seed=config["seed"], run_id=run_id, optimizer_steps=step,
                         compute_proxy=relative_compute(config["method"],
                                                        selected["n_sources"], 256, step),
                         metric=metric, value=value)
                    for metric, value in values.items()
                ]
                atomic_json(cache, dict(identity=identity, rows=one))
                rows.extend(one)
                print(f"TEST {run_id} checkpoint step {step}", flush=True)
    return rows


def summary_table(rows, plan):
    summary = summarize(rows, ("method", "batch_size", "optimizer_steps", "metric"))
    lookup = {r["method"]: r["n_sources"] for r in plan["selections"]}
    for row in summary:
        row["compute_proxy"] = relative_compute(row["method"],
                                                 lookup[row["method"]],
                                                 int(row["batch_size"]),
                                                 int(row["optimizer_steps"]))
    return summary


def plot(summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    labels = {"mse":"MSE B256", "gaussian_nll":"Gaussian NLL B256",
              "td_typed":"TD typed B256", "td_pairwise":"TD pairwise B256"}
    grouped = defaultdict(list)
    for row in summary:
        grouped[(row["method"], row["metric"])].append(row)
    figures = output / "figures" / "compute_proxy"
    figures.mkdir(parents=True, exist_ok=True)
    for metric, ylabel in (("rmse_cycles", "TEST RMSE (cycles)"),
                           ("rul_score_sum", "TEST RUL score (sum)")):
        for view in ("zoom", "full"):
            fig, ax = plt.subplots(figsize=(9.4,5.2))
            for method in METHODS:
                seq = sorted(grouped[(method,metric)],
                             key=lambda x:int(x["optimizer_steps"]))
                if not seq:
                    raise ValueError(f"Missing {method} test-curve results")
                x = np.array([float(row["compute_proxy"]) for row in seq])
                y = np.array([float(row["mean"]) for row in seq])
                se = np.array([float(row["standard_error"]) for row in seq])
                ax.plot(x, np.where(np.isfinite(y),y,np.nan),
                        lw=1.9,marker="o",markersize=3,label=labels[method])
                lower,upper = y-se,y+se
                if np.any(np.isfinite(lower)&np.isfinite(upper)):
                    ax.fill_between(x,lower,upper,
                                    where=np.isfinite(lower)&np.isfinite(upper),
                                    alpha=.11)
            # Complete training trajectory on x; only the y-axis is zoomed.
            ax.set_xlim(0,2.04)
            ax.set_xlabel("Approximate predictor training compute (MSE B256 at 100 epochs = 1)")
            ax.set_ylabel(ylabel)
            if metric=="rmse_cycles" and view=="zoom":
                ax.set_ylim(25,40)
            if metric=="rul_score_sum":
                ax.set_yscale("symlog",linthresh=1000)
                if view=="zoom":
                    ax.set_ylim(3000,250000)
            ax.axvline(1,ls="--",lw=1,alpha=.55,
                       label="B256 baseline 100-epoch compute")
            ax.grid(alpha=.25)
            ax.legend(loc="best",fontsize=8)
            ax.set_title("Held-out TEST predictions; compute proxy, not measured seconds",
                         fontsize=10)
            fig.tight_layout()
            name=f"fd001_b256_test_{metric}_vs_compute_proxy"
            if view=="full":
                name+="_full"
            for ext in ("pdf","png","svg"):
                fig.savefig(figures/f"{name}.{ext}",dpi=180,bbox_inches="tight")
            plt.close(fig)
    return figures


def main():
    import torch
    from fulltd.data import load_dataset
    cpu_environment()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    data, results, primary = dataset_paths("fd001")
    output = primary / "test_compute"
    with file_lock(output / ".analysis.lock"):
        plan = validation_only_plan(results)
        lock = output / "validation_locked_b256_hps.json"
        if lock.is_file():
            if read_json(lock) != plan:
                raise ValueError("Previously locked selection differs; do not overwrite")
        else:
            atomic_json(lock, plan)
        # Test data are opened strictly AFTER the validation-only plan is locked.
        test = load_dataset("fd001", data, "test")
        rows = evaluate_test_curve(plan, results, output, test)
        summary = summary_table(rows, plan)
        write_csv(output / "test_learning_curve_per_seed.csv", rows)
        write_csv(output / "test_learning_curve_summary.csv", summary)
        write_csv(output / "b256_selected_hyperparameters.csv",[
            {k: v for k,v in selected.items() if k!="optimizer_steps"}
            for selected in plan["selections"]])
        figures = plot(summary, primary)
    print("TEST B256 checkpoint curves saved:", figures)
    print("Test means/SE and cached measurements saved:", output)
    print("X is approximate predictor training work; measured wall-clock timestamps do not exist.")


if __name__ == "__main__":
    main()
