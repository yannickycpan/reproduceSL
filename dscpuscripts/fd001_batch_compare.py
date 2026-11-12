"""FD001 batch-specific and equal-update comparisons; no training/HP rule changes.

Full budget: select each method/batch with the canonical final-quarter, five-seed
validation-MSE rule and evaluate the actual epoch-100 final checkpoint.
Equal updates: only for TD batch-128, truncate the *selection timeline* to the
batch-256 100-epoch update count. Select with the same final-quarter rule and
evaluate the saved checkpoint at that matched update, not the epoch-100 model.

Use the original {1,5} TD lag set for the existing batch comparison.
All outputs live separately in fd001_outputs/batch_comparison/.
"""
from __future__ import annotations

import argparse
import math
import statistics

from dscpuscripts.common import (atomic_json, data_identity, file_hash,
                                  file_lock, read_json, write_csv)
from dscpuscripts.fd001_analysis import (OUT_METRICS, audit_and_select,
                                         expected_steps, selection_update_steps,
                                         summarize)
from dscpuscripts.fd001_protocol import (EVAL_UPDATE_PERIOD, METHODS,
                                         PROTOCOL_ID, SEEDS, make_manifest)
from dscpuscripts.fd001_train import build_objective, code_hash, json_safe
from dscpuscripts.result_paths import paths as result_paths
from fulltd.protocol import digest

TD_METHODS = ("td_typed", "td_pairwise")
ANALYSIS_REVISION = "fd001-batch-comparison-v1"


def _rank(item):
    return (item["score"], item["alpha"], item["beta"] or 0.0,
            item["configuration"])


def _choose(rows, method, batch, regime, final_update):
    lag = "1-5" if method.startswith("td_") else "none"
    pool = [r for r in rows if r["method"] == method and r.get("lag_set", lag) == lag
            and r["batch_size"] == batch]
    if not pool:
        raise ValueError(f"Missing hyperparameter grid: {method}, batch {batch}")
    winner = min(pool, key=_rank)
    if not math.isfinite(winner["score"]):
        raise ValueError(f"No finite five-seed HP option: {method}, batch {batch}")
    run_ids = winner["run_ids"]
    if len(run_ids) != len(SEEDS):
        raise ValueError("Selected HP does not include all five seeds")
    return dict(regime=regime, method=method, batch_size=batch,
                configuration=winner["configuration"], alpha=winner["alpha"],
                beta=winner["beta"], selection_mse=winner["score"],
                optimizer_steps=final_update, run_ids=run_ids)


def _matched_score(results, record, update_limit):
    """Select using ONLY observed validation MSE up to update_limit."""
    run_ids = record["run_ids"]
    step_window = selection_update_steps(update_limit, EVAL_UPDATE_PERIOD)
    per_seed = []
    for run_id in run_ids:
        folder = results / run_id
        status = read_json(folder / "status.json")
        if status["state"] == "numerical_failure":
            return math.inf
        if status["state"] != "complete":
            raise ValueError(f"Run not complete: {run_id}")
        observations = read_json(folder / "update_curve.json")
        eligible = {int(x["optimizer_steps"]): float(x["mse_standardized"])
                    for x in observations if int(x["optimizer_steps"]) <= update_limit}
        if any(step not in eligible or not math.isfinite(eligible[step])
               for step in step_window):
            raise ValueError(f"Missing/nonfinite matched-update validation: {run_id}")
        per_seed.append(statistics.mean(eligible[step] for step in step_window))
    return statistics.mean(per_seed)


def compute_plan(results):
    """Audit every candidate BEFORE any test predictions are opened."""
    original, scores = audit_and_select(results)
    manifest = make_manifest()
    if original["counts"].get("complete") != len(manifest):
        raise ValueError("Comparison requires every prescribed run complete; no failures")
    reference = next(c for c in manifest if c["method"] == "mse" and c["seed"] == 0)
    status = read_json(results / reference["run_id"] / "status.json")
    sources = status["n_sources"]
    matched_updates = expected_steps(sources, 256)
    full_128_updates = expected_steps(sources, 128)
    if (matched_updates >= full_128_updates or matched_updates % EVAL_UPDATE_PERIOD):
        raise ValueError("Cannot identify exact measured matched-update boundary")
    candidates = []
    # Full 100-epoch comparison: same HP selection rule, separately for B128/B256.
    for method in METHODS:
        for batch in ((128, 256) if method in TD_METHODS else (256,)):
            limit = expected_steps(sources, batch)
            candidates.append(_choose(scores, method, batch, "full_100_epochs", limit))
    # Secondary comparison: same criterion, applied to shorter available
    # validation timeline (up to the B256 update budget).
    matched_scores = []
    for method in TD_METHODS:
        for row in scores:
            if row["method"] == method and row["lag_set"] == "1-5" and row["batch_size"] == 128:
                matched_scores.append({**row, "score": _matched_score(results, row, matched_updates)})
        candidates.append(_choose(matched_scores, method, 128,
                                  "matched_updates", matched_updates))
    choices = [item for item in candidates]
    plan = dict(analysis_revision=ANALYSIS_REVISION, protocol_id=PROTOCOL_ID,
                original_joint_selection=[row for row in original["selected"]
                                           if row["lag_set"] in ("none", "1-5")],
                original_criterion=original["criterion"],
                matched_update_budget=matched_updates,
                matched_selection_steps=list(selection_update_steps(matched_updates, EVAL_UPDATE_PERIOD)),
                full_128_update_budget=full_128_updates,
                trained_scientific_code_sha256=code_hash(),
                data_sha256=original["data_sha256"],
                validation_sha256=original["validation_sha256"],
                choices=choices)
    # Confirm selected weights are available before locking any selections.
    for item in choices:
        for run_id in item["run_ids"]:
            folder = results / run_id
            checkpoint = (folder / "final.pt" if item["regime"] == "full_100_epochs"
                          else folder / "checkpoints" / f"step_{item['optimizer_steps']:07d}.pt")
            if not checkpoint.is_file():
                raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
    return plan, matched_scores


def _score_metrics(model, dataset):
    from fulltd.metrics import evaluate
    observations, _ = evaluate(model, dataset, batch_size=128,
                               device="cpu", allow_nonfinite=True)
    values = {}
    for units, metric in OUT_METRICS:
        if metric == "mae":
            continue
        candidates = [r for r in observations if r["units"] == units
                      and r["metric"] == metric and str(r["horizon"]) == "all"]
        if len(candidates) != 1:
            raise ValueError(f"Missing unique metric {metric}/{units}")
        values[metric] = float(candidates[0]["value"])
    return values


def _evaluate_choice(item, plan, root, val, test, out):
    import torch
    from fulltd.train import seed_runtime
    by_id = {r["run_id"]: r for r in make_manifest()}
    rows = []
    for run_id in item["run_ids"]:
        config = by_id[run_id]
        folder = root / run_id
        ckpt_path = (folder / "final.pt" if item["regime"] == "full_100_epochs"
                     else folder / "checkpoints" / f"step_{item['optimizer_steps']:07d}.pt")
        identity = dict(analysis_revision=ANALYSIS_REVISION,
                        plan_sha256=digest(plan), run_id=run_id,
                        ckpt_sha256=file_hash(ckpt_path),
                        validation_sha256=data_identity(val),
                        test_sha256=data_identity(test),
                        update=item["optimizer_steps"])
        cache = out / ".cache" / (run_id + "_step" + str(item["optimizer_steps"])
                                  + "_" + digest(identity)[:16] + ".json")
        if cache.is_file():
            cached = read_json(cache)
            if cached.get("identity") != identity:
                raise ValueError(f"Cache identity changed: {cache}")
            rows.extend(cached["rows"])
            continue
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        status = read_json(folder / "status.json")
        if (ckpt.get("identity") != status.get("identity")
                or ckpt.get("config") != config
                or status["state"] != "complete"
                or int(ckpt["progress"]["optimizer_steps"]) != item["optimizer_steps"]):
            raise ValueError(f"Checkpoint/status mismatch: {ckpt_path}")
        curve = read_json(folder / "update_curve.json")
        snapshot_curve = json_safe(ckpt["progress"]["curve"])
        if snapshot_curve != curve[:len(snapshot_curve)]:
            raise ValueError(f"Snapshot validation evidence differs from history: {run_id}")
        if (not snapshot_curve or
                snapshot_curve[-1]["optimizer_steps"] != item["optimizer_steps"]):
            raise ValueError(f"Selected step has no matching validation: {run_id}")
        seed_runtime(config["seed"])
        objective = build_objective(config)
        objective.load_state_dict(ckpt["objective"])
        one = []
        for split, dataset in (("validation", val), ("test", test)):
            measurements = _score_metrics(objective.model, dataset)
            for metric, value in measurements.items():
                one.append(dict(regime=item["regime"], method=item["method"],
                                batch_size=item["batch_size"],
                                optimizer_steps=item["optimizer_steps"],
                                run_id=run_id, seed=config["seed"], split=split,
                                metric=metric, value=value))
        atomic_json(cache, dict(identity=identity, rows=one))
        rows.extend(one)
        print(f"Evaluated {item['regime']} {run_id} step={item['optimizer_steps']}", flush=True)
    return rows


def build_comparison_table(choices, summary):
    grouped = {(r["regime"], r["method"], int(r["batch_size"]), r["split"], r["metric"]): r
               for r in summary}
    output = []
    for item in choices:
        row = {k: item[k] for k in ("regime", "method", "batch_size", "alpha", "beta",
                                  "configuration", "selection_mse", "optimizer_steps")}
        for split in ("test", "validation"):
            for metric in ("rmse", "rul_score_sum"):
                v = grouped[item["regime"], item["method"], item["batch_size"], split, metric]
                row[f"{split}_{metric}_mean"] = v["mean"]
                row[f"{split}_{metric}_se"] = v["standard_error"]
                row[f"{split}_{metric}_n"] = v["n"]
        output.append(row)
    return output


def _pretty(value, metric):
    if not math.isfinite(float(value)):
        return str(value)
    return f"{value:,.0f}" if metric == "rul_score_sum" else f"{value:.2f}"


def make_markdown(table, plan):
    lines = ["# FD001 batch-specific and matched-update performance", "",
             "The canonical original selection rule is unchanged. All results use five seeds",
             "and mean ± standard error. Lower is better. Test data were not used for selection.",
             "", "| Method | α | β | Update | Test RMSE | Test RUL score | Validation RMSE | Validation RUL score |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in table:
        kind = "100 epochs" if row["regime"] == "full_100_epochs" else "matched updates"
        name = f"{row['method']} B{row['batch_size']} ({kind})"
        cols = [name, f"{row['alpha']:.5g}",
                "—" if row["beta"] is None else f"{row['beta']:.5g}",
                str(row["optimizer_steps"])]
        for split, metric in (("test", "rmse"), ("test", "rul_score_sum"),
                              ("validation", "rmse"), ("validation", "rul_score_sum")):
            mean, se = row[f"{split}_{metric}_mean"], row[f"{split}_{metric}_se"]
            cols.append(f"{_pretty(mean, metric)} ± {_pretty(se, metric)}")
        lines.append("| " + " | ".join(cols) + " |")
    lines.extend(["", "The original B256 100-epoch update budget is "
                  f"{plan['matched_update_budget']}; B128 has "
                  f"{plan['full_128_update_budget']} updates in 100 epochs.",
                  "Matched B128 uses its checkpoint at the B256 update budget, and selects",
                  "each B128 HP by five-seed mean validation MSE on the final 25% of the",
                  "observations available up to that exact matched checkpoint: "
                  f"{plan['matched_selection_steps']}.",
                  "The original full-budget results still use their own final-quarter rule.",
                  "No training or checkpoints were modified; outputs are separate.", ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("check", "run"), help="check locks selection; run evaluates")
    args = parser.parse_args()
    data_dir, results, primary = result_paths("fd001")
    output = primary / "batch_comparison"
    output.mkdir(parents=True, exist_ok=True)
    with file_lock(output / ".analysis.lock"):
        plan, matched_scores = compute_plan(results)
        lock = output / "selections.json"
        if lock.is_file():
            if read_json(lock) != plan:
                raise RuntimeError("Locked batch selection disagrees with source data; investigate")
        elif args.mode == "run":
            raise RuntimeError("Run 'check' first to lock the validation-only selection")
        else:
            atomic_json(lock, plan)
        if args.mode == "check":
            write_csv(output / "matched_hp_scores.csv", matched_scores)
            for choice in plan["choices"]:
                print(choice["regime"], choice["method"], choice["batch_size"],
                      choice["alpha"], choice["beta"], "updates", choice["optimizer_steps"],
                      "validation selection MSE", choice["selection_mse"])
            print("Validation-only choices locked. No test evaluation performed.")
            return
        # Access to test labels occurs only after locked selections have been checked.
        import torch
        from fulltd.data import load_dataset
        from fulltd.sampling import training_universe
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        train, _, _ = training_universe(load_dataset("fd001", data_dir, "train"), (1,5))
        val = load_dataset("fd001", data_dir, "val")
        if (data_identity(train) != plan["data_sha256"] or
                data_identity(val) != plan["validation_sha256"]):
            raise RuntimeError("Prepared training/validation data differ from training campaign")
        test = load_dataset("fd001", data_dir, "test")
        rows = []
        for choice in plan["choices"]:
            rows.extend(_evaluate_choice(choice, plan, results, val, test, output))
        summary = summarize(rows, ("regime", "method", "batch_size", "split", "metric"))
        table = build_comparison_table(plan["choices"], summary)
        write_csv(output / "per_seed_metrics.csv", rows)
        write_csv(output / "performance_summary.csv", summary)
        write_csv(output / "comparison_table.csv", table)
        (output / "comparison_table.md").write_text(make_markdown(table, plan), encoding="utf-8")
        print(make_markdown(table, plan))
        print("Wrote:", output)


if __name__ == "__main__":
    main()
