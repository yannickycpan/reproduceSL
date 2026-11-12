"""Selection leakage, exact endpoints, five-seed summaries and report integration."""
from __future__ import annotations

import copy
import math

import numpy as np
import pytest

from dscpuscripts import application_analysis as analysis
from dscpuscripts.application_protocol import (PROTOCOL_ID, SEEDS, digest,
                                               evaluation_steps, make_manifest, update_budgets)
from dscpuscripts.common import atomic_json, read_json
from fulltd.sampling import sequence_hash, sources_for_round


def test_final_quarter_counts_only_positive_measurements_before_endpoint():
    steps = (0, 70, 140, 200, 210, 280, 300)
    assert analysis.selection_update_steps(steps, 200) == (200,)
    assert analysis.selection_update_steps(steps, 300) == (280, 300)
    assert analysis.selection_update_steps((0, 39), 39) == (39,)
    with pytest.raises(ValueError, match="exact"):
        analysis.selection_update_steps(steps, 190)
    with pytest.raises(ValueError, match="unique"):
        analysis.selection_update_steps((0, 70, 70, 200), 200)


def _fake_campaign(root, monkeypatch):
    """Small grid with realistic 100-epoch source accounting and off-grid endpoints."""
    manifest = [copy.deepcopy(c) for c in make_manifest() if c["dataset"] == "metr_la"
                and c["method"] in ("mse", "td_pairwise") and c["configuration"] in (0, 1)]
    for config in manifest:
        config["source_fraction"] = 1.0
        config["source_sampling"] = "random_reshuffling"
        config["learning_curve_updates"] = config["snapshot_updates"] = 70
    monkeypatch.setattr(analysis, "manifest_for", lambda campaign: manifest)
    monkeypatch.setattr(analysis, "training_code_hash", lambda config: "fixture-scientific-code")
    n = 257
    subset = dict(policy="complete_fixed_validation_v1", population_size=10, sampled_size=10,
                  indices_sha256=sequence_hash(np.arange(10)))
    hashes = {(seed, epoch): sequence_hash(sources_for_round("metr_la", n, seed, epoch))
              for seed in SEEDS for epoch in range(1, 101)}
    for config in manifest:
        folder = root / config["run_id"]
        folder.mkdir(parents=True)
        budget = update_budgets(config, n)
        identity = dict(config_sha256=digest(config), protocol_id=PROTOCOL_ID,
                        code_sha256="fixture-scientific-code", data_sha256="train-data",
                        validation_sha256="validation-data", validation_subset_sha256=digest(subset))
        status = dict(config=config, identity=identity, state="complete", n_sources=n,
                      epochs_completed=100, next_epoch=101, cursor=0,
                      optimizer_steps=budget["total_updates"], update_opportunities=budget["total_updates"],
                      source_draws=100 * n, initial_mean_sha256=f"initial-{config['seed']}",
                      validation_subset=subset, **budget)
        atomic_json(folder / "status.json", status)
        atomic_json(folder / "source_universe.json", dict(n_sources=n, evaluation_period=70,
                                                           validation_subset=subset, **budget))
        steps = evaluation_steps(budget["total_updates"], 70)
        curve = []
        for step in steps:
            setting = config["configuration"]
            # Setting 1 wins; update zero does not enter selection.
            mse = 1e6 if step == 0 else (0.8 if setting == 0 else 0.2)
            curve.append(dict(optimizer_steps=step, epoch=min(100, step), mse_standardized=mse,
                              rmse_standardized=math.sqrt(mse), mae_standardized=math.sqrt(mse) * .8,
                              rmse_raw=math.sqrt(mse) * 2, mae_raw=math.sqrt(mse) * 1.6,
                              raw_units="mph", validation_subset=subset))
            checkpoint = folder / "checkpoints" / f"step_{step:07d}.pt"
            checkpoint.parent.mkdir(exist_ok=True)
            checkpoint.write_bytes(f"fixture:{config['run_id']}:{step}".encode())
        atomic_json(folder / "update_curve.json", curve)
        td = config["method"].startswith("td_")
        atomic_json(folder / "rounds.json", [
            dict(epoch=epoch, source_draws=n, source_sha256=hashes[config["seed"], epoch],
                 partner_sha256=f"pair-{config['seed']}-{epoch}" if td else None,
                 type_counts=[n, 0] if td else [],
                 optimizer_steps_cumulative=epoch * budget["updates_per_epoch"])
            for epoch in range(1, 101)])
        (folder / "final.pt").write_bytes(f"final:{config['run_id']}".encode())
    return manifest


def test_only_b256_full_budget_is_selected_without_reading_tests(tmp_path, monkeypatch):
    manifest = _fake_campaign(tmp_path, monkeypatch)
    plan, _ = analysis.compute_plan("metr_la", tmp_path)
    choices = {(c["method"], c["batch_size"], c["regime"]): c for c in plan["choices"]}
    assert {c["batch_size"] for c in plan["choices"]} == {256}
    assert {c["regime"] for c in plan["choices"]} == {"full_budget"}
    chosen = choices["td_pairwise", 256, "full_budget"]
    assert chosen["configuration"] == 1 and chosen["optimizer_steps"] == 200
    assert chosen["selection_steps"] == [200]
    assert len(chosen["run_ids"]) == 5
    for config in manifest:
        atomic_json(tmp_path / config["run_id"] / "endpoint_metrics.json",
                    {"test_rmse": 1e20 if config["configuration"] == 1 else 0.})
    same, _ = analysis.compute_plan("metr_la", tmp_path)
    assert same == plan


def test_incomplete_seed_and_wrong_actual_update_budget_refuse_selection(tmp_path, monkeypatch):
    manifest = _fake_campaign(tmp_path, monkeypatch)
    config = manifest[0]
    path = tmp_path / config["run_id"] / "status.json"
    status = read_json(path)
    atomic_json(path, {**status, "state": "checkpointed"})
    with pytest.raises(ValueError, match="unfinished"):
        analysis.compute_plan("metr_la", tmp_path)
    atomic_json(path, {**status, "optimizer_steps": status["optimizer_steps"] - 1})
    with pytest.raises(ValueError, match="final budget"):
        analysis.compute_plan("metr_la", tmp_path)


def test_missing_off_grid_snapshot_is_not_approximated(tmp_path, monkeypatch):
    manifest = _fake_campaign(tmp_path, monkeypatch)
    config = next(c for c in manifest if c["batch_size"] == 256)
    (tmp_path / config["run_id"] / "checkpoints" / "step_0000200.pt").unlink()
    with pytest.raises(FileNotFoundError, match="measurement checkpoint"):
        analysis.compute_plan("metr_la", tmp_path)


def test_numerical_failure_disqualifies_full_budget():
    candidate = dict(state="numerical_failure", curve=[
        dict(optimizer_steps=0, mse_standardized=1e9),
        dict(optimizer_steps=70, mse_standardized=.4),
        dict(optimizer_steps=140, mse_standardized=.3),
        dict(optimizer_steps=200, mse_standardized=.2)])
    assert analysis._candidate_score(candidate, 200, "full_budget")[0] == math.inf
    with pytest.raises(ValueError, match="full-budget"):
        analysis._candidate_score(candidate, 200, "matched_updates")


def test_standard_error_is_across_all_five_seeds_without_filtering():
    rows = [dict(method="mse", seed=seed, value=value) for seed, value in enumerate((1., 2., 3., 4., 5.))]
    summary = analysis.summarize(rows, ("method",))[0]
    assert summary["mean"] == 3. and summary["n"] == 5
    assert summary["standard_error"] == pytest.approx(math.sqrt(.5))
    with pytest.raises(ValueError, match="five distinct"):
        analysis.summarize(rows[:-1], ("method",))
    with pytest.raises(ValueError, match="five distinct"):
        analysis.summarize(rows[:-1] + [dict(rows[-2])], ("method",))
    rows[-1]["value"] = math.inf
    summary = analysis.summarize(rows, ("method",))[0]
    assert summary["n"] == 5 and summary["finite_seeds"] == 4
    assert math.isinf(summary["mean"]) and math.isnan(summary["standard_error"])


def test_report_end_to_end_locks_before_test_and_reuses_overlap(tmp_path, monkeypatch):
    """Real training/snapshot I/O on a small model exercises both reporting budgets."""
    import torch
    from dscpuscripts import application_train as train
    from dscpuscripts.application_objectives import make_objective
    from dscpuscripts.test_application_training import ToyData, ToyMean
    import fulltd.data

    class ScoringData(ToyData):
        def get_batch(self, indices):
            return dict(x=self.x[indices], y=self.y[indices], mask=self.mask[indices])

    configs = [copy.deepcopy(c) for c in make_manifest() if c["dataset"] == "metr_la"
               and c["method"] == "td_pairwise" and c["batch_size"] == 256 and c["configuration"] == 0]
    for config in configs:
        # This report/caching fixture intentionally tests the full-epoch path;
        # the 10% case is covered by the dedicated sampling/recovery tests.
        config.update(epochs=2, learning_curve_updates=5, snapshot_updates=5,
                      source_fraction=1.0, source_sampling="random_reshuffling")
    monkeypatch.setattr(analysis, "manifest_for", lambda campaign: configs)
    monkeypatch.setattr(analysis, "training_code_hash", lambda config: "fixture-real-io")
    monkeypatch.setattr(train, "code_hash", lambda config: "fixture-real-io")
    monkeypatch.setattr(train, "validate_config", lambda c: c)
    monkeypatch.setattr(train, "build_objective", lambda c, data: make_objective(c["method"], ToyMean()))
    datasets = {"train": ScoringData("metr_la", "train", n=300),
                "val": ScoringData("metr_la", "val", n=35),
                "test": ScoringData("metr_la", "test", n=39)}
    results, output = tmp_path / "results", tmp_path / "output"
    original_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for config in configs:
            done = train.train_candidate(config, datasets["train"], datasets["val"], results, microbatch=32)
            assert done["state"] == "complete"
        opened = []

        def load(name, directory, split):
            assert name == "metr_la"
            if split == "test":
                assert (output / "selections.json").is_file(), "Test opened before locking validation choices"
            opened.append(split)
            return datasets[split]

        monkeypatch.setattr(fulltd.data, "load_dataset", load)
        point_calls = []
        actual_point_metrics = analysis._point_metrics

        def count_metrics(model, data, batch_size):
            point_calls.append(data.split)
            return actual_point_metrics(model, data, batch_size)

        monkeypatch.setattr(analysis, "_point_metrics", count_metrics)
        # Selection never loads a dataset or computes a test prediction.
        analysis.run("metr_la", "check", results_dir=results, data_dir=tmp_path, output=output)
        assert not opened and not point_calls
        locked = read_json(output / "selections.json")
        first_run = locked["choices"][0]["run_ids"][0]
        individual = read_json(results / first_run / "status.json")["identity"]["code_sha256"]
        # Regression for the critical manifest-hash versus candidate-hash bug.
        assert individual != locked["code_sha256"]

        # Fast table path must evaluate only five final checkpoints, without
        # pretending that the full held-out learning curves are ready.
        analysis.run("metr_la", "endpoints", results_dir=results,
                     data_dir=tmp_path, output=output, batch_size=16)
        assert opened == ["train", "val", "test"]
        assert point_calls == ["test"] * 5
        assert (output / "endpoint_manifest.json").is_file()
        assert not (output / "evaluation_manifest.json").exists()
        assert not (output / "learning_curves_per_seed.csv").exists()

        # A controllable executor verifies per-run task partitioning and cache
        # reuse without spawning subprocesses that cannot inherit mock data.
        import concurrent.futures

        class RecordingPool:
            def __init__(self, max_workers, mp_context):
                assert max_workers == 2
                assert mp_context.get_start_method() == "spawn"
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return False
            def map(self, fn, tasks, chunksize):
                assert fn is analysis._evaluate_one_run_worker
                assert chunksize == 1
                for task in tasks:
                    yield analysis._evaluate_one_run(
                        task, datasets["train"], datasets["test"], task["batch_size"])

        with monkeypatch.context() as patch:
            patch.setattr(concurrent.futures, "ProcessPoolExecutor", RecordingPool)
            analysis.run("metr_la", "run", results_dir=results,
                         data_dir=tmp_path, output=output,
                         batch_size=16, workers=2)
        assert opened == ["train", "val", "test"] * 2
        # One B256 full-budget trajectory: update zero and final update four.
        # Endpoint metrics are reused from endpoint-only evaluation.
        assert point_calls == ["test"] * 10
        assert len(list((output / ".cache").rglob("*.json"))) == 10
        for metric in ("rmse", "mae"):
            text = (output / f"graph_wavenet_{metric}_table.md").read_text()
            assert "| Algorithm | Test | Validation |" in text
            assert "±" in text and "matched updates" not in text
            for extension in ("pdf", "png"):
                figure = output / "figures" / f"graph_wavenet_{metric}_learning_curves.{extension}"
                assert figure.stat().st_size > 1000
        # Rerunning evaluation reuses all point-metric caches, including endpoint overlap.
        analysis.run("metr_la", "evaluate", results_dir=results, data_dir=tmp_path, output=output, batch_size=16)
        assert len(point_calls) == 10
        artifact = output / "per_seed_metrics.csv"
        original = artifact.read_bytes()
        lines = original.decode().splitlines()
        lines[1] = lines[1].rsplit(",", 1)[0] + ",999999"
        artifact.write_text("\n".join(lines) + "\n")
        with pytest.raises(ValueError, match="changed or are incomplete"):
            analysis.run("metr_la", "plots", results_dir=results, data_dir=tmp_path, output=output)
        artifact.write_bytes(original)
        locked = read_json(output / "selections.json")
        locked["choices"][0]["selection_mse"] += .01
        atomic_json(output / "selections.json", locked)
        with pytest.raises(ValueError, match="Locked validation-only selection changed"):
            analysis.run("metr_la", "run", results_dir=results, data_dir=tmp_path, output=output)
        assert len(point_calls) == 10
    finally:
        torch.set_num_threads(original_threads)
