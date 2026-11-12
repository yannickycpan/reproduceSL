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


def test_historical_electricity_identity_is_read_only_pinned_and_never_guessed(
        tmp_path, monkeypatch):
    """Full old Electricity sweep can be checked with the new scoped analyzer."""
    manifest = [copy.deepcopy(c) for c in make_manifest()
                if c["dataset"] == "electricity" and c["method"] == "mse"
                and c["configuration"] == 0]
    assert len(manifest) == 5
    monkeypatch.setattr(analysis, "training_code_hash",
                        lambda c: "new-scoped-scientific-identity")
    old_hash = analysis.LEGACY_ELECTRICITY_CODE_SHA256
    for config in manifest:
        directory = tmp_path / config["run_id"]
        directory.mkdir()
        atomic_json(directory / "status.json",
                    dict(identity=dict(code_sha256=old_hash), config=config))
    selected = analysis._expected_codes(manifest, tmp_path)
    assert set(selected.values()) == {old_hash}
    # A newly recomputed scoped identity is NOT the old hash and must never
    # be silently substituted into pre-refactor checkpoint validation.
    assert old_hash != "new-scoped-scientific-identity"
    assert analysis._plan_code_hash(manifest, selected) == old_hash
    assert digest([[c["run_id"], selected[c["run_id"]]] for c in manifest]) != old_hash

    # If the first candidate is not the known pinned historical code,
    # the analyzer reverts to current per-candidate scoped identity.
    first = tmp_path / manifest[0]["run_id"] / "status.json"
    atomic_json(first, dict(config=manifest[0],
                            identity=dict(code_sha256="unrecognized-old-code")))
    current = analysis._expected_codes(manifest, tmp_path)
    assert set(current.values()) == {"new-scoped-scientific-identity"}


def test_historical_electricity_selection_lock_stays_unchanged(
        tmp_path, monkeypatch):
    """The original Electricity selection/code hash must survive analysis unchanged."""
    manifest = [copy.deepcopy(c) for c in make_manifest()
                if c["dataset"] == "electricity" and c["method"] == "mse"
                and c["configuration"] == 0]
    old_hash = analysis.LEGACY_ELECTRICITY_CODE_SHA256
    monkeypatch.setattr(analysis, "_verify_processed_data",
                        lambda campaign, path: "validated synthetic bytes")
    results, output = tmp_path / "results", tmp_path / "outputs"
    for config in manifest:
        folder = results / config["run_id"]
        folder.mkdir(parents=True)
        atomic_json(folder / "status.json",
                    dict(config=config, identity=dict(code_sha256=old_hash)))
    monkeypatch.setattr(analysis, "manifest_for", lambda campaign: manifest)
    expected = analysis._expected_codes(manifest, results)
    assert analysis._plan_code_hash(manifest, expected) == old_hash
    plan = dict(dataset="electricity", code_sha256=old_hash,
                manifest_sha256=digest(manifest), choices=[])
    monkeypatch.setattr(analysis, "compute_plan",
                        lambda campaign, results_path: (plan, []))
    output.mkdir()
    atomic_json(output / "selections.json", plan)
    analysis.run("electricity", "check", data_dir=tmp_path,
                 results_dir=results, output=output)
    assert read_json(output / "selections.json") == plan
    assert not (output / "selections_pre_scoped_hash.json").exists()

    changed = dict(plan, choices=[{"changed": "scientific selection"}])
    monkeypatch.setattr(analysis, "compute_plan",
                        lambda campaign, results_path: (changed, []))
    with pytest.raises(ValueError, match="Locked validation-only selection changed"):
        analysis.run("electricity", "check", data_dir=tmp_path,
                     results_dir=results, output=output)
    assert read_json(output / "selections.json") == plan


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
    monkeypatch.setattr(analysis, "_verify_processed_data",
                        lambda campaign, path: "validated synthetic bytes")
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


def test_electricity_point_metrics_keep_original_evaluation_call(monkeypatch):
    """Do not change the original report evaluator's numerical behavior."""
    from dscpuscripts.application_analysis import _point_metrics
    import fulltd.metrics
    calls = []

    class Dataset:
        dataset = "electricity"

    def fake_evaluate(model, dataset, **kwargs):
        calls.append(kwargs)
        return [
            dict(units="standardized", horizon="all", metric="rmse", value=1.5),
            dict(units="standardized", horizon="all", metric="mae", value=1.2)
        ], []

    monkeypatch.setattr(fulltd.metrics, "evaluate", fake_evaluate)
    assert _point_metrics(object(), Dataset(), batch_size=19) == {
        "rmse": 1.5, "mae": 1.2}
    assert len(calls) == 1
    assert "include_raw" not in calls[0]
    assert calls[0]["batch_size"] == 19


def test_metr_la_point_metrics_still_include_mph(monkeypatch):
    from dscpuscripts.application_analysis import _point_metrics
    import fulltd.metrics
    calls = []

    class Dataset:
        dataset = "metr_la"

    def fake_evaluate(model, dataset, **kwargs):
        calls.append(kwargs)
        return [
            dict(units="mph", horizon="all", metric="rmse", value=3.1),
            dict(units="mph", horizon="all", metric="mae", value=2.6)
        ], []

    monkeypatch.setattr(fulltd.metrics, "evaluate", fake_evaluate)
    assert _point_metrics(object(), Dataset(), batch_size=11) == {
        "rmse": 3.1, "mae": 2.6}
    assert "include_raw" not in calls[0]


def test_cache_evaluator_hash_changes_with_implementation(monkeypatch):
    original = analysis._point_metric_implementation_hash()

    def replacement(model, dataset, batch_size):
        return {"rmse": 5., "mae": 4.}

    with monkeypatch.context() as patch:
        patch.setattr(analysis, "_point_metrics", replacement)
        assert analysis._point_metric_implementation_hash() != original


def test_real_spawn_workers_match_serial_with_actual_graph_wavenet(tmp_path):
    """Real OS spawn, model inference, mmap load, metric caches and shared data.

    Separate synthetic processed METR-LA data; no production training and no
    monkeypatched functions are needed inside spawned workers.
    """
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context
    from pathlib import Path
    import pandas as pd
    import torch
    from dscpuscripts.application_train import build_objective, evaluation_validation_view
    from dscpuscripts.common import file_hash, data_identity
    from fulltd.data import prepare_metr_la, load_dataset
    from fulltd.train import state_hash

    ids = ["101", "102", "103"]
    rng = np.random.default_rng(405)
    frame = pd.DataFrame(
        40 + 2 * rng.normal(size=(180, 3)),
        index=pd.date_range("2024-02-01", periods=180, freq="5min"),
        columns=ids)
    prepare_metr_la(tmp_path, frame=frame,
                    graph=(ids, np.eye(3, dtype=np.float32)))
    assert analysis._verify_processed_data("metr_la", tmp_path)

    train = load_dataset("metr_la", tmp_path, "train")
    validation = load_dataset("metr_la", tmp_path, "val")
    test = load_dataset("metr_la", tmp_path, "test")
    _, subset = evaluation_validation_view(validation)
    root = Path(__file__).resolve().parents[1]
    configs = [copy.deepcopy(c) for c in make_manifest()
               if c["dataset"] == "metr_la" and c["method"] == "mse"
               and c["configuration"] == 0 and c["seed"] in (0, 1)]
    assert len(configs) == 2
    tasks = []
    result_root = tmp_path / "results"
    for config in configs:
        folder = result_root / config["run_id"]
        (folder / "checkpoints").mkdir(parents=True)
        objective = build_objective(config, train).cpu()
        mean_hash = state_hash(objective.model.state_dict())
        identity = dict(
            protocol_id=config["protocol_id"],
            code_sha256="synthetic-scoped-source",
            config_sha256=digest(config), data_sha256=data_identity(train),
            validation_sha256=data_identity(validation),
            validation_subset_sha256=digest(subset))
        record = dict(optimizer_steps=0, validation_subset=subset,
                      mse_standardized=1., rmse_standardized=1.,
                      mae_standardized=1., rmse_raw=1., mae_raw=1.)
        atomic_json(folder / "status.json", dict(
            config=config, identity=identity,
            initial_mean_sha256=mean_hash))
        atomic_json(folder / "update_curve.json", [record])
        checkpoint = folder / "checkpoints" / "step_0000000.pt"
        torch.save(dict(config=config, identity=identity,
                        initial_mean_sha256=mean_hash,
                        objective=objective.state_dict(),
                        progress=dict(optimizer_steps=0, curve=[record])),
                   checkpoint)
        tasks.append(dict(
            campaign="metr_la", config=config, results=str(result_root),
            output=str(tmp_path / "serial_output"), data_dir=str(tmp_path),
            endpoint=0, total=1, expected_code="synthetic-scoped-source",
            endpoint_sha256=file_hash(checkpoint),
            subset=subset, data_sha256=data_identity(train),
            validation_sha256=data_identity(validation),
            test_sha256=data_identity(test),
            metric_code_sha256=file_hash(root / "fulltd/metrics.py"),
            data_loader_code_sha256=file_hash(root / "fulltd/data.py"),
            point_metric_code_sha256=analysis._point_metric_implementation_hash(),
            manifest_code_sha256="synthetic-manifest",
            batch_size=8, backbone="graph_wavenet", regime="full_budget",
            method="mse", endpoints_only=True))
    torch.set_num_threads(1)
    serial = [analysis._evaluate_one_run_worker(task) for task in tasks]
    parallel_tasks = [dict(task, output=str(tmp_path / "parallel_output"))
                      for task in tasks]
    with ProcessPoolExecutor(max_workers=2, mp_context=get_context("spawn")) as pool:
        parallel = list(pool.map(
            analysis._evaluate_one_run_worker, parallel_tasks, chunksize=1))
    assert serial == parallel
    assert len(list((tmp_path / "parallel_output" / ".cache").rglob("*.json"))) == 2
    for path in (tmp_path / "parallel_output" / ".cache").rglob("*.json"):
        stored = read_json(path)
        assert stored["metrics_sha256"] == digest(stored["metrics"])
        assert stored["snapshot_audit_sha256"]

    # Analysis must detect a modified npy file despite unmodified metadata.
    array_file = tmp_path / "processed" / "metr_la" / "test" / "values.npy"
    with array_file.open("r+b") as stream:
        stream.seek(-1, 2)
        old_byte = stream.read(1)
        stream.seek(-1, 2)
        stream.write(bytes([old_byte[0] ^ 1]))
    with pytest.raises(ValueError, match="Processed data missing or modified"):
        analysis._verify_processed_data("metr_la", tmp_path)


def test_analysis_workers_must_be_positive(tmp_path):
    with pytest.raises(ValueError, match="workers"):
        analysis.evaluate_selected(
            {"dataset": "metr_la", "code_sha256": "invalid",
             "manifest_sha256": "invalid"},
            tmp_path, tmp_path, tmp_path, workers=0)


def test_report_end_to_end_locks_before_test_and_reuses_overlap(tmp_path, monkeypatch):
    """Real training/snapshot I/O on a small model exercises both reporting budgets."""
    monkeypatch.setattr(analysis, "_verify_processed_data",
                        lambda campaign, path: "validated synthetic bytes")
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
        with pytest.raises(ValueError, match="Only endpoint tables"):
            analysis.run("metr_la", "plots", results_dir=results,
                         data_dir=tmp_path, output=output)

        # Simulate an endpoint computed by the previous aggregate-hash cache
        # layout. It must migrate without another expensive test inference.
        old_cache = next((output / ".cache" / first_run).glob("*.json"))
        cached = read_json(old_cache)
        # Rebuild a genuine pre-v2 cache key: it had no point-function or
        # data-loader hash and no checksum of the saved metric values.
        original_identity = cached["identity"]
        legacy_identity = {key: original_identity[key] for key in (
            "metric_code_sha256", "checkpoint_sha256", "test_sha256",
            "batch_size", "run_id", "optimizer_steps")}
        legacy_identity.update(version="application-point-metrics-v1",
                               code_sha256=locked["code_sha256"])
        legacy_path = old_cache.with_name(
            f"step_{legacy_identity['optimizer_steps']:07d}_"
            f"{digest(legacy_identity)[:20]}.json")
        atomic_json(legacy_path, dict(identity=legacy_identity,
                                      metrics=cached["metrics"]))
        old_cache.unlink()

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
        assert len(list((output / ".cache").rglob("*.json"))) == 11
        # Ten current checkpoint caches plus the original legacy-format
        # endpoint evidence retained for provenance.
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
        # Metrics are authenticated independently of the run/checkpoint key.
        current = [p for p in (output / ".cache").rglob("*.json")
                   if "metrics_sha256" in read_json(p)]
        assert len(current) == 10
        corrupt = current[0]
        original_cache = read_json(corrupt)
        tampered = copy.deepcopy(original_cache)
        tampered["metrics"]["rmse"] = 123456.0
        atomic_json(corrupt, tampered)
        with pytest.raises(ValueError, match="checksum mismatch"):
            analysis.run("metr_la", "evaluate", results_dir=results,
                         data_dir=tmp_path, output=output, batch_size=16)
        atomic_json(corrupt, original_cache)

        # A fully audited cache does not deserialize Adam tensors again.
        with monkeypatch.context() as patch:
            def should_not_load(*args, **kwargs):
                raise AssertionError("A valid cache must not reload checkpoint tensors")
            patch.setattr(torch, "load", should_not_load)
            analysis.run("metr_la", "evaluate", results_dir=results,
                         data_dir=tmp_path, output=output, batch_size=16)
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
