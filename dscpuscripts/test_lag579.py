"""Non-destructive checks of the separate 5/7/9 experiment."""
from __future__ import annotations

from collections import Counter
import copy
import torch

from .lag579_protocol import DATASETS, METHODS, LAG_SETS, make_manifest, validate_config
from .lag579_train import code_hash, build_objective
from .lag579_runner import paths


def test_lag_sets_and_grid():
    assert LAG_SETS["fd001"] == (
        (1,3,5,10,15), (1,3,5,10,15,20,30),
        (1,3,5,10,15,20,30,40,60))
    assert LAG_SETS["electricity"] == ((1,12,24,36,48),)
    for dataset in DATASETS:
        configs = make_manifest(dataset)
        expected_sizes = (7,9) if dataset == "fd001" else (5,)
        total = 750 if dataset == "fd001" else 375
        assert len(configs) == len({c["run_id"] for c in configs}) == total
        assert Counter((c["method"], len(c["lags"])) for c in configs) == {
            (method, number): 125 for method in METHODS for number in expected_sizes}
        assert all(c["batch_size"] == 256 and c["seed"] in range(5)
                   and c["configuration"] in range(25) for c in configs)
        assert {tuple(c["lags"]) for c in configs} == set(
            LAG_SETS[dataset][1:] if dataset == "fd001" else LAG_SETS[dataset])
        assert all("lag579v1" in c["run_id"] for c in configs)
        assert all(validate_config(c) == c for c in configs)


def test_manifest_does_not_change_original_five_lag_protocol():
    from .fd001_protocol import make_manifest as fd001_original
    from .application_protocol import make_manifest as app_original
    before = [c for c in fd001_original() if c["method"] == "td_typed" and c["batch_size"] == 256]
    before += [c for c in app_original() if c["dataset"] == "electricity" and c["method"] == "td_typed"]
    snap = copy.deepcopy(before)
    make_manifest("fd001")
    make_manifest("electricity")
    assert before == snap
    old_ids = {c["run_id"] for c in before}
    assert not old_ids.intersection(c["run_id"] for d in DATASETS for c in make_manifest(d))


def test_exact_mean_and_operator_dimensions_fd001():
    for count in (7, 9):
        for method in METHODS:
            c = next(c for c in make_manifest("fd001")
                     if c["method"] == method and len(c["lags"]) == count)
            objective = build_objective(c)
            assert objective.operator.shape == (count,)
            assert torch.count_nonzero(objective.operator) == 0
            if method == "td_nonnegative_typed":
                x = torch.zeros(1,24,30)
                t = torch.tensor([count-1], dtype=torch.int64)
                gamma = objective.gamma_values(x,x,t)
                assert 0 < gamma.item() < .001


def test_scientific_hash_method_specific_and_lag_stable():
    configs = make_manifest("fd001")
    first_by_method = {m:next(c for c in configs if c["method"] == m
                                               and len(c["lags"]) == 7) for m in METHODS}
    hashes = {m:code_hash(c) for m,c in first_by_method.items()}
    assert len(set(hashes.values())) == 3
    assert all(len(h) == 64 for h in hashes.values())
    for m,c in first_by_method.items():
        later = next(t for t in configs if t["method"] == m and len(t["lags"]) == 9)
        assert code_hash(later) == hashes[m]


def test_new_paths_never_overlap_old_campaign():
    for dataset in DATASETS:
        data, path = paths(dataset)
        assert dataset in path.name
        assert "lag579" in path.name
        assert str(data) and str(path) and path != data


def test_fd001_audit_rejects_mismatched_source_and_partner_evidence(tmp_path):
    import json
    import math
    import pytest
    from fulltd.sampling import sources_for_round, sequence_hash
    from .lag579_analysis import _audit_fd001_candidate

    config = next(c for c in make_manifest("fd001")
                  if c["method"] == "td_typed" and len(c["lags"]) == 7)
    n = 20
    epoch = 1
    per_epoch = math.ceil(n / config["batch_size"])
    folder = tmp_path / config["run_id"]
    folder.mkdir()
    universe = dict(n_sources=n, updates_per_epoch=per_epoch,
                    total_updates=per_epoch * config["epochs"],
                    evaluation_period=config["learning_curve_updates"],
                    isolated_excluded=[])
    rounds = [dict(epoch=epoch, source_draws=n,
                   source_sha256=sequence_hash(
                       sources_for_round("fd001", n, config["seed"], epoch)),
                   partner_sha256="a" * 64,
                   optimizer_steps_cumulative=epoch * per_epoch,
                   type_counts=[n] + [0] * (len(config["lags"]) - 1))]
    (folder / "source_universe.json").write_text(json.dumps(universe))
    (folder / "rounds.json").write_text(json.dumps(rounds))
    status = dict(n_sources=n, initial_mean_sha256="initial",
                  identity=dict(data_sha256="train", validation_sha256="validation"))
    shared, source_hashes = {}, {}
    _audit_fd001_candidate(config, folder, status, shared, source_hashes)
    assert shared[("initial_mean", config["seed"])] == "initial"
    rounds[0]["type_counts"] = [n - 1] + [0] * (len(config["lags"]) - 1)
    (folder / "rounds.json").write_text(json.dumps(rounds))
    with pytest.raises(ValueError, match="lag-pair counts"):
        _audit_fd001_candidate(config, folder, status, {}, {})
    rounds[0]["type_counts"] = [n] + [0] * (len(config["lags"]) - 1)
    rounds[0]["source_sha256"] = "f" * 64
    (folder / "rounds.json").write_text(json.dumps(rounds))
    with pytest.raises(ValueError, match="source/update"):
        _audit_fd001_candidate(config, folder, status, {}, {})


def test_fd001_existing_checkpoint_hashes_remain_compatible():
    from .lag579_train import _FD001_LEGACY_CODES
    expected = {
        "td_typed": "afe696da123d0cb443e93d5bb4cbea483c80d12e9a0209e48a4f29bbfb0a0e33",
        "td_semi_typed": "45bc20a2918263763b1774ccc2fd2e6351f6aa0f735e6bdde33444ffb4763722",
        "td_nonnegative_typed": "d28ba41bb85c3c9c98a8c3634711ca564720b86bb02da2b9c322ac94170dedba",
    }
    assert _FD001_LEGACY_CODES == expected
    from fulltd.protocol import digest
    assert digest(make_manifest("fd001")) == (
        "c67c6738c6c58036a552b82c1544252aa68b16030aeb17221d032ce93b5fce25")
    for cfg in make_manifest("fd001"):
        assert code_hash(cfg) == expected[cfg["method"]]


def test_electricity_five_lag_only_uses_three_typed_methods():
    from .typed_ablation_train import code_hash as old_ablation_hash
    electricity = make_manifest("electricity")
    assert len(electricity) == 375
    assert {c["method"] for c in electricity} == set(METHODS)
    assert {tuple(c["lags"]) for c in electricity} == {(1,12,24,36,48)}
    assert {len(c["lags"]) for c in electricity} == {5}
    for method in METHODS:
        cfg = next(c for c in electricity if c["method"] == method)
        objective = build_objective(cfg)
        assert tuple(objective.operator.shape) == (5,)
        assert not torch.count_nonzero(objective.operator)
        assert code_hash(cfg) != old_ablation_hash(cfg)
