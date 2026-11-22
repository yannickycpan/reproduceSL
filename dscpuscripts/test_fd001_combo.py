"""Checks for the FD001 MSE -> Pairwise Gamma -> full-gradient TD baseline."""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from torch import nn

from .fd001_combo import ComboObjective, _plateau
from .fd001_protocol import (
    ALPHAS, BASELINE_ALPHAS, BETAS, BATCH_SIZES, EPOCHS,
    LAG_SETS, METHODS, MSE_STAGE_ALPHAS, make_manifest, validate_config,
)


class TinyPredictor(nn.Module):
    feature_dim = 2
    output_dim = 1

    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(24, 2)
        self.head = nn.Linear(2, 1)

    def features(self, x):
        return self.encoder(x.mean(dim=-1))

    def forward(self, x):
        return self.head(self.features(x))


def test_frozen_lag_and_full_new_candidate_grid():
    manifest = make_manifest()
    assert len(manifest) == len({c["run_id"] for c in manifest}) == 850
    assert LAG_SETS == ((1, 5),)
    assert MSE_STAGE_ALPHAS == (0.01,)
    assert set(METHODS) == {"mse", "gaussian_nll", "td_typed", "td_pairwise", "td_mse_combo"}
    assert {(c["method"], c["batch_size"]): 1 for c in manifest} == {
        ("mse",256):1, ("gaussian_nll",256):1, ("td_typed",128):1,
        ("td_typed",256):1, ("td_pairwise",128):1,
        ("td_pairwise",256):1, ("td_mse_combo",128):1, ("td_mse_combo",256):1}
    assert sum(c["method"] == "td_mse_combo" for c in manifest) == 250
    for config in manifest:
        assert validate_config(config) == config
        assert config["epochs"] == EPOCHS == 100
        assert tuple(config["lags"]) == (1, 5)
        if config["method"] == "td_mse_combo":
            assert config["alpha"] in ALPHAS["td_pairwise"]
            assert config["beta"] in BETAS
            assert config["mse_alpha"] == 0.01
            assert config["gamma_max_epochs"] == 100
            assert config["predictor_max_epochs"] == 100
    assert 0.01 in BASELINE_ALPHAS
    assert BATCH_SIZES["td_mse_combo"] == (128, 256)


def _tensors():
    gen = torch.Generator().manual_seed(91)
    x = torch.randn((4, 24, 30), generator=gen)
    other = torch.randn((4, 24, 30), generator=gen)
    source = dict(x=x, y=torch.randn((4,1),generator=gen),
                  mask=torch.ones(4,1,dtype=torch.bool))
    target = dict(x=other, y=torch.randn((4,1),generator=gen),
                  mask=torch.ones(4,1,dtype=torch.bool))
    return source, target, torch.tensor([0,1,0,1], dtype=torch.long)


def test_gamma_stage_freezes_both_encoder_and_predictor():
    torch.manual_seed(5)
    objective = ComboObjective(TinyPredictor(),2)
    objective.set_phase("gamma")
    a, b, types = _tensors()
    loss, count = objective.loss_sum(a,b,types)
    assert count == 4
    loss.backward()
    assert all(x.grad is None for x in objective.model.parameters())
    assert all(x.grad is None for x in objective.teacher.parameters())
    assert objective.gamma_head.linear.weight.grad is not None
    assert objective.gamma_head.linear.weight.grad.abs().max().item() > 0


def test_stage3_uses_both_current_residuals_and_frozen_gamma():
    torch.manual_seed(6)
    objective = ComboObjective(TinyPredictor(),2)
    with torch.no_grad():
        objective.gamma_head.linear.bias.copy_(torch.tensor([0.2,-0.3]))
        objective.gamma_head.linear.weight.normal_(std=.05)
    objective.set_phase("predictor")
    a,b,types = _tensors()
    g_before = objective.gamma_values(a["x"],b["x"],types).detach().clone()
    current, _ = objective.loss_sum(a,b,types)
    predictor_params = list(objective.model.parameters())
    current_grad = torch.autograd.grad(current, predictor_params)
    # A mathematical full-gradient reference, with BOTH predictions live.
    residual_a = a["y"] - objective.model(a["x"])
    residual_b = b["y"] - objective.model(b["x"])
    manual = .5*(residual_a - g_before[:,None]*residual_b).square().sum()
    manual_grad = torch.autograd.grad(manual, predictor_params)
    torch.testing.assert_close(current,manual,rtol=1e-5,atol=1e-6)
    for found, expected in zip(current_grad,manual_grad):
        torch.testing.assert_close(found,expected,rtol=1e-5,atol=1e-6)
    optim = torch.optim.SGD(predictor_params, lr=.1)
    optim.zero_grad()
    current.backward()
    optim.step()
    after = objective.gamma_values(a["x"],b["x"],types)
    torch.testing.assert_close(after,g_before,rtol=0,atol=0)
    assert all(param.grad is None for param in objective.teacher.parameters())
    assert all(param.grad is None for param in objective.gamma_head.parameters())


def test_non_improving_training_loss_stops_after_patience():
    progress = dict(best_loss=math.inf,bad_epochs=0,
                    epoch_loss=5.,epoch_valid=10,
                    phase_epochs={"gamma":0,"predictor":0},
                    phase_steps={"gamma":1,"predictor":0},
                    gamma_rounds=[],predictor_rounds=[],
                    source_sha256="s",partner_sha256="p")
    config = dict(stop_patience=5,stop_relative_min_decrease=1e-4)
    assert not _plateau(progress,"gamma",config)
    assert math.isclose(progress["best_loss"],.5)
    for _ in range(4):
        assert not _plateau(progress,"gamma",config)
    assert _plateau(progress,"gamma",config)
    assert progress["phase_epochs"]["gamma"] == 6


def test_runner_accepts_completed_early_stopped_combo(tmp_path):
    from .common import atomic_json
    from .standard_runner import classify
    from .fd001_train import code_hash
    from .fd001_protocol import digest
    config = next(c for c in make_manifest() if c["method"]=="td_mse_combo")
    run_dir = tmp_path/config["run_id"]
    run_dir.mkdir()
    for name in ("final.pt","update_curve.json","rounds.json","gamma_rounds.json"):
        (run_dir/name).touch()
    atomic_json(run_dir/"status.json",dict(config=config,
        identity=dict(config_sha256=digest(config),code_sha256=code_hash()),
        state="complete",phase="complete",epochs_completed=9,
        stop_reasons=dict(gamma="training_loss_plateau",
                          predictor="training_loss_plateau")))
    assert classify(config,tmp_path,code_hash())=="complete"


def test_wrong_phase_cannot_update_gamma_during_predictor_training():
    objective = ComboObjective(TinyPredictor(),2)
    objective.set_phase("predictor")
    assert all(not p.requires_grad for p in objective.teacher.parameters())
    assert all(not p.requires_grad for p in objective.gamma_head.parameters())
    assert all(p.requires_grad for p in objective.model.parameters())


class _TinyFD001Dataset:
    def __init__(self, split):
        self.dataset, self.split = "fd001", split
        n = 12
        self.entity_ids = np.zeros(n,dtype=np.int64)
        self.origins = np.arange(1,n+1,dtype=np.int64)
        self.metadata = dict(fingerprint="synthetic-combo-resume-v1")
        rng = np.random.default_rng(707)
        self.x = rng.normal(size=(n,24,30)).astype(np.float32)*.05
        self.y = rng.normal(size=(n,1)).astype(np.float32)*.05

    def __len__(self):
        return len(self.x)

    def get_batch(self, indices):
        return dict(x=self.x[indices], y=self.y[indices],
                    mask=np.ones_like(self.y[indices],dtype=bool))


def _install_mock_completed_mse(root, reference_config, train, validation):
    from .fd001_train import build_objective, code_hash
    from .common import atomic_json, data_identity
    from fulltd.train import state_hash
    from fulltd.sampling import training_universe
    from fulltd.protocol import digest
    mean = build_objective(reference_config)
    initial = state_hash(mean.model.state_dict())
    train_sources,_,_ = training_universe(train, (1,5))
    identity = dict(code_sha256=code_hash(),
                    protocol_id=reference_config["protocol_id"],
                    config_sha256=digest(reference_config),
                    data_sha256=data_identity(train_sources),
                    validation_sha256=data_identity(validation))
    folder = root/reference_config["run_id"]
    folder.mkdir(parents=True,exist_ok=True)
    torch.save(dict(config=reference_config,identity=identity,
                    initial_mean_sha256=initial,
                    objective=mean.state_dict(),
                    progress=dict(epoch=101)),
               folder/"final.pt")
    atomic_json(folder/"status.json",dict(
        config=reference_config,identity=identity,
        initial_mean_sha256=initial, state="complete",
        epochs_completed=100, optimizer_steps=100,
        runtime_seconds=2.,mean_update_seconds=.02))


def test_gamma_phase_resume_matches_uninterrupted_updates(tmp_path):
    """A two-update interruption must not repeat source/partner draws."""
    from .fd001_combo import train_candidate
    combos = make_manifest()
    combo = next(c for c in combos if
                 c["method"]=="td_mse_combo" and c["batch_size"]==128
                 and c["seed"]==0 and c["configuration"]==0)
    ref = next(c for c in combos if c["method"]=="mse"
               and c["seed"]==0 and c["alpha"]==combo["mse_alpha"])
    train,validation = _TinyFD001Dataset("train"),_TinyFD001Dataset("val")
    left, right = tmp_path/"uninterrupted",tmp_path/"interrupted"
    for root in (left,right):
        _install_mock_completed_mse(root,ref,train,validation)
    completed = train_candidate(combo,train,validation,left,
                                microbatch=128,stop_after_updates=4)
    first = train_candidate(combo,train,validation,right,
                            microbatch=128,stop_after_updates=2)
    resumed = train_candidate(combo,train,validation,right,
                              microbatch=128,stop_after_updates=4)
    assert completed["state"] == first["state"] == resumed["state"] == "checkpointed"
    assert resumed["phase_steps"]["gamma"] == completed["phase_steps"]["gamma"] == 4
    one = torch.load(left/combo["run_id"]/"recovery.pt",
                     weights_only=False,map_location="cpu")
    two = torch.load(right/combo["run_id"]/"recovery.pt",
                     weights_only=False,map_location="cpu")
    assert one["progress"]["gamma_rounds"] == two["progress"]["gamma_rounds"]
    for key in one["objective"]:
        torch.testing.assert_close(one["objective"][key],two["objective"][key],
                                   rtol=0,atol=0)
