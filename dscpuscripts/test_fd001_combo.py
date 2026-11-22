"""FD001 sequential TD-MSE combo: scientific and restart regression tests."""
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


def _tensors():
    gen = torch.Generator().manual_seed(91)
    x = torch.randn((4, 24, 30), generator=gen)
    other = torch.randn((4, 24, 30), generator=gen)
    source = dict(x=x, y=torch.randn((4, 1), generator=gen),
                  mask=torch.ones(4, 1, dtype=torch.bool))
    target = dict(x=other, y=torch.randn((4, 1), generator=gen),
                  mask=torch.ones(4, 1, dtype=torch.bool))
    return source, target, torch.tensor([0, 1, 0, 1], dtype=torch.long)


def test_fixed_lag_four_method_manifest():
    manifest = make_manifest()
    assert len(manifest) == len({c["run_id"] for c in manifest}) == 600
    assert LAG_SETS == ((1, 3, 5),)
    assert MSE_STAGE_ALPHAS == (.001,)
    assert set(METHODS) == {"mse", "gaussian_nll", "td_pairwise", "td_mse_combo"}
    expected = {("mse", 256): 50, ("gaussian_nll", 256): 50,
                ("td_pairwise", 128): 125, ("td_pairwise", 256): 125,
                ("td_mse_combo", 128): 125, ("td_mse_combo", 256): 125}
    assert {(method, batch): sum(c["method"] == method and c["batch_size"] == batch
                                 for c in manifest)
            for method, batch in expected} == expected
    for config in manifest:
        assert validate_config(config) == config
        assert config["epochs"] == EPOCHS == 100
        assert tuple(config["lags"]) == (1, 3, 5)
        if config["method"] == "td_mse_combo":
            assert config["alpha"] in ALPHAS["td_pairwise"]
            assert config["beta"] in BETAS
            assert config["mse_alpha"] == .001
            assert config["mse_epochs"] == 100
            assert config["gamma_max_epochs"] == 100
            assert config["predictor_max_epochs"] == 100
    assert MSE_STAGE_ALPHAS[0] in BASELINE_ALPHAS
    assert BATCH_SIZES["td_mse_combo"] == (128, 256)


def test_stage1_is_fresh_trainable_mse_and_matches_ordinary_mse_gradient():
    torch.manual_seed(5)
    objective = ComboObjective(TinyPredictor(), 3)
    objective.set_phase("mse")
    source, _, _ = _tensors()
    loss, count = objective.loss_sum(source)
    assert count == 4
    reference = .5*(source["y"]-objective.model(source["x"])).square().sum()
    torch.testing.assert_close(loss, reference)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().max().item() > 0
               for p in objective.model.parameters())
    assert all(p.grad is None for p in objective.teacher.parameters())
    assert all(p.grad is None for p in objective.gamma_head.parameters())


def test_stage2_freezes_predictor_and_teacher_but_trains_pairwise_gamma():
    torch.manual_seed(5)
    objective = ComboObjective(TinyPredictor(), 3)
    objective.set_phase("gamma")
    source, target, kinds = _tensors()
    loss, count = objective.loss_sum(source, target, kinds)
    assert count == 4
    loss.backward()
    assert all(p.grad is None for p in objective.model.parameters())
    assert all(p.grad is None for p in objective.teacher.parameters())
    assert objective.gamma_head.linear.weight.grad is not None
    assert objective.gamma_head.linear.weight.grad.abs().max().item() > 0


def test_stage3_full_gradient_and_frozen_gamma_features():
    torch.manual_seed(6)
    objective = ComboObjective(TinyPredictor(), 3)
    with torch.no_grad():
        objective.gamma_head.linear.bias.copy_(torch.tensor([.2, -.3, .5]))
        objective.gamma_head.linear.weight.normal_(std=.05)
    objective.set_phase("predictor")
    source, target, kinds = _tensors()
    gamma_before = objective.gamma_values(source["x"],target["x"],kinds).detach().clone()
    current, _ = objective.loss_sum(source,target,kinds)
    params = list(objective.model.parameters())
    actual = torch.autograd.grad(current,params,retain_graph=True)
    res_source = source["y"]-objective.model(source["x"])
    res_partner = target["y"]-objective.model(target["x"])
    expected = .5*(res_source-gamma_before[:,None]*res_partner).square().sum()
    reference = torch.autograd.grad(expected,params)
    torch.testing.assert_close(current,expected)
    for left,right in zip(actual,reference):
        torch.testing.assert_close(left,right,rtol=1e-6,atol=1e-6)
    opt = torch.optim.SGD(params,lr=.05)
    opt.zero_grad()
    current.backward()
    opt.step()
    torch.testing.assert_close(
        objective.gamma_values(source["x"],target["x"],kinds),
        gamma_before,rtol=0,atol=0)
    assert all(p.grad is None for p in objective.teacher.parameters())
    assert all(p.grad is None for p in objective.gamma_head.parameters())


def test_gamma_plateau_but_other_stages_must_not_early_stop():
    progress = dict(best_loss=math.inf,bad_epochs=0,epoch_loss=5.,epoch_valid=10)
    config = dict(stop_patience=5,stop_relative_min_decrease=1e-4)
    assert not _plateau(progress,"gamma",config)
    assert progress["best_loss"] == .5
    for _ in range(4):
        assert not _plateau(progress,"gamma",config)
    assert _plateau(progress,"gamma",config)
    with pytest.raises(ValueError):
        _plateau(progress,"predictor",config)
    with pytest.raises(ValueError):
        _plateau(progress,"mse",config)


def test_runner_accepts_only_full_100_epoch_stage3(tmp_path):
    from .common import atomic_json
    from .standard_runner import classify
    from .fd001_train import code_hash
    from .fd001_protocol import digest
    c = next(c for c in make_manifest() if c["method"] == "td_mse_combo")
    folder = tmp_path / c["run_id"]
    folder.mkdir()
    for file in ("final.pt","update_curve.json","rounds.json",
                 "mse_rounds.json","gamma_rounds.json"):
        (folder / file).touch()
    base = dict(config=c,
                identity=dict(config_sha256=digest(c),code_sha256=code_hash()),
                state="complete",phase="complete",epochs_completed=100,
                phase_epochs=dict(mse=100,gamma=9,predictor=100),
                stop_reasons=dict(gamma="training_loss_plateau",
                                  predictor="max_epochs"))
    atomic_json(folder/"status.json",base)
    assert classify(c,tmp_path,code_hash()) == "complete"
    atomic_json(folder/"status.json",dict(base,epochs_completed=30,
                                        phase_epochs=dict(mse=100,gamma=9,predictor=30)))
    assert classify(c,tmp_path,code_hash()) == "resume"


class TinyFD001:
    def __init__(self, split, n=12):
        self.dataset, self.split = "fd001", split
        self.entity_ids = np.zeros(n,dtype=np.int64)
        self.origins = np.arange(1,n+1,dtype=np.int64)
        self.metadata = dict(fingerprint="synthetic-combo-resume-v2")
        rng = np.random.default_rng(707)
        self.x = rng.normal(size=(n,24,30)).astype(np.float32)*.05
        self.y = rng.normal(size=(n,1)).astype(np.float32)*.05

    def __len__(self):
        return len(self.x)

    def get_batch(self, indices):
        return dict(x=self.x[indices],y=self.y[indices],
                    mask=np.ones_like(self.y[indices],dtype=bool))


def test_resume_exactly_matches_fresh_mse_updates_without_checkpoint_dependency(tmp_path):
    from .fd001_combo import train_candidate
    c = next(x for x in make_manifest() if
             x["method"] == "td_mse_combo" and x["batch_size"] == 128
             and x["seed"] == 0 and x["configuration"] == 0)
    tr, va = TinyFD001("train"), TinyFD001("val")
    left, right = tmp_path/"uninterrupted",tmp_path/"interrupted"
    # There are NO baseline MSE checkpoints anywhere in either root.
    a = train_candidate(c,tr,va,left,microbatch=128,stop_after_updates=4)
    first = train_candidate(c,tr,va,right,microbatch=128,stop_after_updates=2)
    b = train_candidate(c,tr,va,right,microbatch=128,stop_after_updates=4)
    assert a["state"] == first["state"] == b["state"] == "checkpointed"
    assert a["phase_steps"]["mse"] == b["phase_steps"]["mse"] == 4
    one = torch.load(left/c["run_id"]/"recovery.pt",map_location="cpu",weights_only=False)
    two = torch.load(right/c["run_id"]/"recovery.pt",map_location="cpu",weights_only=False)
    assert one["progress"]["mse_rounds"] == two["progress"]["mse_rounds"]
    for key in one["objective"]:
        torch.testing.assert_close(one["objective"][key],two["objective"][key],
                                   rtol=0,atol=0)


def test_stage1_retrains_then_stage2_can_stop_but_stage3_runs_100_epochs(
        tmp_path, monkeypatch):
    from . import fd001_train as regular
    from . import fd001_combo as combo
    c = next(x for x in make_manifest() if
             x["method"] == "td_mse_combo" and x["batch_size"] == 256
             and x["seed"] == 0 and x["configuration"] == 0)
    tr,va = TinyFD001("train"),TinyFD001("val")
    def fake_validation(model,val_data,train,law,physical,steps,epoch,td):
        return dict(epoch=epoch,optimizer_steps=steps,
                    mse_standardized=1.,rmse_cycles=1.,mae_cycles=1.,
                    rul_score_sum=1.,gamma_probe={"types":[]})
    monkeypatch.setattr(regular,"capture_validation",fake_validation)
    # Only the stage-2 plateau evaluator is patched; stage-3 has no stop test.
    monkeypatch.setattr(combo,"_plateau",lambda progress,phase,config: True)
    result = combo.train_candidate(c,tr,va,tmp_path,microbatch=256)
    assert result["state"] == "complete"
    assert result["phase_epochs"] == dict(mse=100,gamma=1,predictor=100)
    assert result["phase"] == "complete"
    assert result["stop_reasons"] == dict(gamma="training_loss_plateau",
                                          predictor="max_epochs")
    assert result["phase_steps"] == dict(mse=100,gamma=1,predictor=100)
    assert result["stage1_model_sha256"]
    assert result["timing"]["mse_mean_update_seconds"] > 0
    assert result["timing"]["gamma_mean_update_seconds"] > 0
    assert result["timing"]["predictor_mean_update_seconds"] > 0
    assert len(list((tmp_path/c["run_id"]).glob("*.pt"))) >= 1
