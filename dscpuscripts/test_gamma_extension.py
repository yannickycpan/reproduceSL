"""Extension protocol, pairwise features, full gradients and old-HEAD preservation."""
from __future__ import annotations
import copy
from pathlib import Path
import math

import pytest
import torch
from torch import nn

from fd001_gamma.objectives import GammaTD
from fd001_gamma.protocol import make_manifest as original_manifest
from fd001_gamma.common import scientific_hash as original_scientific_hash
from fd001_gamma_ext.protocol import (make_manifest as extension_manifest,
                                      ADDITIONAL_ALPHAS, PAIRWISE_ALPHAS, BETAS)
from fd001_gamma_ext.objectives import PairwiseGammaHead, PairwiseGammaTD
from fd001_gamma_ext.runtime import (scientific_hash, campaign_paths,
                                     bind_original_training_engine)


class TinyMean(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.feature_dim, self.output_dim = d, 1
        self.hidden = nn.Linear(24, d)
        self.head = nn.Linear(d, 1)

    def features(self, x):
        return torch.tanh(self.hidden(x[:, :, -1]))

    def forward(self, x):
        return self.head(self.features(x))


def inputs():
    torch.manual_seed(34)
    x = torch.randn(4, 24, 30, dtype=torch.float64)
    p = torch.randn(4, 24, 30, dtype=torch.float64)
    a = dict(x=x, y=torch.randn(4, 1, dtype=torch.float64), mask=torch.ones(4, 1, dtype=torch.bool))
    b = dict(x=p, y=torch.randn(4, 1, dtype=torch.float64), mask=torch.ones(4, 1, dtype=torch.bool))
    return a, b, torch.tensor([0, 1, 0, 1])


def test_manifest_exact_extension_and_original_unmodified():
    before = original_manifest()
    extension = extension_manifest()
    assert len(before) == 180 and len(extension) == 140
    assert {c["run_id"] for c in before}.isdisjoint({c["run_id"] for c in extension})
    assert len([c for c in extension if c["gamma_variant"] == "representation"]) == 40
    assert len([c for c in extension if c["gamma_variant"] == "pairwise"]) == 100
    for variant, alphas in (("representation", ADDITIONAL_ALPHAS),
                            ("pairwise", PAIRWISE_ALPHAS)):
        subset = [c for c in extension if c["gamma_variant"] == variant]
        assert {c["alpha"] for c in subset} == set(alphas)
        assert {c["beta"] for c in subset} == set(BETAS)
        assert {c["seed"] for c in subset} == set(range(5))
        assert all(c["epochs"] == 100 and c["batch_size"] == 128 and
                   c["selection_epochs"] == [80,90,100] and
                   c["gradient_clipping"] is None and c["weight_decay"] == 0
                   for c in subset)
    assert original_manifest() == before


@pytest.mark.parametrize("d", [5, 11, 64])
def test_pair_features_correct_and_hidden_size_not_hardcoded(d):
    head = PairwiseGammaHead(d).double()
    hi, hj = torch.randn(4,d,dtype=torch.float64), torch.randn(4,d,dtype=torch.float64)
    s = head.pair_features(hi,hj)
    assert s.shape == (4, d+2)
    torch.testing.assert_close(s[:, :1], (hi*hj).sum(dim=1,keepdim=True)/d)
    torch.testing.assert_close(s[:, 1:-1], (hi-hj).abs())
    torch.testing.assert_close(s[:, -1:], (hi-hj).norm(dim=1,keepdim=True)/math.sqrt(d))
    torch.testing.assert_close(s, head.pair_features(hj,hi), rtol=0, atol=0)
    assert sum(p.numel() for p in head.parameters()) == 2*(d+2)+2
    assert not torch.any(head(hi,hj,torch.tensor([0,1,0,1])))
    with pytest.raises(ValueError, match="matching"):
        head.pair_features(hi, torch.randn(4,d+1))


def test_full_gradient_through_both_pairwise_representations():
    a,b,t = inputs()
    model = TinyMean(5).double()
    objective = PairwiseGammaTD(model).double()
    with torch.no_grad():
        objective.gamma_head.linear.weight.copy_(torch.randn_like(objective.gamma_head.linear.weight)*.1)
    hi = model.features(a["x"]).detach().requires_grad_(True)
    hj = model.features(b["x"]).detach().requires_grad_(True)
    g = objective.gamma_head(hi,hj,t)
    gi,gj = torch.autograd.grad(g.sum(),(hi,hj))
    assert gi.abs().sum() > 0 and gj.abs().sum() > 0
    for v in (gi,gj):
        assert torch.isfinite(v).all()
    (loss, n) = objective.loss_sum(a,b,t)
    (loss/n).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in objective.parameters())
    groups = objective.parameter_groups(.0003,.00003)
    first = {id(x) for x in groups[0]["params"]}
    second = {id(x) for x in groups[1]["params"]}
    assert first.isdisjoint(second)
    assert first|second == {id(x) for x in objective.parameters()}


def test_pairwise_all_head_parameters_have_correct_gradients():
    a,b,t=inputs()
    model=TinyMean(5).double()
    obj=PairwiseGammaTD(model).double()
    with torch.no_grad():
        for p in obj.operator_parameters():
            p.copy_(torch.linspace(-.05,.05,p.numel(),dtype=p.dtype).reshape_as(p))
    loss,count=obj.loss_sum(a,b,t)
    (loss/count).backward()
    for name,param in obj.named_parameters():
        # Test every head coordinate and both endpoints of the predictor weights.
        ix=range(param.numel()) if "gamma_head." in name else (0,param.numel()-1)
        analytic=param.grad.flatten().clone()
        for i in ix:
            with torch.no_grad():
                old=param.flatten()[i].item()
                param.flatten()[i]=old+1e-6
            plus=obj.loss_sum(a,b,t)[0].item()/count
            with torch.no_grad():
                param.flatten()[i]=old-1e-6
            minus=obj.loss_sum(a,b,t)[0].item()/count
            with torch.no_grad():
                param.flatten()[i]=old
            assert analytic[i].item() == pytest.approx((plus-minus)/2e-6, rel=3e-5, abs=1e-8), (name,i)


def test_extended_representation_is_exact_old_head():
    a,b,t=inputs()
    model=TinyMean(64).double()
    old=GammaTD(copy.deepcopy(model), "representation").double()
    candidate=GammaTD(copy.deepcopy(model), "representation").double()
    for v in (old,candidate):
        loss,n=v.loss_sum(a,b,t)
        (loss/n).backward()
    for (_,op),(_,cp) in zip(old.named_parameters(),candidate.named_parameters()):
        torch.testing.assert_close(op, cp, rtol=0, atol=0)
        torch.testing.assert_close(op.grad,cp.grad,rtol=0,atol=0)


def test_training_engine_bindings_are_restored_after_context():
    import fd001_gamma.train as original_train
    original_factory=original_train.make_objective
    original_validate=original_train.validate_config
    original_proto=original_train.PROTOCOL_ID
    before=original_scientific_hash()
    with bind_original_training_engine() as trained:
        assert trained.validate_config == __import__("fd001_gamma_ext.protocol",fromlist=["validate_config"]).validate_config
        assert trained.scientific_hash() == scientific_hash()
        assert trained.PROTOCOL_ID != original_proto
    assert original_train.make_objective is original_factory
    assert original_train.validate_config is original_validate
    assert original_train.PROTOCOL_ID == original_proto
    assert original_scientific_hash() == before


def test_extension_results_in_separate_directory(tmp_path,monkeypatch):
    monkeypatch.setenv("FULLTD_FD001_GAMMA_EXT_RESULTS_DIR", str(tmp_path/"ext"))
    monkeypatch.setenv("FULLTD_FD001_GAMMA_EXT_OUTPUT_DIR", str(tmp_path/"plots"))
    assert campaign_paths()[1] == tmp_path/"ext"
    assert campaign_paths()[2] == tmp_path/"plots"
    assert not (tmp_path/"ext").exists()


def test_direct_runner_preserves_original_default_and_has_extension_mode():
    from dscpuscripts import direct_runner as d
    import inspect
    content=inspect.getsource(d.main)
    assert 'default="original"' in content
    assert 'ext_runtime.scientific_hash' in content
    assert '"--campaign", getattr(args, "campaign", "original")' in inspect.getsource(d.run_campaign)
