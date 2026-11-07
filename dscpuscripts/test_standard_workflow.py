"""Tests for unchanged campaign scope and explicit metric conventions."""
import argparse
from collections import Counter
import copy
import json
import math
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
from dscpuscripts import standard_runner as runner, fd001_report as report
from fd001_gamma.common import atomic_json
from fulltd.protocol import make_manifest, digest


@pytest.mark.parametrize('name,count', [('fd001_baselines',60),('electricity',300),('metr_la',210)])
def test_canonical_manifest_is_filtered_not_rewritten(name,count):
    configs = runner.configurations(name)
    assert len(configs) == count
    originals = {c['run_id']: c for c in make_manifest()}
    assert all(c == originals[c['run_id']] for c in configs)
    assert all(c['epochs'] == 100 for c in configs)
    assert len({c['run_id'] for c in configs}) == count
    if name == 'fd001_baselines':
        assert {c['method'] for c in configs} == {'mse','gaussian_nll'}
        assert {c['batch_size'] for c in configs} == {256}
        assert len({c['alpha'] for c in configs}) == 6


def test_dataset_results_paths_share_fd001_and_isolate_other_datasets(tmp_path,monkeypatch):
    monkeypatch.setenv('FULLTD_RESULTS_ROOT',str(tmp_path / 'fulltd'))
    monkeypatch.setenv('FULLTD_DATA_DIR',str(tmp_path / 'data'))
    monkeypatch.delenv('FULLTD_FD001_ROOT', raising=False)
    monkeypatch.delenv('FULLTD_METR_LA_ROOT', raising=False)
    baseline=runner.paths('fd001_baselines')
    traffic=runner.paths('metr_la')
    assert baseline[0]==traffic[0]==tmp_path/'data'
    assert baseline[1]!=traffic[1]
    assert baseline[1]==tmp_path/'fulltd/fd001_results'
    assert baseline[2]==tmp_path/'fulltd/fd001_results/outputs/baselines'
    assert traffic[1]==tmp_path/'fulltd/metr_la_results'
    assert not baseline[1].exists()


def test_resumption_requires_identity_and_all_final_evidence(tmp_path):
    config=runner.configurations('fd001_baselines')[0]
    assert runner.classify(config,tmp_path,'code')=='missing'
    d=tmp_path/config['run_id'];d.mkdir()
    status=dict(config=config,identity=dict(config_sha256=digest(config),code_sha256='code'),state='complete',epochs_completed=100)
    atomic_json(d/'status.json',status)
    assert runner.classify(config,tmp_path,'code')=='resume'
    for f in ('final.pt','rounds.json','validation.json'):(d/f).touch()
    assert runner.classify(config,tmp_path,'code')=='complete'
    with pytest.raises(ValueError,match='Incompatible'):
        runner.classify(config,tmp_path,'other')
    status['state']='numerical_failure';atomic_json(d/'status.json',status)
    assert runner.classify(config,tmp_path,'code')=='numerical_failure'


def test_worker_uses_existing_trainer_without_parameter_changes(tmp_path,monkeypatch):
    config=runner.configurations('fd001_baselines')[0]
    root=tmp_path/'results'
    monkeypatch.setattr(runner,'paths',lambda _: (tmp_path/'data',root,tmp_path/'out'))
    train=ModuleType('fulltd.train');train.code_hash=lambda:'code'
    captured=[]
    def fit(c,training,validation,results,**kwargs):
        captured.append((c,training,validation,results,kwargs))
        atomic_json(results/c['run_id']/'status.json',dict(state='complete'))
    train.train_candidate=fit
    data=ModuleType('fulltd.data');data.load_dataset=lambda name,path,split: (name,split)
    monkeypatch.setitem(sys.modules,'fulltd.train',train)
    monkeypatch.setitem(sys.modules,'fulltd.data',data)
    import torch
    monkeypatch.setattr(torch,'set_num_threads',lambda _:None)
    monkeypatch.setattr(torch,'set_num_interop_threads',lambda _:None)
    args=argparse.Namespace(campaign='fd001_baselines',expected_code='code',run_id=config['run_id'],microbatch=128)
    assert runner.execute_worker(args)==0
    c,tr,va,dest,kw=captured[0]
    assert c==config and tr==('fd001','train') and va==('fd001','val')
    assert kw['device']=='cpu' and kw['microbatch']==128
    assert c['batch_size']==256  # Microbatch is NOT the logical batch.
    assert not kw['stop_requested']()


def test_rmse_computed_per_seed_not_sqrt_of_pooled_mse():
    # The reporting transform is explicitly per-seed; this example separates
    # the two competing definitions (mean sqrt MSE = 3, sqrt mean MSE > 3).
    assert report.moments([math.sqrt(v) for v in [1,4,9,16,25]])[0]==3
    assert math.sqrt(sum([1,4,9,16,25])/5)>3


def test_nonfinite_metrics_remain_visible_not_filtered():
    mean,sd=report.moments([1,2,3,4,math.inf])
    assert mean==math.inf and math.isnan(sd)


def test_analysis_has_no_optimization_or_refit():
    text=(Path(__file__).with_name('standard_analysis.py')).read_text()
    assert 'optimizer.step' not in text and 'refit_readouts' not in text
    assert 'train_candidate(' not in text
    assert 'actual epoch-100 mean network only' in text


def test_baseline_selection_same_late_rule_and_only_baseline_candidates(tmp_path,monkeypatch):
    from dscpuscripts import standard_analysis as a
    import fulltd.protocol as protocol
    checked=[]
    fake=ModuleType('fulltd.analysis')
    def validate(c,path):
        checked.append(c)
        score=abs(c['configuration']-2)+0.1+c['seed']*.01
        records=[dict(epoch=e,mse=score,mae=.2) for e in protocol.EVALUATIONS]
        status=dict(state='complete',identity={'fixture':1},initial_mean_sha256=str(c['seed']),epochs_completed=100)
        return score,records,status
    fake._validate_candidate=validate
    fake.audit_campaign=lambda manifest,*a,**k:dict(runs=len(manifest))
    fake.group_id=lambda c:c['method']
    fake._manifest_hash=lambda x:'fixture-hash'
    monkeypatch.setitem(sys.modules,'fulltd.analysis',fake)
    s=a.select_baselines(tmp_path/'results',tmp_path/'outputs')
    assert len(checked)==60 and {c['method'] for c in checked}=={'mse','gaussian_nll'}
    assert s['selection_epochs']==[80,90,100]
    assert all(g['configuration']==2 and len(g['run_ids'])==5 for g in s['groups'])
