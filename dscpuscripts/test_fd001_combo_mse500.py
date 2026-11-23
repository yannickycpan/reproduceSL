"""Test add-only, same-branch Combo MSE-500 experiment without changing legacy."""
from collections import Counter
import json

import numpy as np
import torch
from torch import nn

from dscpuscripts.fd001_combo_mse500_protocol import (
    make_manifest, validate_config, MSE_STAGE_ALPHA, MSE_STAGE_UPDATES,
)
from dscpuscripts.fd001_protocol import make_manifest as original_manifest
from dscpuscripts.fd001_combo_mse500_runner import classify
from dscpuscripts.fd001_combo_mse500_train import code_hash


def test_additive_manifest_preserves_original():
    old, new = original_manifest(), make_manifest()
    assert len(old) == 1600
    assert len(new) == len({x['run_id'] for x in new}) == 750
    assert not {x['run_id'] for x in old} & {x['run_id'] for x in new}
    assert MSE_STAGE_ALPHA == .007 and MSE_STAGE_UPDATES == 500
    assert {x['method'] for x in new} == {'td_mse_combo'}
    assert Counter((x['batch_size'], tuple(x['lags'])) for x in new) == {
        (batch, lags): 125 for batch in (128, 256)
        for lags in ((1,5), (1,3,5), (1,3,5,10,15))
    }
    assert all(x['mse_alpha'] == .007 and x['mse_max_updates'] == 500
               and x['gamma_max_epochs'] == x['predictor_max_epochs'] == 100
               for x in new)
    assert validate_config(new[0]) == new[0]
    from dscpuscripts.fd001_train import SCIENTIFIC_SOURCES
    assert not any('combo_mse500' in name for name in SCIENTIFIC_SOURCES)
    assert len(code_hash()) == 64


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


class TinyData:
    def __init__(self, split, n=1300):
        self.dataset, self.split = 'fd001', split
        self.entity_ids = np.zeros(n, dtype=np.int64)
        self.origins = np.arange(1, n+1, dtype=np.int64)
        self.metadata = dict(fingerprint='mse500-toy')
        self.x = np.zeros((n, 24, 30), dtype=np.float32)
        self.y = np.zeros((n, 1), dtype=np.float32)

    def __len__(self):
        return len(self.x)

    def get_batch(self, indices):
        return dict(x=self.x[indices], y=self.y[indices],
                    mask=np.ones_like(self.y[indices], dtype=bool))


def test_exact_500_updates_even_mid_epoch_and_resume(tmp_path, monkeypatch):
    from dscpuscripts import fd001_combo_mse500_train as stage
    from dscpuscripts.fd001_combo import ComboObjective
    from dscpuscripts import fd001_train as regular
    c = next(x for x in make_manifest()
             if x['batch_size'] == 256 and x['configuration'] == 0 and x['seed'] == 0)
    monkeypatch.setattr(stage, 'build_objective',
                        lambda cfg: ComboObjective(TinyPredictor(), len(cfg['lags'])))
    monkeypatch.setattr(stage, 'code_hash', lambda: 'mse500-toy-code')
    def fake_validation(*args, **kwargs):
        return dict(epoch=args[6], optimizer_steps=args[5], mse_standardized=1.,
                    rmse_cycles=1., mae_cycles=1., rul_score_sum=1.,
                    gamma_probe={'types': []})
    monkeypatch.setattr(regular, 'capture_validation', fake_validation)
    tr, val = TinyData('train'), TinyData('val', n=30)
    partial = stage.train_candidate(c, tr, val, tmp_path, microbatch=256,
                                    stop_after_updates=499)
    assert partial['state'] == 'checkpointed'
    continued = stage.train_candidate(c, tr, val, tmp_path, microbatch=256,
                                      stop_after_updates=501)
    assert continued['state'] == 'checkpointed'
    assert continued['phase'] == 'gamma'
    assert continued['phase_steps']['mse'] == 500
    assert continued['phase_steps']['gamma'] == 1
    assert continued['phase_epochs']['mse'] == 84   # 83 full + 1 partial
    assert continued['mse_partial_final_epoch'] is True
    assert continued['stage1_model_sha256']
    rows = json.loads((tmp_path/c['run_id']/'mse_rounds.json').read_text())
    assert len(rows) == 84 and rows[-1]['partial_epoch']
    assert rows[-1]['source_draws'] == 2*256
    assert sum(r['source_draws'] for r in rows) == 83*1300 + 2*256
    assert classify(c,tmp_path,'mse500-toy-code') == 'resume'
