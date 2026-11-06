"""Gradient, protocol, recovery and scheduling checks for the isolated Gamma sweep."""
import argparse
import contextlib
import copy
import csv
import io
import json
import math
import os
from pathlib import Path
import shlex
import signal
import subprocess
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from torch import nn

from agents.fulltd_objectives import TDObjective
from fulltd.models import build_model, parameter_count
from fulltd.sampling import LocalPairLaw, sources_for_round, sequence_hash
from fd001_gamma import PROTOCOL_ID
from fd001_gamma import analysis, hpc, train
from fd001_gamma.common import atomic_json, read_json, scientific_hash, file_lock
from fd001_gamma.objectives import GammaTD, make_objective
from fd001_gamma.protocol import (ALPHAS, BETAS, VARIANTS, SEEDS, EVALUATIONS,
                                  digest, make_manifest, validate_config, validate_manifest)


@pytest.fixture(autouse=True)
def single_thread():
    old = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)
    torch.use_deterministic_algorithms(deterministic)


class TinyMean(nn.Module):
    feature_dim, output_dim = 64, 1

    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(24, 64)
        self.head = nn.Linear(64, 1)

    def features(self, x):
        return torch.tanh(self.hidden(x[:, :, -1]))

    def forward(self, x):
        return self.head(self.features(x))


def tiny_objective(variant, dtype=torch.float32):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(555)
        return GammaTD(TinyMean(), variant).to(dtype=dtype)


def config(variant="typed", configuration=0, seed=0):
    return next(c for c in make_manifest() if c["gamma_variant"] == variant
                and c["configuration"] == configuration and c["seed"] == seed)


def batch(n=7, dtype=torch.float64, masked=True):
    g = torch.Generator().manual_seed(29)
    x = torch.randn(n, 24, 30, generator=g, dtype=dtype)
    y = torch.randn(n, 1, generator=g, dtype=dtype)
    mask = torch.ones(n, 1, dtype=torch.bool)
    other_mask = mask.clone()
    if masked:
        mask[1] = False
        other_mask[4] = False
    return (dict(x=x, y=y, mask=mask),
            dict(x=.3 * x + .2, y=.4 * y + .1, mask=other_mask),
            torch.arange(n) % 2)


def set_nonzero(obj):
    with torch.no_grad():
        if obj.gamma_variant == "typed":
            obj.operator.copy_(torch.tensor([.2, -.15], dtype=obj.operator.dtype))
        elif obj.gamma_variant == "raw26":
            obj.operator.copy_(torch.linspace(-.03, .02, 26, dtype=obj.operator.dtype))
        else:
            obj.gamma_head.weight.copy_(torch.linspace(-.03, .04, 128, dtype=obj.gamma_head.weight.dtype).reshape(2, 64))
            obj.gamma_head.bias.copy_(torch.tensor([.1, -.2], dtype=obj.gamma_head.bias.dtype))


def test_manifest_exact_cartesian_grid_and_disjoint_ids():
    runs = make_manifest()
    assert len(runs) == 180 and len({c['run_id'] for c in runs}) == 180
    assert ALPHAS == (3e-4, 1e-3, 3e-3)
    assert BETAS == (3e-5, 1e-4, 3e-4, 1e-3)
    assert {(c['gamma_variant'], c['alpha'], c['beta'], c['seed']) for c in runs} == {
        (v, a, b, s) for v in VARIANTS for a in ALPHAS for b in BETAS for s in SEEDS}
    assert all(c['batch_size'] == 128 and c['epochs'] == 100 and c['lags'] == [1, 5] for c in runs)
    assert all(c['gradient_clipping'] is None and c['learning_rate_schedule'] is None and c['weight_decay'] == 0 for c in runs)
    validate_manifest(runs, complete=True)
    with pytest.raises(ValueError):
        validate_config(dict(runs[0], beta=1e-5))
    with pytest.raises(ValueError):
        validate_manifest(runs[:-1], complete=True)
    with pytest.raises(ValueError):
        validate_manifest([runs[0], runs[0]])


@pytest.mark.parametrize('variant,count', [('typed', 2), ('raw26', 26), ('representation', 130)])
def test_parameter_counts_initialization_rng_and_groups(variant, count):
    torch.manual_seed(9)
    before = torch.get_rng_state().clone()
    objective = make_objective(config(variant))
    assert torch.equal(before, torch.get_rng_state())
    reference = build_model('fd001', 'tcn', 0)
    assert parameter_count(reference) == 92801
    assert parameter_count(objective) == 92801 + count
    assert all(torch.equal(t, objective.model.state_dict()[k]) for k, t in reference.state_dict().items())
    assert all(torch.count_nonzero(p) == 0 for p in objective.operator_parameters())
    groups = objective.parameter_groups(3e-4, 3e-5)
    p0, p1 = [{id(p) for p in g['params']} for g in groups]
    assert p0.isdisjoint(p1) and p0 | p1 == {id(p) for p in objective.parameters()}
    assert [g['lr'] for g in groups] == [3e-4, 3e-5]


def test_raw26_uses_latest_standardized_source_minus_partner_and_lag_one_hot():
    source, other, types = batch()
    objective = tiny_objective('raw26', torch.float64)
    phi = objective.raw_features(source['x'], other['x'], types)
    assert phi.shape == (7, 26)
    torch.testing.assert_close(phi[:, :2], torch.nn.functional.one_hot(types, 2).double())
    torch.testing.assert_close(phi[:, 2:], source['x'][:, :, -1] - other['x'][:, :, -1])
    changed = source['x'].clone()
    changed[:, :, :-1] += 100
    torch.testing.assert_close(phi, objective.raw_features(changed, other['x'], types), rtol=0, atol=0)
    set_nonzero(objective)
    torch.testing.assert_close(objective.gamma_values(source['x'], other['x'], types), phi @ objective.operator)


def test_representation_two_outputs_source_only_signed_unrestricted():
    source, other, types = batch()
    obj = tiny_objective('representation', torch.float64)
    with torch.no_grad():
        obj.gamma_head.bias.copy_(torch.tensor([-2., 3.]))
    actual = obj.gamma_values(source['x'], other['x'], types)
    torch.testing.assert_close(actual, torch.tensor([-2., 3.], dtype=torch.float64)[types])
    torch.testing.assert_close(actual, obj.gamma_values(source['x'], other['x'] + 42, types))
    assert obj.gamma_head.weight.shape == (2, 64)
    with pytest.raises(ValueError):
        obj.gamma_values(source['x'], other['x'], types.float())


@pytest.mark.parametrize('variant', VARIANTS)
def test_zero_gamma_matches_half_mse_and_mean_gradient(variant):
    source, other, types = batch(masked=False)
    obj = tiny_objective(variant, torch.float64)
    baseline = copy.deepcopy(obj.model)
    loss, count = obj.loss_sum(source, other, types)
    reference = .5 * (source['y'] - baseline(source['x'])).square().mean()
    torch.testing.assert_close(loss / count, reference, atol=0, rtol=0)
    (loss / count).backward()
    reference.backward()
    for a, b in zip(obj.model.parameters(), baseline.parameters()):
        torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
    assert any(p.grad.abs().sum() > 0 for p in obj.operator_parameters())


@pytest.mark.parametrize('variant', VARIANTS)
def test_full_gradients_finite_difference_and_partner_path(variant):
    source, other, types = batch()
    obj = tiny_objective(variant, torch.float64)
    set_nonzero(obj)
    other['x'] = other['x'].clone().requires_grad_()
    total, count = obj.loss_sum(source, other, types)
    (total / count).backward()
    assert other['x'].grad.abs().sum() > 0
    for name, parameter in obj.named_parameters():
        indices = range(parameter.numel()) if not name.startswith('model.') else [0, parameter.numel() // 2, parameter.numel() - 1]
        analytic = parameter.grad.detach().flatten().clone()
        for index in indices:
            with torch.no_grad():
                flat = parameter.reshape(-1)
                original = flat[index].item()
                flat[index] = original + 1e-6
            plus = obj.loss_sum(source, other, types)[0].item() / count
            with torch.no_grad():
                flat[index] = original - 1e-6
            minus = obj.loss_sum(source, other, types)[0].item() / count
            with torch.no_grad():
                flat[index] = original
            assert analytic[index].item() == pytest.approx((plus - minus) / 2e-6, abs=2e-9, rel=1e-5), (name, index)


def test_representation_gradient_contains_gamma_to_backbone_path():
    source, other, types = batch()
    obj = tiny_objective('representation', torch.float64)
    set_nonzero(obj)
    loss, count = obj.loss_sum(source, other, types)
    exact, = torch.autograd.grad(loss / count, obj.model.hidden.weight)
    h = obj.model.features(source['x'])
    residual = source['y'] - obj.model.head(h)
    partner_residual = other['y'] - obj.model(other['x'])
    gamma_detached_h = obj.gamma_head(h.detach()).gather(1, types[:, None])
    mask = source['mask'] & other['mask']
    wrong = .5 * torch.where(mask, residual - gamma_detached_h * partner_residual, torch.zeros_like(residual)).square().sum() / count
    detached, = torch.autograd.grad(wrong, obj.model.hidden.weight)
    assert torch.linalg.norm(exact - detached) > 1e-4


def test_original_typed_td_bitwise_equivalence_over_adam_updates():
    source, other, types = batch(dtype=torch.float32)
    current = tiny_objective('typed')
    old = TDObjective(copy.deepcopy(current.model), 2)
    optimizers = [torch.optim.Adam(o.parameter_groups(3e-4, 3e-5), betas=(.9, .999), eps=1e-8, weight_decay=0)
                  for o in (current, old)]
    for _ in range(5):
        for objective, optimizer in zip((current, old), optimizers):
            optimizer.zero_grad(set_to_none=True)
            loss, count = objective.loss_sum(source, other, types)
            (loss / count).backward()
            optimizer.step()
        for a, b in zip(current.model.parameters(), old.model.parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        torch.testing.assert_close(current.operator, old.theta, rtol=0, atol=0)


@pytest.mark.parametrize('variant', VARIANTS)
def test_microbatch_gradient_and_one_adam_step(variant):
    source, other, types = batch(n=17, dtype=torch.float32)
    first = tiny_objective(variant)
    set_nonzero(first)
    second = copy.deepcopy(first)
    source_np = {k: v.numpy() for k, v in source.items()}
    other_np = {k: v.numpy() for k, v in other.items()}
    optimizers = [torch.optim.Adam(o.parameter_groups(3e-4, 3e-5), eps=1e-8) for o in (first, second)]
    for objective, chunk in ((first, 17), (second, 6)):
        total, count = train.logical_backward(objective, source_np, other_np, types.numpy(), 'cpu', chunk)
        assert count == 15
    for a, b in zip(first.parameters(), second.parameters()):
        torch.testing.assert_close(a.grad, b.grad, rtol=3e-5, atol=2e-7)
    for optimizer in optimizers:
        optimizer.step()
    for a, b in zip(first.parameters(), second.parameters()):
        torch.testing.assert_close(a, b, rtol=2e-5, atol=1e-7)


@pytest.mark.parametrize('variant', VARIANTS)
def test_masked_nan_and_empty_pair_gradients(variant):
    source, other, types = batch()
    source['y'][~source['mask']] = float('nan')
    other['y'][~other['mask']] = float('nan')
    obj = tiny_objective(variant, torch.float64)
    set_nonzero(obj)
    loss, count = obj.loss_sum(source, other, types)
    assert count == 5 and torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in obj.parameters())
    obj.zero_grad()
    other['mask'].fill_(False)
    loss, count = obj.loss_sum(source, other, types)
    assert count == 0 and loss == 0
    loss.backward()
    assert all(torch.count_nonzero(p.grad) == 0 for p in obj.parameters())


class FixtureData:
    dataset = 'fd001'

    def __init__(self, n=259, split='train'):
        self.split = split
        self.metadata = dict(fingerprint=f'fixture-{n}-{split}')
        self.entity_ids = np.zeros(n, dtype=np.int64)
        self.origins = np.arange(n, dtype=np.int64) + 30
        rng = np.random.default_rng(567)
        self.x = rng.normal(size=(n, 24, 30)).astype('float32') * .2
        self.y = (.5 * self.x[:, 0, -1] + .2 * self.x[:, 1, -1])[:, None]
        self.reads = 0

    def __len__(self):
        return len(self.origins)

    def get_batch(self, ids):
        if self.split == 'test':
            raise AssertionError('Training read test labels')
        self.reads += 1
        return dict(x=self.x[ids].copy(), y=self.y[ids].copy(), mask=np.ones_like(self.y[ids], dtype=bool))

    def inverse(self, values, ids):
        return np.asarray(values) * 30 + 70


@pytest.fixture
def tiny_training(monkeypatch):
    monkeypatch.setattr(train, 'make_objective', lambda c: tiny_objective(c['gamma_variant']))


@pytest.mark.parametrize('variant', VARIANTS)
def test_100_epochs_and_exact_resume(tmp_path, tiny_training, variant):
    c = config(variant)
    data, val = FixtureData(), FixtureData(11, 'val')
    with contextlib.redirect_stdout(io.StringIO()):
        complete = train.train_candidate(c, data, val, tmp_path / 'full', microbatch=64)
        interrupted = train.train_candidate(c, data, val, tmp_path / 'resumed', microbatch=64, stop_after_batches=4)
        assert interrupted['next_epoch'] == 2 and interrupted['optimizer_steps'] == 3
        assert interrupted['state'] == 'checkpointed'
        finished = train.train_candidate(c, data, val, tmp_path / 'resumed', microbatch=64)
    assert complete['epochs_completed'] == finished['epochs_completed'] == 100
    assert finished['source_draws'] == finished['partner_draws'] == 25900
    assert finished['optimizer_steps'] == 300
    a = torch.load(tmp_path / 'full' / c['run_id'] / 'final.pt', weights_only=False)
    b = torch.load(tmp_path / 'resumed' / c['run_id'] / 'final.pt', weights_only=False)
    assert a['validation'] == b['validation']
    assert a['round_records'] == b['round_records']
    assert [v['epoch'] for v in b['validation']] == list(EVALUATIONS)
    assert len(b['round_records']) == 100
    for key, value in a['objective'].items():
        torch.testing.assert_close(value, b['objective'][key], rtol=0, atol=0)
    for key, state in a['optimizer']['state'].items():
        for name, value in state.items():
            torch.testing.assert_close(value, b['optimizer']['state'][key][name], rtol=0, atol=0)
    interruptions = read_json(tmp_path / 'resumed' / c['run_id'] / 'interruptions.json')
    assert interruptions[0]['discarded_counters']['optimizer_steps'] == 1


@pytest.mark.parametrize('variant', VARIANTS)
def test_real_tcn_100_small_epochs(tmp_path, variant):
    c = config(variant)
    with contextlib.redirect_stdout(io.StringIO()):
        result = train.train_candidate(c, FixtureData(8), FixtureData(4, 'val'), tmp_path, microbatch=4)
    assert result['state'] == 'complete' and result['optimizer_steps'] == 100
    assert result['total_parameter_count'] == {'typed':92803, 'raw26':92827, 'representation':92931}[variant]


def test_warning_before_first_epoch_no_checkpoint(tmp_path, tiny_training):
    c = config()
    result = train.train_candidate(c, FixtureData(8), FixtureData(4, 'val'), tmp_path, stop_requested=lambda: True)
    assert result['state'] == 'stopped_no_checkpoint' and result['optimizer_steps'] == 0
    assert not (tmp_path / c['run_id'] / 'recovery.pt').exists()


def test_validation_interruption_replays_and_repairs_final_sidecars(tmp_path, tiny_training):
    c = config('raw26')
    flag = False
    class Val(FixtureData):
        def get_batch(self, ids):
            nonlocal flag
            result = super().get_batch(ids)
            flag = True
            return result
    with contextlib.redirect_stdout(io.StringIO()):
        partial = train.train_candidate(c, FixtureData(8), Val(4, 'val'), tmp_path, stop_requested=lambda: flag)
        assert partial['next_epoch'] == 10
        train.train_candidate(c, FixtureData(8), FixtureData(4, 'val'), tmp_path)
    folder = tmp_path / c['run_id']
    (folder / 'validation.json').unlink()
    (folder / 'rounds.json').unlink()
    before = (folder / 'final.pt').read_bytes()
    restored = train.train_candidate(c, FixtureData(8), FixtureData(4, 'val'), tmp_path)
    assert restored['state'] == 'complete' and (folder / 'final.pt').read_bytes() == before
    assert len(read_json(folder / 'validation.json')) == 10


def test_failure_and_identity_are_never_relabelled(tmp_path, tiny_training):
    c = config()
    bad = FixtureData(8)
    bad.y[:] = np.nan
    result = train.train_candidate(c, bad, FixtureData(4, 'val'), tmp_path)
    assert result['state'] == 'numerical_failure'
    good = FixtureData(8)
    assert train.train_candidate(c, good, FixtureData(4, 'val'), tmp_path)['state'] == 'numerical_failure'
    assert good.reads == 0
    changed = FixtureData(4, 'val')
    changed.metadata = {'fingerprint': 'changed'}
    original = (tmp_path / c['run_id'] / 'status.json').read_bytes()
    with pytest.raises(ValueError, match='Incompatible'):
        train.train_candidate(c, good, changed, tmp_path)
    assert (tmp_path / c['run_id'] / 'status.json').read_bytes() == original


def args(**changes):
    values = dict(mode='--submit', workers=90, concurrency=2, microbatch=128, memory='400G',
                  time='02:00:00', partition='batch', signal_seconds=180)
    values.update(changes)
    return argparse.Namespace(**values)


@pytest.fixture
def submit_env(tmp_path, monkeypatch):
    data, results, out = tmp_path / 'data space', tmp_path / 'results space', tmp_path / 'outputs'
    monkeypatch.setattr(hpc, 'campaign_paths', lambda: (data, results, out))
    monkeypatch.setattr(hpc, 'queued_jobs', lambda: [])
    import fulltd.data
    monkeypatch.setattr(fulltd.data, 'load_dataset', lambda name, path, split: object())
    return data, results, out


def test_submission_correct_limits_and_no_legacy_module(submit_env, monkeypatch):
    monkeypatch.setattr(hpc.subprocess, 'check_output', lambda command, **kwargs: '123456;cluster\n')
    receipt = hpc.submit(args())
    command = receipt['command']
    assert '--ntasks=90' in command and '--signal=B:USR1@180' in command
    for key, value in [('--time', '02:00:00'), ('--mem', '400G'), ('--array', '0-1%2')]:
        assert command[command.index(key) + 1] == value
    wrapped = shlex.split(command[-1])
    assert wrapped[:3] == ['exec', 'bash', '-c']
    assert 'fd001_gamma.hpc' in wrapped[3] and 'fulltd.hpc' not in wrapped[3]
    assert str(submit_env[1]) in wrapped[3]
    assert receipt['candidates'] == 180
    manifests = list((submit_env[1] / 'submissions').glob('*manifest.json'))
    assert len(manifests) == 1 and len(read_json(manifests[0])) == 180


def test_dry_run_writes_nothing_and_duplicate_refusal(submit_env, monkeypatch):
    monkeypatch.setattr(hpc.subprocess, 'check_output', lambda *a, **kw: pytest.fail('Must not submit'))
    hpc.submit(args(mode='--dry-run'))
    assert not submit_env[1].exists()
    monkeypatch.setattr(hpc, 'queued_jobs', lambda: [('123', hpc.JOB_NAME, 'RUNNING')])
    with pytest.raises(RuntimeError, match='already queued'):
        hpc.submit(args())
    assert not submit_env[1].exists()


@pytest.mark.parametrize(('stdout', 'expected_ids'), [
    ('', []),
    ('Last update in 0 seconds ago\n', []),
    ('739296\nLast update in 13 seconds ago\n', ['739296']),
    ('718378_2\n718378_3\nLast update in 1 second ago\n', ['718378_2', '718378_3']),
    ('739296_[0-5]\nLast update in 16 seconds ago\n', ['739296_[0-5]']),
    ('739296\n', ['739296']),
])
def test_gamma_scheduler_query_filters_other_jobs_and_accepts_arrays(monkeypatch, stdout, expected_ids):
    monkeypatch.setenv('USER', 'fixture')
    def fake_run(command, **kwargs):
        assert command == ['squeue', '--array', '--noheader', '--user', 'fixture',
                           '--name', hpc.JOB_NAME, '--format', '%i']
        assert kwargs == dict(check=True, text=True, capture_output=True)
        return SimpleNamespace(stdout=stdout)
    monkeypatch.setattr(hpc.subprocess, 'run', fake_run)
    assert hpc.queued_jobs() == [(job_id, hpc.JOB_NAME, 'queued') for job_id in expected_ids]


@pytest.mark.parametrize('stdout', ['JOBID\n', 'bogus|name|RUNNING\n',
                                    'Last update in unknown seconds ago\n',
                                    'Last update in 13 seconds ago\nmalformed-id\n'])
def test_gamma_scheduler_rejects_unexpected_output_with_diagnostics(monkeypatch, stdout):
    monkeypatch.setenv('USER', 'fixture')
    monkeypatch.setattr(hpc.subprocess, 'run', lambda *args, **kwargs: SimpleNamespace(stdout=stdout))
    with pytest.raises(RuntimeError, match='Raw response'):
        hpc.queued_jobs()


def test_scheduler_failure_closed(monkeypatch):
    monkeypatch.setenv('USER', 'fixture')
    monkeypatch.setattr(hpc.subprocess, 'run', Mock(side_effect=subprocess.CalledProcessError(1, ['squeue'])))
    with pytest.raises(subprocess.CalledProcessError):
        hpc.queued_jobs()


def test_requeue_only_on_durable_progress():
    terminal = [dict(state='complete')]
    assert hpc.completion_action(terminal, 0, False) == 'complete'
    progress = [dict(state='checkpointed', before_epoch=1, next_epoch=3)]
    assert hpc.completion_action(progress, 0, True) == 'requeue'
    for outcomes, code, warned in [(progress, 1, True), (progress, 0, False),
                                   ([dict(state='checkpointed', before_epoch=3, next_epoch=3)], 0, True),
                                   ([dict(state='stopped_no_checkpoint', before_epoch=1, next_epoch=1)], 0, True),
                                   ([dict(state='missing')], 0, True)]:
        with pytest.raises(RuntimeError):
            hpc.completion_action(outcomes, code, warned)


def test_locks_and_manifest_code_identity(tmp_path):
    with file_lock(tmp_path / 'lock'):
        with pytest.raises(RuntimeError, match='concurrent'):
            with file_lock(tmp_path / 'lock'):
                pass
    with file_lock(tmp_path / 'lock'):
        pass
    manifest = make_manifest()[:2]
    path = tmp_path / 'manifest.json'
    atomic_json(path, manifest)
    a = SimpleNamespace(manifest=str(path), digest=digest(manifest), expected_code_hash=scientific_hash())
    assert hpc.frozen_configs(a) == manifest
    a.expected_code_hash = 'wrong'
    with pytest.raises(ValueError, match='code changed'):
        hpc.frozen_configs(a)


def completed_campaign(root):
    code = scientific_hash()
    for c in make_manifest():
        n = 8
        folder = root / c['run_id']
        rounds = [dict(epoch=e, source_draws=n,
                       source_sha256=sequence_hash(sources_for_round('fd001', n, c['seed'], e)),
                       partner_sha256=f"pair-{c['seed']}-{e}", type_counts=[4,4], optimizer_steps_cumulative=e)
                  for e in range(1, 101)]
        score = 1 + VARIANTS.index(c['gamma_variant']) + abs(c['configuration'] - 7) + c['seed'] * .01
        vals = [dict(epoch=e, mse=score if e >= 80 else 100 - score, mae=1,
                     optimizer_steps=e, validation_sampling='full', validation_population_size=4)
                for e in EVALUATIONS]
        status = dict(config=c, protocol_id=PROTOCOL_ID, state='complete', epochs_completed=100, next_epoch=101,
                      identity=dict(protocol_id=PROTOCOL_ID, code_sha256=code, config_sha256=digest(c),
                                    data_sha256='same_train', validation_sha256='same_val'),
                      n_sources=n, initial_mean_sha256=f"same-mean-{c['seed']}", source_draws=800, partner_draws=800,
                      optimizer_steps=100, update_opportunities=100, empty_mask_batches=0)
        atomic_json(folder / 'status.json', status)
        atomic_json(folder / 'validation.json', vals)
        atomic_json(folder / 'rounds.json', rounds)
        (folder / 'final.pt').write_bytes(b'fixture')


def test_joint_categorical_selection_uses_pooled_late_mse_only(tmp_path):
    root = tmp_path / 'runs'
    completed_campaign(root)
    selected = analysis.select(root, tmp_path / 'out')
    assert selected['primary_selection']['gamma_variant'] == 'typed'
    assert all(w['configuration'] == 7 for w in selected['per_variant'])
    assert all(len(w['run_ids']) == 5 for w in selected['per_variant'])
    assert digest(selected) == digest(read_json(tmp_path / 'out' / 'selections.json'))
    c = config('typed', 7, 4)
    path = root / c['run_id'] / 'status.json'
    s = read_json(path)
    s['state'] = 'numerical_failure'
    atomic_json(path, s)
    changed = analysis.select(root)
    assert changed['primary_selection']['configuration'] != 7
    s['state'] = 'running'
    atomic_json(path, s)
    with pytest.raises(ValueError, match='incomplete'):
        analysis.select(root)


def test_aggregation_preserves_all_seeds_and_failure():
    rows = [dict(gamma_variant=v, split='test', units='cycles', horizon='all', metric='rmse', seed=s,
                 value=float(s + i)) for i, v in enumerate(VARIANTS) for s in SEEDS]
    summaries = analysis.aggregate(rows)
    assert all(r['n'] == 5 and r['sample_sd'] == pytest.approx(math.sqrt(2.5)) for r in summaries)
    pairs = analysis.paired_comparisons(rows)
    assert len(pairs) == 3 and all(r['sample_sd_difference'] == 0 for r in pairs)
    rows[0]['value'] = math.inf
    summary = next(r for r in analysis.aggregate(rows) if r['gamma_variant'] == 'typed')
    assert math.isinf(summary['mean']) and summary['finite_seeds'] == 4 and summary['n'] == 5


def test_original_preprocessing_used_and_training_never_loads_test(tmp_path, monkeypatch):
    from fulltd.data import prepare_fd001, load_dataset
    def engines(count):
        return np.asarray([np.r_[eid, t, np.arange(24) + .01 * t + eid]
                           for eid in range(1, count+1) for t in range(1, 36+eid)])
    prepare_fd001(tmp_path / 'data', engines(10), engines(3), np.array([12, 20, 5]))
    fitting = load_dataset('fd001', tmp_path / 'data', 'train')
    assert fitting.get_batch([0, 1])['x'].shape == (2, 24, 30)
    for path in (tmp_path / 'data' / 'processed' / 'fd001' / 'test').glob('*.npy'):
        path.unlink()
    calls = []
    monkeypatch.setattr(train, 'train_candidate', lambda c, fit, val, root, **kw: calls.append((fit.split, val.split)))
    train.run_config(config(), tmp_path / 'data', tmp_path / 'results')
    assert calls == [('train', 'val')]
