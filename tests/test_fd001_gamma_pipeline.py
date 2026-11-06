"""End-to-end synthetic endpoint/plot audit and actual shell/Slurm entrypoint checks."""
import contextlib
import csv
import io
import os
from pathlib import Path
import shlex
import subprocess
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from fd001_gamma import analysis, hpc, train
from fd001_gamma.common import atomic_json, read_json, scientific_hash
from fd001_gamma.protocol import digest, make_manifest, VARIANTS, SEEDS
from test_fd001_gamma import FixtureData, config, single_thread, tiny_objective


class EndpointFixture(FixtureData):
    def get_batch(self, ids):
        self.reads += 1
        return dict(x=self.x[ids].copy(), y=self.y[ids].copy(), mask=np.ones_like(self.y[ids], dtype=bool))


def test_full_training_selection_evaluation_cache_and_plot_pipeline(tmp_path, monkeypatch):
    root, out = tmp_path / 'runs', tmp_path / 'output'
    # Small 64-feature fixture for the full 15-model pipeline; separate tests
    # above run the actual 92,801-parameter TCN for 100 epochs per variant.
    import fd001_gamma.objectives
    factory = lambda c: tiny_objective(c['gamma_variant'])
    monkeypatch.setattr(train, 'make_objective', factory)
    monkeypatch.setattr(fd001_gamma.objectives, 'make_objective', factory)
    completed = {}
    with contextlib.redirect_stdout(io.StringIO()):
        for variant in VARIANTS:
            for seed in SEEDS:
                c = config(variant, seed=seed)
                completed[(variant, seed)] = train.train_candidate(c, EndpointFixture(8), EndpointFixture(4, 'val'), root, microbatch=4)
                assert completed[(variant, seed)]['state'] == 'complete'
    # The remaining settings are explicitly synthetic failed-candidate fixtures,
    # not results claimed for any real dataset. They test the complete grid gate.
    for c in make_manifest():
        if c['configuration'] == 0:
            continue
        reference = completed[(c['gamma_variant'], c['seed'])]
        failed = dict(reference, config=c, state='numerical_failure', next_epoch=1, epochs_completed=0,
                      identity=dict(reference['identity'], config_sha256=digest(c)))
        atomic_json(root / c['run_id'] / 'status.json', failed)
    selected = analysis.select(root, out)
    assert len(selected['per_variant']) == 3 and selected['primary_selection'] is not None
    opened = []
    def load(name, data_dir, split):
        assert name == 'fd001'
        if split == 'test':
            assert (out / 'selections.json').exists()
        opened.append(split)
        return EndpointFixture(8 if split == 'train' else 4, split)
    import fulltd.data
    monkeypatch.setattr(fulltd.data, 'load_dataset', load)
    rows, gamma = analysis.evaluate_selected(root, out, 'unused')
    assert opened == ['train', 'val', 'test']
    assert len({r['run_id'] for r in rows}) == 15
    assert all(r['variant'] == 'trained' and r['batch_size'] == 128 for r in rows)
    assert len(gamma) == 15 * 10 * 2
    for c in [c for c in make_manifest() if c['configuration'] == 0]:
        predictions = root / c['run_id'] / 'test_engine_predictions.csv'
        assert len(list(csv.DictReader(predictions.open()))) == 4
        endpoint_validation = next(r for r in rows if r['run_id'] == c['run_id'] and r['split'] == 'validation'
                                   and r['units'] == 'standardized' and r['horizon'] == 'all' and r['metric'] == 'mse')
        saved = read_json(root / c['run_id'] / 'validation.json')[-1]
        assert endpoint_validation['value'] == pytest.approx(saved['mse'], rel=1e-5, abs=1e-9)
    import fulltd.metrics
    monkeypatch.setattr(fulltd.metrics, 'evaluate', Mock(side_effect=AssertionError('Cache should avoid reevaluation')))
    cached, _ = analysis.evaluate_selected(root, out, 'unused')
    assert cached == rows
    analysis.figures(out, rows, gamma)
    pdfs = sorted((out / 'figures').glob('*.pdf'))
    assert len(pdfs) == 7
    for pdf in pdfs:
        assert pdf.read_bytes().startswith(b'%PDF')
        assert '<svg' in pdf.with_suffix('.svg').read_text()
        assert pdf.with_suffix('.csv').is_file()
    # The primary figure sidecar must be drawn from the numerical summary.
    rmse = list(csv.DictReader((out / 'figures' / 'primary_fd001_gamma_rmse.csv').open()))
    summary = analysis.aggregate(rows)
    for row in rmse:
        expected = next(x for x in summary if x['gamma_variant'] == row['gamma_variant'] and x['split'] == 'test'
                        and x['metric'] == 'rmse' and x['units'] == 'cycles' and x['horizon'] == 'all')
        assert float(row['mean']) == expected['mean']
    # Preserve this generated fixture for visual QA in the isolated test workspace.
    import shutil
    qa = Path(os.environ.get('GAMMA_TEST_QA_DIR', tmp_path / 'qa'))
    shutil.copytree(out, qa, dirs_exist_ok=True)


def test_supervisor_forwards_warning_and_requeues_after_workers_drain(tmp_path, monkeypatch):
    manifest = make_manifest()[:2]
    path = tmp_path / 'manifest.json'
    atomic_json(path, manifest)
    args = SimpleNamespace(manifest=str(path), digest=digest(manifest), expected_code_hash=scientific_hash(),
                           workers=2, microbatch=128, results_dir=str(tmp_path), data_dir=str(tmp_path / 'data'))
    for name, value in {'SLURM_ARRAY_JOB_ID':'100', 'SLURM_ARRAY_TASK_ID':'0', 'SLURM_JOB_NUM_NODES':'1',
                        'SLURM_NTASKS':'2', 'SLURM_CPUS_PER_TASK':'1'}.items():
        monkeypatch.setenv(name, value)
    signalled = []
    class Process:
        def __init__(self, command):
            assert command[0] == 'srun' and 'fd001_gamma.hpc' in command
            self.attempt = Path(command[command.index('--attempt') + 1])
            for rank in range(2):
                atomic_json(self.attempt / f'ready_{rank}.json', dict(rank=rank))
            self.code = None
        def send_signal(self, value):
            signalled.append(value)
        def wait(self, timeout):
            for rank in range(2):
                atomic_json(self.attempt / f'rank_{rank}.json', dict(rank=rank, state='checkpointed', before_epoch=1, next_epoch=3))
            self.code = 0
            return 0
        def poll(self):
            return self.code
    monkeypatch.setattr(hpc.subprocess, 'Popen', Process)
    requeues = []
    monkeypatch.setattr(hpc.subprocess, 'run', lambda command, **kw: requeues.append(command))
    receipt = hpc.supervise(args, lambda: True)
    assert signalled and receipt['state'] == 'requeue_accepted'
    assert requeues == [['scontrol', 'requeue', '100_0']]


def test_worker_runs_exact_mapped_candidate_and_records_failure(tmp_path, monkeypatch):
    manifest = make_manifest()[:2]
    path = tmp_path / 'manifest.json'
    atomic_json(path, manifest)
    attempt = tmp_path / 'attempt'
    attempt.mkdir()
    args = SimpleNamespace(manifest=str(path), digest=digest(manifest), expected_code_hash=scientific_hash(),
                           attempt=str(attempt), offset=0, task_count=2, microbatch=128,
                           results_dir=str(tmp_path / 'runs'), data_dir='unused')
    monkeypatch.setenv('SLURM_PROCID', '1')
    monkeypatch.setenv('SLURM_CPUS_PER_TASK', '1')
    seen = []
    monkeypatch.setattr(train, 'run_config', lambda c, *a, **kw: (seen.append(c) or dict(state='complete', next_epoch=101)))
    result = hpc.worker(args, lambda: False)
    assert seen == [manifest[1]] and result['state'] == 'complete'
    assert (attempt / 'ready_1.json').exists()
    monkeypatch.setattr(train, 'run_config', Mock(side_effect=ValueError('fixture failure')))
    with pytest.raises(ValueError, match='fixture failure'):
        hpc.worker(args, lambda: False)
    assert read_json(attempt / 'rank_1.json')['state'] == 'operational_failure'


def test_shell_training_entrypoint_runs_real_dry_run(tmp_path):
    root = Path(__file__).resolve().parents[1]
    bindir = tmp_path / 'bin'
    bindir.mkdir()
    for name, body in {'module':'#!/bin/sh\nexit 0\n', 'squeue':'#!/bin/sh\nexit 0\n'}.items():
        p = bindir / name
        p.write_text(body)
        p.chmod(0o755)
    import sys
    venv = tmp_path / 'venv with spaces' / 'bin'
    venv.mkdir(parents=True)
    (venv / 'python').symlink_to(sys.executable)
    results = tmp_path / 'new results'
    env = dict(os.environ, PATH=str(bindir) + os.pathsep + os.environ['PATH'], USER='fixture',
               FULLTD_ENV=str(venv.parent), FULLTD_FD001_GAMMA_RESULTS_DIR=str(results),
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    run = subprocess.run(['bash', str(root / 'scripts/submit_fulltd_fd001_gamma.sh'), '--dry-run'],
                         env=env, text=True, capture_output=True, check=True)
    assert '180 runs' in run.stdout and '02:00:00' in run.stdout
    assert not results.exists()


def test_shell_analysis_launcher_quotes_and_fails_closed(tmp_path):
    root = Path(__file__).resolve().parents[1]
    bindir = tmp_path / 'bin'
    bindir.mkdir()
    # This shell test verifies scheduling/quoting. The actual analysis itself is
    # exercised above with real synthetic trained checkpoints and metrics.
    log = tmp_path / 'sbatch.txt'
    for name, body in {'module':'#!/bin/sh\nexit 0\n', 'squeue':'#!/bin/sh\nexit "${MOCK_QUEUE_RC:-0}"\n',
                       'sbatch':'#!/usr/bin/env python3\nimport json,os,sys\nopen(os.environ["MOCK_LOG"],"w").write(json.dumps(sys.argv[1:]))\nprint("12345")\n'}.items():
        p = bindir / name
        p.write_text(body)
        p.chmod(0o755)
    venv = tmp_path / 'venv with spaces' / 'bin'
    venv.mkdir(parents=True)
    py = venv / 'python'
    py.write_text('#!/bin/sh\nexit 0\n')
    py.chmod(0o755)
    env = dict(os.environ, PATH=str(bindir) + os.pathsep + os.environ['PATH'], USER='fixture',
               FULLTD_ENV=str(venv.parent), FULLTD_FD001_GAMMA_RESULTS_DIR=str(tmp_path / 'results space'),
               FULLTD_FD001_GAMMA_OUTPUT_DIR=str(tmp_path / 'outputs space'), MOCK_LOG=str(log))
    subprocess.run(['bash', str(root / 'scripts/submit_fulltd_fd001_gamma_analysis.sh')],
                   env=env, text=True, capture_output=True, check=True)
    import json
    command = json.loads(log.read_text())
    wrap = next(x.split('=',1)[1] for x in command if x.startswith('--wrap='))
    pieces = shlex.split(wrap)
    assert pieces[:3] == ['exec', 'bash', '-c']
    assert 'fd001_gamma.analysis run' in pieces[3]
    assert shlex.split(pieces[3].split('; exec ', 1)[1]) == [str(py), '-m', 'fd001_gamma.analysis', 'run']
    log.unlink()
    run = subprocess.run(['bash', str(root / 'scripts/submit_fulltd_fd001_gamma_analysis.sh')],
                         env=dict(env, MOCK_QUEUE_RC='1'), capture_output=True)
    assert run.returncode != 0 and not log.exists()
