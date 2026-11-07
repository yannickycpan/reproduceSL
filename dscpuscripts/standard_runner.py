"""Direct CPU execution of unchanged legacy campaigns; no Slurm or time limit.

fd001_baselines = original MSE/NLL grid (60 runs, B256).
electricity = original 300 runs; metr_la = original 210 runs.
All dataset methods share run directories directly in ~/scratch/fulltd/<dataset>_results/;
analysis outputs remain isolated by campaign.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

CAMPAIGNS = ('fd001_baselines', 'electricity', 'metr_la', 'fd001_v2')
DEFAULTS = {'fd001_baselines': (16, 256), 'electricity': (8, 128), 'metr_la': (8, 32),
            'fd001_v2': (32, 256)}


def configurations(campaign):
    from fulltd.protocol import make_manifest
    if campaign not in CAMPAIGNS:
        raise ValueError(f'Unknown campaign: {campaign}')
    if campaign == 'fd001_v2':
        from dscpuscripts.fd001_v2_protocol import make_manifest as v2_manifest
        return v2_manifest()
    if campaign == 'fd001_baselines':
        return [c for c in make_manifest() if c['dataset'] == 'fd001' and c['method'] in ('mse', 'gaussian_nll')]
    return [c for c in make_manifest() if c['dataset'] == campaign]


def paths(campaign):
    from dscpuscripts.result_paths import paths as dataset_paths
    if campaign not in CAMPAIGNS:
        raise ValueError(f"Unknown campaign: {campaign}")
    dataset = 'fd001' if campaign in ('fd001_baselines', 'fd001_v2') else campaign
    return dataset_paths(dataset, 'baselines' if campaign == 'fd001_baselines' else None)


def classify(config, results, expected_code):
    from fd001_gamma.common import read_json
    from fulltd.protocol import digest
    directory = Path(results) / config['run_id']
    status_path = directory / 'status.json'
    if not status_path.exists():
        return 'resume' if (directory / 'recovery.pt').exists() else 'missing'
    status = read_json(status_path)
    identity = status.get('identity', {})
    if (status.get('config') != config or identity.get('config_sha256') != digest(config)
            or identity.get('code_sha256') != expected_code):
        raise ValueError(f'Incompatible prior result preserved: {directory}')
    if status.get('state') == 'numerical_failure':
        return 'numerical_failure'
    if (status.get('state') == 'complete' and status.get('epochs_completed') == 100
            and all((directory / f).is_file() for f in ('final.pt', 'validation.json', 'rounds.json'))):
        return 'complete'
    return 'resume'


@contextmanager
def stop_flag():
    requested = [False]
    def handler(signum, frame):
        requested[0] = True
    signals = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
    old = {}
    for s in signals:
        previous = signal.getsignal(s)
        # Respect nohup's inherited SIGHUP ignore disposition.
        if s == signal.SIGHUP and previous == signal.SIG_IGN:
            continue
        old[s] = signal.signal(s, handler)
    try:
        yield lambda: requested[0]
    finally:
        for s, h in old.items():
            signal.signal(s, h)


def execute_worker(args):
    # Install handlers before importing torch, so an interrupt during imports
    # is remembered and passed to the existing epoch-checkpoint trainer.
    with stop_flag() as stopped:
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        from fulltd.data import load_dataset
        if args.campaign == 'fd001_v2':
            from dscpuscripts.fd001_v2_train import train_candidate, code_hash
        else:
            from fulltd.train import train_candidate, code_hash
        from fd001_gamma.common import file_lock, read_json
        if code_hash() != args.expected_code:
            raise ValueError('Training implementation changed after launch; no fitting performed')
        configs = {c['run_id']: c for c in configurations(args.campaign)}
        config = configs[args.run_id]
        data, results, _ = paths(args.campaign)
        folder = results / args.run_id
        with file_lock(folder / '.worker.lock'):
            train_candidate(config, load_dataset(config['dataset'], data, 'train'),
                            load_dataset(config['dataset'], data, 'val'), results,
                            device='cpu', microbatch=args.microbatch, stop_requested=stopped)
        result = read_json(folder / 'status.json')
        if result['state'] in ('complete', 'numerical_failure'):
            return 0
        if stopped():
            return 75
        raise RuntimeError(f'Unexpected nonterminal worker state: {result["state"]}')


def dispatch(args):
    from fd001_gamma.common import atomic_json, file_lock
    if args.campaign == 'fd001_v2':
        from dscpuscripts.fd001_v2_train import code_hash
    else:
        from fulltd.train import code_hash
    from dscpuscripts.direct_runner import choose_workers, cpu_idle_fraction
    data, results, _ = paths(args.campaign)
    configs = configurations(args.campaign)
    code = code_hash()
    states = {c['run_id']: classify(c, results, code) for c in configs}
    pending = [c for c in configs if states[c['run_id']] not in ('complete', 'numerical_failure')]
    if args.max_runs:
        pending = pending[:args.max_runs]
    workers, physical, idle = choose_workers(args.workers)
    print(f'{args.campaign}: {len(configs)} canonical runs; states={dict(Counter(states.values()))}', flush=True)
    print(f'Pending now: {len(pending)}; workers={workers}; physical microbatch={args.microbatch}', flush=True)
    print(f'Data: {data}\nResults: {results}\nNo Slurm, new optimizer settings, or time limit.', flush=True)
    if args.mode in ('status', 'dry-run') or not pending:
        return 0
    from dscpuscripts.result_paths import (dataset_root, guard_unmigrated_runs)
    dataset_name = 'fd001' if args.campaign in ('fd001_baselines', 'fd001_v2') else args.campaign
    # Both the earlier nested layout and the old per-campaign direct layout
    # may contain checkpoints. Never silently start them over in the flat root.
    guard_unmigrated_runs(results, configs, (
        dataset_root(dataset_name) / 'results',
        Path.home() / 'scratch/fulltd/direct' / args.campaign / 'results',
    ))
    from fulltd.data import verify_dataset, load_dataset
    from fulltd.sampling import training_universe
    from fulltd.protocol import LAGS
    from fulltd.train import data_identity
    from fd001_gamma.common import read_json
    dataset = configs[0]['dataset']
    verify_dataset(dataset, data)
    train, _, _ = training_universe(load_dataset(dataset, data, 'train'), LAGS[dataset])
    val = load_dataset(dataset, data, 'val')
    expected_data, expected_val = data_identity(train), data_identity(val)
    for config in configs:
        previous = results / config['run_id'] / 'status.json'
        if previous.exists():
            identity = read_json(previous).get('identity', {})
            if identity.get('data_sha256') != expected_data or identity.get('validation_sha256') != expected_val:
                raise ValueError(f'Prepared data changed for {config["run_id"]}; preserve old results separately')
    results.mkdir(parents=True, exist_ok=True)
    with file_lock(results / '.direct_campaign.lock'), stop_flag() as stopped:
        # The lock is separate from each scientific run lock and protects the
        # entire dispatcher, including the startup race between two terminals.
        pending = [c for c in pending if classify(c, results, code) not in ('complete', 'numerical_failure')]
        logs = results / 'direct_logs'
        logs.mkdir(exist_ok=True)
        receipt_path = results / 'direct_launches' / f'{uuid.uuid4().hex}.json'
        record = dict(campaign=args.campaign, started=datetime.now(timezone.utc).isoformat(),
                      pid=os.getpid(), code_sha256=code, workers=workers, microbatch=args.microbatch,
                      state='running', completed=[], failed=[])
        atomic_json(receipt_path, record)
        active, next_index, stopping = {}, 0, False
        next_check, budget = 0., 0
        try:
            while active or (next_index < len(pending) and not stopping):
                stopping = stopping or stopped()
                # Reap first: no new work is launched after observing a worker
                # error. Scientific numerical failures remain terminal records.
                for run_id, process in list(active.items()):
                    exitcode = process.poll()
                    if exitcode is None:
                        continue
                    del active[run_id]
                    config = next(c for c in configs if c['run_id'] == run_id)
                    if exitcode == 0 and classify(config, results, code) in ('complete', 'numerical_failure'):
                        record['completed'].append(run_id)
                        print(f'Finished {run_id}', flush=True)
                    elif not (exitcode == 75 and stopping):
                        record['failed'].append(dict(run_id=run_id, exit_code=exitcode))
                        stopping = True
                    atomic_json(receipt_path, record)
                if stopping:
                    for process in active.values():
                        if process.poll() is None:
                            process.send_signal(signal.SIGTERM)
                    if not active:
                        break
                    time.sleep(1)
                    continue
                if not args.adaptive:
                    budget = workers - len(active)
                elif time.monotonic() >= next_check:
                    idle = cpu_idle_fraction()
                    budget = min(workers - len(active), int(.8 * physical * idle))
                    next_check = time.monotonic() + args.check_seconds
                    print(f'Adaptive check: active={len(active)}, idle={idle:.1%}, new slots={budget}', flush=True)
                while budget > 0 and len(active) < workers and next_index < len(pending) and not stopped():
                    c = pending[next_index]
                    next_index += 1
                    cmd = [sys.executable, '-m', 'dscpuscripts.standard_runner', args.campaign,
                           '--worker', '--run-id', c['run_id'], '--expected-code', code,
                           '--microbatch', str(args.microbatch)]
                    with (logs / (c['run_id'] + '.log')).open('a', encoding='utf-8') as log:
                        process = subprocess.Popen(cmd, cwd=Path(__file__).resolve().parents[1],
                                                   stdout=log, stderr=subprocess.STDOUT,
                                                   start_new_session=True,
                                                   env=dict(os.environ, PYTHONUNBUFFERED='1'))
                    active[c['run_id']] = process
                    budget -= 1
                    print(f'Started {c["run_id"]}: PID {process.pid}', flush=True)
                if active or next_index < len(pending):
                    time.sleep(1)
        except BaseException as error:
            record['failed'].append(dict(dispatcher_error=f'{type(error).__name__}: {error}'))
            raise
        finally:
            for process in active.values():
                if process.poll() is None:
                    process.send_signal(signal.SIGTERM)
            for process in active.values():
                process.wait()
            record['state'] = 'failed' if record['failed'] else 'interrupted' if stopping or stopped() else 'complete'
            atomic_json(receipt_path, record)
    return 0 if record['state'] == 'complete' else 1


def main():
    from fd001_gamma.common import cpu_environment
    cpu_environment()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('campaign', choices=CAMPAIGNS)
    action = parser.add_mutually_exclusive_group(required=True)
    for mode in ('run', 'dry-run', 'status', 'worker'):
        action.add_argument('--' + mode, dest='mode', action='store_const', const=mode)
    parser.add_argument('--workers', default=None)
    parser.add_argument('--microbatch', type=int, default=None)
    parser.add_argument('--adaptive', action='store_true')
    parser.add_argument('--check-seconds', type=float, default=600.)
    parser.add_argument('--max-runs', type=int, default=0)
    parser.add_argument('--run-id')
    parser.add_argument('--expected-code')
    args = parser.parse_args()
    defaults = DEFAULTS[args.campaign]
    args.workers = str(defaults[0]) if args.workers is None else args.workers
    args.microbatch = defaults[1] if args.microbatch is None else args.microbatch
    if args.microbatch < 1 or args.max_runs < 0 or not math.isfinite(args.check_seconds) or args.check_seconds < 5:
        parser.error('Invalid microbatch, max-runs or check interval')
    if args.mode == 'worker':
        if not args.run_id or not args.expected_code:
            parser.error('Internal worker requires run-id and expected-code')
        return execute_worker(args)
    return dispatch(args)


if __name__ == '__main__':
    raise SystemExit(main())
