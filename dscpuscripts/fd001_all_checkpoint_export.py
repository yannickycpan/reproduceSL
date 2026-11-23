"""Export every saved FD001 Stage-1-MSE-alpha0.01 Combo checkpoint and prediction.

Read-only postprocessing. Does not train, select hyperparameters, or modify runs.
Outputs CSV + per-run compressed numpy arrays under fd001_outputs/combo_stage1opt_all_checkpoints.
Run-level atomic cache makes it safe to interrupt/restart. The TEST export must
NOT be used to choose the paper's hyperparameters, lags, or stopping times.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import math
import multiprocessing as mp
import os
from pathlib import Path
import tempfile

import numpy as np

from dscpuscripts.common import atomic_json, data_identity, read_json, write_csv
from dscpuscripts.fd001_protocol import EPOCHS, PROTOCOL_ID, make_manifest
from dscpuscripts.result_paths import paths
from fulltd.protocol import digest

REVISION = 'fd001-combo-stage1opt-all-checkpoints-test-export-v1'
N_RUNS = 750
_WORKER_TEST = None
_WORKER_TEST_DIR = None


def checkpoint_steps(config, n_sources, curve):
    """All scheduled snapshots, including update zero and final update."""
    total = EPOCHS * math.ceil(n_sources / int(config['batch_size']))
    interval = int(config['learning_curve_updates'])
    expected = list(range(0, total + 1, interval))
    if expected[-1] != total:
        expected.append(total)
    observed = [int(row['optimizer_steps']) for row in curve]
    if observed != expected:
        raise ValueError(f'Incomplete/unexpected validation snapshots: {config["run_id"]}: '
                         f'{observed[:3]}...{observed[-3:]}')
    return expected


def _validate_run(root, config, scientific_sha, validation_sha):
    run = root / config['run_id']
    status = read_json(run / 'status.json')
    identity = status.get('identity', {})
    if (status.get('state') != 'complete'
            or status.get('config') != config
            or identity.get('protocol_id') != PROTOCOL_ID
            or identity.get('config_sha256') != digest(config)
            or identity.get('code_sha256') != scientific_sha
            or identity.get('validation_sha256') != validation_sha):
        raise ValueError(f'Incompatible/incomplete run: {run}')
    curve = read_json(run / 'update_curve.json')
    steps = checkpoint_steps(config, int(status['n_sources']), curve)
    if status['optimizer_steps'] != steps[-1]:
        raise ValueError(f'Final optimizer step differs from saved history: {run}')
    if config['method'] == 'td_mse_combo' and status.get('phase') != 'complete':
        raise ValueError(f'Incomplete staged training: {run}')
    return status, curve, steps


def _worker_setup():
    from dscpuscripts.common import cpu_environment
    cpu_environment()
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def _cached(record_path, prediction_path, signature, steps, n_test):
    if not record_path.is_file():
        return None
    metadata = read_json(record_path)
    if metadata.get('signature') != signature:
        raise ValueError(f'Previously exported run has different identity: {record_path}')
    if not prediction_path.is_file():
        return None
    try:
        with np.load(prediction_path, allow_pickle=False) as packed:
            if (not np.array_equal(packed['steps'], steps)
                    or packed['predictions'].shape != (len(steps), n_test)
                    or packed['target_rul_cycles'].shape != (n_test,)
                    or packed['engine_ids'].shape != (n_test,)):
                return None
    except (OSError, ValueError, KeyError):
        return None
    if len(metadata.get('rows', [])) != len(steps):
        return None
    return metadata


def export_one(job):
    """One run per worker; reuse its predictor across all saved checkpoints."""
    config, raw_root, raw_out, data_dir, scientific_sha, validation_sha, test_sha = job
    from dscpuscripts.fd001_train import build_objective, json_safe
    from fulltd.data import load_dataset
    from fulltd.metrics import evaluate
    import torch

    root = Path(raw_root)
    out = Path(raw_out)
    status, curve, steps = _validate_run(root, config, scientific_sha, validation_sha)
    global _WORKER_TEST, _WORKER_TEST_DIR
    if _WORKER_TEST is None or _WORKER_TEST_DIR != data_dir:
        _WORKER_TEST = load_dataset('fd001', Path(data_dir), 'test')
        _WORKER_TEST_DIR = data_dir
    test = _WORKER_TEST
    if data_identity(test) != test_sha:
        raise ValueError('Test data identity differs across workers')
    n_test = len(test)
    run_id = config['run_id']
    signature = dict(revision=REVISION, run_id=run_id,
                     status_identity=status['identity'], test_data_sha256=test_sha,
                     curve_sha256=digest(curve), steps=steps)
    record = out / 'records' / (run_id + '.json')
    prediction_file = out / 'predictions' / (run_id + '.npz')
    if _cached(record, prediction_file, signature, np.asarray(steps), n_test):
        return run_id, 'cached', len(steps)

    objective = build_objective(config).cpu()
    objective.eval()
    per_step = []
    predictions = []
    engine_ids = None
    target_rul = None
    for step, val in zip(steps, curve):
        path = root / run_id / 'checkpoints' / f'step_{step:07d}.pt'
        if not path.is_file():
            raise FileNotFoundError(f'Missing checkpoint: {path}')
        # Only locally produced training checkpoints; never deserialize untrusted files.
        saved = torch.load(path, map_location='cpu', weights_only=False)
        progress = saved['progress']
        actual_step = (progress['phase_steps']['predictor']
                       if config['method'] == 'td_mse_combo'
                       else progress['optimizer_steps'])
        if (saved.get('config') != config
                or saved.get('identity') != status['identity']
                or int(actual_step) != step):
            raise ValueError(f'Checkpoint identity/step mismatch: {path}')
        snapshot_curve = json_safe(progress['curve'])
        if (snapshot_curve != curve[:len(snapshot_curve)]
                or int(snapshot_curve[-1]['optimizer_steps']) != step):
            raise ValueError(f'Checkpoint validation history mismatch: {path}')
        objective.load_state_dict(saved['objective'], strict=True)
        # FD001 test has exactly one endpoint per engine; evaluate() uses
        # equal-engine RMSE and the official asymmetric untruncated RUL score.
        with torch.no_grad():
            observations, endpoints = evaluate(objective.model, test, batch_size=128,
                                               device='cpu', allow_nonfinite=True)
        lookup = {(r['units'], str(r['horizon']), r['metric']): float(r['value'])
                  for r in observations}
        ids = np.asarray([r['engine_id'] for r in endpoints], dtype=np.int64)
        y = np.asarray([r['target'] for r in endpoints], dtype=np.float64)
        prediction = np.asarray([r['prediction'] for r in endpoints], dtype=np.float64)
        if len(ids) != n_test or len(np.unique(ids)) != n_test:
            raise ValueError(f'Missing/duplicate test engine endpoint: {run_id}')
        order = np.argsort(ids)
        ids, y, prediction = ids[order], y[order], prediction[order]
        if engine_ids is None:
            engine_ids, target_rul = ids, y
        elif not np.array_equal(ids, engine_ids) or not np.array_equal(y, target_rul):
            raise ValueError(f'Test endpoint order or labels changed: {run_id}')
        measured_rmse = lookup[('cycles', 'all', 'rmse')]
        measured_score = lookup[('score', 'all', 'rul_score_sum')]
        expected_rmse = float(np.sqrt(np.mean((prediction-y)**2, dtype=np.float64)))
        if not np.isclose(measured_rmse, expected_rmse, rtol=1e-7, atol=1e-6,
                          equal_nan=True):
            raise ValueError(f'Test RMSE disagrees with endpoint errors: {run_id}/{step}')
        predictions.append(prediction)
        per_step.append(dict(run_id=run_id, method=config['method'],
            batch_size=config['batch_size'], lag_set='-'.join(map(str, config['lags']))
            if config['method'].startswith('td_') else 'none',
            seed=config['seed'], alpha=config['alpha'], beta=config['beta'],
            mse_alpha=config.get('mse_alpha'), optimizer_steps=step,
            epoch=val['epoch'],
            checkpoint_elapsed_seconds=float(saved['runtime_seconds']),
            validation_mse_standardized=float(val['mse_standardized']),
            validation_rmse_cycles=float(val['rmse_cycles']),
            validation_rul_score_sum=float(val['rul_score_sum']),
            test_rmse_cycles=measured_rmse,
            test_rul_score_sum=measured_score,
            finite_test_rmse=math.isfinite(measured_rmse),
            finite_test_rul_score=math.isfinite(measured_score)))

    # Atomic array and manifest output; no training or checkpoint file is altered.
    prediction_file.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('wb', dir=prediction_file.parent,
                                     suffix='.npz', delete=False) as stream:
        tmp = Path(stream.name)
        try:
            np.savez_compressed(stream, steps=np.asarray(steps, dtype=np.int32),
                engine_ids=engine_ids, target_rul_cycles=target_rul,
                predictions=np.stack(predictions))
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
    os.replace(tmp, prediction_file)
    atomic_json(record, dict(signature=signature, rows=per_step))
    return run_id, 'exported', len(steps)


def finalize(out, root, manifest, scientific_sha, val_sha, test_sha, n_test):
    all_rows = []
    gamma_rows = []
    stage_rows = []
    counters = Counter()
    if len({c['run_id'] for c in manifest}) != N_RUNS:
        raise ValueError('Duplicate run IDs or missing configurations')
    for c in manifest:
        run = c['run_id']
        record = out / 'records' / (run + '.json')
        if not record.is_file() or not (out / 'predictions' / (run + '.npz')).is_file():
            raise RuntimeError(f'Not all {N_RUNS} run outputs are complete: {run}')
        exported = read_json(record)
        signature = exported['signature']
        if (signature['revision'] != REVISION or signature['run_id'] != run
                or signature['test_data_sha256'] != test_sha
                or signature['status_identity']['code_sha256'] != scientific_sha
                or signature['status_identity']['validation_sha256'] != val_sha):
            raise ValueError(f'Export manifest identity mismatch: {run}')
        all_rows.extend(exported['rows'])
        counters[c['method']] += 1
        status = read_json(root / run / 'status.json')
        timings = status.get('timing', {})
        stage_epochs = status.get('phase_epochs', {})
        stage_steps = status.get('phase_steps', {})
        stage_rows.append(dict(run_id=run, method=c['method'],
            batch_size=c['batch_size'], lag_set='-'.join(map(str, c['lags'])),
            seed=c['seed'], alpha=c['alpha'], beta=c['beta'],
            mse_alpha=c['mse_alpha'], total_wall_seconds=status['runtime_seconds'],
            stage1_mse_epochs=stage_epochs.get('mse'),
            stage2_gamma_epochs=stage_epochs.get('gamma'),
            stage3_td_epochs=stage_epochs.get('predictor'),
            stage1_mse_updates=stage_steps.get('mse'),
            stage2_gamma_updates=stage_steps.get('gamma'),
            stage3_td_updates=stage_steps.get('predictor'),
            stage1_mse_wall_seconds=timings.get('mse_wall_seconds'),
            stage2_gamma_wall_seconds=timings.get('gamma_wall_seconds'),
            stage3_td_wall_seconds=timings.get('predictor_wall_seconds'),
            stage2_stop_reason=status.get('stop_reasons', {}).get('gamma')))
        # Source-side training gamma observations, if present, are exported for
        # optional mechanism analysis; no new fit or model selection is done.
        curve = read_json(root / run / 'update_curve.json')
        for v in curve:
            for typ in v.get('gamma_probe', {}).get('types', []):
                gamma_rows.append(dict(run_id=run, method=c['method'],
                    batch_size=c['batch_size'], lag_set='-'.join(map(str,c['lags'])),
                    seed=c['seed'], optimizer_steps=v['optimizer_steps'], **typ))
    write_csv(out / 'checkpoint_metrics.csv', all_rows)
    write_csv(out / 'checkpoint_gamma.csv', gamma_rows)
    write_csv(out / 'stage_training.csv', stage_rows)
    count = len(all_rows)
    atomic_json(out / 'export_manifest.json', dict(
        revision=REVISION, n_runs=len(manifest), n_checkpoints=count,
        n_engine_predictions=count * n_test, methods=dict(counters),
        protocol_id=PROTOCOL_ID, scientific_code_sha256=scientific_sha,
        validation_data_sha256=val_sha, test_data_sha256=test_sha,
        storage='predictions/<run_id>.npz: steps, engine_ids, '
                'target_rul_cycles, predictions (steps x engines)',
        method='Read-only test inference at every checkpoint. '
               'Hyperparameters/stopping MUST be chosen using validation data only. '
               'Full model checkpoints are NOT in the export.'))
    (out / 'README.txt').write_text(
        'FD001 all-checkpoint export. Every saved step includes step 0 and final.\n'
        'checkpoint_metrics.csv: validation MSE, RMSE, RUL score; test RMSE and RUL score; '
        'elapsed checkpoint wall time, alpha/beta/lags/batch/seed.\n'
        'checkpoint_gamma.csv: saved training-probe Gamma diagnostics.\n'
        'stage_training.csv: all stages epochs, optimizer updates, wall-clock times, and Gamma early-stopping reason.\n'
        'predictions/<run_id>.npz: numpy arrays steps, engine_ids, target_rul_cycles, '
        'predictions (checkpoint x engine). Raw signed error = predictions - target.\n'
        'RUL contribution = exp(error/10)-1 if error>=0, else exp(-error/13)-1.\n'
        'Use numpy.load(..., allow_pickle=False). These contain every run and checkpoint.\n'
        'IMPORTANT: DO NOT choose model hyperparameters/lag/stopping using held-out TEST results.\n',
        encoding='utf-8')
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', nargs='?', default='run', choices=('run', 'status', 'smoke'))
    parser.add_argument('--workers', type=int, default=24)
    args = parser.parse_args()
    if not 1 <= args.workers <= 128:
        parser.error('--workers must be from 1 to 128')
    from dscpuscripts.common import cpu_environment, file_lock
    from dscpuscripts.fd001_train import code_hash
    from fulltd.data import load_dataset, verify_dataset
    cpu_environment()
    data_dir, root, outputs = paths('fd001')
    out = outputs / 'combo_stage1opt_all_checkpoints'
    manifest = make_manifest()
    if len(manifest) != N_RUNS:
        raise ValueError(f'Unexpected manifest length {len(manifest)}')
    verify_dataset('fd001', data_dir)
    scientific_sha = code_hash()
    val_sha = data_identity(load_dataset('fd001', data_dir, 'val'))
    test_data = load_dataset('fd001', data_dir, 'test')
    test_sha = data_identity(test_data)
    out.mkdir(parents=True, exist_ok=True)
    if args.mode == 'status':
        statuses = Counter('cached' if (out/'records'/(c['run_id']+'.json')).is_file()
                           and (out/'predictions'/(c['run_id']+'.npz')).is_file()
                           else 'pending' for c in manifest)
        print(f'{N_RUNS} runs: {dict(statuses)}; output={out}', flush=True)
        return
    if args.mode == 'smoke':
        _worker_setup()
        for method, batch in (('td_mse_combo', 128), ('td_mse_combo', 256)):
            config = next(c for c in manifest if c['method'] == method
                          and c['batch_size'] == batch and c['seed'] == 0)
            result = export_one((config, str(root), str(out), str(data_dir),
                                 scientific_sha, val_sha, test_sha))
            print('SMOKE:', result, flush=True)
        return
    with file_lock(out / '.all_checkpoints.lock'):
        for config in manifest:
            _validate_run(root, config, scientific_sha, val_sha)
        print(f'Exporting {N_RUNS} completed runs; {args.workers} workers; '
              f'no training; output={out}',flush=True)
        args_by_run = [(c, str(root), str(out), str(data_dir),
                        scientific_sha, val_sha, test_sha) for c in manifest]
        states = Counter()
        steps = 0
        with ProcessPoolExecutor(max_workers=args.workers,
                                 mp_context=mp.get_context('spawn'),
                                 initializer=_worker_setup) as pool:
            futures = [pool.submit(export_one, job) for job in args_by_run]
            for idx, future in enumerate(as_completed(futures), start=1):
                run_id, state, n = future.result()
                states[state] += 1
                steps += n
                if idx % 25 == 0 or idx == N_RUNS:
                    print(f'{idx}/{N_RUNS} runs: {dict(states)}, '
                          f'{steps} checkpoints',flush=True)
        # Avoid mixing results from different dataset roots even when users set
        # FULLTD_FD001_ROOT to a nondefault directory.
        count = finalize(out, root, manifest, scientific_sha, val_sha, test_sha, len(test_data))
        print(f'COMPLETE: {N_RUNS} runs, {count} checkpoints; {out}',flush=True)


if __name__ == '__main__':
    main()
