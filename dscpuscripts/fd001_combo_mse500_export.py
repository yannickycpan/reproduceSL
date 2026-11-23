"""Export every saved FD001 Combo MSE-500 checkpoint and test prediction.

Read-only postprocessing. Does not train, select hyperparameters, or modify runs.
Stores the 750 new runs alongside the original 1600 runs in
fd001_outputs/all_checkpoints/{records,predictions}. Retains separate per-
protocol CSV/manifest files and creates jointly merged reporting CSVs.
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
from dscpuscripts.fd001_combo_mse500_protocol import PROTOCOL_ID, make_manifest
from dscpuscripts.fd001_protocol import EPOCHS
from dscpuscripts.result_paths import paths
from fulltd.protocol import digest

REVISION = 'fd001-combo-mse500-test-export-v1'
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
    from dscpuscripts.fd001_combo_mse500_train import build_objective
    from dscpuscripts.fd001_train import json_safe
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
        timing = status.get('timing', {})
        phases = status.get('phase_steps', {})
        stage_rows.append(dict(run_id=run, seed=c['seed'],
            batch_size=c['batch_size'], lag_set='-'.join(map(str, c['lags'])),
            mse_alpha=c['mse_alpha'], alpha=c['alpha'], beta=c['beta'],
            stage1_mse_updates=phases.get('mse'),
            stage1_mse_partial_epoch=status.get('mse_partial_final_epoch'),
            stage2_gamma_epochs=status.get('phase_epochs', {}).get('gamma'),
            stage3_td_epochs=status.get('phase_epochs', {}).get('predictor'),
            stage1_mse_wall_seconds=timing.get('mse_wall_seconds'),
            stage2_gamma_wall_seconds=timing.get('gamma_wall_seconds'),
            stage3_td_wall_seconds=timing.get('predictor_wall_seconds'),
            total_wall_seconds=status['runtime_seconds']))
        # Source-side training gamma observations, if present, are exported for
        # optional mechanism analysis; no new fit or model selection is done.
        curve = read_json(root / run / 'update_curve.json')
        for v in curve:
            for typ in v.get('gamma_probe', {}).get('types', []):
                gamma_rows.append(dict(run_id=run, method=c['method'],
                    batch_size=c['batch_size'], lag_set='-'.join(map(str,c['lags'])),
                    seed=c['seed'], optimizer_steps=v['optimizer_steps'], **typ))
    # Never overwrite the original 1600-run exports in the shared directory.
    write_csv(out / 'combo_mse500_checkpoint_metrics.csv', all_rows)
    write_csv(out / 'combo_mse500_checkpoint_gamma.csv', gamma_rows)
    write_csv(out / 'combo_mse500_stage_training.csv', stage_rows)
    count = len(all_rows)
    atomic_json(out / 'combo_mse500_export_manifest.json', dict(
        revision=REVISION, n_runs=len(manifest), n_checkpoints=count,
        n_engine_predictions=count * n_test, methods=dict(counters),
        protocol_id=PROTOCOL_ID, scientific_code_sha256=scientific_sha,
        validation_data_sha256=val_sha, test_data_sha256=test_sha,
        storage='predictions/<run_id>.npz: steps, engine_ids, '
                'target_rul_cycles, predictions (steps x engines)',
        method='Read-only test inference at every checkpoint. '
               'Hyperparameters/stopping MUST be chosen using validation data only. '
               'Full model checkpoints are NOT in the export.'))
    (out / 'README_combo_mse500.txt').write_text(
        'FD001 all-checkpoint export. Every saved step includes step 0 and final.\n'
        'combo_mse500_checkpoint_metrics.csv: validation MSE, RMSE, RUL score; test RMSE and RUL score; '
        'elapsed checkpoint wall time, alpha/beta/lags/batch/seed.\n'
        'combo_mse500_checkpoint_gamma.csv: saved training-probe Gamma diagnostics.\n'
        'combo_mse500_stage_training.csv: all stage optimizer updates and wall-clock times.\n'
        'predictions/<run_id>.npz: numpy arrays steps, engine_ids, target_rul_cycles, '
        'predictions (checkpoint x engine). Raw signed error = predictions - target.\n'
        'RUL contribution = exp(error/10)-1 if error>=0, else exp(-error/13)-1.\n'
        'Use numpy.load(..., allow_pickle=False). These contain every run and checkpoint.\n'
        'IMPORTANT: DO NOT choose model hyperparameters/lag/stopping using held-out TEST results.\n',
        encoding='utf-8')
    return count




def reuse_previous_separate_export(outputs, merged_dir):
    """Copy/hard-link earlier split export into the common directory if present.

    Older code wrote the new 750-run results into a separate output directory.
    Reuse those files without touching either version or any training checkpoint.
    """
    import shutil

    previous = Path(outputs) / "combo_mse500_all_checkpoints"
    merged_dir = Path(merged_dir)
    if not previous.is_dir() or previous.resolve() == merged_dir.resolve():
        return 0
    copied = 0
    for kind, pattern in (("records", "*.json"), ("predictions", "*.npz")):
        for source in (previous / kind).glob(pattern):
            target = merged_dir / kind / source.name
            if target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(source, target)
            except OSError:
                shutil.copy2(source, target)
            copied += 1
    summaries = {
        "checkpoint_metrics.csv": "combo_mse500_checkpoint_metrics.csv",
        "checkpoint_gamma.csv": "combo_mse500_checkpoint_gamma.csv",
        "stage_training.csv": "combo_mse500_stage_training.csv",
        "export_manifest.json": "combo_mse500_export_manifest.json",
        "README.txt": "README_combo_mse500.txt",
    }
    for old_name, new_name in summaries.items():
        src, dest = previous / old_name, merged_dir / new_name
        if src.is_file() and not dest.exists():
            shutil.copy2(src, dest)
            copied += 1
    return copied

def merge_original_and_combo_exports(out):
    """Combine two immutable protocol exports without rewriting either original.

    Shared records/ and predictions/ contain all distinct run IDs. Only the
    combined CSVs and combined manifest are rewritten; original CSVs, per-run
    prediction arrays, and scientific checkpoints are never modified.
    """
    import csv
    from dscpuscripts.common import write_csv

    out = Path(out)
    manifests = [
        ("original_lagscanv3", out / "export_manifest.json",
         out / "checkpoint_metrics.csv", out / "checkpoint_gamma.csv", 1600),
        ("combo_stage1_mse500", out / "combo_mse500_export_manifest.json",
         out / "combo_mse500_checkpoint_metrics.csv",
         out / "combo_mse500_checkpoint_gamma.csv", 750),
    ]
    identities = []
    combined_metrics = []
    combined_gamma = []
    seen_ids = set()
    for experiment, manifest_path, metrics_path, gamma_path, required in manifests:
        if not manifest_path.is_file() or not metrics_path.is_file():
            raise FileNotFoundError(
                f"Missing {experiment} export: {manifest_path} or {metrics_path}. "
                "Run both original and new checkpoint exporters first.")
        manifest = read_json(manifest_path)
        if manifest.get("n_runs") != required:
            raise ValueError(f"{experiment}: expected {required} completed runs")
        if manifest.get("n_checkpoints", 0) < required:
            raise ValueError(f"{experiment}: invalid checkpoint count")
        identities.append((experiment, manifest))
        with metrics_path.open(encoding="utf-8", newline="") as stream:
            records = list(csv.DictReader(stream))
        if len(records) != manifest["n_checkpoints"]:
            raise ValueError(f"{experiment}: checkpoint row count mismatch")
        run_ids = {r["run_id"] for r in records}
        if len(run_ids) != required or run_ids & seen_ids:
            raise ValueError(f"{experiment}: missing/duplicate run IDs")
        for run_id in run_ids:
            if not (out / "records" / (run_id + ".json")).is_file():
                raise FileNotFoundError(f"Missing exported record: {run_id}")
            if not (out / "predictions" / (run_id + ".npz")).is_file():
                raise FileNotFoundError(f"Missing exported predictions: {run_id}")
        seen_ids.update(run_ids)
        combined_metrics.extend(dict(experiment=experiment, **r) for r in records)
        if gamma_path.is_file():
            with gamma_path.open(encoding="utf-8", newline="") as stream:
                combined_gamma.extend(dict(experiment=experiment, **r)
                                      for r in csv.DictReader(stream))

    old, new = (item[1] for item in identities)
    if old["test_data_sha256"] != new["test_data_sha256"]:
        raise ValueError("Test data identity differs between experiments")
    if old["validation_data_sha256"] != new["validation_data_sha256"]:
        raise ValueError("Validation identity differs between experiments")

    n_runs = len(seen_ids)
    n_checkpoints = len(combined_metrics)
    if n_runs != 2350 or n_checkpoints != sum(m["n_checkpoints"] for _, m in identities):
        raise ValueError("Combined checkpoint count is incomplete")
    write_csv(out / "combined_checkpoint_metrics.csv", combined_metrics)
    write_csv(out / "combined_checkpoint_gamma.csv", combined_gamma)
    atomic_json(out / "combined_export_manifest.json", dict(
        experiments=[dict(name=name, manifest=manifest) for name, manifest in identities],
        n_runs=n_runs, n_checkpoints=n_checkpoints,
        n_engine_predictions=sum(int(m["n_engine_predictions"]) for _, m in identities),
        record_directory="records/", prediction_directory="predictions/",
        test_data_sha256=old["test_data_sha256"],
        validation_data_sha256=old["validation_data_sha256"],
        selection_warning="Held-out TEST records are for evaluation only. "
                          "Choose hyperparameters, lag sets, checkpoints, and "
                          "stopping policies from validation data, never TEST."))
    (out / "README_combined.txt").write_text(
        "This directory contains ALL 2350 FD001 experiment runs.\n"
        "Original 1600-run metrics: checkpoint_metrics.csv\n"
        "New 750-run Combo metrics: combo_mse500_checkpoint_metrics.csv\n"
        "All 2350 runs: combined_checkpoint_metrics.csv (experiment column).\n"
        "All 2350 runs: records/<run_id>.json and predictions/<run_id>.npz.\n"
        "New three-stage timing: combo_mse500_stage_training.csv.\n"
        "Do not select hyperparameters/checkpoints by TEST results.\n",
        encoding="utf-8")
    return dict(n_runs=n_runs, n_checkpoints=n_checkpoints)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', nargs='?', default='run', choices=('run', 'status', 'smoke', 'merge-only'))
    parser.add_argument('--workers', type=int, default=24)
    args = parser.parse_args()
    if not 1 <= args.workers <= 128:
        parser.error('--workers must be from 1 to 128')
    from dscpuscripts.common import cpu_environment, file_lock
    from dscpuscripts.fd001_combo_mse500_train import code_hash
    from fulltd.data import load_dataset, verify_dataset
    cpu_environment()
    data_dir, root, outputs = paths('fd001')
    out = outputs / 'all_checkpoints'
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
    if args.mode == 'merge-only':
        with file_lock(out / '.all_checkpoints.lock'):
            copied = reuse_previous_separate_export(outputs, out)
            merged = merge_original_and_combo_exports(out)
        print(f'MERGED: {merged["n_runs"]} runs, {merged["n_checkpoints"]} checkpoints; '
              f'{copied} files reused from separate export; output={out}', flush=True)
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
        copied = reuse_previous_separate_export(outputs, out)
        if copied:
            print(f'Reused {copied} files from earlier separate Combo export',
                  flush=True)
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
        joint = merge_original_and_combo_exports(out)
        print(f'COMPLETE: {N_RUNS} new runs, {count} checkpoints; '
              f'{joint["n_runs"]} combined runs; output={out}', flush=True)


if __name__ == '__main__':
    main()
