"""Unified FD001 analysis and independent Electricity/METR-LA reporting, no refit."""
from __future__ import annotations
import argparse
from pathlib import Path

from dscpuscripts.standard_runner import CAMPAIGNS, configurations, paths


def selection(campaign, results, out):
    from fulltd.analysis import select_candidates
    from fulltd.protocol import make_manifest
    return select_candidates(make_manifest(), results, out, dataset=campaign)


def evaluate_campaign(campaign, data_dir, results, out, locked):
    import torch
    from fulltd.data import load_dataset
    from fulltd.protocol import LAGS
    from fulltd.sampling import training_universe
    from fulltd.train import make_objective_for, seed_runtime
    from fulltd.validation import validation_view
    from fulltd.evaluate import validate_final_checkpoint, evaluate_endpoint
    from dscpuscripts.common import read_json, atomic_json, write_csv, file_hash, data_identity
    from fulltd.protocol import digest
    configs = {c['run_id']: c for c in configurations(campaign)}
    dataset = campaign
    train, _, _ = training_universe(load_dataset(dataset, data_dir, 'train'), LAGS[dataset])
    val = load_dataset(dataset, data_dir, 'val')
    test, rows = None, []
    hashes = {}
    code = file_hash(__file__)
    for group in locked['groups']:
        if group['state'] != 'selected':
            continue
        for rid in group['run_ids']:
            config = configs[rid]
            checkpoint_path = results / rid / 'final.pt'
            ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            status = read_json(results / rid / 'status.json')
            validate_final_checkpoint(ckpt, config, status, train, val)
            if (ckpt['validation'] != read_json(results / rid / 'validation.json')
                    or ckpt['round_records'] != read_json(results / rid / 'rounds.json')):
                raise ValueError('Selected evidence and actual final checkpoint differ')
            endpoint_val, subset = validation_view(val, 100, rid)
            if test is None:
                test = load_dataset(dataset, data_dir, 'test')
            hashes[rid] = file_hash(checkpoint_path)
            identity = dict(selection=digest(locked), checkpoint=hashes[rid], test=data_identity(test), code=code)
            cache = out / 'endpoint_cache' / (rid + '_' + digest(identity)[:16] + '.json')
            if cache.exists():
                saved = read_json(cache)
                if saved['identity'] != identity:
                    raise ValueError('Endpoint cache identity mismatch')
                rows.extend(saved['rows'])
                continue
            seed_runtime(config['seed'])
            objective = make_objective_for(config, train)
            objective.load_state_dict(ckpt['objective'])
            metadata = {k: config[k] for k in ('run_id', 'dataset', 'backbone', 'method', 'batch_size', 'seed', 'alpha', 'beta')}
            run_rows, failures = [], []
            eval_batch = 32 if dataset == 'metr_la' else 128
            for split, ds in [('train', train), ('validation', endpoint_val), ('test', test)]:
                metrics, endpoints = evaluate_endpoint(objective.model, ds, eval_batch, 'cpu', failures,
                                                       dict(run_id=rid, split=split, variant='trained'))
                run_rows.extend(dict(metadata, split=split, variant='trained', **m) for m in metrics)
                if endpoints and split == 'test':
                    write_csv(out / 'predictions' / rid / 'test_engine_predictions.csv', endpoints)
            atomic_json(cache, dict(identity=identity, rows=run_rows, failures=failures))
            rows.extend(run_rows)
            print(f'Evaluated final checkpoint: {rid}', flush=True)
    for split in ('val', 'test'):
        ds = load_dataset(dataset, data_dir, split)
        if split == 'val':
            ds, _ = validation_view(ds, 100, dataset + '_persistence')
        failures = []
        metrics, _ = evaluate_endpoint(None, ds, 32 if dataset == 'metr_la' else 128, 'cpu', failures,
                                       dict(run_id=dataset + '_persistence', split=split), persistence=True)
        rows.extend(dict(run_id=dataset + '_persistence', dataset=dataset, backbone='persistence',
                         method='daily_persistence' if dataset == 'electricity' else 'last_observation',
                         batch_size=256, seed=-1, alpha=None, beta=None, variant='trained',
                         split='validation' if split == 'val' else 'test', **m) for m in metrics)
    write_csv(out / 'metrics.csv', rows)
    atomic_json(out / 'evaluation_manifest.json', dict(protocol_id=locked['protocol_id'],
        selection_sha256=digest(locked), checkpoint_hashes=hashes,
        data_fingerprints={dataset: train.metadata}, refit_scope='disabled',
        point_prediction='actual epoch-100 mean network only'))
    return rows


def main():
    from dscpuscripts.common import cpu_environment, file_lock
    cpu_environment()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('campaign', choices=CAMPAIGNS)
    parser.add_argument('mode', choices=('check', 'run', 'plots'), nargs='?', default='run')
    args = parser.parse_args()
    if args.campaign == 'fd001':
        from dscpuscripts.fd001_analysis import run as analyze_fd001
        return analyze_fd001(args.mode)
    data, results, out = paths(args.campaign)
    from fulltd.analysis import analyze_results, create_figures
    if args.mode == 'check':
        selection(args.campaign, results, out)
        print('READY: canonical candidate records audited')
        return
    with file_lock(out / '.analysis.lock'):
        if args.mode == 'plots':
            create_figures(out, out / 'figures')
        else:
            import torch
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
            from fulltd.data import verify_dataset
            verify_dataset(args.campaign, data)
            locked = selection(args.campaign, results, out)
            evaluate_campaign(args.campaign, data, results, out, locked)
            analyze_results(out / 'metrics.csv', out, out / 'figures')
    print('Analysis complete (no refit):', out)


if __name__ == '__main__':
    main()
