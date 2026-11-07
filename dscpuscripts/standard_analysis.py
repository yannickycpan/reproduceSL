"""Audit and evaluate unchanged direct-server campaigns, without any refit.

Uses the original validation selector for complete dataset campaigns and the
same rule restricted to the two original FD001 baselines. Test targets are
opened only after all candidates in the requested campaign are terminal.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import math
from pathlib import Path

from dscpuscripts.standard_runner import CAMPAIGNS, configurations, paths


def select_baselines(results, output=None):
    from fulltd.analysis import _validate_candidate, audit_campaign, group_id, _manifest_hash
    from fulltd.protocol import EVALUATIONS, SELECTION_EPOCHS, PROTOCOL_ID, SEEDS
    from fd001_gamma.common import atomic_json, write_csv
    manifest = configurations('fd001_baselines')
    groups, evidence, curves, accounting = defaultdict(list), defaultdict(list), [], []
    for config in manifest:
        score, records, status = _validate_candidate(config, Path(results))
        group = group_id(config)
        groups[group].append((config, score, status))
        evidence[group].append(dict(config=config, state=status['state'], validation=records,
                                    identity=status.get('identity'), initial_mean_sha256=status.get('initial_mean_sha256')))
        curves.extend(dict(config, **record) for record in records)
        accounting.append(dict(config, state=status['state'], late_validation_mse=score,
                               epochs_completed=status.get('epochs_completed')))
    audit = audit_campaign(manifest, results, dataset='fd001')
    choices, scores, selected = [], [], []
    for name, entries in sorted(groups.items()):
        by_setting = defaultdict(list)
        for entry in entries:
            by_setting[entry[0]['configuration']].append(entry)
        ranked = []
        for setting, members in sorted(by_setting.items()):
            members.sort(key=lambda x: x[0]['seed'])
            if tuple(x[0]['seed'] for x in members) != SEEDS:
                raise ValueError('Baseline selection requires all five seeds')
            c = members[0][0]
            score = math.fsum(x[1] for x in members) / 5 if all(math.isfinite(x[1]) for x in members) else math.inf
            scores.append(dict(dataset='fd001', backbone='tcn', method=c['method'], batch_size=256,
                               configuration=setting, alpha=c['alpha'], beta=None, score=score))
            ranked.append((score, c['alpha'], setting, members))
        best = min(ranked, key=lambda x: x[:3])
        c = best[3][0][0]
        choice = dict(dataset='fd001', backbone='tcn', method=c['method'], batch_size=256,
                      group_id=name, selection_evidence_sha256=_manifest_hash(evidence[name]),
                      state='selected' if math.isfinite(best[0]) else 'all_configurations_failed',
                      score=best[0] if math.isfinite(best[0]) else 'Infinity', configuration=best[2], alpha=best[1], beta=None,
                      run_ids=[x[0]['run_id'] for x in best[3]] if math.isfinite(best[0]) else [])
        choices.append(choice)
        selected.extend(dict(x[0], pooled_selection_score=best[0]) for x in best[3] if math.isfinite(best[0]))
    result = dict(protocol_id=PROTOCOL_ID, dataset_scope='fd001', method_scope=['mse', 'gaussian_nll'],
                  manifest_sha256=_manifest_hash(manifest), validation_epochs=list(EVALUATIONS),
                  selection_epochs=list(SELECTION_EPOCHS), groups=choices)
    if output is not None:
        out = Path(output)
        atomic_json(out / 'selections.json', result)
        atomic_json(out / 'campaign_audit.json', audit)
        for name, rows in [('scores', scores), ('selected_runs', selected),
                           ('candidate_accounting', accounting), ('validation_records', curves)]:
            write_csv(out / (name + '.csv'), rows)
    return result


def selection(campaign, results, out):
    if campaign == 'fd001_baselines':
        return select_baselines(results, out)
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
    from fd001_gamma.common import read_json, atomic_json, write_csv, file_hash, data_identity
    from fd001_gamma.protocol import digest
    configs = {c['run_id']: c for c in configurations(campaign)}
    dataset = 'fd001' if campaign == 'fd001_baselines' else campaign
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
    if dataset != 'fd001':
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
    from fd001_gamma.common import cpu_environment, file_lock
    cpu_environment()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('campaign', choices=CAMPAIGNS)
    parser.add_argument('mode', choices=('check', 'run', 'plots'), nargs='?', default='run')
    args = parser.parse_args()
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
            verify_dataset('fd001' if args.campaign == 'fd001_baselines' else args.campaign, data)
            locked = selection(args.campaign, results, out)
            evaluate_campaign(args.campaign, data, results, out, locked)
            analyze_results(out / 'metrics.csv', out, out / 'figures')
    print('Analysis complete (no refit):', out)


if __name__ == '__main__':
    main()
