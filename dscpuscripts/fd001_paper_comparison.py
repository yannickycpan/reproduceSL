"""Validation-locked FD001 paper comparison: MSE vs pairwise TD vs TD-MSE combo.

Read ONLY existing lagscanv3 runs and checkpoints. Never fit, modify, or overwrite
trained weights. Output under fd001_outputs/paper_comparison/. No MAE or NLL.

check: audit 1600 runs, independently select alpha/beta/lags for each TD batch,
       and lock complete-budget and approximate equal-wall-time selection.
run:   verify locked choices; infer held-out TEST predictions from selected
       final/checkpoint states; report RMSE, RUL score, time and engine errors.

The matched-time TD models use a single preselected checkpoint step per batch,
chosen via cohort-average full-run wall times, not precise per-checkpoint times.
The original 100-epoch MSE -> Gamma -> 100-epoch TD combo generally cannot
finish even its initial stages within the MSE budget; it is never extrapolated.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import math
from pathlib import Path
import statistics

from dscpuscripts.common import atomic_json, data_identity, file_hash, read_json, write_csv
from dscpuscripts.fd001_analysis import audit_and_select, selection_update_steps
from dscpuscripts.fd001_protocol import EPOCHS, EVAL_UPDATE_PERIOD, METHODS, PROTOCOL_ID, SEEDS, make_manifest
from dscpuscripts.fd001_train import build_objective, code_hash, json_safe
from dscpuscripts.result_paths import paths
from fulltd.protocol import digest

REVISION = 'fd001-paper-three-methods-validation-locked-v1'
SPEC = (('mse', 256), ('td_pairwise', 128), ('td_pairwise', 256),
        ('td_mse_combo', 128), ('td_mse_combo', 256))
TARGETS = ('rmse', 'rul_score_sum')


def _best(options):
    valid = [dict(x) for x in options if math.isfinite(float(x['score']))]
    if not valid:
        raise ValueError('No finite five-seed validation candidates')
    return min(valid, key=lambda x: (float(x['score']), float(x['alpha']),
                                     float(x['beta'] or 0), str(x['lag_set']),
                                     int(x['configuration'])))


def _row(best, method, batch, regime, step, time_budget):
    return dict(regime=regime, method=method, batch_size=batch,
                lag_set=best['lag_set'], alpha=float(best['alpha']),
                beta=None if best['beta'] is None else float(best['beta']),
                mse_alpha=best.get('mse_alpha'),
                configuration=int(best['configuration']),
                selection_mse=float(best['score']), optimizer_steps=int(step),
                estimated_budget_seconds=float(time_budget),
                run_ids=list(best['run_ids']))


def _status(root, run_id):
    return read_json(root / run_id / 'status.json')


def _full_seconds(status):
    # Both fields include wall-clock overhead (checkpoint, validation, staging).
    return float(status['runtime_seconds'])


def _select_plan(root):
    canonical, scores = audit_and_select(root)
    manifest = make_manifest()
    if canonical['counts'].get('complete') != len(manifest) or len(manifest) != 1600:
        raise ValueError('Paper comparison requires 1600 completed lagscanv3 runs')
    if canonical['protocol_id'] != PROTOCOL_ID:
        raise ValueError('Unexpected protocol')
    full_best = {}
    for method, batch in SPEC:
        best = _best(x for x in scores if x['method'] == method and x['batch_size'] == batch)
        full_best[method, batch] = best
    mse = full_best['mse', 256]
    budget = statistics.mean(_full_seconds(_status(root, rid)) for rid in mse['run_ids'])
    n = int(_status(root, mse['run_ids'][0])['n_sources'])
    chosen = []
    for method, batch in SPEC:
        best = full_best[method, batch]
        step = EPOCHS * math.ceil(n / batch)
        chosen.append(_row(best, method, batch, 'full_100_epochs', step, budget))
    # Fixed-step calibration from ALL pairwise configurations, before seeing
    # any truncated validation score or held-out test result.
    calibrations = []
    for batch in (128, 256):
        pool = [c for c in manifest if c['method'] == 'td_pairwise'
                and c['batch_size'] == batch]
        if len(pool) != 375:
            raise ValueError(f'Expected 375 pairwise candidates at batch {batch}')
        mean_full_wall = statistics.mean(_full_seconds(_status(root, c['run_id']))
                                         for c in pool)
        total = EPOCHS * math.ceil(n / batch)
        cutoff = min(total, int(math.floor(total * budget / mean_full_wall
                                           / EVAL_UPDATE_PERIOD)) * EVAL_UPDATE_PERIOD)
        if cutoff < EVAL_UPDATE_PERIOD:
            raise ValueError(f'No positive saved checkpoint fits B{batch} budget')
        selection_steps = selection_update_steps(cutoff, EVAL_UPDATE_PERIOD)
        options = []
        for row in (x for x in scores if x['method'] == 'td_pairwise'
                    and x['batch_size'] == batch):
            measurements = []
            for rid in row['run_ids']:
                curve = read_json(root / rid / 'update_curve.json')
                lookup = {int(v['optimizer_steps']): float(v['mse_standardized'])
                          for v in curve}
                if any(step not in lookup for step in selection_steps):
                    raise ValueError(f'Missing truncated validation curve: {rid}')
                measurements.append(statistics.mean(lookup[step]
                                                    for step in selection_steps))
            options.append(dict(row, score=statistics.mean(measurements)))
        best = _best(options)
        chosen.append(_row(best, 'td_pairwise', batch, 'matched_time_proxy',
                           cutoff, budget))
        calibrations.append(dict(batch_size=batch, MSE_budget_seconds=budget,
                                 TD_full_cohort_mean_seconds=mean_full_wall,
                                 cutoff_updates=cutoff,
                                 cutoff_approx_epochs=cutoff / math.ceil(n / batch),
                                 cohort_estimated_seconds=mean_full_wall * cutoff / total,
                                 note='Not measured timestamp at checkpoint'))
    chosen.append(_row(mse, 'mse', 256, 'matched_time_proxy',
                       EPOCHS * math.ceil(n / 256), budget))
    # Original Combo needs a new stage-budget protocol for matched time.
    feasibility = []
    for batch in (128, 256):
        x = full_best['td_mse_combo', batch]
        total_updates = EPOCHS * math.ceil(n / batch)
        for rid in x['run_ids']:
            status = _status(root, rid)
            timings = status['timing']
            first_two = (float(timings['mse_wall_seconds']) +
                         float(timings['gamma_wall_seconds']))
            first_saved_predictor = (first_two +
                   float(timings['predictor_wall_seconds']) *
                   EVAL_UPDATE_PERIOD / total_updates)
            feasibility.append(dict(method='td_mse_combo', batch_size=batch,
                                    lag_set=x['lag_set'], seed=status['config']['seed'],
                                    MSE_budget_seconds=budget,
                                    stage1_MSE_seconds=timings['mse_wall_seconds'],
                                    stage2_Gamma_seconds=timings['gamma_wall_seconds'],
                                    before_stage3_seconds=first_two,
                                    first_500_update_checkpoint_estimated_seconds=first_saved_predictor,
                                    first_saved_checkpoint_within_budget=first_saved_predictor <= budget,
                                    note='Estimate; no original matched-time three-stage model'))
    return dict(revision=REVISION, protocol_id=PROTOCOL_ID,
                scientific_code_sha256=code_hash(),
                manifest_sha256=digest(manifest),
                validation_data_sha256=canonical['validation_sha256'],
                selection='Five-seed mean standardized validation MSE on last '
                          'ceil(25%) of positive checkpoint evaluations; '
                          'alpha, beta and lags selected separately for B128/B256; '
                          'no test labels used',
                matched_time='Average 100-epoch MSE wall-clock budget; fixed '
                             'pairwise checkpoint per batch, using cohort mean '
                             'full-run time and linear-in-updates approximation; '
                             'individual estimated times may exceed budget',
                mse_budget_seconds=budget,
                calibration=calibrations, choices=chosen,
                combo_matched_time_available=False,
                combo_feasibility=feasibility,
                incomplete_regime='Original Combo Stage 1 and Stage 2 consume the '
                                  'matched MSE budget; re-training a shorter '
                                  'three-stage protocol is required for equal time')


def _metric_lookup(records):
    return {(r['units'], str(r['horizon']), r['metric']): float(r['value'])
            for r in records}


def _score_and_endpoints(model, test):
    from fulltd.metrics import evaluate
    measurements, predictions = evaluate(model, test, batch_size=128,
                                         device='cpu', allow_nonfinite=False)
    lookup = _metric_lookup(measurements)
    return {metric: lookup[('cycles' if metric == 'rmse' else 'score',
                            'all', metric)] for metric in TARGETS}, predictions


def _load_one(root, c, item, dataset, cache_root, plan_id):
    import torch
    from fulltd.train import seed_runtime
    folder = root / c['run_id']
    step = item['optimizer_steps']
    completed = _status(root, c['run_id'])
    final = item['regime'] == 'full_100_epochs' or c['method'] == 'mse'
    path = folder / 'final.pt' if final else folder / 'checkpoints' / f'step_{step:07d}.pt'
    if not path.is_file():
        raise FileNotFoundError(f'Saved checkpoint unavailable: {path}')
    sha = file_hash(path)
    test_sha = data_identity(dataset)
    identity = dict(revision=REVISION, plan_sha256=plan_id, run_id=c['run_id'],
                    checkpoint_sha256=sha, test_data_sha256=test_sha, step=step)
    cache = cache_root / (digest(identity) + '.json')
    if cache.exists():
        saved = read_json(cache)
        if saved['identity'] != identity:
            raise ValueError('Checkpoint cache identity disagreement')
        return saved['metrics'], saved['endpoints'], sha
    # Training-produced local checkpoints only. Do not load arbitrary PT files.
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    if (completed['state'] != 'complete' or ckpt.get('config') != c or
            ckpt.get('identity') != completed['identity']):
        raise ValueError(f'Checkpoint/config identity mismatch: {path}')
    progress = ckpt['progress']
    actual = (progress['phase_steps']['predictor'] if c['method'] == 'td_mse_combo'
              else progress['optimizer_steps'])
    if actual != step:
        raise ValueError(f'Checkpoint step {actual} differs from selected {step}')
    observed = read_json(folder / 'update_curve.json')
    snapshot_curve = json_safe(progress['curve'])
    if (not snapshot_curve or snapshot_curve != observed[:len(snapshot_curve)]
            or snapshot_curve[-1]['optimizer_steps'] != step):
        raise ValueError(f'Saved checkpoint validation trajectory mismatch: {path}')
    seed_runtime(c['seed'])
    objective = build_objective(c)
    objective.load_state_dict(ckpt['objective'])
    objective.model.eval()
    metrics, endpoints = _score_and_endpoints(objective.model, dataset)
    from fulltd.metrics import rul_score
    reconstructed = sum(rul_score([p['prediction']], [p['target']])
                        for p in endpoints)
    if not math.isclose(reconstructed, metrics['rul_score_sum'], rel_tol=1e-9, abs_tol=1e-5):
        raise ValueError(f'Endpoint RUL score does not match metric: {c["run_id"]}')
    atomic_json(cache, dict(identity=identity, metrics=metrics, endpoints=endpoints))
    print(f'TEST {item["regime"]}: {c["run_id"]} step={step}', flush=True)
    return metrics, endpoints, sha


def _summary(rows, keys):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[k] for k in keys)].append(float(row['value']))
    result = []
    for key, vals in sorted(grouped.items()):
        if len(vals) != len(SEEDS):
            raise ValueError(f'Expected five seeds, got {len(vals)} at {key}')
        sd = statistics.stdev(vals)
        result.append(dict(zip(keys, key), n=len(vals), mean=statistics.mean(vals),
                           sd=sd, se=sd / math.sqrt(len(vals))))
    return result


def _paired(rows, choices):
    table = {(r['regime'], r['method'], r['batch_size'], r['seed'], r['metric']): r['value']
             for r in rows}
    comparisons = []
    for regime in ('full_100_epochs', 'matched_time_proxy'):
        for batch in (128, 256):
            for method2, batch2 in [('mse', 256)] + ([('td_mse_combo', batch)] if regime == 'full_100_epochs' else []):
                for metric in TARGETS:
                    a = [table[(regime, 'td_pairwise', batch, seed, metric)] for seed in SEEDS]
                    b = [table[(regime, method2, batch2, seed, metric)] for seed in SEEDS]
                    diff = [x - y for x, y in zip(a, b)]
                    avg = statistics.mean(diff)
                    sem = statistics.stdev(diff) / math.sqrt(len(SEEDS))
                    t95 = 2.7764451051977987
                    comparisons.append(dict(regime=regime, first='td_pairwise', first_batch=batch,
                                            second=method2, second_batch=batch2,
                                            metric=metric, n=5, mean_difference=avg,
                                            standard_error=sem,
                                            ci95_lower=avg - t95 * sem,
                                            ci95_upper=avg + t95 * sem,
                                            wins=sum(x < y for x, y in zip(a,b)),
                                            note='Descriptive paired-seed t interval; not an independent dataset test'))
    return comparisons


def _markdown(plan, summary, minutes):
    lookup = {(r['regime'], r['method'], r['batch_size'], r['metric']): r
              for r in summary}
    items = {(x['regime'], x['method'], x['batch_size']): x for x in plan['choices']}
    lines = ['# FD001: three-method validation-locked comparison', '',
             'Test RMSE (cycles) and untruncated NASA RUL Score, five seeds. '
             'Lower is better. No test-based hyperparameter selection.', '',
             '## Complete 100-epoch training', '',
             '| Method | Batch | Lags | Alpha | Beta | Test RMSE mean ± SE | Test RUL Score mean ± SE | Wall min |',
             '|---|---:|---|---:|---:|---:|---:|---:|']
    def cell(regime, method, batch):
        it = items[regime, method, batch]
        rmse = lookup[regime, method, batch, 'rmse']
        rul = lookup[regime, method, batch, 'rul_score_sum']
        return (f'| {method} | {batch} | {it["lag_set"]} | {it["alpha"]:g} | '
                f'{it["beta"] if it["beta"] is not None else "—"} | '
                f'{rmse["mean"]:.3f} ± {rmse["se"]:.3f} | '
                f'{rul["mean"]:,.1f} ± {rul["se"]:,.1f} | '
                f'{minutes[regime,method,batch]:.2f} |')
    for m,b in SPEC:
        lines.append(cell('full_100_epochs',m,b))
    lines += ['', '## Approximately matched MSE wall-clock budget', '',
              'Pairwise cutoffs from ALL candidate runs by cohort-average time; '
              'not precise per-checkpoint walltime. Individual seeds can exceed budget.', '',
              '| Method | Batch | Lags | Alpha | Beta | Test RMSE mean ± SE | Test RUL Score mean ± SE | Estimated wall min |',
              '|---|---:|---|---:|---:|---:|---:|---:|']
    for m,b in SPEC:
        if m=='td_mse_combo':
            lines.append(f'| {m} | {b} | — | — | — | not available | not available | requires new training |')
        else:
            lines.append(cell('matched_time_proxy',m,b))
    lines += ['', f'MSE 100-epoch reference time: {plan["mse_budget_seconds"]/60:.3f} min.',
              'Combo original three-stage schedule cannot be extrapolated to this time '
              'budget; see combo_match_feasibility.csv for per-seed times.',
              'Only the FULL results form a fair three-method comparison under '
              'the same 100-epoch final predictor training rule.', '']
    return '\n'.join(lines)


def run(mode):
    from dscpuscripts.common import cpu_environment, file_lock
    cpu_environment()
    data_dir, root, out = paths('fd001')
    out = out / 'paper_comparison'
    out.mkdir(parents=True, exist_ok=True)
    with file_lock(out / '.paper_comparison.lock'):
        plan = _select_plan(root)
        selection_file = out / 'validation_locked.json'
        if selection_file.is_file():
            if read_json(selection_file) != plan:
                raise ValueError('Previously locked validation plan changed; do not overwrite')
        elif mode == 'run':
            raise ValueError('Run check first to lock validation-only choices')
        else:
            atomic_json(selection_file, plan)
        write_csv(out / 'combo_match_feasibility.csv', plan['combo_feasibility'])
        write_csv(out / 'time_calibration.csv', plan['calibration'])
        write_csv(out / 'validation_choices.csv', [
            {k:v for k,v in item.items() if k != 'run_ids'} for item in plan['choices']])
        if mode == 'check':
            print('LOCKED: 5 full selections + MSE/Pairwise matched checkpoints; '
                  'no held-out TEST labels loaded.', flush=True)
            for x in plan['choices']:
                print(x['regime'], x['method'], x['batch_size'], x['lag_set'],
                      x['alpha'], x['beta'], 'step', x['optimizer_steps'], flush=True)
            return
        import torch
        from fulltd.data import load_dataset
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        val = load_dataset('fd001', data_dir, 'val')
        if data_identity(val) != plan['validation_data_sha256']:
            raise ValueError('Validation split has changed since training')
        test = load_dataset('fd001', data_dir, 'test')
        configs = {c['run_id']:c for c in make_manifest()}
        cache_dir = out / '.cache'
        rows = []
        errors = []
        detail = []
        minutes = {}
        locked_plan_sha = digest(plan)
        for item in plan['choices']:
            estimates = []
            final_steps = EPOCHS * math.ceil(_status(root, item['run_ids'][0])['n_sources']
                                               / item['batch_size'])
            for rid in item['run_ids']:
                c = configs[rid]
                status = _status(root, rid)
                if c['method'] != item['method'] or c['batch_size'] != item['batch_size']:
                    raise ValueError('Wrong selected configuration')
                metrics, endpoints, sha = _load_one(root, c, item, test, cache_dir,
                                                   locked_plan_sha)
                measured = _full_seconds(status)
                estimated = (measured * item['optimizer_steps']/final_steps
                             if item['regime']=='matched_time_proxy' and c['method']!='mse'
                             else measured)
                estimates.append(estimated)
                detail.append(dict(regime=item['regime'], method=c['method'],
                                   batch_size=c['batch_size'], lag_set=item['lag_set'],
                                   seed=c['seed'], optimizer_steps=item['optimizer_steps'],
                                   full_training_wall_seconds=measured,
                                   estimated_wall_seconds=estimated,
                                   MSE_budget_seconds=plan['mse_budget_seconds'],
                                   estimated_over_budget=estimated > plan['mse_budget_seconds'],
                                   checkpoint_sha256=sha, run_id=rid))
                for metric, value in metrics.items():
                    rows.append(dict(regime=item['regime'], method=c['method'],
                                     batch_size=c['batch_size'], lag_set=item['lag_set'],
                                     seed=c['seed'], metric=metric, value=value))
                for endpoint in endpoints:
                    err = float(endpoint['prediction']) - float(endpoint['target'])
                    score = math.expm1(err/10 if err >= 0 else -err/13)
                    errors.append(dict(regime=item['regime'],method=c['method'],
                                       batch_size=c['batch_size'],lag_set=item['lag_set'],
                                       seed=c['seed'], engine_id=endpoint['engine_id'],
                                       prediction=endpoint['prediction'],target=endpoint['target'],
                                       signed_error=err,overprediction=int(err>0),
                                       rul_score_contribution=score))
            minutes[item['regime'],item['method'],item['batch_size']] = statistics.mean(estimates)/60
        summary = _summary(rows, ('regime','method','batch_size','lag_set','metric'))
        paired = _paired(rows, plan['choices'])
        write_csv(out/'test_per_seed.csv',rows)
        write_csv(out/'test_summary.csv',summary)
        write_csv(out/'paired_comparisons.csv',paired)
        write_csv(out/'run_time_and_checkpoint.csv',detail)
        write_csv(out/'test_engine_errors.csv',errors)
        (out/'paper_tables.md').write_text(_markdown(plan,summary,minutes), encoding='utf-8')
        atomic_json(out/'analysis_notes.json',dict(
            revision=REVISION,
            criteria=plan['selection'],
            timing=plan['matched_time'],
            combo_feasibility=plan['incomplete_regime'],
            note='All TEST measurements evaluated after validation choices were locked; '
                 'no training, no MAE or Gaussian NLL reported'))
    print('WROTE:',out,flush=True)
    print((out/'paper_tables.md').read_text(),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('check','run'))
    args=parser.parse_args()
    run(args.mode)


if __name__ == '__main__':
    main()
