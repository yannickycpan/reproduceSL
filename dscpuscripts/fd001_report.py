"""Consistent FD001 reporting from completed outputs or uploaded ZIPs.

Selection stays mean validation MSE over epochs 80/90/100 and five seeds.
Test tables use the actual epoch-100 RMSE/MAE/RUL score. The additional
learning curve computes sqrt(MSE) per seed BEFORE taking mean and SE;
legacy MSE and MAE curves are not relabeled or removed.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import io
import json
import math
import os
from pathlib import Path
import statistics
import zipfile

LABELS = {'mse': 'MSE (B256)', 'gaussian_nll': 'Gaussian NLL (B256)',
          'typed': 'TD typed (B128)', 'raw26': 'TD raw26 (B128)',
          'representation': 'TD representation (B128)'}
ORDER = tuple(LABELS)


def read_outputs(source):
    path = Path(source).expanduser()
    names = ('selections.json', 'metrics.csv', 'validation_records.csv', 'scores.csv')
    raw = {}
    for name in names:
        if path.is_dir():
            raw[name] = (path / name).read_text(encoding='utf-8-sig')
        else:
            with zipfile.ZipFile(path) as archive:
                matches = [prefix + name for prefix in ('outputs/', 'outputs_fd001/', '') if prefix + name in archive.namelist()]
                if len(matches) != 1:
                    raise ValueError(f'Expected one {name} in {source}')
                raw[name] = archive.read(matches[0]).decode('utf-8-sig')
    return dict(selection=json.loads(raw['selections.json']),
                metrics=list(csv.DictReader(io.StringIO(raw['metrics.csv']))),
                validation=list(csv.DictReader(io.StringIO(raw['validation_records.csv']))),
                scores=list(csv.DictReader(io.StringIO(raw['scores.csv']))))


def moments(values):
    if not values:
        raise ValueError('Empty metric group')
    if all(math.isfinite(v) for v in values):
        return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.
    if any(math.isnan(v) for v in values) or (math.inf in values and -math.inf in values):
        return math.nan, math.nan
    return (math.inf if math.inf in values else -math.inf), math.nan


def target_scale(metrics):
    grouped = defaultdict(dict)
    for r in metrics:
        if r['horizon'] == 'all' and r['metric'] == 'mse' and r['variant'] == 'trained':
            grouped[(r['run_id'], r['split'])][r['units']] = float(r['value'])
    ratios = [math.sqrt(r['cycles'] / r['standardized']) for r in grouped.values()
              if r.get('standardized', 0) > 0 and math.isfinite(r.get('cycles', math.nan))]
    if not ratios or not all(math.isclose(v, ratios[0], rel_tol=1e-7) for v in ratios):
        raise ValueError('Cannot verify one common FD001 target scale from the saved metrics')
    return statistics.mean(ratios)


def selected_records(payload, family):
    selection = payload['selection']
    if selection.get('selection_epochs') != [80, 90, 100]:
        raise ValueError('Unexpected selection policy; not the original late-validation rule')
    groups = selection['per_variant'] if family == 'gamma' else [g for g in selection['groups']
              if g['dataset'] == 'fd001' and g['method'] in ('mse', 'gaussian_nll')]
    expected = {'typed', 'raw26', 'representation'} if family == 'gamma' else {'mse', 'gaussian_nll'}
    if {g['gamma_variant'] if family == 'gamma' else g['method'] for g in groups} != expected:
        raise ValueError('Missing a requested method')
    scale = target_scale(payload['metrics'])
    summary, curves, choices = [], [], []
    for group in groups:
        method = group['gamma_variant'] if family == 'gamma' else group['method']
        ids = set(group['run_ids'])
        if group.get('state') != 'selected' or len(ids) != 5:
            raise ValueError(f'{method} has no valid five-seed selected configuration')
        validation = [r for r in payload['validation'] if r['run_id'] in ids]
        if len(validation) != 50 or {(int(r['seed']), int(r['epoch'])) for r in validation} != {
                (s, e) for s in range(5) for e in range(10, 101, 10)}:
            raise ValueError(f'Missing validation evidence: {method}')
        selected_score = statistics.mean(float(r['mse']) for r in validation if int(r['epoch']) in (80, 90, 100))
        scores = [r for r in payload['scores'] if
                  (r.get('gamma_variant') == method if family == 'gamma' else
                   r['method'] == method and r['batch_size'] == '256')]
        if (not scores or not math.isclose(selected_score, float(group['score']), rel_tol=1e-9)
                or selected_score > min(float(r['score']) for r in scores) + 1e-10):
            raise ValueError(f'{method} is not the validation-selected best configuration')
        choices.append(dict(method=method, label=LABELS[method], alpha=group['alpha'], beta=group.get('beta'),
                            batch_size=128 if family == 'gamma' else 256,
                            validation_selection_mse=selected_score, test_checkpoint_epoch=100))
        for metric in ('rmse', 'mae', 'rul_score_sum'):
            rows = sorted([r for r in payload['metrics'] if r['run_id'] in ids and r['split'] == 'test'
                           and r['horizon'] == 'all' and r['metric'] == metric and r['variant'] == 'trained'
                           and r['units'] == ('score' if metric == 'rul_score_sum' else 'cycles')], key=lambda r: int(r['seed']))
            if [int(r['seed']) for r in rows] != list(range(5)):
                raise ValueError('Missing or duplicate test seed')
            mean, sd = moments([float(r['value']) for r in rows])
            summary.append(dict(method=method, label=LABELS[method], metric=metric,
                                units='score' if metric == 'rul_score_sum' else 'cycles', n=5,
                                mean=mean, sample_sd=sd, standard_error=sd / math.sqrt(5)))
        for epoch in range(10, 101, 10):
            mses = [float(r['mse']) for r in validation if int(r['epoch']) == epoch]
            if any(v < 0 for v in mses):
                raise ValueError('MSE cannot be negative')
            rmses = [math.sqrt(v) * scale for v in mses]
            mean, sd = moments(rmses)
            update_values = {int(r["optimizer_steps"]) for r in validation
                             if int(r["epoch"]) == epoch}
            if len(update_values) != 1:
                raise ValueError(f"{method}: missing or nonmatching optimizer update counts at epoch {epoch}")
            curves.append(dict(method=method, label=LABELS[method], epoch=epoch,
                               optimizer_steps=update_values.pop(), metric='rmse', units='cycles',
                               n=5, mean=mean, sample_sd=sd, standard_error=sd / math.sqrt(5)))
    return summary, curves, choices, scale


def plot_learning_curve(curves, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    for method in ORDER:
        rows = sorted([r for r in curves if r['method'] == method], key=lambda r: r['epoch'])
        if not rows:
            continue
        x, mean, se = [[r[k] for r in rows] for k in ('optimizer_steps', 'mean', 'standard_error')]
        ax.plot(x, mean, marker='o', markersize=3.5, linewidth=1.8, label=LABELS[method])
        ax.fill_between(x, [m-s for m,s in zip(mean,se)], [m+s for m,s in zip(mean,se)], alpha=.10)
    ax.set_xlabel('Cumulative Adam optimizer updates (B128 vs B256)')
    ax.set_ylabel('Validation RMSE (cycles)')

    ax.grid(alpha=.2)
    ax.legend(fontsize=9)
    ax.set_title('FD001: each method at its validation-selected hyperparameters')
    fig.tight_layout()
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(destination.with_suffix('.' + ext), dpi=180, bbox_inches='tight')
    plt.close(fig)


def run(gamma_source, output, baseline_source=None):
    from fd001_gamma.common import write_csv, atomic_json
    summary, curves, choices, scale = selected_records(read_outputs(gamma_source), 'gamma')
    if baseline_source:
        other, other_curves, other_choices, old_scale = selected_records(read_outputs(baseline_source), 'baseline')
        if not math.isclose(scale, old_scale, rel_tol=1e-7):
            raise ValueError('Baseline and Gamma target scales differ; cannot merge the curves')
        summary += other
        curves += other_curves
        choices += other_choices
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / 'test_performance.csv', summary)
    write_csv(out / 'selected_hyperparameters.csv', choices)
    write_csv(out / 'validation_rmse.csv', curves)
    atomic_json(out / 'metric_conventions.json', dict(
        selection='mean validation MSE over epochs 80,90,100 and five seeds; unchanged',
        test='actual epoch-100 checkpoint; mean/SD of per-seed metrics',
        curve='mean/SE of per-seed sqrt(validation MSE) times target scale, x=actual cumulative Adam updates',
        target_scale=scale, units='cycles (RMSE/MAE), dimensionless summed RUL score',
        baseline_batch=256, gamma_batch=128,
        caveat='Matching target scale is checked here; use the campaign audits for complete dataset identity verification'))
    plot_learning_curve(curves, out / 'validation_rmse_fd001_tcn')
    print('FD001 report:', out)


def main():
    from fd001_gamma.common import campaign_paths
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('gamma', 'baselines', 'compare'))
    parser.add_argument('--gamma')
    parser.add_argument('--legacy', '--baseline', dest='baseline')
    parser.add_argument('--output-dir')
    args = parser.parse_args()
    if args.mode == 'baselines':
        from dscpuscripts.standard_runner import paths
        from fd001_gamma.common import write_csv
        source = Path(args.baseline).expanduser() if args.baseline else paths('fd001_baselines')[2]
        _, curves, _, _ = selected_records(read_outputs(source), 'baseline')
        out = Path(args.output_dir).expanduser() if args.output_dir else source / 'figures'
        out.mkdir(parents=True, exist_ok=True)
        write_csv(out / 'validation_rmse_fd001_tcn.csv', curves)
        plot_learning_curve(curves, out / 'validation_rmse_fd001_tcn')
        return
    gamma = Path(args.gamma).expanduser() if args.gamma else campaign_paths()[2]
    baseline = args.baseline
    if args.mode == 'compare' and not baseline:
        from dscpuscripts.standard_runner import paths
        local = paths('fd001_baselines')[2]
        if (local / 'selections.json').is_file():
            baseline = local
        else:
            from dscpuscripts.combined_fd001 import locate_source
            baseline = locate_source(None, campaign_paths()[1])
    output = Path(args.output_dir).expanduser() if args.output_dir else gamma / ('rmse_report' if args.mode == 'gamma' else 'comparison_report')
    run(gamma, output, baseline if args.mode == 'compare' else None)


if __name__ == '__main__':
    main()
