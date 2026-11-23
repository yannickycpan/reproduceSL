"""No-retraining regression checks for validation-locked FD001 paper reporting."""
import pytest

from dscpuscripts.fd001_paper_comparison import _best, _paired, _summary, _markdown


def test_best_uses_validation_score_not_lag_order():
    candidates = [
        dict(method='td_pairwise', lag_set='1-5', score=.241, alpha=.001,
             beta=.003, configuration=1),
        dict(method='td_pairwise', lag_set='1-3-5', score=.240, alpha=.0003,
             beta=.0001, configuration=2),
        dict(method='td_pairwise', lag_set='1-3-5-10-15', score=float('inf'),
             alpha=.0001, beta=.0001, configuration=0)]
    assert _best(candidates)['lag_set'] == '1-3-5'
    assert _best(candidates)['configuration'] == 2
    assert candidates[0]['score'] == .241


def test_seed_pairing_and_five_seed_summary():
    data = []
    for regime in ('full_100_epochs', 'matched_time_proxy'):
        methods = (('mse', 256), ('td_pairwise', 128), ('td_pairwise', 256))
        if regime == 'full_100_epochs':
            methods += (('td_mse_combo', 128), ('td_mse_combo', 256))
        for method, batch in methods:
            for seed in range(5):
                for metric in ('rmse', 'rul_score_sum'):
                    score = (1 if method == 'td_pairwise' else 3)
                    data.append(dict(regime=regime, method=method,
                                     batch_size=batch, lag_set='1-5',
                                     seed=seed, metric=metric, value=score + seed/10))
    summary = _summary(data, ('regime','method','batch_size','lag_set','metric'))
    assert len(summary) == 16
    comparisons = _paired(data, [])
    assert len(comparisons) == 12
    assert all(x['wins'] == 5 for x in comparisons)
    assert all(x['mean_difference'] == pytest.approx(-2) for x in comparisons)


def test_summary_requires_five_seeds():
    rows = [dict(seed=i, metric='rmse', value=5.) for i in range(4)]
    with pytest.raises(ValueError, match='Expected five seeds'):
        _summary(rows, ('metric',))
