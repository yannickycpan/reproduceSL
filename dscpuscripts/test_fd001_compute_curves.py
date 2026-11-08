"""Offline regression checks for FD001 B256 TEST-vs-compute reporting."""
import math

import pytest

from dscpuscripts.fd001_compute_curves import processed_sources, relative_compute


def test_equal_epoch_source_count_and_relative_budget():
    n = 13989
    assert processed_sources(n, 256, 0) == 0
    assert processed_sources(n, 256, 55) == n
    assert processed_sources(n, 256, 5500) == 100*n
    assert relative_compute("mse", n, 256, 5500) == 1
    assert relative_compute("gaussian_nll", n, 256, 5500) == 1
    assert relative_compute("td_typed", n, 256, 5500) == 2
    assert relative_compute("td_pairwise", n, 256, 5500) == 2
    assert relative_compute("td_pairwise", n, 128, 5500) == 1
    with pytest.raises(ValueError):
        relative_compute("invalid", n, 256, 100)


def test_test_curve_plot_names_and_both_views(tmp_path):
    pytest.importorskip("matplotlib")
    from dscpuscripts.fd001_protocol import METHODS
    from dscpuscripts.fd001_compute_curves import plot
    summary = []
    for idx, method in enumerate(METHODS):
        for metric in ("rmse_cycles", "rul_score_sum"):
            for step in (0, 500, 5500):
                x = relative_compute(method, 13989, 256, step)
                value = ((33-0.002*step+idx) if metric=="rmse_cycles"
                         else (150000-10*step+idx*5000))
                summary.append(dict(method=method, metric=metric, batch_size=256,
                                    optimizer_steps=step, compute_proxy=x,
                                    mean=value, standard_error=1,
                                    finite_seeds=5))
    folder = plot(summary,tmp_path)
    for metric in ("rmse_cycles","rul_score_sum"):
        for suffix in ("","_full"):
            for ext in ("pdf","png","svg"):
                output = folder/f"fd001_b256_test_{metric}_vs_compute_proxy{suffix}.{ext}"
                assert output.is_file() and output.stat().st_size > 500
