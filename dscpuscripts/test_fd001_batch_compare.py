"""Offline checks for fixed FD001 selection rules and matched-update truncation."""
import json

from dscpuscripts.fd001_analysis import selection_update_steps
from dscpuscripts.fd001_batch_compare import _choose, _matched_score


def test_equal_update_window():
    # FD001 13,989 sources, 100 passes: B256=5,500, B128=11,000 updates.
    # Eleven positive evaluations up to 5,500 -> final ceil(25%) = 3.
    assert selection_update_steps(5500, 500) == (4500, 5000, 5500)


def test_matched_score_uses_only_early_validation_and_all_seeds(tmp_path):
    rows = []
    for setting in (0, 1):
        run_ids = []
        for seed in range(5):
            run_id = f"test_h{setting}_s{seed}"
            folder = tmp_path / run_id
            folder.mkdir()
            (folder / "status.json").write_text(json.dumps({"state": "complete"}))
            # Setting 1 wins at the matched boundary but setting 0 would win
            # if post-boundary evaluation records were improperly included.
            observations = [
                dict(optimizer_steps=step,
                     mse_standardized=(0.3 if (setting == 0 and step <= 5500)
                                       or (setting == 1 and step > 5500) else 0.2))
                for step in range(0, 11001, 500)
            ]
            (folder / "update_curve.json").write_text(json.dumps(observations))
            run_ids.append(run_id)
        rows.append(dict(method="td_typed", batch_size=128, configuration=setting,
                         alpha=0.0001, beta=(setting + 1)*0.0001,
                         run_ids=run_ids))
    scored = [dict(row, score=_matched_score(tmp_path, row, 5500))
              for row in rows]
    assert scored[0]["score"] == 0.3
    assert scored[1]["score"] == 0.2
    winner = _choose(scored, "td_typed", 128, "matched_updates", 5500)
    assert winner["configuration"] == 1
    assert len(winner["run_ids"]) == 5
