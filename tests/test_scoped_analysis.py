"""Combined dataset selection and recorded-only MSE/MAE validation curves."""
import numpy as np
import pytest

from fulltd import analysis
from fulltd.protocol import make_manifest
from fulltd.sampling import sequence_hash, sources_for_round
from fulltd.validation import FULL_POLICY, indices_hash
from test_fulltd_analysis import fixture_accounting


@pytest.fixture(scope="module")
def combined_campaign(tmp_path_factory):
    root = tmp_path_factory.mktemp("combined-campaign")
    manifest = make_manifest()
    for config in manifest:
        name = config["dataset"]
        if name == "electricity":
            continue
        directory = root / config["run_id"]
        directory.mkdir()
        status, rounds = fixture_accounting(config)
        for record in rounds:
            record["source_sha256"] = sequence_hash(sources_for_round(name, 9, config["seed"], record["epoch"]))
        analysis.write_json(directory / "status.json", status)
        analysis.write_json(directory / "rounds.json", rounds)
        records = [dict(epoch=epoch, mse=config["configuration"] + 1 + .1 * config["seed"],
                        mae=7 - config["configuration"] + .2 * config["seed"],
                        validation_subset=dict(policy=FULL_POLICY, epoch=epoch, population_size=11,
                                               sampled_size=11, indices_sha256=indices_hash(np.arange(11))))
                   for epoch in range(10, 101, 10)]
        analysis.write_json(directory / "validation.json", records)
        (directory / "final.pt").touch()
    return root, manifest


def test_combined_scope_selects_330_candidates_without_reading_electricity(combined_campaign, tmp_path, monkeypatch):
    root, manifest = combined_campaign
    original_read = analysis.read_json
    reads = []

    def checked_read(path):
        assert not any(part.startswith("electricity_") for part in path.parts)
        reads.append(path)
        return original_read(path)

    monkeypatch.setattr(analysis, "read_json", checked_read)
    selected = analysis.select_candidates(manifest, root, tmp_path, "metr_la_fd001")
    assert len(manifest) == 630
    assert selected["dataset_scope"] == "metr_la_fd001"
    assert len(selected["groups"]) == 11
    assert {g["dataset"] for g in selected["groups"]} == {"metr_la", "fd001"}
    assert all(g["configuration"] == 0 for g in selected["groups"])
    assert reads and not list(root.glob("electricity_*"))
    assert original_read(tmp_path / "campaign_audit.json")["runs"] == 330
    assert len(analysis.read_csv(tmp_path / "validation_records.csv")) == 3300
    selected_rows = analysis.read_csv(tmp_path / "selected_runs.csv")
    assert len(selected_rows) == 55
    assert all(row[f"validation_mae_{epoch}"] for row in selected_rows for epoch in (80, 90, 100))
    # The separate all-data release still requires Electricity evidence.
    with pytest.raises(ValueError, match="Missing candidate status"):
        analysis.select_candidates(manifest, root, tmp_path / "all", "all")


def test_one_incomplete_combined_candidate_blocks_selection(combined_campaign, tmp_path):
    root, manifest = combined_campaign
    config = next(c for c in manifest if c["dataset"] == "metr_la")
    path = root / config["run_id"] / "status.json"
    original = analysis.read_json(path)
    try:
        analysis.write_json(path, dict(original, state="running"))
        with pytest.raises(ValueError, match="Operationally incomplete"):
            analysis.select_candidates(manifest, root, tmp_path, "metr_la_fd001")
        assert not (tmp_path / "selections.json").exists()
    finally:
        analysis.write_json(path, original)


def test_mae_changes_do_not_change_the_mse_selector(combined_campaign, tmp_path):
    root, manifest = combined_campaign
    baseline = analysis.select_candidates(manifest, root, tmp_path / "before", "metr_la_fd001", write_outputs=False)
    originals = {}
    try:
        for config in manifest:
            if config["dataset"] == "electricity":
                continue
            path = root / config["run_id"] / "validation.json"
            records = analysis.read_json(path)
            originals[path] = records
            analysis.write_json(path, [dict(record, mae=1e20 if config["configuration"] == 0 else 0.)
                                       for record in records])
        changed = analysis.select_candidates(manifest, root, tmp_path / "after", "metr_la_fd001", write_outputs=False)
        fields = ("group_id", "configuration", "alpha", "beta", "score", "run_ids")
        assert [{k: g[k] for k in fields} for g in changed["groups"]] == [
            {k: g[k] for k in fields} for g in baseline["groups"]]
    finally:
        for path, records in originals.items():
            analysis.write_json(path, records)


def curve_records():
    return [dict(run_id=f"fixture_s{seed}", dataset="fd001", backbone="tcn", method="mse",
                 batch_size=256, seed=seed, epoch=epoch, mse=seed + epoch / 10,
                 mae=10 * seed + epoch / 100)
            for seed in range(5) for epoch in range(10, 101, 10)]


def prepare_curves(directory, records):
    analysis.write_csv(directory / "summary_metrics.csv", [], fields=["split"])
    analysis.write_csv(directory / "validation_records.csv", records)
    analysis.write_json(directory / "selections.json", {
        "dataset_scope": "fd001", "groups": [{"run_ids": [f"fixture_s{s}" for s in range(5)]}]})


def test_mse_and_mae_curves_use_their_recorded_values_and_identify_metric(tmp_path):
    tables, figures = tmp_path / "tables", tmp_path / "figures"
    prepare_curves(tables, curve_records())
    analysis.create_figures(tables, figures)
    for metric, stem, expected in (("mse", "validation_fd001_tcn", 3.),
                                   ("mae", "validation_mae_fd001_tcn", 20.1)):
        rows = analysis.read_csv(figures / f"{stem}.csv")
        assert len(rows) == 10 and {r["metric"] for r in rows} == {metric}
        assert [int(r["epoch"]) for r in rows] == list(range(10, 101, 10))
        assert float(rows[0]["mean"]) == pytest.approx(expected)
        assert (figures / f"{stem}.pdf").read_bytes().startswith(b"%PDF")
        assert "<svg" in (figures / f"{stem}.svg").read_text(encoding="utf-8")


@pytest.mark.parametrize("missing", ["all", "one"])
def test_legacy_records_skip_mae_curve_explicitly_without_inferred_values(tmp_path, missing):
    records = curve_records()
    for record in records if missing == "all" else records[:1]:
        record.pop("mae")
    tables, figures = tmp_path / "tables", tmp_path / "figures"
    prepare_curves(tables, records)
    with pytest.warns(RuntimeWarning, match="Skipping MAE validation curve.*legacy validation records"):
        analysis.create_figures(tables, figures)
    assert (figures / "validation_fd001_tcn.csv").exists()
    assert not list(figures.glob("validation_mae_*"))
