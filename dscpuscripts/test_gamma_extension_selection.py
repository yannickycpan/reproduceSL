"""Synthetic end-to-end validation-selector accounting for all 320 Gamma candidates."""
from __future__ import annotations
import copy

import pytest
from fd001_gamma.common import atomic_json
from fd001_gamma.protocol import make_manifest as original_manifest
from fd001_gamma_ext.protocol import make_manifest as extension_manifest
import fd001_gamma.analysis as original_analysis
import fd001_gamma_ext.analysis as extension_analysis


def fake_records(tmp_path, mismatch=False):
    old, new = tmp_path / "old", tmp_path / "new"
    for root, manifest in ((old, original_manifest()), (new, extension_manifest())):
        for config in manifest:
            d = root / config["run_id"]
            d.mkdir(parents=True)
            variant, alpha = config["gamma_variant"], config["alpha"]
            # Best original representation is 0.0003. A new 0.00003
            # candidate should win globally after the grid extension.
            score = (.1 if variant == "representation" and alpha == .00003
                     else .2 if variant == "pairwise" and alpha == .0001
                     else .3 if variant == "representation" else .4)
            val = [dict(epoch=e, mse=score, mae=.5) for e in (80,90,100)]
            atomic_json(d / "status.json", dict(
                config=config, state="complete",
                identity=dict(data_sha256="data-identical" if not (
                    mismatch and root == new and config["seed"] == 0) else "different",
                    validation_sha256="validation-identical"),
                initial_mean_sha256=f"matched-seed-{config['seed']}"))
            atomic_json(d / "validation.json", val)
    return old, new


def test_extended_hp_selection_is_joint_not_separate(tmp_path,monkeypatch):
    old,new = fake_records(tmp_path)
    # This test exercises cross-protocol candidate aggregation, not the
    # source-sequence audit (which has its own original and extension calls).
    monkeypatch.setattr(original_analysis, "select", lambda path: dict(sentinel="audited"))
    selected, scores, shared = extension_analysis.audit_and_select(old, new)
    winners = {c["gamma_variant"]: c for c in selected["per_variant"]}
    assert len(scores) == 64  # 12 + 12 + 20 + 20
    assert winners["representation"]["origin"] == "extension"
    assert winners["representation"]["alpha"] == .00003
    assert winners["pairwise"]["origin"] == "extension"
    assert winners["pairwise"]["alpha"] == .0001
    assert winners["typed"]["origin"] == "original"
    assert winners["raw26"]["origin"] == "original"
    assert all(len(c["run_ids"]) == 5 for c in selected["per_variant"])
    assert shared["data_sha256"] == "data-identical"


def test_cross_campaign_data_mismatch_blocks_selection(tmp_path,monkeypatch):
    old,new = fake_records(tmp_path,mismatch=True)
    monkeypatch.setattr(original_analysis, "select", lambda path: dict(sentinel="audited"))
    with pytest.raises(ValueError, match="data mismatch"):
        extension_analysis.audit_and_select(old,new)


def test_old_scientific_modules_not_modified():
    from pathlib import Path
    from fd001_gamma.common import scientific_hash as original_hash
    from fd001_gamma_ext.runtime import scientific_hash as extended_hash
    assert len(original_hash()) == len(extended_hash()) == 64
    assert original_hash() != extended_hash()
    root = Path(__file__).resolve().parents[1]
    source = (root/"fd001_gamma_ext"/"protocol.py").read_text()
    assert "len(runs) == 140" in source
    source = (root/"fd001_gamma_ext"/"objectives.py").read_text()
    assert "math.sqrt(dimension)" in source
    assert "detach()" not in source.split("def loss_sum",1)[1].split("def operator_state",1)[0]
