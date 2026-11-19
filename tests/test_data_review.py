"""Regression tests for the second independent data-integrity review."""
import json
from pathlib import Path

import numpy as np
import pytest

from fulltd.data import (prepare_fd001, load_dataset, verify_dataset,
                         metadata_fingerprint, DATA_VERSION)


def _engines(count):
    rows = []
    for engine in range(1, count + 1):
        for cycle in range(1, 51 + engine):
            rows.append([engine, cycle, *(
                np.arange(24, dtype=float) + cycle * .1 + engine)])
    return np.asarray(rows, dtype=np.float64)


@pytest.fixture
def prepared(tmp_path):
    prepare_fd001(tmp_path, _engines(10), _engines(3), np.array([10., 20., 30.]))
    return tmp_path


@pytest.mark.parametrize("field", ["mean", "scale", "fit_engine_ids"])
def test_mutated_scalers_or_split_cannot_reuse_stored_identity(prepared, field):
    path = prepared/"processed/fd001/metadata.json"
    metadata = json.loads(path.read_text())
    if field == "fit_engine_ids":
        metadata[field] = metadata[field][:-1]
    else:
        metadata[field] += 123
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        verify_dataset("fd001", prepared)
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        load_dataset("fd001", prepared, "train")


def test_split_sidecar_is_verified_on_loading_its_split(prepared):
    path = prepared/"processed/fd001/train/split.json"
    sidecar = json.loads(path.read_text())
    sidecar["engine_ids"] = [999]
    path.write_text(json.dumps(sidecar))
    with pytest.raises(ValueError, match="missing or modified"):
        load_dataset("fd001", prepared, "train")
    with pytest.raises(ValueError, match="missing or modified"):
        verify_dataset("fd001", prepared)


def test_integrity_check_does_not_open_test_payloads_during_training(prepared):
    for path in (prepared/"processed/fd001/test").glob("*.npy"):
        path.unlink()
    fitting = load_dataset("fd001", prepared, "train")
    assert fitting.get_batch([0])["mask"].all()
    with pytest.raises(ValueError, match="missing or modified"):
        verify_dataset("fd001", prepared)


def test_fingerprint_excludes_only_acquisition_date(prepared):
    path = prepared/"processed/fd001/metadata.json"
    metadata = json.loads(path.read_text())
    assert metadata["version"] == DATA_VERSION
    assert metadata_fingerprint(metadata) == metadata["fingerprint"]
    metadata["acquisition"] = [{"source_url": "https://example.test/public-data", "retrieved_utc": "old"}]
    before = metadata_fingerprint(metadata)
    metadata["acquisition"][0]["retrieved_utc"] = "new"
    assert metadata_fingerprint(metadata) == before
    metadata["acquisition"][0]["source_url"] += "-changed"
    assert metadata_fingerprint(metadata) != before


def test_old_metadata_version_fails_explicitly(prepared):
    path = prepared/"processed/fd001/metadata.json"
    metadata = json.loads(path.read_text())
    metadata["version"] = "td-v7-data-1"
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="rerun prepare"):
        load_dataset("fd001", prepared, "train")


def test_split_metadata_has_portable_utf8_lf_bytes(prepared):
    for split in ("train", "val", "test"):
        payload = (prepared/f"processed/fd001/{split}/split.json").read_bytes()
        assert b"\r\n" not in payload
        assert payload.endswith(b"\n")
        json.loads(payload.decode("utf-8"))
