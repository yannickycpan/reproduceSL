"""Safely import disjoint, finished Gamma run directories from another server.

First transfer the other server's results directory into a local staging
directory (rsync or scp). Import only canonical FD001 Gamma run directories.
No old fulltd files, shared subdirectories, or local candidate results are
overwritten. Incoming result/data/scientific-code identities must match.
"""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import shutil
import sys
import uuid

from fd001_gamma.common import (campaign_paths, data_identity, file_lock,
                                read_json, scientific_hash)
from fd001_gamma.protocol import digest, make_manifest


def tree_digest(directory: Path) -> str:
    h = hashlib.sha256()
    for file in sorted(directory.rglob("*")):
        if file.is_symlink():
            raise ValueError(f"Symlink in result directory not accepted: {file}")
        if not file.is_file():
            continue
        h.update(str(file.relative_to(directory)).encode("utf-8"))
        with file.open("rb") as stream:
            for part in iter(lambda: stream.read(1024 * 1024), b""):
                h.update(part)
    return h.hexdigest()


def validate_incoming(directory, config, code_hash, train_hash, val_hash):
    status_file = directory / "status.json"
    if not status_file.is_file():
        raise ValueError(f"No status.json in {directory}")
    status = read_json(status_file)
    ident = status.get("identity") or {}
    if status.get("config") != config or status.get("protocol_id") != config["protocol_id"]:
        raise ValueError(f"Incorrect configuration/protocol in {directory}")
    if (ident.get("code_sha256") != code_hash
            or ident.get("config_sha256") != digest(config)
            or ident.get("data_sha256") != train_hash
            or ident.get("validation_sha256") != val_hash):
        raise ValueError(f"Different scientific code or processed FD001 data: {directory}")
    state = status.get("state")
    if state == "complete":
        if (status.get("epochs_completed") != 100
                or not all((directory / x).is_file() for x in
                           ("final.pt", "validation.json", "rounds.json"))):
            raise ValueError(f"Completed run is missing evidence: {directory}")
    elif state != "numerical_failure":
        raise ValueError(f"Only terminal runs can be imported: {directory} ({state})")
    for path in directory.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Refusing symlink in incoming run: {path}")
    return state


def import_runs(source: Path, target: Path, *, dry_run=False):
    source, target = source.expanduser().resolve(), target.expanduser().resolve()
    if not source.is_dir() or source == target:
        raise ValueError("Source must be an existing separate local staging directory")
    from fulltd.data import load_dataset, verify_dataset
    from fulltd.sampling import training_universe
    from fd001_gamma.protocol import LAGS
    data, _, _ = campaign_paths()
    verify_dataset("fd001", data)
    train, _, _ = training_universe(load_dataset("fd001", data, "train"), LAGS)
    val = load_dataset("fd001", data, "val")
    train_hash, val_hash = data_identity(train), data_identity(val)
    expected = {c["run_id"]: c for c in make_manifest()}
    candidates = sorted((p for p in source.iterdir() if p.is_dir() and p.name in expected),
                        key=lambda p: p.name)
    if not candidates:
        raise ValueError(f"No canonical FD001 Gamma run directories under {source}")
    code_hash = scientific_hash()
    print(f"Import source: {source}; target: {target}; candidates: {len(candidates)}")
    imported, identical = [], []
    validated = []
    for candidate in candidates:
        config = expected[candidate.name]
        validate_incoming(candidate, config, code_hash, train_hash, val_hash)
        dest = target / candidate.name
        if dest.exists():
            if not dest.is_dir() or tree_digest(dest) != tree_digest(candidate):
                raise RuntimeError(f"Local result already exists and differs; refusing overwrite: {dest}")
            identical.append(candidate.name)
        else:
            validated.append(candidate)
    print(f"Ready to import: {len(validated)}; already byte-identical: {len(identical)}")
    if dry_run:
        return imported, identical
    target.mkdir(parents=True, exist_ok=True)
    with file_lock(target / ".direct_import.lock"):
        for candidate in validated:
            dest = target / candidate.name
            if dest.exists():
                raise RuntimeError(f"Destination appeared during import; nothing overwritten: {dest}")
            tmp = target / ("." + candidate.name + ".import-" + uuid.uuid4().hex)
            try:
                shutil.copytree(candidate, tmp, symlinks=False)
                if tree_digest(tmp) != tree_digest(candidate):
                    raise IOError(f"Staged copy checksum mismatch: {candidate}")
                if dest.exists():
                    raise RuntimeError(f"Unexpected concurrent result at {dest}")
                tmp.rename(dest)
                imported.append(candidate.name)
                print(f"Imported {candidate.name}", flush=True)
            finally:
                if tmp.exists():
                    shutil.rmtree(tmp)
    print(f"Import done: {len(imported)} added; {len(identical)} identical skipped")
    return imported, identical


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True,
                        help="Local staging directory containing another server's results/ run dirs")
    parser.add_argument("--destination", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    _, default_results, _ = campaign_paths()
    import_runs(args.source, args.destination or default_results, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
