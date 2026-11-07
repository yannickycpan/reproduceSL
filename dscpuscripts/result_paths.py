"""Consistent dataset-specific results layout across direct CPU/HPC environments.

Each dataset owns one <dataset>_results directory. Different algorithms use
non-overlapping canonical run IDs directly in <dataset>_results/; analysis
outputs have separate subdirectories to avoid overwriting selections.
No scientific code depends on this operational module.
"""
from __future__ import annotations
import os
from pathlib import Path

DATASETS = ("fd001", "electricity", "metr_la")


def base_root():
    return Path(os.environ.get("FULLTD_RESULTS_ROOT",
                Path.home() / "scratch" / "fulltd")).expanduser().resolve()


def dataset_root(dataset):
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset for results path: {dataset}")
    name = "FULLTD_" + dataset.upper() + "_ROOT"
    return Path(os.environ.get(name, base_root() / f"{dataset}_results")).expanduser().resolve()


def paths(dataset, output_kind=None):
    """Return (dataset-independent data directory, results, analysis outputs).

    Canonical run directories live directly in <dataset>_results/ (flat),
    as on the user's server. Only analysis output files live in outputs/.
    The frozen run IDs are never renamed.
    """
    root = dataset_root(dataset)
    data = Path(os.environ.get("FULLTD_DATA_DIR", base_root() / "data")).expanduser().resolve()
    # New analyses live beside the data/results folder, never inside it.
    # Historical outputs under <dataset>_results/outputs are left untouched.
    out = root.parent / f"{dataset}_outputs"
    if output_kind:
        if not (output_kind.replace("_", "").isalnum() and "/" not in output_kind):
            raise ValueError("Invalid output directory kind")
        out = out / output_kind
    return data, root, out


def configure_environment():
    """Populate old scientific APIs' path env variables without changing their hashes.

    The frozen Gamma v1 and Gamma extension runtimes retain their original
    historical default code for checkpoint provenance. Operational entrypoints
    call this before using those APIs, so every machine obtains dataset_results.
    Explicit user environment overrides always take precedence.
    """
    data, results, gamma_out = paths("fd001")
    extension_out = gamma_out / "extension"
    values = dict(FULLTD_DATA_DIR=str(data),
                  FULLTD_FD001_GAMMA_RESULTS_DIR=str(results),
                  FULLTD_FD001_GAMMA_OUTPUT_DIR=str(gamma_out),
                  FULLTD_FD001_GAMMA_EXT_RESULTS_DIR=str(results),
                  FULLTD_FD001_GAMMA_EXT_OUTPUT_DIR=str(extension_out))
    for name, value in values.items():
        os.environ.setdefault(name, value)
    return values


def guard_unmigrated_runs(results_dir, configs, previous_roots):
    """Refuse retraining when recorded runs exist in a former result directory.

    The destination may already hold completed or interrupted runs; the guard
    only checks for *missing* run IDs with evidence in another known location.
    It never moves, deletes or overwrites any result or treats failures as
    missing. Run IDs are assumed to be prevalidated canonical names.
    """
    results = Path(results_dir).expanduser().resolve()
    for previous in previous_roots:
        old = Path(previous).expanduser().resolve()
        if old == results or not old.is_dir():
            continue
        unmigrated = []
        for config in configs:
            run_id = config["run_id"]
            source = old / run_id
            destination = results / run_id
            markers = ("status.json", "recovery.pt", "final.pt")
            if (any((source / name).exists() for name in markers)
                    and not any((destination / name).exists() for name in markers)):
                unmigrated.append(run_id)
        if unmigrated:
            raise RuntimeError(
                f"{len(unmigrated)} existing run records still live at {old}, "
                f"but are absent from {results}. Copy the original run "
                "directories without overwriting checkpoints before rerunning. "
                f"Example run: {unmigrated[0]}.")
