"""Consistent dataset-specific results layout across direct CPU/HPC environments.

Every dataset uses one sibling results/outputs pair. Canonical run IDs
are stored directly inside <dataset>_results, never in nested results folders.
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
    The run IDs are set by each dataset's current scientific protocol.
    """
    root = dataset_root(dataset)
    data = Path(os.environ.get("FULLTD_DATA_DIR", base_root() / "data")).expanduser().resolve()
    # Analysis is a sibling of results, not nested inside training checkpoints.
    out = root.parent / f"{dataset}_outputs"
    if output_kind:
        if not (output_kind.replace("_", "").isalnum() and "/" not in output_kind):
            raise ValueError("Invalid output directory kind")
        out = out / output_kind
    return data, root, out


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
