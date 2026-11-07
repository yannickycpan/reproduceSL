"""Consistent dataset-specific results layout across direct CPU/HPC environments.

Each dataset owns one <dataset>_results directory. Different algorithms use
non-overlapping canonical run IDs in the same results/ directory; analysis
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

    All methods for a dataset share results/ but their output folders differ.
    Existing FD001 Gamma-v1 outputs intentionally stay directly in outputs/.
    """
    root = dataset_root(dataset)
    data = Path(os.environ.get("FULLTD_DATA_DIR", base_root() / "data")).expanduser().resolve()
    out = root / "outputs"
    if output_kind:
        if not (output_kind.replace("_", "").isalnum() and "/" not in output_kind):
            raise ValueError("Invalid output directory kind")
        out = out / output_kind
    return data, root / "results", out


def configure_environment():
    """Populate old scientific APIs' path env variables without changing their hashes.

    The frozen Gamma v1 and Gamma extension runtimes retain their original
    historical default code for checkpoint provenance. Operational entrypoints
    call this before using those APIs, so every machine obtains dataset_results.
    Explicit user environment overrides always take precedence.
    """
    data, results, gamma_out = paths("fd001")
    extension_out = dataset_root("fd001") / "outputs" / "extension"
    values = dict(FULLTD_DATA_DIR=str(data),
                  FULLTD_FD001_GAMMA_RESULTS_DIR=str(results),
                  FULLTD_FD001_GAMMA_OUTPUT_DIR=str(gamma_out),
                  FULLTD_FD001_GAMMA_EXT_RESULTS_DIR=str(results),
                  FULLTD_FD001_GAMMA_EXT_OUTPUT_DIR=str(extension_out))
    for name, value in values.items():
        os.environ.setdefault(name, value)
    return values
