"""Simple sibling data/results/outputs roots, with METR-LA fractions in the name."""
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
    override = os.environ.get(name)
    if dataset == "metr_la":
        from .metr_la_sampling import SOURCE_FRACTION, fraction_tag
        tag = fraction_tag(SOURCE_FRACTION)
        if override:
            root = Path(override).expanduser().resolve()
            # An inherited old environment can otherwise accidentally reuse
            # the former untagged directory for the new 10%-sampling campaign.
            if root.name == "metr_la_results":
                raise ValueError(
                    "Refusing legacy METR-LA result root; unset FULLTD_METR_LA_ROOT "
                    "to select the fraction-specific results directory")
            if root.name.startswith("metr_la_f") and root.name.endswith("_results"):
                expected = f"metr_la_{tag}_results"
                if root.name != expected:
                    raise ValueError(f"Result root fraction conflicts with {tag}: {root}")
            return root
        return (base_root() / f"metr_la_{tag}_results").resolve()
    if override:
        return Path(override).expanduser().resolve()
    return (base_root() / f"{dataset}_results").resolve()


def paths(dataset, output_kind=None):
    """Return the processed-data directory and sibling results/output folders."""
    root = dataset_root(dataset)
    data = Path(os.environ.get("FULLTD_DATA_DIR", base_root() / "data")).expanduser().resolve()
    if dataset == "metr_la":
        # Results: metr_la_f10pct_results; outputs: metr_la_f10pct_outputs.
        suffix = root.name[:-len("_results")] if root.name.endswith("_results") else "metr_la"
        out = root.parent / (suffix + "_outputs")
    else:
        out = root.parent / f"{dataset}_outputs"
    if output_kind:
        if not (output_kind.replace("_", "").isalnum() and "/" not in output_kind):
            raise ValueError("Invalid output directory kind")
        out = out / output_kind
    return data, root, out


def guard_unmigrated_runs(results_dir, configs, previous_roots):
    """Protect checkpoints in historical nested locations; no file mutation."""
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
