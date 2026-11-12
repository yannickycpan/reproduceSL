"""Scientific checkpoint identity scoped to the selected dataset and method.

Dataset-only sampler/model changes should not invalidate another dataset.
Objective-only changes should not invalidate unrelated algorithms. Genuinely
shared source changes must invalidate all consumers; FD001 remains independent.
"""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path

SHARED_FILES = (
    "dscpuscripts/application_identity.py",
    "dscpuscripts/application_train.py",
    "dscpuscripts/common.py",
    "fulltd/train.py",
    "fulltd/sampling.py",
    "fulltd/metrics.py",
    "agents/fulltd_objectives.py",
)
MODEL_NODES = {
    "electricity": ("initialization_seed", "initialize_modules",
                    "ResidualAttentionLayer", "PatchTST"),
    "metr_la": ("initialization_seed", "initialize_modules",
                "row_normalized_adjacency", "DiffusionConvolution", "GraphWaveNet"),
}
DATA_COMMON_NODES = ("canonical_name", "sha256", "_save_json", "_save_split",
                     "_time_windows", "_finish_metadata", "metadata_fingerprint",
                     "_read_metadata", "verify_dataset", "LazyDataset",
                     "load_dataset")
DATA_SPECIFIC_NODES = {
    "electricity": ("aggregate_electricity", "prepare_electricity"),
    "metr_la": ("_load_graph", "prepare_metr_la"),
}
OBJECTIVE_NODES = {
    "mse": (),
    "gaussian_nll": (),
    "mae": (),
    "td_typed": ("check_pairs", "_rates", "pair_delta", "TypedGammaTD"),
    "td_pairwise": ("check_pairs", "_rates", "pair_delta", "PairwiseGammaTD"),
    "td_pairwise_mae": ("check_pairs", "_rates", "pair_delta", "PairwiseGammaTD"),
}
METHOD_FILES = {
    "mse": "agents/fulltd_mse.py",
    "gaussian_nll": "agents/fulltd_gaussian_nll.py",
    "mae": "agents/fulltd_mae.py",
}
DATASET_FILES = {
    "electricity": ("dscpuscripts/electricity_sampling.py",
                    "dscpuscripts/electricity_model.py"),
    "metr_la": ("dscpuscripts/metr_la_sampling.py",
                "dscpuscripts/metr_la_model.py"),
}


def _add(hasher, label, payload):
    tag = label.encode("utf-8")
    hasher.update(len(tag).to_bytes(4, "big"))
    hasher.update(tag)
    hasher.update(len(payload).to_bytes(8, "big"))
    hasher.update(payload)


def _file_bytes(root, name):
    return (root / name).read_bytes().replace(b"\r\n", b"\n")


def _selected_nodes(root, name, requested):
    source = _file_bytes(root, name).decode("utf-8")
    tree = ast.parse(source, filename=name)
    nodes = {n.name: n for n in tree.body
             if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
    if any(n not in nodes for n in requested):
        raise ValueError(f"Missing dependency in {name}: {set(requested) - set(nodes)}")
    return [(n, ast.get_source_segment(source, nodes[n]).encode("utf-8"))
            for n in requested]


def code_hash(config, *, root=None):
    """Hash exactly the selected dependencies; config_sha256 separately guards HPs."""
    dataset, method = config["dataset"], config["method"]
    if dataset not in MODEL_NODES or method not in OBJECTIVE_NODES:
        raise ValueError("Unsupported dataset/algorithm identity scope")
    if config["backbone"] != ("patchtst" if dataset == "electricity" else "graph_wavenet"):
        raise ValueError("Unsupported application backbone")
    root = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    h = hashlib.sha256()
    _add(h, "identity-schema", b"applications-scoped-v2")
    _add(h, "dataset-method", f"{dataset}:{method}".encode())
    for name in SHARED_FILES + DATASET_FILES[dataset]:
        _add(h, name, _file_bytes(root, name))
    # Global budget/cadence logic is genuinely shared; dataset-specific laws
    # are in electricity_sampling.py / metr_la_sampling.py above.
    for node, payload in _selected_nodes(
            root, "dscpuscripts/application_protocol.py", ("update_budgets", "evaluation_steps")):
        _add(h, "dscpuscripts/application_protocol.py::" + node, payload)
    for path, names in (
            ("fulltd/models.py", MODEL_NODES[dataset]),
            ("fulltd/data.py", DATA_COMMON_NODES + DATA_SPECIFIC_NODES[dataset]),
            ("dscpuscripts/application_objectives.py",
             ("make_objective",) + OBJECTIVE_NODES[method])):
        for node, payload in _selected_nodes(root, path, names):
            _add(h, path + "::" + node, payload)
    if method in METHOD_FILES:
        name = METHOD_FILES[method]
        _add(h, name, _file_bytes(root, name))
    if method in ("td_pairwise", "td_pairwise_mae"):
        for node, payload in _selected_nodes(
                root, "dscpuscripts/fd001_objectives.py", ("PairwiseGammaHead",)):
            _add(h, "dscpuscripts/fd001_objectives.py::" + node, payload)
    return h.hexdigest()
