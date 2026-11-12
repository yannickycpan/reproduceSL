"""Dependency-scoped application scientific identity (v3).

Never hash an entire mixed-dataset module merely because one dataset uses a
function within it. Hash the AST of the actually reachable dataset/method
branches, as well as every genuinely shared training/data/metric dependency.
FD001 keeps its original bytewise scientific hash in fd001_train.py.

The v3 scheme intentionally changes application code identities once. The
completed pre-refactor Electricity dataset remains readable through its
separately pinned, read-only historic analysis identity. Old v2 checkpoints
cannot be resumed as v3 merely by relabeling their hash.
"""
from __future__ import annotations

import ast
import copy
import hashlib
from pathlib import Path


HASH_SCHEME = "applications-scoped-v3"

# Only genuinely shared file-level scientific dependencies live here.
SHARED_FILES = (
    "dscpuscripts/common.py",
    "fulltd/train.py",
    "fulltd/sampling.py",
    "fulltd/metrics.py",
)
TRAIN_SHARED_NODES = (
    "atomic_torch", "rng_snapshot", "rng_restore",
    "_EvaluationInterrupted", "_InterruptibleView",
    "capture_validation", "train_candidate",
    "evaluation_validation_view", "build_objective",
)
TRAIN_DATASET_NODES = {
    "electricity": ("_ValidationView",),
    "metr_la": (),
}
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
                    "dscpuscripts/electricity_model.py",
                    "dscpuscripts/electricity_validation.py"),
    "metr_la": ("dscpuscripts/metr_la_sampling.py",
                "dscpuscripts/metr_la_model.py",
                "dscpuscripts/metr_la_validation.py"),
}


def _add(hasher, label, payload):
    tag = label.encode("utf-8")
    hasher.update(len(tag).to_bytes(4, "big"))
    hasher.update(tag)
    hasher.update(len(payload).to_bytes(8, "big"))
    hasher.update(payload)


def _file_bytes(root, name):
    return (root / name).read_bytes().replace(b"\r\n", b"\n")


def _lookup_constant(node, config):
    """Resolve only literal, reproducible selectors; all others remain unknown."""
    unknown = _UNKNOWN
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, (ast.Tuple, ast.List)):
        values = [_lookup_constant(v, config) for v in node.elts]
        return tuple(values) if unknown not in values else unknown
    if isinstance(node, ast.Name) and node.id == "method":
        return config["method"]
    if (isinstance(node, ast.Attribute) and node.attr == "dataset"
            and isinstance(node.value, ast.Name) and node.value.id == "self"):
        return config["dataset"]
    if (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name)
            and node.value.id == "config" and isinstance(node.slice, ast.Constant)):
        return config.get(node.slice.value, unknown)
    return unknown


_UNKNOWN = object()


def _choose_condition(test, config):
    """Prune only statically decidable dataset/method comparisons."""
    if isinstance(test, ast.BoolOp):
        parts = [_choose_condition(v, config) for v in test.values]
        if isinstance(test.op, ast.And):
            return False if False in parts else (True if all(x is True for x in parts) else None)
        if isinstance(test.op, ast.Or):
            return True if True in parts else (False if all(x is False for x in parts) else None)
    if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
        value = _choose_condition(test.operand, config)
        return None if value is None else not value
    if isinstance(test, ast.Compare) and len(test.ops) == 1:
        lhs = _lookup_constant(test.left, config)
        rhs = _lookup_constant(test.comparators[0], config)
        if lhs is _UNKNOWN or rhs is _UNKNOWN:
            return None
        op = test.ops[0]
        if isinstance(op, ast.Eq):
            return lhs == rhs
        if isinstance(op, ast.NotEq):
            return lhs != rhs
        if isinstance(op, ast.In):
            return lhs in rhs
        if isinstance(op, ast.NotIn):
            return lhs not in rhs
    return None


class _ScopeBranches(ast.NodeTransformer):
    def __init__(self, config):
        self.config = config

    def visit_FunctionDef(self, node):
        # After pruning a selected return branch, code below it is unreachable
        # for this particular dataset/method and must not widen its hash.
        body = []
        for statement in node.body:
            changed = self.visit(statement)
            parts = changed if isinstance(changed, list) else (
                [] if changed is None else [changed])
            body.extend(parts)
            if parts and isinstance(parts[-1], (ast.Return, ast.Raise)):
                break
        node.body = body
        return node

    def visit_ImportFrom(self, node):
        # Factory imports of unrelated baseline heads are not dependencies.
        if node.module and node.module.startswith("agents.fulltd_"):
            expected = METHOD_FILES.get(self.config["method"], "")
            return node if expected.removesuffix(".py").replace("/", ".") == node.module else None
        return node

    def visit_Assign(self, node):
        if (len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "classes"
                and isinstance(node.value, ast.Dict)):
            # The baseline dispatch table contains other methods' entries.
            # Retain only the entry actually used by the selected objective.
            choices = [(k, v) for k, v in zip(node.value.keys, node.value.values)
                       if isinstance(k, ast.Constant) and k.value == self.config["method"]]
            if choices:
                node.value.keys = [choices[0][0]]
                node.value.values = [choices[0][1]]
        return self.generic_visit(node)

    def visit_If(self, node):
        decision = _choose_condition(node.test, self.config)
        if decision is True:
            return self._selected(node.body)
        if decision is False:
            return self._selected(node.orelse)
        return self.generic_visit(node)

    def _selected(self, branch):
        result = []
        for statement in branch:
            changed = self.visit(statement)
            if isinstance(changed, list):
                result.extend(changed)
            elif changed is not None:
                result.append(changed)
        return result


def _selected_nodes(root, path, names, config):
    """Canonical AST, with irrelevant dataset/algorithm branches removed."""
    source = _file_bytes(root, path).decode("utf-8")
    tree = ast.parse(source, filename=path)
    found = {node.name: node for node in tree.body
             if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
    if missing := set(names) - found.keys():
        raise ValueError(f"Missing scientific code nodes in {path}: {sorted(missing)}")
    scoped = _ScopeBranches(config)
    for name in names:
        selected = scoped.visit(copy.deepcopy(found[name]))
        if selected is None or isinstance(selected, list):
            raise ValueError(f"Invalid scientific source scope for {path}:{name}")
        yield name, ast.dump(selected, include_attributes=False).encode("utf-8")


def code_hash(config, *, root=None):
    """Stable hash over actual per-candidate scientific dependencies.

    Hyperparameters, dataset fractions, lags and checkpoint budgets are
    separately guarded by the exact config_sha256 saved with each run.
    """
    dataset, method = config["dataset"], config["method"]
    if dataset not in MODEL_NODES or method not in OBJECTIVE_NODES:
        raise ValueError("Unsupported application dataset/algorithm identity")
    if config["backbone"] != ("patchtst" if dataset == "electricity" else "graph_wavenet"):
        raise ValueError("Unsupported application mean backbone")
    root = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    h = hashlib.sha256()
    _add(h, "hash-scheme", HASH_SCHEME.encode("ascii"))
    _add(h, "dataset-method", f"{dataset}:{method}".encode("ascii"))
    for name in SHARED_FILES + DATASET_FILES[dataset]:
        _add(h, name, _file_bytes(root, name))

    scoped_sources = (
        ("dscpuscripts/application_train.py",
         TRAIN_SHARED_NODES + TRAIN_DATASET_NODES[dataset]),
        ("dscpuscripts/application_protocol.py", ("update_budgets", "evaluation_steps")),
        ("agents/fulltd_objectives.py", ("Objective", "masked_residual")),
        ("fulltd/models.py", MODEL_NODES[dataset]),
        ("fulltd/data.py", DATA_COMMON_NODES + DATA_SPECIFIC_NODES[dataset]),
        ("dscpuscripts/application_objectives.py",
         ("make_objective",) + OBJECTIVE_NODES[method]),
    )
    for path, names in scoped_sources:
        for node, payload in _selected_nodes(root, path, names, config):
            _add(h, path + "::" + node, payload)
    if method in METHOD_FILES:
        name = METHOD_FILES[method]
        _add(h, name, _file_bytes(root, name))
    if method in ("td_pairwise", "td_pairwise_mae"):
        for node, payload in _selected_nodes(
                root, "dscpuscripts/fd001_objectives.py", ("PairwiseGammaHead",), config):
            _add(h, "dscpuscripts/fd001_objectives.py::" + node, payload)
    return h.hexdigest()
