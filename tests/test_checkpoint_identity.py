"""Regression checks for the fresh FD001 lag sweep and scoped application hashes.

The three-lag campaign intentionally changes FD001 and shared sampler code.
It must have a NEW scientific checkpoint identity; old checkpoints cannot be
silently resumed. Application hashes remain scoped by dataset and method.
"""
from pathlib import Path
import hashlib

from dscpuscripts import application_train
from dscpuscripts import fd001_train


def test_fd001_scientific_identity_is_complete_reproducible_and_new():
    from dscpuscripts.fd001_protocol import LAG_SETS, make_manifest

    root = Path(__file__).resolve().parents[1]
    h = hashlib.sha256()
    for name in fd001_train.SCIENTIFIC_SOURCES:
        raw = (root / name).read_bytes().replace(b"\r\n", b"\n")
        assert raw, name
        h.update(name.encode())
        h.update(raw)
    assert fd001_train.code_hash() == h.hexdigest()
    assert fd001_train.code_hash() != (
        "c6820c53e26b4ab47be164cb0e5e3a4c62a8f865e01a8b918148980ea840ef21")
    configs = make_manifest()
    assert len(configs) == 1600
    assert {tuple(c["lags"]) for c in configs if c["method"].startswith("td_")} == set(LAG_SETS)
    assert len({c["run_id"] for c in configs}) == len(configs)


def test_application_hash_scopes_dataset_algorithm_and_common_code(tmp_path):
    import shutil
    from dscpuscripts.application_identity import (
        code_hash, SHARED_FILES, DATASET_FILES, METHOD_FILES)
    from dscpuscripts.application_protocol import make_manifest

    root = Path(__file__).resolve().parents[1]
    files = set(SHARED_FILES)
    files.update(name for paths in DATASET_FILES.values() for name in paths)
    files.update(METHOD_FILES.values())
    files.update(("fulltd/models.py", "fulltd/data.py",
                  "dscpuscripts/application_train.py", "agents/fulltd_objectives.py",
                  "dscpuscripts/application_protocol.py",
                  "dscpuscripts/application_objectives.py",
                  "dscpuscripts/fd001_objectives.py"))
    for name in sorted(files):
        dest = tmp_path / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / name, dest)

    manifest = make_manifest()
    electricity = next(c for c in manifest
                       if c["dataset"] == "electricity" and c["method"] == "mse")
    metr_mse = next(c for c in manifest
                    if c["dataset"] == "metr_la" and c["method"] == "mse")
    metr_pair = next(c for c in manifest
                     if c["dataset"] == "metr_la" and c["method"] == "td_pairwise")

    old_elec = code_hash(electricity, root=tmp_path)
    old_metr = code_hash(metr_mse, root=tmp_path)

    # Editing only METR-LA sampling cannot invalidate Electricity.
    sampling = tmp_path / "dscpuscripts/metr_la_sampling.py"
    sampling.write_text(sampling.read_text() + "\n# METR-only change\n")
    assert code_hash(electricity, root=tmp_path) == old_elec
    assert code_hash(metr_mse, root=tmp_path) != old_metr

    # A change to the pairwise objective must not invalidate ordinary MSE.
    objective = tmp_path / "dscpuscripts/application_objectives.py"
    original = objective.read_text()
    assert 'gamma_variant = "pairwise"' in original
    mse_before = code_hash(metr_mse, root=tmp_path)
    pair_before = code_hash(metr_pair, root=tmp_path)
    objective.write_text(original.replace('gamma_variant = "pairwise"',
                                          'gamma_variant = "pairwise-changed"'))
    assert code_hash(metr_mse, root=tmp_path) == mse_before
    assert code_hash(metr_pair, root=tmp_path) != pair_before

    # Shared evaluation code does invalidate genuinely dependent models.
    metrics = tmp_path / "fulltd/metrics.py"
    metrics.write_text(metrics.read_text() + "\n# Shared-code change\n")
    assert code_hash(electricity, root=tmp_path) != old_elec


def test_metr_la_model_change_does_not_invalidate_electricity(tmp_path):
    import shutil
    from dscpuscripts.application_identity import (
        code_hash, SHARED_FILES, DATASET_FILES, METHOD_FILES)
    from dscpuscripts.application_protocol import make_manifest

    root = Path(__file__).resolve().parents[1]
    paths = set(SHARED_FILES)
    paths.update(name for names in DATASET_FILES.values() for name in names)
    paths.update(METHOD_FILES.values())
    paths.update(("fulltd/models.py", "fulltd/data.py",
                  "dscpuscripts/application_train.py", "agents/fulltd_objectives.py",
                  "dscpuscripts/application_protocol.py",
                  "dscpuscripts/application_objectives.py",
                  "dscpuscripts/fd001_objectives.py"))
    for name in paths:
        dest = tmp_path / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / name, dest)
    manifest = make_manifest()
    electricity = next(c for c in manifest
                       if c["dataset"] == "electricity" and c["method"] == "mse")
    metr = next(c for c in manifest
                if c["dataset"] == "metr_la" and c["method"] == "mse")
    e0, m0 = code_hash(electricity, root=tmp_path), code_hash(metr, root=tmp_path)
    models = tmp_path / "fulltd/models.py"
    content = models.read_text()
    assert "self.feature_dim = 192" in content
    models.write_text(content.replace("self.feature_dim = 192",
                                      "self.feature_dim = 191"))
    assert code_hash(electricity, root=tmp_path) == e0
    assert code_hash(metr, root=tmp_path) != m0


def test_mixed_application_module_is_hashed_by_reachable_dataset_and_method(
        tmp_path):
    """Changing a METR-LA-only branch must not invalidate Electricity."""
    import shutil
    from dscpuscripts.application_identity import (
        code_hash, SHARED_FILES, DATASET_FILES, METHOD_FILES)
    from dscpuscripts.application_protocol import make_manifest

    source_root = Path(__file__).resolve().parents[1]
    paths = set(SHARED_FILES)
    paths.update(x for names in DATASET_FILES.values() for x in names)
    paths.update(METHOD_FILES.values())
    paths.update(("dscpuscripts/application_train.py",
                  "dscpuscripts/application_protocol.py",
                  "dscpuscripts/application_objectives.py",
                  "dscpuscripts/fd001_objectives.py",
                  "agents/fulltd_objectives.py",
                  "fulltd/data.py", "fulltd/models.py"))
    for name in paths:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_root / name, target)

    manifest = make_manifest()
    electricity = next(c for c in manifest if c["dataset"] == "electricity"
                       and c["method"] == "mse")
    metr = next(c for c in manifest if c["dataset"] == "metr_la"
                and c["method"] == "mse")
    pair = next(c for c in manifest if c["dataset"] == "metr_la"
                and c["method"] == "td_pairwise")
    typed = next(c for c in manifest if c["dataset"] == "metr_la"
                 and c["method"] == "td_typed")

    before = {name: code_hash(config, root=tmp_path) for name, config in
              (("electricity", electricity), ("metr", metr),
               ("pair", pair), ("typed", typed))}
    train = tmp_path / "dscpuscripts/application_train.py"
    original = train.read_text()
    source = 'model = build_metr_la_model(config["seed"], adjacency)'
    assert source in original
    train.write_text(original.replace(
        source, 'model = build_metr_la_model(config["seed"], adjacency + 1)'))
    assert code_hash(electricity, root=tmp_path) == before["electricity"]
    assert code_hash(metr, root=tmp_path) != before["metr"]
    train.write_text(original)

    # Dataset-specific validation view routing is also isolated.
    original_train = train.read_text()
    selection = "return metr_la_validation_selection(dataset)"
    assert selection in original_train
    train.write_text(original_train.replace(
        selection, "return metr_la_validation_selection(dataset, extra=True)"))
    assert code_hash(electricity, root=tmp_path) == before["electricity"]
    assert code_hash(metr, root=tmp_path) != before["metr"]
    train.write_text(original_train)

    # Different dataset-specific LazyDataset.get_batch branches are
    # independent even though FD001 freezes the containing shared file.
    data = tmp_path / "fulltd/data.py"
    original = data.read_text()
    assert "speed = self.values[history].transpose(0,2,1)" in original
    data.write_text(original.replace(
        "speed = self.values[history].transpose(0,2,1)",
        "speed = (self.values[history] * 1.0).transpose(0,2,1)"))
    assert code_hash(electricity, root=tmp_path) == before["electricity"]
    assert code_hash(metr, root=tmp_path) != before["metr"]
    data.write_text(original)

    # The selected algorithm class must be tracked, but not other methods.
    objective = tmp_path / "dscpuscripts/application_objectives.py"
    original = objective.read_text()
    assert 'gamma_variant = "typed"' in original
    objective.write_text(original.replace(
        'gamma_variant = "typed"', 'gamma_variant = "typed-changed"'))
    assert code_hash(metr, root=tmp_path) == before["metr"]
    assert code_hash(pair, root=tmp_path) == before["pair"]
    assert code_hash(typed, root=tmp_path) != before["typed"]
    objective.write_text(original)

    # Baseline factory dispatch must ignore another baseline's import path.
    assert "from agents.fulltd_gaussian_nll import GaussianNLL" in original
    objective.write_text(original.replace(
        "from agents.fulltd_gaussian_nll import GaussianNLL",
        "from agents.fulltd_gaussian_nll import AnotherGaussianNLL"))
    assert code_hash(metr, root=tmp_path) == before["metr"]
    objective.write_text(original)

    # Generic training semantics still invalidate every dependent run.
    original_train = train.read_text()
    assert 'seed_runtime(config["seed"])' in original_train
    train.write_text(original_train.replace(
        'seed_runtime(config["seed"])', 'seed_runtime(config["seed"] + 1)'))
    assert code_hash(electricity, root=tmp_path) != before["electricity"]
    assert code_hash(metr, root=tmp_path) != before["metr"]
