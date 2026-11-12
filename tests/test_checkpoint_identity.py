"""Protect completed FD001 runs and the current application checkpoint identity.

The FD001 hashes are the LF-normalized scientific source bytes from fullTD
commit 24b92960d0fb4441f620349e303b1dca9358a117, before the new dataset work.
They deliberately cover shared dependencies as well as FD001-specific files.
"""
from pathlib import Path
import hashlib

from dscpuscripts import application_train
from dscpuscripts import fd001_train


FD001_BASELINE_SOURCES = {
    "dscpuscripts/fd001_protocol.py": "9f57b25cf392d918d0c4d3f7b6cb07d9f8f3fd6af6dc19327ac1bc6d12893ff1",
    "dscpuscripts/fd001_train.py": "0a6b6880d9ec26de8093e8f5e85ce00642a36fca4ead32ba89403394fce54003",
    "dscpuscripts/fd001_objectives.py": "7f557b59a15af5f511b3c18d0a1079b539665c62f99d7a0916f0f73ae94b73f6",
    "dscpuscripts/common.py": "e6a6536d83ea9758aeb12d7a9d93bd7bdcdb66ffe1f2dbd4beae6adfc66f7aed",
    "agents/fulltd_objectives.py": "26a77cff9e5f88944d607ed5c5e9bfd27f1206b2337eb162247489706a5636ec",
    "agents/fulltd_mse.py": "53f709dadf15ece0519c22505726c9ca42dad01279da51409fb6e947a5495819",
    "agents/fulltd_gaussian_nll.py": "b00bcb5803fc1e8df32f7d6f06fd9849bdc72bf43650dfeca2fa3bd85dd9be61",
    "fulltd/protocol.py": "df64b72df06be53c11f5e04daa8f948e696054724f3b9591170d971e276e548b",
    "fulltd/train.py": "03b578ae68bf5b436fb708af55af409c18411bed9f506f67fc34fd9f65fa6cad",
    "fulltd/data.py": "9705d5d15bc8cfe32773d20c9781ae3ceb0c87512fa56e7ff14544413dace564",
    "fulltd/metrics.py": "557e4c824a54ffcc4823247d4278ba55211cc077fd07775235d06cd875077ffb",
    "fulltd/models.py": "dde2daa6539e976d209658e0eabd1dd79815c5c6519650e33654ddfa41a26db9",
    "fulltd/sampling.py": "a337e828803897212653a104aad4bb549d6d8863c803e60e45e792089471d953",
}
FD001_BASELINE_CODE_HASH = "c6820c53e26b4ab47be164cb0e5e3a4c62a8f865e01a8b918148980ea840ef21"


def test_existing_fd001_scientific_bytes_and_identity_are_unchanged():
    root = Path(__file__).resolve().parents[1]
    assert tuple(FD001_BASELINE_SOURCES) == fd001_train.SCIENTIFIC_SOURCES
    for name, expected in FD001_BASELINE_SOURCES.items():
        source = (root / name).read_bytes().replace(b"\r\n", b"\n")
        assert hashlib.sha256(source).hexdigest() == expected, name
    assert fd001_train.code_hash() == FD001_BASELINE_CODE_HASH


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
