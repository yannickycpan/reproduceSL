"""Changing scientific code invalidates states; operational edits do not."""
from fulltd import train


def test_scientific_hash_compatibility_boundary(tmp_path, monkeypatch):
    package = tmp_path / "fulltd"
    package.mkdir()
    agents = tmp_path / "agents"
    agents.mkdir()
    for name in ("__init__.py", "data.py", "metrics.py", "models.py", "protocol.py",
                 "sampling.py", "train.py", "hpc.py", "analysis.py"):
        (package / name).write_bytes(b"original\n")
    objective = agents / "fulltd_mse.py"
    objective.write_bytes(b"objective\n")
    monkeypatch.setattr(train, "__file__", str(package / "train.py"))
    original = train.code_hash()
    (package / "hpc.py").write_bytes(b"new scheduling\n")
    (package / "analysis.py").write_bytes(b"new plotting\n")
    assert train.code_hash() == original
    (package / "models.py").write_bytes(b"original\r\n")
    assert train.code_hash() == original  # Windows and Linux checkouts agree
    (package / "models.py").write_bytes(b"changed architecture\n")
    assert train.code_hash() != original
    (package / "models.py").write_bytes(b"original\n")
    objective.write_bytes(b"changed objective\n")
    assert train.code_hash() != original
