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


def test_current_application_scientific_hash_boundary(tmp_path, monkeypatch):
    for name in application_train.SCIENTIFIC_SOURCES:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"original\n")
    monkeypatch.setattr(application_train, "__file__",
                        str(tmp_path / "dscpuscripts" / "application_train.py"))
    original = application_train.code_hash()
    (tmp_path / "dscpuscripts" / "standard_runner.py").write_bytes(b"different scheduling\n")
    (tmp_path / "dscpuscripts" / "application_analysis.py").write_bytes(b"different reporting\n")
    assert application_train.code_hash() == original
    objective = tmp_path / "dscpuscripts" / "application_objectives.py"
    objective.write_bytes(b"original\r\n")
    assert application_train.code_hash() == original
    objective.write_bytes(b"changed scientific objective\n")
    assert application_train.code_hash() != original
