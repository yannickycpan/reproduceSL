"""Offline regression tests for repairing an existing CPU venv without pip."""
from pathlib import Path
import os
import shutil
import subprocess
import textwrap

import pytest


STUB_PYTHON = r"""#!/usr/bin/env python3
import os
from pathlib import Path
import sys

args = sys.argv[1:]
state = Path(os.environ["MOCK_PIP_STATE"])
log = Path(os.environ["MOCK_PIP_LOG"])

def record(message):
    with log.open("a", encoding="utf-8") as f:
        f.write(message + "\n")

if args[:2] == ["-m", "pip"]:
    if args[2:] == ["--version"]:
        sys.exit(0 if state.exists() else 1)
    if not state.exists():
        sys.exit("pip was called before bootstrap")
    record("pip " + " ".join(args[2:]))
    if args[2:] == ["freeze"]:
        print("fake-package==1")
    sys.exit(0)

if args[:2] == ["-m", "ensurepip"]:
    record("ensurepip " + " ".join(args[2:]))
    if os.environ.get("MOCK_FAIL_ENSUREPIP") == "1":
        sys.exit(1)
    state.write_text("bootstrapped")
    sys.exit(0)

if args[:1] == ["-c"]:
    sys.exit(0)

sys.exit(f"Unexpected Python operation: {args!r}")
"""


def run_setup(tmp_path, *, fail_ensurepip=False):
    root = Path(__file__).resolve().parents[1]
    fixture_root = tmp_path / "copy"
    scripts = fixture_root / "dscpuscripts"
    scripts.mkdir(parents=True)
    shutil.copy2(root / "dscpuscripts" / "setup_server.sh", scripts / "setup_server.sh")
    shutil.copy2(root / "dscpuscripts" / "env.sh", scripts / "env.sh")
    (fixture_root / "requirements-fulltd.txt").write_text("fake-package==1\n")
    interpreter = tmp_path / "fake-python3.10"
    interpreter.write_text(STUB_PYTHON)
    interpreter.chmod(0o755)
    venv = tmp_path / "existing-venv"
    (venv / "bin").mkdir(parents=True)
    shutil.copy2(interpreter, venv / "bin" / "python")
    (venv / "bin" / "python").chmod(0o755)
    pip_state = tmp_path / "pip-installed"
    log = tmp_path / "mock.log"
    env = os.environ.copy()
    env.update(FULLTD_PYTHON=str(interpreter), FULLTD_ENV=str(venv),
               FULLTD_DATA_DIR=str(tmp_path / "data"),
               MOCK_PIP_STATE=str(pip_state), MOCK_PIP_LOG=str(log))
    if fail_ensurepip:
        env["MOCK_FAIL_ENSUREPIP"] = "1"
    command = ["bash", str(scripts / "setup_server.sh")]
    return command, env, pip_state, log, venv


def test_missing_pip_is_bootstrapped_once_and_existing_venv_preserved(tmp_path):
    command, env, marker, log, venv = run_setup(tmp_path)
    sentinel = venv / "important-preexisting-file.txt"
    sentinel.write_text("preserve me")
    first = subprocess.run(command, env=env, capture_output=True, text=True)
    assert first.returncode == 0, first.stderr
    assert marker.exists() and sentinel.read_text() == "preserve me"
    assert log.read_text().splitlines() == [
        "ensurepip --upgrade",
        "pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu",
        f"pip install -r {tmp_path / 'copy' / 'requirements-fulltd.txt'}",
        "pip check",
        "pip freeze",
    ]
    assert (tmp_path / "data" / "environment-freeze.txt").read_text() == "fake-package==1\n"
    second = subprocess.run(command, env=env, capture_output=True, text=True)
    assert second.returncode == 0, second.stderr
    assert log.read_text().count("ensurepip") == 1
    assert sentinel.read_text() == "preserve me"


def test_missing_ensurepip_fails_with_actionable_ubuntu_message(tmp_path):
    command, env, marker, log, venv = run_setup(tmp_path, fail_ensurepip=True)
    failed = subprocess.run(command, env=env, capture_output=True, text=True)
    assert failed.returncode != 0
    assert "python3.10-venv" in failed.stderr
    assert "Do not delete" in failed.stderr
    assert not marker.exists()
    assert log.read_text() == "ensurepip --upgrade\n"
