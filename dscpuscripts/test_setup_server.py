"""Offline tests: CPU setup must work without pip and without ensurepip."""
from pathlib import Path
import os
import shutil
import subprocess

import pytest


PYTHON_STUB = r"""#!/usr/bin/env python3
import sys
if sys.argv[1:2] == ["-c"]:
    sys.exit(0)
if sys.argv[1:3] in (["-m", "pip"], ["-m", "ensurepip"]):
    sys.exit("pip/ensurepip must never be called")
sys.exit(f"Unexpected Python operation: {sys.argv[1:]!r}")
"""

UV_STUB = r"""#!/usr/bin/env python3
from pathlib import Path
import os
import shutil
import sys

args = sys.argv[1:]
log = Path(os.environ["MOCK_UV_LOG"])
if args == ["--version"]:
    print("uv 0.13.7")
    sys.exit(0)
if args[:1] == ["venv"]:
    env = Path(args[-1])
    (env / "bin").mkdir(parents=True)
    shutil.copy2(os.environ["MOCK_PYTHON"], env / "bin" / "python")
    (env / "pyvenv.cfg").write_text("home = /usr/bin\n")
    with log.open("a") as f:
        f.write("venv\n")
    sys.exit(0)
if args[:2] == ["pip", "install"]:
    assert "--python" in args
    assert Path(args[args.index("--python")+1]).is_file()
    with log.open("a") as f:
        f.write("install " + " ".join(args[2:]) + "\n")
    sys.exit(0)
if args[:2] == ["pip", "check"]:
    with log.open("a") as f:
        f.write("check\n")
    sys.exit(0)
if args[:2] == ["pip", "freeze"]:
    with log.open("a") as f:
        f.write("freeze\n")
    print("torch==2.5.1")
    sys.exit(0)
sys.exit(f"Unexpected uv operation: {args!r}")
"""


def setup_fixture(tmp_path, *, existing_env=True, auto_install_uv=False):
    root = Path(__file__).resolve().parents[1]
    fixture_root = tmp_path / "copy"
    scripts = fixture_root / "dscpuscripts"
    scripts.mkdir(parents=True)
    shutil.copy2(root / "dscpuscripts" / "setup_server.sh", scripts / "setup_server.sh")
    shutil.copy2(root / "dscpuscripts" / "env.sh", scripts / "env.sh")
    (fixture_root / "requirements-fulltd.txt").write_text("torch==2.5.1\n")

    python = tmp_path / "fake-python3.10"
    python.write_text(PYTHON_STUB)
    python.chmod(0o755)
    uv = tmp_path / "fake-uv"
    uv.write_text(UV_STUB)
    uv.chmod(0o755)
    venv = tmp_path / "existing-env"
    if existing_env:
        (venv / "bin").mkdir(parents=True)
        shutil.copy2(python, venv / "bin" / "python")
        (venv / "pyvenv.cfg").write_text("home = /usr/bin\n")
    log = tmp_path / "uv.log"
    env = dict(os.environ)
    env.update(
        FULLTD_PYTHON=str(python),
        FULLTD_ENV=str(venv),
        FULLTD_DATA_DIR=str(tmp_path / "data"),
        MOCK_PYTHON=str(python),
        MOCK_UV_LOG=str(log),
    )
    if not auto_install_uv:
        env["FULLTD_UV"] = str(uv)
    else:
        env.pop("FULLTD_UV", None)
        # Fake curl writes a script that copies uv to the user's own bin dir,
        # proving installation needs neither pip nor administrator privileges.
        curl = tmp_path / "bin" / "curl"
        curl.parent.mkdir()
        curl.write_text(r"""#!/usr/bin/env python3
from pathlib import Path
import sys
args = sys.argv[1:]
assert "https://astral.sh/uv/install.sh" in args
out = Path(args[args.index("-o") + 1])
out.write_text('#!/bin/sh\nset -e\nmkdir -p "$UV_UNMANAGED_INSTALL"\ncp "$MOCK_UV_SOURCE" "$UV_UNMANAGED_INSTALL/uv"\nchmod +x "$UV_UNMANAGED_INSTALL/uv"\n')
""")
        curl.chmod(0o755)
        env.update(HOME=str(tmp_path), FULLTD_UV_DIR=str(tmp_path / "uv-bin"),
                   MOCK_UV_SOURCE=str(uv),
                   PATH=str(curl.parent) + os.pathsep + "/usr/bin:/bin")
    return (["bash", str(scripts / "setup_server.sh")], env, venv, log)


def test_preexisting_pipless_venv_preserved_and_dependencies_installed(tmp_path):
    command, env, venv, log = setup_fixture(tmp_path)
    sentinel = venv / "important-file.txt"
    sentinel.write_text("must survive")
    first = subprocess.run(command, env=env, text=True, capture_output=True)
    assert first.returncode == 0, first.stderr
    assert sentinel.read_text() == "must survive"
    lines = log.read_text().splitlines()
    assert len(lines) == 4
    assert lines[0].startswith("install ") and "--index-url https://download.pytorch.org/whl/cpu" in lines[0]
    assert "torch==2.5.1" in lines[0]
    assert "-r " in lines[1] and "requirements-fulltd.txt" in lines[1]
    assert lines[2:] == ["check", "freeze"]
    assert (tmp_path / "data" / "environment-freeze.txt").read_text() == "torch==2.5.1\n"
    second = subprocess.run(command, env=env, text=True, capture_output=True)
    assert second.returncode == 0, second.stderr
    assert sentinel.read_text() == "must survive"
    assert len(log.read_text().splitlines()) == 8


def test_create_new_venv_without_pip_or_ensurepip(tmp_path):
    command, env, venv, log = setup_fixture(tmp_path, existing_env=False)
    completed = subprocess.run(command, env=env, text=True, capture_output=True)
    assert completed.returncode == 0, completed.stderr
    assert (venv / "bin" / "python").is_file()
    assert log.read_text().splitlines()[0] == "venv"


def test_uv_standalone_bootstrap_without_pip_or_sudo(tmp_path):
    command, env, venv, log = setup_fixture(tmp_path, auto_install_uv=True)
    completed = subprocess.run(command, env=env, text=True, capture_output=True)
    assert completed.returncode == 0, completed.stderr
    assert (tmp_path / "uv-bin" / "uv").is_file()
    assert "Downloading official Astral installer" in completed.stdout
    assert log.read_text().splitlines()[0].startswith("install ")
