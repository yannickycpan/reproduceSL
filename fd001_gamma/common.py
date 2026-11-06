"""Atomic records, provenance and CPU resource setup for the separate campaign."""
from __future__ import annotations

import contextlib
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import uuid

from .protocol import digest

THREAD_VARIABLES = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS")


def cpu_environment():
    for name in THREAD_VARIABLES:
        os.environ[name] = "1"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path, value):
    def clean(x):
        if isinstance(x, dict):
            return {k: clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [clean(v) for v in x]
        if isinstance(x, float) and not math.isfinite(x):
            return "Infinity" if x > 0 else "-Infinity" if x < 0 else "NaN"
        return x
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with tmp.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(clean(value), stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def write_csv(path, rows):
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with tmp.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def scientific_hash():
    root = Path(__file__).resolve().parents[1]
    paths = ["fd001_gamma/" + name for name in
             ("__init__.py", "protocol.py", "objectives.py", "common.py", "train.py")]
    paths += ["fulltd/" + name for name in ("__init__.py", "data.py", "models.py", "sampling.py", "metrics.py")]
    paths += ["agents/fulltd_objectives.py"]
    h = hashlib.sha256()
    for name in paths:
        h.update(name.encode())
        h.update((root / name).read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()


def data_identity(dataset):
    metadata = dataset.metadata
    return digest(dict(fingerprint=metadata.get("fingerprint", digest(metadata)),
                       split=getattr(dataset, "split", "unknown"),
                       common_source_count=metadata.get("common_source_count"),
                       source_index_sha256=metadata.get("source_index_sha256"),
                       isolated_excluded_count=metadata.get("isolated_excluded_count")))


@contextlib.contextmanager
def file_lock(path):
    """OS-released locks, including after a killed worker; never stale PID files."""
    import fcntl  # This launcher targets the Linux Slurm cluster.
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another process holds {path}; refusing concurrent writes") from exc
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def campaign_paths():
    home = Path.home() / "scratch" / "fulltd"
    return (Path(os.environ.get("FULLTD_DATA_DIR", home / "data")).expanduser().resolve(),
            Path(os.environ.get("FULLTD_FD001_GAMMA_RESULTS_DIR", home / "fd001_gamma_v1" / "results")).expanduser().resolve(),
            Path(os.environ.get("FULLTD_FD001_GAMMA_OUTPUT_DIR", home / "fd001_gamma_v1" / "outputs")).expanduser().resolve())
