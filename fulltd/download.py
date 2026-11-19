"""Acquire original public data, without substituting filtered benchmark variants.

Downloads are recorded with provenance, retrieval time, size, and SHA-256.
Existing files must still match their recorded hash. Manual copies are allowed
only with a user-supplied provenance URL, which is recorded as such.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import urllib.request
import zipfile

from .data import NAMES, canonical_name, sha256

DCRNN_COMMIT = "602afd9d767d3aa1c9b3eac51710d6aeee12c227"
ELECTRICITY_URL = "https://archive.ics.uci.edu/static/public/321/electricityloaddiagrams20112014.zip"
FD001_URL = "https://data.nasa.gov/docs/legacy/CMAPSSData.zip"
METR_FOLDER = "https://drive.google.com/drive/folders/10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX"
GRAPH_URL = f"https://raw.githubusercontent.com/liyaguang/DCRNN/{DCRNN_COMMIT}/data/sensor_graph/adj_mx.pkl"


def _record(path, url, provenance, extra=None):
    record = {"file": path.name, "source_url": url, "provenance": provenance,
              "retrieved_utc": datetime.now(timezone.utc).isoformat(),
              "size_bytes": path.stat().st_size, "sha256": sha256(path)}
    if extra:
        record.update(extra)
    path.with_name(path.name + ".source.json").write_text(json.dumps(record,indent=2)+"\n")


def _existing(path):
    sidecar = path.with_name(path.name + ".source.json")
    if not path.exists():
        return False
    if not sidecar.exists():
        raise RuntimeError(f"Unrecorded raw file {path}. Use --register-local with its original source URL")
    record = json.loads(sidecar.read_text())
    if sha256(path) != record["sha256"]:
        raise RuntimeError(f"Raw file changed since acquisition: {path}")
    return True


def _download(url, path, provenance):
    if _existing(path):
        return
    path.parent.mkdir(parents=True,exist_ok=True)
    partial = path.with_name(path.name+".part")
    request = urllib.request.Request(url, headers={"User-Agent":"reproduceSL-fullTD/1.0"})
    with urllib.request.urlopen(request, timeout=120) as source, open(partial,"wb") as target:
        shutil.copyfileobj(source,target,1024*1024)
    # ZIP/HDF validation happens before preprocessing. Reject common HTML error
    # pages now, rather than silently retaining them with a data-file suffix.
    with open(partial,"rb") as stream:
        first=stream.read(128).lstrip().lower()
    if first.startswith((b"<!doctype html",b"<html")):
        raise RuntimeError(f"Server returned HTML instead of dataset: {url}")
    partial.replace(path)
    _record(path,url,provenance)


def _extract(archive, filename, directory):
    target=directory/filename
    if _existing(target):
        return
    with zipfile.ZipFile(archive) as zipped:
        names=[n for n in zipped.namelist() if Path(n).name==filename and "__MACOSX" not in n]
        if len(names)!=1:
            raise RuntimeError(f"Expected exactly one {filename} in {archive}")
        # Copy a named member only; do not trust archive paths.
        with zipped.open(names[0]) as src, open(target,"wb") as dst:
            shutil.copyfileobj(src,dst,1024*1024)
    source=json.loads(archive.with_name(archive.name+".source.json").read_text())
    _record(target,source["source_url"],source["provenance"],
            {"archive_sha256":source["sha256"],"archive_member":names[0]})


def acquire(dataset,data_dir):
    dataset=canonical_name(dataset)
    directory=Path(data_dir)/"raw"/dataset
    directory.mkdir(parents=True,exist_ok=True)
    if dataset=="electricity":
        archive=directory/"electricityloaddiagrams20112014.zip"
        _download(ELECTRICITY_URL,archive,"UCI original 370-customer matrix; doi:10.24432/C58C86; CC BY 4.0")
        _extract(archive,"LD2011_2014.txt",directory)
    elif dataset=="fd001":
        archive=directory/"CMAPSSData.zip"
        _download(FD001_URL,archive,"NASA Open Data official CMAPSS resource; catalog license unspecified")
        for name in ("train_FD001.txt","test_FD001.txt","RUL_FD001.txt","readme.txt"):
            _extract(archive,name,directory)
    elif dataset=="metr_la":
        target=directory/"metr-la.h5"
        if not _existing(target):
            import gdown
            # Discover the file from the actual folder linked by DCRNN authors.
            # skip_download enumerates names/IDs without downloading PEMS-BAY.
            entries=gdown.download_folder(url=METR_FOLDER,skip_download=True,quiet=False)
            matches=[entry for entry in (entries or []) if Path(entry.path).name=="metr-la.h5"]
            if len(matches)!=1:
                raise RuntimeError("Author Drive folder did not expose one metr-la.h5; no mirror substituted")
            url=f"https://drive.google.com/uc?id={matches[0].id}"
            partial=target.with_name(target.name+".part")
            result=gdown.download(url=url,output=str(partial),quiet=False)
            if not result:
                raise RuntimeError("METR-LA author download failed")
            with open(partial,"rb") as stream:
                if stream.read(8)!=b"\x89HDF\r\n\x1a\n":
                    raise RuntimeError("METR-LA download is not HDF5")
            partial.replace(target)
            _record(target,url,"DCRNN author-linked Google Drive folder",{"folder_url":METR_FOLDER,"reference_commit":DCRNN_COMMIT})
        _download(GRAPH_URL,directory/"adj_mx.pkl",f"DCRNN supplied sensor graph, commit {DCRNN_COMMIT}")
    else:
        raise ValueError(f"Unknown dataset {dataset}")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset",choices=["all",*NAMES],default="all")
    parser.add_argument("--data-dir",default="data/fulltd")
    parser.add_argument("--register-local",type=Path,help="Register an existing raw file explicitly")
    parser.add_argument("--source-url",help="Original provenance URL for --register-local")
    args=parser.parse_args()
    if args.register_local:
        if not args.source_url or not args.register_local.is_file():
            parser.error("--register-local requires an existing file and --source-url")
        root=(Path(args.data_dir)/"raw").resolve()
        if root not in args.register_local.resolve().parents:
            parser.error("Registered file must be inside --data-dir/raw")
        _record(args.register_local,args.source_url,"User-supplied original public file; provenance asserted by user")
        return
    for dataset in NAMES if args.dataset=="all" else [args.dataset]:
        acquire(dataset,args.data_dir)
        print(f"Acquired and hashed {dataset}")


if __name__=="__main__":
    main()
