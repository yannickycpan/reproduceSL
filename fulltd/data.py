"""Training-only preprocessing and lazy windows for the fixed neural TD study.

Arrays contain readings, never all overlapping windows. A training Dataset opens
only its split directory; test labels are not loaded by candidate training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import warnings

import numpy as np

DATA_VERSION = "td-v7-data-2"
NAMES = ("electricity", "metr_la", "fd001")


def canonical_name(name):
    key = name.lower().replace("-", "_")
    return {"metrla": "metr_la", "cmapss": "fd001"}.get(key, key)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _save_json(path, obj):
    # Split sidecars participate in the content identity; use the same bytes on
    # Windows verification hosts and Linux HPC nodes.
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n",
                          encoding="utf-8", newline="\n")


def _save_split(base, split, arrays, details):
    directory = Path(base) / split
    directory.mkdir(parents=True, exist_ok=True)
    for name, array in arrays.items():
        np.save(directory / (name + ".npy"), array, allow_pickle=False)
    _save_json(directory / "split.json", details)


def _time_windows(times, history, horizon, start, end, earliest=0):
    """Origin rows with all targets in [start,end), no input/target time gaps."""
    origin = np.arange(max(history - 1 + earliest, start - 1), end - horizon,
                       dtype=np.int64)
    if not len(origin):
        return origin
    # times is in native hours or five-minute units; differences other than 1
    # must never be treated as a shorter elapsed-time interval.
    bad = np.r_[0, np.cumsum(np.diff(times) != 1)]
    return origin[bad[origin + horizon] == bad[origin - history + 1]]


def aggregate_electricity(frame):
    """Four consecutive quarter-hour power readings -> hour-ending mean kW.

    Group on nominal hour ends, preserving UCI's nominal Portuguese calendar
    and its documented March zeros/October combined consumption. An incomplete
    nominal hour is rejected, recorded, and leaves a gap in the hourly grid.
    """
    import pandas as pd
    index = pd.DatetimeIndex(frame.index)
    if index.tz is not None or not index.is_monotonic_increasing or index.has_duplicates:
        raise ValueError("Electricity timestamps must be unique increasing nominal times")
    if np.any(index.second != 0) or np.any(index.minute % 15 != 0):
        raise ValueError("Electricity timestamps are not quarter-hour aligned")
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError("UCI data contain nonfinite readings; do not omit or impute them")
    end = index.ceil("h")
    grouped = frame.groupby(end, sort=True)
    counts = grouped.size()
    complete = counts.index[counts.to_numpy() == 4]
    hourly = grouped.mean().loc[complete]
    incomplete = [{"hour_end": str(t), "readings": int(c)}
                  for t, c in counts.items() if c != 4]
    # Exactly four distinct aligned readings in a nominal hour implies :15,
    # :30, :45, and :00. Uniqueness plus alignment means a count of four verifies
    # all four positions without scanning the 140K-row index once per hour.
    if not np.isfinite(hourly.to_numpy()).all():
        raise ValueError("UCI data contain nonfinite readings; do not impute them")
    return hourly, incomplete


def prepare_electricity(data_dir, frame=None):
    import pandas as pd
    data_dir = Path(data_dir)
    raw = data_dir / "raw" / "electricity" / "LD2011_2014.txt"
    fixture = frame is not None
    if frame is None:
        frame = pd.read_csv(raw, sep=";", decimal=",", index_col=0,
                            parse_dates=True, dtype=np.float64)
        if frame.shape != (140256, 370):
            raise ValueError(f"Expected original UCI shape (140256,370), got {frame.shape}")
    raw_shape = list(frame.shape)
    hourly, incomplete = aggregate_electricity(frame)
    time = hourly.index.to_numpy().astype("datetime64[h]").astype(np.int64)
    values = hourly.to_numpy(np.float64)
    T, C = values.shape
    bounds = [0, int(np.floor(.7 * T)), int(np.floor(.8 * T)), T]
    active, keep, means, scales, excluded = [], [], [], [], []
    for column in range(C):
        nonzero = np.flatnonzero(values[:bounds[1], column] != 0)
        reason = None
        if not len(nonzero):
            reason = "no_nonzero_training_reading"
        else:
            first = int(nonzero[0])
            available = _time_windows(time, 168, 24, 0, bounds[1], first)
            scale = values[first:bounds[1], column].std(ddof=0)
            if not len(available):
                reason = "no_eligible_training_window"
            elif scale <= 1e-8:
                reason = "constant_active_training_interval"
        if reason:
            excluded.append({"id": str(hourly.columns[column]), "reason": reason})
            continue
        keep.append(column)
        active.append(first)
        means.append(float(values[first:bounds[1], column].mean()))
        scales.append(float(scale))
    if not keep:
        raise ValueError("No eligible Electricity customers")
    means, scales = np.asarray(means), np.asarray(scales)
    standardized = ((values[:, keep] - means) / scales).astype(np.float32)
    base = data_dir / "processed" / "electricity"
    base.mkdir(parents=True, exist_ok=True)
    split_counts, split_target_spans = {}, {}
    for split, start, end in zip(("train", "val", "test"), bounds[:-1], bounds[1:]):
        rows, entities = [], []
        for entity, first in enumerate(active):
            origin = _time_windows(time, 168, 24, start, end, first)
            rows.append(origin)
            entities.append(np.full(len(origin), entity, dtype=np.int64))
        rows = np.concatenate(rows)
        entities = np.concatenate(entities)
        offset = max(0, start - 168)
        _save_split(base, split, {
            "values": standardized[offset:end], "time": time[offset:end],
            "rows": rows - offset, "origins": time[rows], "entity_ids": entities,
        }, {"row_offset": offset, "target_start": start, "target_end_exclusive": end,
            "examples": len(rows), "target_components": int(len(rows) * 24)})
        split_counts[split] = len(rows)
        split_target_spans[split] = [str(hourly.index[start]), str(hourly.index[end-1])]
    metadata = {
        "version": DATA_VERSION, "dataset": "electricity", "fixture": fixture,
        "raw_shape": raw_shape, "hourly_shape": [T, len(keep)],
        "history": 168, "horizon": 24, "target_shape": [24], "input_channels": 1,
        "entity_names": [str(hourly.columns[c]) for c in keep], "activation_rows": active,
        "excluded_customers": excluded, "mean": means.tolist(), "scale": scales.tolist(),
        "split_boundaries": bounds, "split_target_spans": split_target_spans,
        "split_counts": split_counts, "incomplete_hours": incomplete,
        "hourly_grid_gaps": int(np.sum(np.diff(time) != 1)),
        "units": "kW", "aggregation": "equal_customer", "raw_shape_expected": [140256, 370],
        "scaling_fit": "unique hourly readings of each active training interval; ddof=0",
        "calendar": "nominal Portuguese 24-hour; UCI DST artifacts preserved",
        "source": "https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014",
        "license": "CC BY 4.0; Artur Trindade (2015), doi:10.24432/C58C86",
        "raw_sha256": sha256(raw) if raw.exists() and not fixture else None,
    }
    _finish_metadata(base, metadata)
    return metadata


def _load_graph(path):
    # Only the exact downloaded author graph is deserialized; never consume
    # an arbitrary run-produced pickle as graph input.
    import pickle
    with open(path, "rb") as stream:
        sensor_ids, mapping, adjacency = pickle.load(stream, encoding="latin1")
    sensor_ids = [str(x) for x in sensor_ids]
    if len(set(sensor_ids)) != len(sensor_ids):
        raise ValueError("Duplicate graph sensor IDs")
    if any(mapping.get(s, mapping.get(int(s), None)) != i for i, s in enumerate(sensor_ids)):
        raise ValueError("Graph index mapping does not match graph sensor order")
    return sensor_ids, np.asarray(adjacency, dtype=np.float32)


def prepare_metr_la(data_dir, frame=None, graph=None):
    import pandas as pd
    data_dir = Path(data_dir)
    raw = data_dir / "raw" / "metr_la"
    fixture = frame is not None
    if frame is None:
        frame = pd.read_hdf(raw / "metr-la.h5")
    if graph is None:
        graph = _load_graph(raw / "adj_mx.pkl")
    ids, adjacency = graph
    ids = [str(x) for x in ids]
    if [str(x) for x in frame.columns] != ids:
        raise ValueError("Traffic HDF columns do not match supplied graph sensor order")
    if adjacency.shape != (len(ids), len(ids)) or not np.isfinite(adjacency).all():
        raise ValueError("Invalid supplied graph")
    if not fixture and len(ids) != 207:
        raise ValueError("METR-LA must contain exactly 207 sensors")
    index = pd.DatetimeIndex(frame.index)
    if index.tz is not None or index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError("METR-LA requires unique increasing naive timestamps")
    seconds = index.to_numpy().astype("datetime64[s]").astype(np.int64)
    if np.any(seconds % 300):
        raise ValueError("METR-LA timestamps must align to five-minute grid")
    time = seconds // 300
    values = frame.to_numpy(dtype=np.float64)
    T, C = values.shape
    if T != 34272 and not fixture:
        warnings.warn(f"METR-LA expected 34272 time rows, found {T}; recorded discrepancy")
    bounds = [0, int(np.floor(.7*T)), int(np.floor(.8*T)), T]
    # Graph WaveNet engine.py passes null_val=0.0 on raw-speed labels.
    mask = np.isfinite(values) & (values != 0)
    observed_train = values[:bounds[1]][mask[:bounds[1]]]
    if not len(observed_train):
        raise ValueError("No observed traffic training targets")
    mean, scale = float(observed_train.mean()), float(observed_train.std(ddof=0))
    if scale <= 1e-8:
        raise ValueError("Traffic training speed is constant")
    standardized = np.where(mask, (values - mean) / scale, 0).astype(np.float32)
    tod = ((seconds % 86400) / 86400).astype(np.float32)
    base = data_dir / "processed" / "metr_la"
    base.mkdir(parents=True, exist_ok=True)
    np.save(base / "adjacency.npy", adjacency, allow_pickle=False)
    split_counts, missing = {}, {}
    for split, start, end in zip(("train", "val", "test"), bounds[:-1], bounds[1:]):
        rows = _time_windows(time, 12, 12, start, end)
        offset = max(0, start - 12)
        _save_split(base, split, {
            "values": standardized[offset:end], "mask": mask[offset:end],
            "tod": tod[offset:end], "time": time[offset:end],
            "rows": rows-offset, "origins": time[rows],
            "entity_ids": np.zeros(len(rows), dtype=np.int64),
        }, {"row_offset": offset, "target_start": start, "target_end_exclusive": end,
            "examples": len(rows)})
        split_counts[split] = len(rows)
        missing[split] = int((~mask[start:end]).sum())
    metadata = {
        "version": DATA_VERSION, "dataset": "metr_la", "fixture": fixture,
        "raw_shape": [T, C], "expected_raw_shape": [34272, 207],
        "time_count_discrepancy": T-34272, "sensor_ids": ids, "entity_names": ["network"],
        "history": 12, "horizon": 12, "target_shape": [C, 12], "input_channels": 2,
        "mean": mean, "scale": scale, "split_boundaries": bounds, "split_counts": split_counts,
        "split_target_spans": {s: [str(index[a]), str(index[b-1])] for s,a,b in
                               zip(("train","val","test"),bounds[:-1],bounds[1:])},
        "time_gaps": [{"before": str(index[k]), "after": str(index[k+1])}
                      for k in np.flatnonzero(np.diff(time)!=1)],
        "missing_raw_values": missing, "mask_rule": "isfinite(raw_speed) and raw_speed != 0",
        "input_missing_rule": "training observed-speed mean; standardized zero",
        "scaling_fit": "one global mean/std on unique observed training values; ddof=0",
        "units": "mph", "aggregation": "valid_scalar", "protocol": "boundary-audited reference protocol",
        "source": "https://github.com/liyaguang/DCRNN", "license": "See DCRNN author dataset attribution; dataset license not separately specified",
        "raw_sha256": {p.name: sha256(p) for p in (raw / "metr-la.h5", raw / "adj_mx.pkl") if p.exists()} if not fixture else {},
    }
    _finish_metadata(base, metadata)
    return metadata


def _validate_engine_rows(array):
    if array.ndim != 2 or array.shape[1] != 26 or not np.isfinite(array).all():
        raise ValueError("FD001 requires 26 finite columns: ID, cycle, 3 settings, 21 sensors")
    if not np.equal(array[:, :2], np.floor(array[:, :2])).all():
        raise ValueError("Engine/cycle IDs must be integers")
    array = array[np.lexsort((array[:, 1], array[:, 0]))]
    for eid in np.unique(array[:, 0]):
        cycles = array[array[:, 0] == eid, 1]
        if not np.array_equal(cycles, np.arange(1, len(cycles)+1)):
            raise ValueError(f"Engine {eid} has missing, repeated, or noncontiguous cycles")
    return array


def prepare_fd001(data_dir, train=None, test=None, rul=None):
    data_dir = Path(data_dir)
    raw = data_dir / "raw" / "fd001"
    fixture = train is not None
    if train is None:
        train = np.loadtxt(raw / "train_FD001.txt", dtype=np.float64)
        test = np.loadtxt(raw / "test_FD001.txt", dtype=np.float64)
        rul = np.loadtxt(raw / "RUL_FD001.txt", dtype=np.float64)
    train, test = _validate_engine_rows(train), _validate_engine_rows(test)
    rul = np.asarray(rul, dtype=np.float64).reshape(-1)
    ids = np.unique(train[:, 0]).astype(np.int64)
    test_ids = np.unique(test[:, 0]).astype(np.int64)
    if len(test_ids) != len(rul) or not np.isfinite(rul).all() or np.any(rul < 0):
        raise ValueError("Invalid test-engine endpoint RUL vector")
    if not fixture and (not np.array_equal(ids,np.arange(1,101)) or
                        not np.array_equal(test_ids,np.arange(1,101))):
        raise ValueError("Official FD001 training/test engine IDs must both be 1..100")
    shuffled = np.random.default_rng(12345).permutation(np.sort(ids))
    n_fit = int(np.floor(.8*len(ids)))
    fit_ids, val_ids = shuffled[:n_fit], shuffled[n_fit:]
    fitting = train[np.isin(train[:, 0], fit_ids)]
    mean = fitting[:, 2:].mean(0)
    original_std = fitting[:, 2:].std(0, ddof=0)
    scale = np.where(original_std <= 1e-8, 1., original_std)
    training_targets = np.concatenate([len(a)-a[a[:,1]>=30,1] for eid in fit_ids
        for a in [train[train[:,0]==eid]]])
    if not len(training_targets):
        raise ValueError("No eligible FD001 training windows")
    target_mean, target_scale = float(training_targets.mean()), float(training_targets.std(ddof=0))
    if target_scale <= 1e-8:
        raise ValueError("Degenerate FD001 training-window RUL scaling")
    base = data_dir / "processed" / "fd001"
    base.mkdir(parents=True, exist_ok=True)
    split_counts, padding = {}, 0
    for split, selected, data in (("train",np.sort(fit_ids),train),
                                 ("val",np.sort(val_ids),train), ("test",test_ids,test)):
        values, targets, rows, origins, entities, starts = [], [], [], [], [], []
        offset = 0
        for eid in selected:
            a = data[data[:,0] == eid]
            T = len(a)
            if split == "train":
                endpoints = np.arange(30, T+1, dtype=np.int64)
            elif split == "val":
                if T < 31:
                    raise ValueError("Validation engines need at least 31 cycles for prescribed prefixes")
                endpoints = np.unique(np.clip(np.floor(np.array([.4,.5,.6,.7,.8])*T).astype(np.int64),30,T-1))
            else:
                endpoints = np.array([T], dtype=np.int64)
                padding += int(T < 30)
            values.append(((a[:,2:] - mean) / scale).astype(np.float32))
            raw_y = T - np.arange(1,T+1,dtype=np.float64)
            if split == "test":
                raw_y += rul[np.flatnonzero(test_ids==eid)[0]]
            targets.append(((raw_y-target_mean)/target_scale).astype(np.float32))
            rows.extend((offset+endpoints-1).tolist())
            origins.extend(endpoints.tolist())
            entities.extend([int(eid)]*len(endpoints))
            starts.extend([offset]*len(endpoints))
            offset += T
        arrays = {"values": np.concatenate(values), "targets": np.concatenate(targets),
                  "rows": np.asarray(rows,dtype=np.int64), "origins": np.asarray(origins,dtype=np.int64),
                  "entity_ids": np.asarray(entities,dtype=np.int64), "starts": np.asarray(starts,dtype=np.int64)}
        _save_split(base, split, arrays, {"engine_ids": selected.tolist(), "examples": len(rows)})
        split_counts[split] = len(rows)
    metadata = {
        "version": DATA_VERSION, "dataset": "fd001", "fixture": fixture,
        "raw_shapes": {"train": list(train.shape), "test": list(test.shape), "rul": list(rul.shape)},
        "fit_engine_ids": fit_ids.tolist(), "val_engine_ids": val_ids.tolist(), "test_engine_ids": test_ids.tolist(),
        "engine_split_seed": 12345, "history": 30, "horizon": 1, "target_shape": [1], "input_channels": 24,
        "input_mean": mean.tolist(), "input_scale": scale.tolist(),
        "constant_channels": np.flatnonzero(original_std<=1e-8).tolist(),
        "mean": target_mean, "scale": target_scale, "target_cap": None,
        "split_counts": split_counts, "test_padded_engines": padding,
        "input_channels_order": [f"setting_{i}" for i in range(1,4)] + [f"sensor_{i}" for i in range(1,22)],
        "units": "cycles", "aggregation": "equal_engine", "training_evaluation": "all eligible fitting-engine windows",
        "source": "https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data",
        "license": "NASA public dataset; catalog does not specify a license",
        "raw_sha256": {p.name: sha256(p) for p in (raw/"train_FD001.txt",raw/"test_FD001.txt",raw/"RUL_FD001.txt") if p.exists()} if not fixture else {},
    }
    _finish_metadata(base, metadata)
    return metadata


def _finish_metadata(base, metadata):
    # Content identity includes preprocessing outputs, not file modification times.
    raw_dir = Path(base).parent.parent / "raw" / metadata["dataset"]
    metadata["acquisition"] = [json.loads(p.read_text()) for p in sorted(raw_dir.glob("*.source.json"))]
    files = (sorted(Path(base).glob("*/*.npy")) + sorted(Path(base).glob("*.npy"))
             + sorted(Path(base).glob("*/split.json")))
    metadata["processed_sha256"] = {str(p.relative_to(base)).replace("\\", "/"): sha256(p) for p in files}
    metadata["fingerprint"] = metadata_fingerprint(metadata)
    _save_json(Path(base)/"metadata.json", metadata)


def metadata_fingerprint(metadata):
    """Identify scalers, splits and payload hashes, excluding retrieval dates.

    The stored digest is not itself an input. Recomputing this digest on reads
    prevents an edited scaler from being accepted under an old checkpoint's
    unchanged identity field.
    """
    identity = {key: value for key, value in metadata.items() if key != "fingerprint"}
    # Reacquiring identical public bytes on another machine must not invalidate
    # a compatible checkpoint solely because its download date differs.
    identity["acquisition"] = [{k:v for k,v in entry.items() if k != "retrieved_utc"}
                               for entry in metadata["acquisition"]]
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def _read_metadata(base):
    metadata = json.loads((Path(base)/"metadata.json").read_text())
    if metadata.get("version") != DATA_VERSION:
        raise ValueError("Incompatible preprocessing version; rerun prepare (raw data are reusable)")
    if metadata.get("fingerprint") != metadata_fingerprint(metadata):
        raise ValueError("Processed metadata fingerprint mismatch: scaler/split/provenance metadata changed")
    expected_splits = {f"{split}/split.json" for split in ("train", "val", "test")}
    if not expected_splits.issubset(metadata.get("processed_sha256", {})):
        raise ValueError("Processed metadata lacks split-file integrity records; rerun prepare")
    return metadata


def verify_dataset(dataset, data_dir):
    """Verify processed bytes once before launching the campaign."""
    base = Path(data_dir)/"processed"/canonical_name(dataset)
    metadata = _read_metadata(base)
    for relative, expected in metadata["processed_sha256"].items():
        path = base/relative
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Processed data missing or modified: {path}")
    return metadata["fingerprint"]


class LazyDataset:
    def __init__(self, dataset, data_dir, split):
        self.dataset = canonical_name(dataset)
        self.split = {"validation": "val"}.get(split, split)
        if self.dataset not in NAMES or self.split not in ("train", "val", "test"):
            raise ValueError("Unknown dataset or split")
        self.base = Path(data_dir)/"processed"/self.dataset
        self.metadata = _read_metadata(self.base)
        self.directory = self.base/self.split
        split_path = self.directory/"split.json"
        if sha256(split_path) != self.metadata["processed_sha256"][f"{self.split}/split.json"]:
            raise ValueError(f"Processed split metadata missing or modified: {split_path}")
        self.split_metadata = json.loads(split_path.read_text())
        def load(name):
            return np.load(self.directory/(name+".npy"), mmap_mode="r", allow_pickle=False)
        self.values, self.rows = load("values"), load("rows")
        self.entity_ids, self.origins = load("entity_ids"), load("origins")
        self.history, self.horizon = self.metadata["history"], self.metadata["horizon"]
        self.mean, self.scale = np.asarray(self.metadata["mean"]), np.asarray(self.metadata["scale"])
        if self.dataset == "metr_la":
            self.mask, self.tod = load("mask"), load("tod")
            self.adjacency = np.load(self.base/"adjacency.npy",allow_pickle=False)
        elif self.dataset == "fd001":
            self.targets, self.starts = load("targets"), load("starts")

    def __len__(self):
        return len(self.rows)

    def get_batch(self, indices):
        ids = np.asarray(indices, dtype=np.int64).reshape(-1)
        if np.any(ids < 0) or np.any(ids >= len(self)):
            raise IndexError("Window index out of range")
        rows, entities = self.rows[ids], self.entity_ids[ids]
        history = rows[:,None]+np.arange(1-self.history,1)
        future = rows[:,None]+np.arange(1,self.horizon+1)
        if self.dataset == "electricity":
            x = self.values[history,entities[:,None]][:,None,:]
            y = self.values[future,entities[:,None]]
            mask = np.ones_like(y,dtype=bool)
        elif self.dataset == "metr_la":
            speed = self.values[history].transpose(0,2,1)
            tod = np.broadcast_to(self.tod[history][:,None,:], speed.shape)
            x = np.stack((speed,tod),axis=1)
            y = self.values[future].transpose(0,2,1)
            mask = self.mask[future].transpose(0,2,1)
        else:
            history = np.maximum(history,self.starts[ids,None])
            x = self.values[history].transpose(0,2,1)
            y = self.targets[rows,None]
            mask = np.ones_like(y,dtype=bool)
        return {"x": np.ascontiguousarray(x,dtype=np.float32), "y":np.ascontiguousarray(y,dtype=np.float32),
                "mask":np.ascontiguousarray(mask), "entity_ids":entities, "origins":self.origins[ids]}

    def inverse(self, values, indices):
        values = np.asarray(values,dtype=np.float64)
        if self.dataset == "electricity":
            entities = self.entity_ids[np.asarray(indices,dtype=np.int64)]
            return values*self.scale[entities,None]+self.mean[entities,None]
        return values*self.scale+self.mean

    def persistence(self, indices):
        x = self.get_batch(indices)["x"]
        if self.dataset == "electricity":
            return x[:,0,-24:].copy()
        if self.dataset == "metr_la":
            return np.repeat(x[:,0,:,-1,None],12,axis=-1)
        raise ValueError("No persistence baseline is prescribed for FD001")


def load_dataset(dataset, data_dir, split):
    return LazyDataset(dataset,data_dir,split)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "verify"])
    parser.add_argument("--dataset", choices=["all",*NAMES],default="all")
    parser.add_argument("--data-dir", default="data/fulltd")
    args=parser.parse_args()
    for name in NAMES if args.dataset=="all" else [args.dataset]:
        if args.command == "verify":
            print(json.dumps({"dataset":name,"verified":verify_dataset(name,args.data_dir)}))
            continue
        result = globals()["prepare_"+name](args.data_dir)
        print(json.dumps({"dataset":name,"counts":result["split_counts"],"fingerprint":result["fingerprint"]}))


if __name__ == "__main__":
    main()
