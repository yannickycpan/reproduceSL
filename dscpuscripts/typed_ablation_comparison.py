"""Read-only four-method typed-TD comparison; never runs training.

Uses the existing Full TD/MSE runs plus the new semi/nonnegative typed runs.
The old B256 Full TD hyperparameters are selected independently for each
original lag set using only saved validation traces. Test is opened only after
both old and new validation selections have been locked.

    python -m dscpuscripts.typed_ablation_comparison fd001 --select
    python -m dscpuscripts.typed_ablation_comparison fd001 --evaluate

Only this new analysis module is installed; no scientific training source changes.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
import math
from pathlib import Path
from statistics import mean, stdev

from .common import atomic_json, file_hash, read_json, write_csv
from .result_paths import paths as old_paths
from .typed_ablation_protocol import make_manifest as new_manifest
from .typed_ablation_runner import paths as new_paths
from .typed_ablation_train import code_hash as new_code_hash

REFERENCE_METHODS = ("mse", "td_typed")
ABLATIONS = ("td_semi_typed", "td_nonnegative_typed")
LABELS = {"mse":"MSE", "td_typed":"Full TD (typed)",
          "td_semi_typed":"Semi-gradient TD (typed)",
          "td_nonnegative_typed":"Nonnegative Full TD (typed)"}
REVISION = "typed-four-method-comparison-v1"


def _digest(obj):
    from fulltd.protocol import digest
    return digest(obj)


def lag_key(config):
    return "-".join(map(str, config["lags"])) if config["method"] != "mse" else "none"


def _metric_keys(dataset):
    if dataset == "fd001":
        return (("cycles","rmse"),("cycles","mae"),("score","rul_score_sum"))
    return (("kW" if dataset == "electricity" else "mph", "rmse"),
            ("kW" if dataset == "electricity" else "mph", "mae"))


def _original_manifest(dataset):
    if dataset == "fd001":
        from .fd001_protocol import make_manifest
        manifest = make_manifest()
    else:
        from .application_protocol import make_manifest
        manifest = make_manifest()
    return [c for c in manifest if c["dataset"] == dataset
            and c["method"] in REFERENCE_METHODS and c["batch_size"] == 256]


def _original_hash(config):
    if config["dataset"] == "fd001":
        from .fd001_train import code_hash
        return code_hash()
    from .application_train import code_hash
    return code_hash(config)


def _audit_reference(config, root, shared, source_hashes):
    folder = root / config["run_id"]
    if not (folder / "status.json").is_file():
        raise FileNotFoundError(f"Missing original B256 training result: {folder}")
    status = read_json(folder / "status.json")
    identity = status.get("identity", {})
    if (status.get("config") != config or
            identity.get("protocol_id") != config["protocol_id"] or
            identity.get("config_sha256") != _digest(config)):
        raise ValueError(f"Original configuration differs: {config['run_id']}")
    allowed = {_original_hash(config)}
    if config["dataset"] == "electricity":
        from .application_analysis import LEGACY_ELECTRICITY_CODE_SHA256
        allowed.add(LEGACY_ELECTRICITY_CODE_SHA256)
    if identity.get("code_sha256") not in allowed:
        raise ValueError(f"Original scientific code mismatch: {config['run_id']}")
    if config["dataset"] != "fd001":
        from .application_analysis import _audit_candidate, _candidate_score
        checked = _audit_candidate(config, root, identity["code_sha256"], shared, source_hashes)
        score, _ = _candidate_score(checked, checked["budgets"]["total_updates"], "full_budget")
        return score
    from .fd001_protocol import EPOCHS, EVAL_UPDATE_PERIOD
    from .fd001_analysis import selection_update_steps
    from fulltd.sampling import sources_for_round, sequence_hash
    for field in ("data_sha256", "validation_sha256"):
        if not identity.get(field) or (field in shared and shared[field] != identity[field]):
            raise ValueError(f"Inconsistent FD001 {field}: {config['run_id']}")
        shared[field] = identity[field]
    init_key = ("initial_mean", config["seed"])
    if init_key in shared and shared[init_key] != status.get("initial_mean_sha256"):
        raise ValueError(f"Original predictor initialization differs: {config['run_id']}")
    shared[init_key] = status.get("initial_mean_sha256")
    state = status.get("state")
    if state == "numerical_failure":
        return math.inf
    if state != "complete":
        raise RuntimeError(f"Original B256 candidate unfinished: {config['run_id']}: {state}")
    n = status.get("n_sources")
    if not isinstance(n, int) or n <= 0:
        raise ValueError("Invalid original FD001 source count")
    total = math.ceil(n / config["batch_size"]) * EPOCHS
    if (status.get("optimizer_steps") != total or
            status.get("update_opportunities") != total or
            status.get("epochs_completed") != EPOCHS or
            status.get("next_epoch") != EPOCHS+1 or
            not (folder / "final.pt").is_file()):
        raise ValueError(f"Incorrect original FD001 completion evidence: {config['run_id']}")
    curve = read_json(folder / "update_curve.json")
    checkpoints = list(range(0, total+1, EVAL_UPDATE_PERIOD))
    if [row.get("optimizer_steps") for row in curve] != checkpoints:
        raise ValueError(f"Missing original FD001 validation checkpoints: {config['run_id']}")
    rounds = read_json(folder / "rounds.json")
    if [r.get("epoch") for r in rounds] != list(range(1,EPOCHS+1)):
        raise ValueError(f"Missing original FD001 epochs: {config['run_id']}")
    for rr in rounds:
        key = config["seed"], rr["epoch"]
        if key not in source_hashes:
            source_hashes[key] = sequence_hash(sources_for_round("fd001", n, *key))
        if (rr.get("source_sha256") != source_hashes[key] or
                rr.get("optimizer_steps_cumulative") != rr["epoch"]*math.ceil(n/256)):
            raise ValueError(f"Original FD001 source/update mismatch: {config['run_id']}")
    scores = {row["optimizer_steps"]: float(row["mse_standardized"]) for row in curve}
    if len(scores) != len(curve) or any(not math.isfinite(x) or x < 0 for x in scores.values()):
        raise ValueError(f"Invalid original FD001 validation values: {config['run_id']}")
    return mean(scores[step] for step in selection_update_steps(total))


def _select_old(dataset, root):
    manifest = _original_manifest(dataset)
    groups = defaultdict(list)
    shared, source_hashes = {}, {}
    for config in manifest:
        score = _audit_reference(config, root, shared, source_hashes)
        groups[(config["method"], lag_key(config), config["configuration"])].append((config,score))
    options = defaultdict(list)
    for (method, lag, setting), records in groups.items():
        records.sort(key=lambda item: item[0]["seed"])
        if [c["seed"] for c, _ in records] != list(range(5)):
            raise ValueError(f"Incomplete original five-seed grid: {method}/{lag}/{setting}")
        scores = [v for _,v in records]
        score = mean(scores) if all(math.isfinite(x) for x in scores) else math.inf
        config = records[0][0]
        options[(method,lag)].append((score,config["alpha"],config["beta"] or 0,setting,records))
    choices=[]
    expected_groups = {("mse","none")} | {
        ("td_typed",lag_key(c)) for c in manifest if c["method"]=="td_typed"}
    if set(options) != expected_groups:
        raise ValueError(f"Original method/lag groups incomplete: {set(options)^expected_groups}")
    for (method,lag), opts in sorted(options.items()):
        selected=min(opts,key=lambda row:row[:4])
        if not math.isfinite(selected[0]):
            raise ValueError(f"No valid original B256 setting for {method}/{lag}")
        records=selected[4]
        paths={c["run_id"]: file_hash(root/c["run_id"]/"final.pt") for c,_ in records}
        choices.append(dict(method=method,lag_set=lag,alpha=records[0][0]["alpha"],
                            beta=records[0][0]["beta"],configuration=selected[3],
                            score=selected[0],run_ids=[c["run_id"] for c,_ in records],
                            endpoint_sha256=paths))
    return choices


def select(dataset):
    from .typed_ablation_analysis import select as select_new
    new_plan=select_new(dataset)
    old_data, old_root, _=old_paths(dataset)
    choices=_select_old(dataset,old_root)
    _,new_root=new_paths(dataset)
    output=new_root/"typed_ablation_outputs"
    plan=dict(revision=REVISION,dataset=dataset,old_method_scope=list(REFERENCE_METHODS),
              new_method_scope=list(ABLATIONS),batch_size=256,
              new_selection_sha256=_digest(new_plan),old_selections=choices)
    target=output/"four_method_selection.json"
    if target.is_file() and read_json(target)!=plan:
        raise RuntimeError("Old/new selection changed; refusing to replace previous lock")
    atomic_json(target,plan)
    print(f"Validation-only selection locked: {len(choices)} old B256 choices, 6 new choices")
    return plan


def _read_csv(path):
    if not path.is_file():
        return []
    with path.open(newline="",encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _existing_old_metrics(output, run_id, config, keys):
    # Reuse recorded old endpoints only when their run IDs and lag/batch match.
    for filename in ("endpoint_per_seed_metrics.csv", "per_seed_metrics.csv"):
        rows=_read_csv(output/filename)
        existing={}
        for r in rows:
            if (r.get("run_id")!=run_id or r.get("split")!="test" or
                    r.get("method")!=config["method"] or
                    r.get("lag_set")!=lag_key(config) or
                    int(r.get("batch_size",0))!=256 or
                    str(r.get("horizon","all"))!="all"):
                continue
            k=r["units"],r["metric"]
            if k in keys and k not in existing:
                existing[k]=float(r["value"])
        if len(existing)==len(keys) and all(math.isfinite(v) for v in existing.values()):
            return existing
    return None


def _fresh_old_metrics(dataset, config, checkpoint, test, keys):
    # Endpoint *inference* only, not old model refitting or hyperparameter tuning.
    import torch
    from fulltd.metrics import evaluate
    if dataset=="fd001":
        from .fd001_train import build_objective
        objective=build_objective(config)
    else:
        from .application_train import build_objective
        objective=build_objective(config,test)
    objective.load_state_dict(checkpoint["objective"])
    objective.eval()
    records,_=evaluate(objective.model,test,batch_size=128,device="cpu")
    found={(r["units"],r["metric"]):r["value"] for r in records
           if str(r["horizon"])=="all" and (r["units"],r["metric"]) in keys}
    if set(found)!=set(keys):
        raise ValueError(f"Missing old endpoint metrics {set(keys)-set(found)}")
    return found


def evaluate(dataset):
    import torch
    from fulltd.data import load_dataset, verify_dataset
    from .typed_ablation_analysis import _digest as new_digest
    new_data,new_root=new_paths(dataset)
    old_data,old_root,old_output=old_paths(dataset)
    output=new_root/"typed_ablation_outputs"
    plan=read_json(output/"four_method_selection.json")
    current_new=read_json(output/"selection.json")
    if (plan["revision"]!=REVISION or plan["dataset"]!=dataset or
            plan["new_selection_sha256"]!=new_digest(current_new)):
        raise ValueError("Old/new selection lock does not match this experiment")
    if plan["old_selections"]!=_select_old(dataset,old_root):
        raise ValueError("Original validation evidence changed since selection")
    if new_data!=old_data:
        raise ValueError("Old and new campaigns use different prepared-data directories")
    keys=_metric_keys(dataset)
    new_records=_read_csv(output/"test_per_seed_metrics.csv")
    if not new_records:
        raise RuntimeError("First run typed_ablation_analysis --evaluate; no new test results")
    selected_new={(c["method"],c["lag_set"]):c for c in current_new["selected"]}
    if len(selected_new)!=6 or len(current_new["selected"])!=6:
        raise ValueError("Expected two new typed methods at all three lag settings")
    new_index={}
    for r in new_records:
        if str(r.get("horizon"))!="all":
            continue
        selection=selected_new.get((r["method"],r["lag_set"]))
        if selection is None:
            raise ValueError("New test results contain an unselected method or lag")
        if (float(r["alpha"])!=float(selection["alpha"]) or
                float(r["beta"])!=float(selection["beta"])):
            raise ValueError("New test results do not match validation-selected rates")
        k=(r["method"],r["lag_set"],int(r["seed"]),r["units"],r["metric"])
        if k in new_index:
            raise ValueError(f"Duplicate new test metric {k}")
        new_index[k]=float(r["value"])
    result=[]
    for (method,lag),choice in selected_new.items():
        if method not in ABLATIONS or len(choice["run_ids"])!=5:
            raise ValueError("Unexpected new method or incomplete selection")
        for seed in range(5):
            for unit,metric in keys:
                k=method,lag,seed,unit,metric
                if k not in new_index or not math.isfinite(new_index[k]):
                    raise ValueError(f"Missing or invalid new test metric: {k}")
                result.append(dict(dataset=dataset,method=method,label=LABELS[method],
                                   lag_set=lag,batch_size=256,seed=seed,units=unit,
                                   metric=metric,value=new_index[k],source="new_saved_test"))
    old_by_id={c["run_id"]:c for c in _original_manifest(dataset)}
    test=None
    torch.set_num_threads(1)
    for choice in plan["old_selections"]:
        for run_id in choice["run_ids"]:
            c=old_by_id[run_id]
            checkpoint_path=old_root/run_id/"final.pt"
            if file_hash(checkpoint_path)!=choice["endpoint_sha256"][run_id]:
                raise ValueError(f"Old checkpoint changed: {run_id}")
            values=_existing_old_metrics(old_output,run_id,c,keys)
            origin="old_saved_test"
            if values is None:
                # Held-out test is opened only after the combined selection lock.
                if test is None:
                    verify_dataset(dataset,old_data)
                    test=load_dataset(dataset,old_data,"test")
                saved=torch.load(checkpoint_path,map_location="cpu",weights_only=False)
                if (saved["config"]!=c or
                        saved["identity"]!=read_json(old_root/run_id/"status.json")["identity"]):
                    raise ValueError(f"Old endpoint identity mismatch: {run_id}")
                values=_fresh_old_metrics(dataset,c,saved,test,keys)
                origin="old_checkpoint_inference"
            for (unit,metric),val in values.items():
                result.append(dict(dataset=dataset,method=c["method"],label=LABELS[c["method"]],
                                   lag_set=lag_key(c),batch_size=256,seed=c["seed"],units=unit,
                                   metric=metric,value=val,source=origin))
    # There should be precisely five seeds per method/lag/metric and no duplicates.
    groups=defaultdict(list)
    for r in result:
        groups[(r["method"],r["lag_set"],r["units"],r["metric"])].append(r)
    expected=3*3+1 # three TD methods at three lags + one MSE
    if len(groups)!=expected*len(keys) or any(sorted(r["seed"] for r in rows)!=list(range(5))
                                               for rows in groups.values()):
        raise ValueError("Four-method comparison is incomplete or contains duplicate seeds")
    summary=[]
    for (method,lag,unit,metric),rs in sorted(groups.items()):
        vals=[float(r["value"]) for r in sorted(rs,key=lambda x:x["seed"])]
        summary.append(dict(dataset=dataset,method=method,label=LABELS[method],
                            lag_set=lag,units=unit,metric=metric,n=5,
                            mean=mean(vals),std_error=stdev(vals)/(5**.5)))
    # Within-lag comparisons are paired by seed; the same MSE reference is reused.
    paired=[]
    for lag in {g[1] for g in groups if g[0]!="mse"}:
        for unit,metric in keys:
            for other in ("mse","td_semi_typed","td_nonnegative_typed"):
                a=sorted(groups[("td_typed",lag,unit,metric)],key=lambda x:x["seed"])
                b=sorted(groups[(other,"none" if other=="mse" else lag,unit,metric)],key=lambda x:x["seed"])
                diff=[float(x["value"])-float(y["value"]) for x,y in zip(a,b)]
                paired.append(dict(dataset=dataset,lag_set=lag,units=unit,metric=metric,
                                   method_a="td_typed",method_b=other,n=5,
                                   mean_paired_difference=mean(diff),
                                   paired_standard_error=stdev(diff)/(5**.5),
                                   win_fraction=sum(v<0 for v in diff)/5))
    # Read-only historical roots. All generated reports live with the new ablations.
    for name,rows in (("four_method_per_seed.csv",result),
                      ("four_method_summary.csv",summary),
                      ("four_method_paired.csv",paired)):
        target=output/name
        if target.is_file() and _read_csv(target)!=[
                {k:str(v) for k,v in r.items()} for r in rows]:
            raise RuntimeError(f"Existing comparison report differs; refuse overwrite: {target}")
        if not target.is_file():
            write_csv(target,rows)
    print(f"Four-method B256 comparison ready: {output}")
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset",choices=("fd001","electricity","metr_la"))
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--select",action="store_true")
    mode.add_argument("--evaluate",action="store_true")
    args=parser.parse_args()
    select(args.dataset) if args.select else evaluate(args.dataset)


if __name__=="__main__":
    main()
