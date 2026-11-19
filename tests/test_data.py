"""Small deterministic preprocessing checks; no scientific outcome tuning."""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from fulltd.data import (aggregate_electricity, prepare_electricity,
                         prepare_metr_la, prepare_fd001, load_dataset, _time_windows)


def test_electricity_nominal_hours_and_train_only_scaling(tmp_path):
    time=pd.date_range("2011-01-01 00:15",periods=4*900,freq="15min")
    base=np.arange(len(time),dtype=float)
    frame=pd.DataFrame({"active":base+1,"late":np.r_[np.zeros(4*700),np.ones(4*200)],
                        "constant":np.ones(len(time)),"starts":np.r_[np.zeros(4*30),base[4*30:]+1]},index=time)
    hourly,incomplete=aggregate_electricity(frame)
    assert hourly.index[0]==pd.Timestamp("2011-01-01 01:00")
    assert hourly.iloc[0,0]==2.5 and incomplete==[]
    metadata=prepare_electricity(tmp_path,frame)
    assert metadata["entity_names"]==["active","starts"]
    assert len(metadata["excluded_customers"])==2
    original_mean=metadata["mean"]
    changed=frame.copy()
    changed.iloc[4*630:,:]+=100000
    assert prepare_electricity(tmp_path/"changed",changed)["mean"]==original_mean
    train=load_dataset("electricity",tmp_path,"train")
    val=load_dataset("electricity",tmp_path,"val")
    test=load_dataset("electricity",tmp_path,"test")
    # Metadata origins are actual native-hour timestamps, no positional wrap.
    assert train.origins.max()+24 < val.origins.min()+1
    assert val.origins.max()+24 < test.origins.min()+1
    batch=train.get_batch([0,1])
    assert batch["x"].shape==(2,1,168) and batch["y"].shape==(2,24)
    assert train.inverse(batch["y"],[0,1])[0,0]==pytest.approx(hourly.iloc[168,0],rel=1e-6)
    assert np.array_equal(train.persistence([0,1]),batch["x"][:,0,-24:])


def test_incomplete_hour_is_logged_and_gap_not_crossed():
    index=pd.date_range("2011-01-01 00:15",periods=32,freq="15min")
    frame=pd.DataFrame({"v":np.arange(32,dtype=float)},index=index).drop(index[13])
    hourly,incomplete=aggregate_electricity(frame)
    assert len(incomplete)==1 and incomplete[0]["readings"]==3
    t=hourly.index.to_numpy().astype("datetime64[h]").astype(np.int64)
    rows=_time_windows(t,2,2,0,len(t))
    for row in rows:
        assert np.all(np.diff(t[row-1:row+3])==1)
    invalid=frame.copy(); invalid.iloc[0,0]=np.nan
    with pytest.raises(ValueError,match="nonfinite"):
        aggregate_electricity(invalid)


def test_metr_order_masks_causal_fill_and_no_gap_windows(tmp_path):
    index=pd.date_range("2012-03-01",periods=400,freq="5min").delete(90)
    values=np.arange(len(index)*3,dtype=float).reshape(-1,3)+1
    values[20,0]=0
    values[21,1]=np.nan
    frame=pd.DataFrame(values,index=index,columns=["1","2","3"])
    graph=(["1","2","3"],np.eye(3,dtype=np.float32))
    info=prepare_metr_la(tmp_path,frame,graph)
    assert len(info["time_gaps"])==1
    train=load_dataset("metr_la",tmp_path,"train")
    batch=train.get_batch([0,1])
    assert batch["x"].shape==(2,2,3,12) and batch["y"].shape==(2,3,12)
    assert not batch["mask"][0,0,8] # origin 11 -> target row 20
    idx=np.flatnonzero(train.rows==20)[0]
    assert train.get_batch([idx])["x"][0,0,0,-1]==0
    assert np.array_equal(train.persistence([idx]),np.repeat(train.get_batch([idx])["x"][:,0,:,-1,None],12,-1))
    changed=frame.copy(); changed.iloc[int(.7*len(frame)):]+=50000
    other=prepare_metr_la(tmp_path/"changed",changed,graph)
    assert other["mean"]==info["mean"] and other["scale"]==info["scale"]
    with pytest.raises(ValueError,match="sensor order"):
        prepare_metr_la(tmp_path/"bad",frame,(["2","1","3"],graph[1]))


def _engines(count=10,short=False):
    rows=[]
    for eid in range(1,count+1):
        length=20 if short and eid==1 else 50+eid
        for cycle in range(1,length+1):
            features=np.arange(24,dtype=float)+eid+cycle*.01
            features[0]=1 # constant channel retained and scaled by one
            rows.append(np.r_[eid,cycle,features])
    return np.asarray(rows)


def test_fd001_split_labels_features_prefixes_and_padding(tmp_path):
    train,test=_engines(),_engines(3,short=True)
    info=prepare_fd001(tmp_path,train,test,np.array([10.,20.,30.]))
    expected=np.random.default_rng(12345).permutation(np.arange(1,11))
    assert info["fit_engine_ids"]==expected[:8].tolist()
    assert info["val_engine_ids"]==expected[8:].tolist()
    assert info["constant_channels"]==[0] and info["input_scale"][0]==1
    fit=load_dataset("fd001",tmp_path,"train")
    val=load_dataset("fd001",tmp_path,"val")
    held=load_dataset("fd001",tmp_path,"test")
    assert set(fit.entity_ids).isdisjoint(val.entity_ids)
    assert fit.get_batch([0])["x"].shape==(1,24,30)
    for eid in info["val_engine_ids"]:
        T=50+eid
        endpoints=np.unique(np.clip(np.floor(np.array([.4,.5,.6,.7,.8])*T),30,T-1))
        assert np.array_equal(val.origins[val.entity_ids==eid],endpoints)
    b=held.get_batch([0])
    assert np.all(b["x"][0,:,:11]==b["x"][0,:,0,None])
    assert held.inverse(b["y"],[0])[0,0]==pytest.approx(10,abs=1e-5)
    assert info["test_padded_engines"]==1
    changed=train.copy()
    changed[np.isin(changed[:,0],info["val_engine_ids"]),2:]+=100000
    other=prepare_fd001(tmp_path/"changed",changed,test,np.array([10.,20.,30.]))
    assert other["input_mean"]==info["input_mean"] and other["mean"]==info["mean"]


def test_training_loader_never_opens_test_values(tmp_path):
    prepare_fd001(tmp_path,_engines(),_engines(3),np.array([10.,20.,30.]))
    # The read path must work even when all held-out test payloads are absent.
    for path in (tmp_path/"processed"/"fd001"/"test").glob("*.npy"):
        path.unlink()
    train=load_dataset("fd001",tmp_path,"train")
    assert len(train)>0 and train.get_batch([0])["mask"].all()
