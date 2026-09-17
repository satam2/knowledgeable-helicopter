"""Budgeted RealMLP with explicit temporal validation and independent refit."""
import torch
import numpy as np
from importlib.metadata import version
from pathlib import Path
import time

from preprocessing import FramePreprocessor

MAX_EPOCHS=64


def estimator_class():
    from pytabkit import RealMLP_TD_Regressor
    return RealMLP_TD_Regressor


def selected_epoch(params):
    value=params["stop_epoch"]
    if isinstance(value,dict):value=value["rmse"]
    value=int(value)
    if not 1<=value<=MAX_EPOCHS:raise ValueError(f"Invalid selected epoch {value}")
    return value


def fit(x,y,tuning=None,*,steps=None,seed=20260916,threads=4,device="cuda",max_epochs=MAX_EPOCHS):
    labels=np.asarray(y,dtype=float)
    if not len(x) or len(x)!=len(labels) or not np.isfinite(labels).all():raise ValueError("Invalid training rows/labels")
    if steps is None and tuning is None:raise ValueError("Selection requires explicit temporal tune data")
    if steps is not None and (tuning is not None or not 1<=steps<=max_epochs):raise ValueError("Refit requires only a valid selected epoch")
    if not 1<=max_epochs<=MAX_EPOCHS:raise ValueError("Epoch budget exceeded")
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    started=time.monotonic()
    processor=FramePreprocessor().fit(x)
    train=processor.transform(x)
    params=dict(device=device,random_state=seed,n_cv=1,n_refit=0,n_epochs=max_epochs,batch_size=1024,
        predict_batch_size=2048,hidden_sizes=[256]*3,n_threads=threads,verbosity=1,
        val_metric_name="rmse",clamp_output=False,use_early_stopping=tuning is not None,
        early_stopping_additive_patience=8,early_stopping_multiplicative_patience=1)
    if steps is not None:params["stop_epoch"]=int(steps)
    estimator=estimator_class()(**params)
    if tuning is not None:
        tx,ty=tuning;ty=np.asarray(ty,dtype=float)
        if len(tx)!=len(ty) or not len(ty) or not np.isfinite(ty).all():raise ValueError("Invalid temporal tune data")
        estimator.fit(train,labels,X_val=processor.transform(tx),y_val=ty,cat_col_names=processor.categorical)
        chosen=selected_epoch(estimator.fit_params_)
    else:
        # Empty explicit validation indices disable the library's random split.
        estimator.fit(train,labels,val_idxs=np.empty(0,dtype=np.int64),cat_col_names=processor.categorical)
        chosen=int(steps)
    evidence={"library":"pytabkit","version":version("pytabkit"),"family":"RealMLP_TD_Regressor_budgeted",
        "steps":chosen,"rows":len(x),"fit_seconds":time.monotonic()-started,"seed":seed,"threads":threads,
        "params":params,"fit_params":getattr(estimator,"fit_params_",{}),"device":device,
        "preprocessing":"fit-only median+missing flags; fixed fit-category vocabulary; native categorical embeddings",
        "objective":"raw MSE with library affine target normalization; clamp_output disabled; labels unchanged",
        "training_scope":"Caller-supplied permitted subset; fit/refit sampled independently from original fold indices"}
    return {"estimator":estimator,"processor":processor,"steps":chosen,"threads":threads},evidence


def predict(model,x):
    torch.set_num_threads(model["threads"])
    result=np.asarray(model["estimator"].predict(model["processor"].transform(x)),dtype=float).reshape(-1)
    if len(result)!=len(x) or not np.isfinite(result).all():raise ValueError("Invalid prediction output")
    return result
