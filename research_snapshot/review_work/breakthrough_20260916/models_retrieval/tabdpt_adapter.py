"""Offline bounded pretrained retrieval: earlier labeled references only."""
import os
os.environ["HF_HUB_OFFLINE"]="1"
os.environ["HF_HUB_DISABLE_TELEMETRY"]="1"
os.environ["TRANSFORMERS_OFFLINE"]="1"

import torch
import hashlib
import json
from pathlib import Path
import time
from importlib.metadata import version
import numpy as np

from preprocessing import RetrievalPreprocessor

ROOT=Path(__file__).resolve().parents[3]
CHECKPOINT=ROOT/"output/breakthrough_20260916/models_retrieval/checkpoint"
REFERENCE_LIMIT=32000
CONTEXT=256
BATCH=16
ENSEMBLES=1


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1048576),b""):h.update(b)
    return h.hexdigest()


def checked_checkpoint():
    receipt=json.loads((CHECKPOINT/"receipt.json").read_text())
    path=Path(receipt["file"])
    if sha(path)!=receipt["sha256"]:raise ValueError("Checkpoint hash mismatch")
    if receipt["license"]!="Apache-2.0":raise ValueError("Unexpected checkpoint license")
    return path,receipt


def estimator_class():
    from tabdpt import TabDPTRegressor
    return TabDPTRegressor


def fit(x,y,tuning=None,*,steps=None,seed=20260916,threads=4,device="cuda",max_epochs=None):
    if len(x)>REFERENCE_LIMIT:raise ValueError("Reference subset exceeds32K predeclared budget")
    labels=np.asarray(y,dtype=float)
    if not len(x) or len(x)!=len(labels) or not np.isfinite(labels).all():raise ValueError("Invalid reference rows/labels")
    if tuning is not None and steps is not None:raise ValueError("Ambiguous fit/refit request")
    # Tuning is deliberately not supplied to the in-context reference bank.
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    import faiss
    faiss.omp_set_num_threads(threads)
    path,receipt=checked_checkpoint()
    start=time.monotonic()
    processor=RetrievalPreprocessor().fit(x)
    encoded=processor.transform(x)
    params=dict(device=device,context_reduction="retrieval",compile=False,use_flash=True,
        model_weight_path=str(path),missing_indicators=False,feature_reduction="pca",verbose=False)
    estimator=estimator_class()(**params)
    estimator.fit(encoded,labels)
    evidence={"library":"tabdpt","version":version("tabdpt"),"steps":1,"family":"TabDPT1.3_reference32K",
        "rows":len(x),"reference_rows":len(x),"fit_seconds":time.monotonic()-start,"seed":seed,"threads":threads,
        "params":params,"prediction_params":{"context_size":min(CONTEXT,len(x)),"batch_size":BATCH,"n_ensembles":ENSEMBLES,"seed":seed},
        "checkpoint_sha256":receipt["sha256"],"checkpoint_revision":receipt["revision"],"encoded_features":encoded.shape[1],
        "checkpoint_max_features":int(estimator.max_features),"pca_used":encoded.shape[1]>estimator.max_features,
        "preprocessing":"fit-reference-only median and missing flags; train-only frequency-capped onehot max16/category; library standardization and PCA if>checkpoint featurelimit",
        "objective":"pretrained regression-bin conditional mean; no task gradient training, no label clipping",
        "tune_labels_in_reference":False,"network_disabled":True,"device":device}
    model={"estimator":estimator,"processor":processor,"steps":1,"threads":threads,"seed":seed,"context_size":min(CONTEXT,len(x))}
    return model,evidence


def predict(model,x):
    torch.set_num_threads(model["threads"])
    result=[]
    for start in range(0,len(x),2048):
        encoded=model["processor"].transform(x.iloc[start:start+2048])
        result.append(np.asarray(model["estimator"].predict(encoded,context_size=model["context_size"],batch_size=BATCH,n_ensembles=ENSEMBLES,seed=model["seed"]),dtype=float).reshape(-1))
    prediction=np.concatenate(result) if result else np.empty(0,dtype=float)
    if len(prediction)!=len(x) or not np.isfinite(prediction).all():raise ValueError("Invalid predictions")
    return prediction
