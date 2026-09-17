"""Frozen lexical covariates added to verified normalized raw-MSE model."""
import os
for key in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"]:
    os.environ[key]="2"
import argparse
import importlib.util
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
from threadpoolctl import threadpool_limits
from run_lexical import ROOT,NUMERIC,CATEGORICAL,lexical_features,decode_flight_tokens

CONTEXT_SOURCE=ROOT / "review_work/tail240_20260916/state/normalized_context/run.py"
spec=importlib.util.spec_from_file_location("normalized_context_reused",CONTEXT_SOURCE)
context=importlib.util.module_from_spec(spec)
spec.loader.exec_module(context)
base,common=context.base,context.common
ID=base.ID
OUT=ROOT / "private_runs/tail240_20260916/forensics/models/normalized_lexical_v1"
LEXICAL=ROOT / "private_runs/tail240_20260916/forensics/models/lexical_v1"
FEATURES=NUMERIC+CATEGORICAL


def selected_lexical(x,cached):
    if not x.index.is_unique or not cached.index.is_unique:
        raise ValueError("Unique IDs required")
    np.testing.assert_array_equal(x.index,cached.index)
    if set(FEATURES).intersection(x) or not set(FEATURES).issubset(cached):
        raise ValueError("Frozen lexical columns must be complete and nonoverlapping")
    result=cached[FEATURES].copy()
    result.index=x.index.copy()
    fresh=lexical_features(decode_flight_tokens(x.flight))
    pd.testing.assert_frame_equal(result,fresh,check_exact=True)
    return result


def declare():
    sources=[Path(__file__),CONTEXT_SOURCE,Path(base.__file__),Path(base.shared.__file__),
        ROOT / "review_work/tail240_20260916/forensics/run_lexical.py",ROOT / "review_work/breakthrough_20260916/models/encoders.py"]
    record={"created_utc":common.utc_now(),"arms":["control76","lexical91"],"lexical_columns":FEATURES,
        "params":base.PARAMS,"round_cap":600,"patience":60,
        "target":"Exact frozen normalized target Y=900+s(X)*Z; same observed schedule scale and s^2/fitmean weights; raw squared loss identity unchanged.",
        "features":"Exact76 airport+IDcontext control versus same76 plus all15 previously frozen flight-only lexical fields. No new feature selection, public data, DST anchors, or label-defined route.",
        "scope":"Original fullmissing purged F1/F3 fit/tune only; no score prediction/refit. Fullmetadata is loaded but scorelabels unused.",
        "control":"Fresh control predictions and selected77/146iterations must exactly reproduce normalized_missing_tune_v1 observed_schedule_scale before evaluating lexical arm.",
        "gate":"Both June/October allmissing rawRMSE improve normalized76 and all UTCday removals improve; report300 daybootstrap and top1/5/10 beneficialrow removal sensitivity. No automatic scoreadvance.",
        "novelty":"Tests whether previously observed source-regime lexical association helps the independently successful magnitude-scaled target parameterization. CatBoost unscaled lexical result was negative againstglobal9; reuse every frozen lexicalfield unchanged.",
        "resource":"2CPU6GiBprocess8GiBavailable reserve",
        "source_hashes":{str(p.relative_to(ROOT)):common.sha256(p) for p in sources},
        "normalized_protocol_sha256":common.sha256(base.OUT / "protocol.json"),
        "lexical_completion_sha256":common.sha256(LEXICAL / "completion.json")}
    path=OUT / "protocol.json"
    if path.exists():
        old=common.read_json(path);record["created_utc"]=old["created_utc"];assert record==old
    else:
        OUT.mkdir(parents=True,exist_ok=False);common.write_json(path,record)
    return record


def prepare():
    x,meta=base.shared.load_missing()
    expected=common.read_json(ROOT / "private_runs/breakthrough_20260916/missing/id_context_v1/feature_manifest.json")["columns"]
    assert list(x)==expected and len(x.columns)==76 and len(x)==22470
    path=LEXICAL / "all_missing_features.parquet"
    assert common.sha256(path)==common.read_json(LEXICAL / "completion.json")["feature_sha256"]
    extra=selected_lexical(x,pd.read_parquet(path).set_index(ID))
    extra.reset_index().to_parquet(OUT / "lexical.parquet",index=False)
    common.write_json(OUT / "cache_manifest.json",{"status":"complete","rows":len(extra),"columns":FEATURES,
        "original_missing_ids_hash":common.object_hash(x.index.tolist()),"feature_sha256":common.sha256(OUT / "lexical.parquet"),
        "frozen_lexical_source_cache_sha256":common.sha256(path),"flight_only_regeneration_exact":True,
        "protocol_sha256":common.sha256(OUT / "protocol.json")})
    return x,extra,meta


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--declare-only",action="store_true");args=parser.parse_args()
    protocol=declare()
    if args.declare_only:
        print("Declared normalized76 versus frozen lexical91, tune only",flush=True);return
    pa.set_cpu_count(2);pa.set_io_thread_count(1)
    with threadpool_limits(2):
        x,extra,meta=prepare()
        context.OUT=OUT
        for fold in ["F1","F3"]:
            context.run(fold,x,extra,meta,protocol)
        records={fold:common.read_json(OUT / fold / "lexical91/metrics.json") for fold in ["F1","F3"]}
        passed=all(r["rmse_improvement"]>0 and r["all_day_removals_improve"] for r in records.values())
        common.write_json(OUT / "summary.json",{"status":"complete","gate_passed":passed,"folds":records,
            "no_score_prediction":True,"source_sha256":common.sha256(__file__),"protocol_sha256":common.sha256(OUT / "protocol.json")})


if __name__=="__main__":main()
