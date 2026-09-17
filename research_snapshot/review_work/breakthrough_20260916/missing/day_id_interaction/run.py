"""Exact original day decomposition with the separately frozen MVT-ID context."""
import argparse
import gc
import sys
import time
import traceback
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
import run_missing_models as base
import run_id_context as context
from taxiout.artifacts import read_json,write_json,sha256,object_hash,utc_now
from taxiout.paths import external_path
from taxiout.schema import ID,TARGET,MOVEMENT

ROOT=HERE.parents[3]
OUT=external_path(ROOT/'private_runs/breakthrough_20260916/missing/day_id_interaction')


def declare(args):
    original=read_json(ROOT/'private_runs/breakthrough_20260916/missing/protocol.json')
    for filename,digest in original['declaration']['source_hashes'].items():
        assert sha256(ROOT/filename)==digest
    cache=read_json(context.CACHE/'audit.json')
    assert sha256(context.CACHE/'features.parquet')==cache['feature_sha256']
    assert sha256(ROOT/'review_work/mechanism_20260916/id_neighborhood.py')==cache['script_sha256']
    declaration=dict(seed=args.seed,threads=args.threads,folds=args.folds,arm='day_decomposition',
        source_hashes={str(p.relative_to(ROOT)):sha256(p) for p in [Path(__file__),Path(base.__file__),Path(context.__file__)]},
        base_protocol_sha256=sha256(ROOT/'private_runs/breakthrough_20260916/missing/protocol.json'),
        id_cache_sha256=cache['feature_sha256'],id_audit_sha256=sha256(context.CACHE/'audit.json'),
        model='Exact original day_decomposition: D=max(0,floor(Y/86400)); R=Y-86400D; fullrawYreconstruction, probabilitydaymean, originaltune-onlyridge calibration.',
        features='Originalairport38 plus exactfrozen38MVT-IDcontext. NOT NMflightIDcache. Sameairport/month peers, selfexcluded, possiblyfuturemovementrows.',
        hypothesis='Interaction ofrare dayclass with source/extractiondate signatures. Adaptive afterknown13day+shorttaxicases; no newlyuntouched validationclaim.',
        tune='Fitmodel onoriginalmissingfit, classifier/remainderstopping+daycalibration onoriginalmissingtune. Reportuncalibrated andcalibratederror; calibratedtune isoptimisticselectionevidence.',
        score='Optionalcentrallyscheduledfullstage invokesfrozenbase.run_arm; completeoriginalscorecohorts, originalrawlabels, V2exactelsewhere, fixed25percentblend.',
        params=base.PARAMS,classifier='Original400depth4lr.05L2=30; exactdayclasses and60roundstopping',
        resources='CPU2threads,noGPU. Preparation onlysynthetic tests anddeclaration; parentlaunchesprivatefit/tune.')
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/'protocol.json'
    if path.exists():
        if read_json(path)['declaration']!=declaration:
            raise ValueError('Frozen dayID protocol changed')
    else:
        write_json(path,dict(created_utc=utc_now(),declaration=declaration))
    return path


def load():
    x,meta=base.load_data()
    peer=pd.read_parquet(context.CACHE/'features.parquet').set_index(ID)
    np.testing.assert_array_equal(peer.index,meta[ID])
    times=meta.set_index(ID).loc[x.index,MOVEMENT]
    extension=context.id_context_features(x,peer.loc[x.index],times)
    if len(extension.columns)!=38:
        raise ValueError('Expected exact38contextfeatures')
    return x,pd.concat([x,extension],axis=1),meta


def rmse(y,p):
    return float(np.sqrt(np.mean((np.asarray(y,float)-np.asarray(p,float))**2)))


def tune_only(args,fold,original,x,meta,protocol):
    path=OUT/'tune'/f'{fold}_s{args.seed}'
    if path.exists():
        old=read_json(path/'manifest.json')
        if old['status']!='complete' or old['protocol_sha256']!=sha256(protocol):
            raise ValueError('Priorincomplete/differenttune preserved')
        for filename,digest in old['outputs'].items():
            assert sha256(path/filename)==digest
        print('REUSED',fold,flush=True)
        return
    idx,split,_=base.common.fold_data(meta,fold,full=True)
    missing=~np.isfinite(meta.proxy_sec.to_numpy(float))
    rows={stage:positions[missing[positions]] for stage,positions in idx.items() if stage in ['fit','tune']}
    frames={stage:x.loc[meta.iloc[positions][ID]] for stage,positions in rows.items()}
    times={stage:pd.Series(pd.to_datetime(meta.iloc[positions][MOVEMENT],utc=True).to_numpy(),index=frames[stage].index) for stage,positions in rows.items()}
    y=meta[TARGET].to_numpy(float)
    path.mkdir(parents=True)
    record=dict(status='running',created_utc=utc_now(),fold=fold,split=split,protocol_sha256=sha256(protocol),
        features=list(x),fit_ids={stage:dict(n=len(positions),hash=object_hash(meta.iloc[positions][ID].tolist())) for stage,positions in rows.items()})
    write_json(path/'manifest.json',record)
    started=time.monotonic()
    try:
        model,evidence=base.fit_arm('day_decomposition',frames['fit'],y[rows['fit']],times['fit'],
            tune=(frames['tune'],y[rows['tune']],times['tune']),seed=args.seed,threads=args.threads)
        joblib.dump(model,path/'fit_model.joblib')
        prediction=base.predict_arm(model,frames['tune'],times['tune'])
        days=base.predict_days(model,frames['tune'])
        remainder=model['model'].predict(frames['tune'],thread_count=args.threads)
        uncalibrated=remainder+base.DAY*days
        replay=float(np.max(np.abs(prediction-base.predict_arm(joblib.load(path/'fit_model.joblib'),frames['tune'],times['tune']))))
        assert replay<=1e-9 and np.isfinite(prediction).all()
        reference_path=ROOT/'private_runs/breakthrough_20260916/missing/models'/f'day_decomposition_{fold}_s{args.seed}'
        reference=read_json(reference_path/'manifest.json')
        assert reference['status']=='complete' and object_hash(reference['split'])==object_hash(split)
        assert sha256(reference_path/'fit_model.joblib')==reference['outputs']['fit_model.joblib']
        baseline=joblib.load(reference_path/'fit_model.joblib')
        original_tune=original.loc[frames['tune'].index]
        baseline_prediction=base.predict_arm(baseline,original_tune,times['tune'])
        baseline_days=base.predict_days(baseline,original_tune)
        baseline_remainder=baseline['model'].predict(original_tune,thread_count=args.threads)
        comparison=dict(base_calibrated=rmse(y[rows['tune']],baseline_prediction),
            base_uncalibrated=rmse(y[rows['tune']],baseline_remainder+base.DAY*baseline_days),
            id_calibrated=rmse(y[rows['tune']],prediction),id_uncalibrated=rmse(y[rows['tune']],uncalibrated))
        pd.DataFrame({ID:frames['tune'].index,'prediction_sec':prediction,'uncalibrated_sec':uncalibrated,
            'expected_day':days,'remainder_sec':remainder,'baseline_prediction_sec':baseline_prediction}).to_parquet(path/'tune_predictions.parquet',index=False)
        record.update(status='complete',completed_utc=utc_now(),tune=evidence,comparison=comparison,
            runtime_sec=time.monotonic()-started,reload_max_abs_delta=replay,score_labels_used=False,
            baseline_manifest_sha256=sha256(reference_path/'manifest.json'))
        record['outputs']={p.name:sha256(p) for p in path.iterdir() if p.is_file() and p.name!='manifest.json'}
        write_json(path/'manifest.json',record)
        print('DAY_ID_TUNE',fold,comparison,flush=True)
    except Exception as exc:
        record.update(status='failed',error=repr(exc),traceback=traceback.format_exc())
        write_json(path/'manifest.json',record)
        raise


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--folds',nargs='+',choices=['F1','F3'],default=['F1','F3'])
    parser.add_argument('--seed',type=int,default=20260916)
    parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--declare-only',action='store_true')
    parser.add_argument('--tune-only',action='store_true')
    args=parser.parse_args()
    protocol=declare(args)
    if args.declare_only:
        print('DECLARED daydecomposition+38MVT-IDcontext; no training',flush=True)
        return
    original,x,meta=load()
    feature_manifest=dict(rows=len(x),columns=list(x),extension_columns=list(x)[len(original.columns):],protocol_sha256=sha256(protocol))
    manifest=OUT/'features_manifest.json'
    if manifest.exists():
        assert read_json(manifest)==feature_manifest
    else:
        write_json(manifest,feature_manifest)
    if not args.tune_only:
        base.OUT=OUT
    for fold in args.folds:
        if args.tune_only:
            tune_only(args,fold,original,x,meta,protocol)
        else:
            base.run_arm('day_decomposition',fold,x,meta,args.seed,args.threads)
        gc.collect()


if __name__=='__main__':
    main()
