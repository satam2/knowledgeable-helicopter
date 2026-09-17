"""Declared disjoint finite/missing expert composition; no fitted score weights."""
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'campaign_20260916'))
import common
from taxiout.metrics import evaluate, paired_stability, season_score

ROOT=common.WORKSPACE/'private_runs/breakthrough_20260916'
OUT=common.external_path(ROOT/'route_composition_v1')
FINITE=ROOT/'models/augmented/source_past__source_twosided'
MISSING=ROOT/'missing/id_context_v1/models'


def declare():
    payload={'hypothesis':'Gains on mutually exclusive observed source-availability routes can be combined without fitting a new gate.',
        'finite':'TabM aobt_allfinite plus source peer context',
        'missing':'Historical template plus retrospective ID context',
        'weights':'Use each expert at fixed25percent within its declared route, originalV2 at75percent.',
        'negative_support':'Raw composition is primary; support-projected companion diagnostic uses fitminimum -12.',
        'selection':'Components selected after exposed F1/F3 development results; no untouched confirmation.',
        'known_risk':'Missing-template improvement depends on two November extreme labels; July worsens against guarded reference.',
        'ranking':'No submission or ranking inference; no projected official gain claim.',
        'source_sha256':common.sha256(__file__)}
    path=OUT/'protocol.json'
    if path.exists():
        assert common.read_json(path)['declaration']==payload
    else:
        common.write_json(path,{'created_utc':common.utc_now(),'declaration':payload})
    return path


def checked(path):
    manifest=common.read_json(path/'manifest.json')
    assert manifest['status']=='complete'
    prediction=path/'blend25.parquet'
    assert common.sha256(prediction)==manifest['outputs'][prediction.name]
    return pd.read_parquet(prediction),common.sha256(path/'manifest.json')


def run(fold):
    dest=OUT/fold
    assert not dest.exists()
    ref,reference_manifest=common.reference(fold)
    finite,finite_hash=checked(FINITE/f'tabm_aobt_allfinite_{fold}_s20260916')
    missing,missing_hash=checked(MISSING/f'historical_template_{fold}_s20260916')
    for frame in [finite,missing]:
        assert np.array_equal(frame[common.ID],ref[common.ID])
        assert np.array_equal(frame[common.TARGET],ref[common.TARGET])
    absent=~np.isfinite(ref.proxy_sec.to_numpy(float))
    assert np.array_equal(finite.loc[absent,'prediction_sec'],ref.loc[absent,'prediction_sec'])
    assert np.array_equal(missing.loc[~absent,'prediction_sec'],ref.loc[~absent,'prediction_sec'])
    values=np.where(absent,missing.prediction_sec,finite.prediction_sec)
    dest.mkdir(parents=True)
    reports={}
    for variant,pred in [('raw',values),('support_projected',np.maximum(values,-12))]:
        frame=ref.drop(columns=[common.TARGET,'error_sec','squared_error','label_bin','month','day'],errors='ignore').copy()
        frame['prediction_sec']=pred
        metrics,errors=evaluate(frame,ref[[common.ID,common.TARGET]])
        errors.to_parquet(dest/f'{variant}.parquet',index=False)
        reports[variant]={'metrics':metrics,'stability':paired_stability(ref,errors,repetitions=2000)}
    record={'status':'complete','fold':fold,'created_utc':common.utc_now(),'protocol_sha256':common.sha256(OUT/'protocol.json'),
        'source_sha256':common.sha256(__file__),'finite_manifest_sha256':finite_hash,'missing_manifest_sha256':missing_hash,
        'reference_manifest_sha256':common.object_hash(reference_manifest),'reports':reports,
        'outputs':{p.name:common.sha256(p) for p in dest.glob('*.parquet')}}
    common.write_json(dest/'manifest.json',record)
    return record


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--declare-only',action='store_true')
    p.add_argument('--folds',nargs='+',default=['F1','F3'])
    args=p.parse_args()
    declare()
    if args.declare_only:
        print('DECLARED route composition; no scores computed',flush=True)
    else:
        records={fold:run(fold) for fold in args.folds}
        if set(records)=={'F1','F3'}:
            scores={variant:season_score(records['F1']['reports'][variant]['metrics']['overall'],records['F3']['reports'][variant]['metrics']['overall']) for variant in ('raw','support_projected')}
            common.write_json(OUT/'summary.json',{'seasonal_rmse':scores,'created_utc':common.utc_now()})
            print(scores,flush=True)
