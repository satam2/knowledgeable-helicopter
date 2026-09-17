"""Frozen V3 release composition; no model selection or ranking truth access."""
import lightgbm
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'review_work/campaign_20260916'))
import common
from taxiout.submission import build_submission
ID=common.ID
OUT=common.external_path(ROOT/'private_runs/tail240_20260916/final_submission_v3_v2')
WEIGHTS=ROOT/'private_runs/breakthrough_20260916/models/context_gate/final_simplex9_v1'
ORDINARY=ROOT/'private_runs/tail240_20260916/final_ordinary/v2'
PLE=ROOT/'private_runs/tail240_20260916/state/final_ple387/model_v2/fit'
MISSING=ROOT/'private_runs/tail240_20260916/forensics/final_missing/v1/models'
EXPERTS={'tabm_combined':'tabm225','tabm_ple8':'ple387','lgb63_sequence8':'lgb449',
         'lgb63_union':'lgb387','catboost_combined':'cat225'}


def release_weights(f1,f3):
    a,b=192122/344841,152719/344841
    original=a*np.asarray(f1,float)+b*np.asarray(f3,float)
    weights=original.copy()
    weights[weights<1e-12]=0.
    weights/=weights.sum()
    assert (weights>=0).all() and np.isclose(weights.sum(),1.)
    return original,weights


def route(proxy,ordinary,finite_other,missing):
    proxy=np.asarray(proxy,float)
    ordinary,finite_other,missing=[np.asarray(v,float) for v in (ordinary,finite_other,missing)]
    if any(v.shape!=proxy.shape for v in (ordinary,finite_other,missing)):
        raise ValueError('Route inputs must have identical shape')
    finite=np.isfinite(proxy)
    regular=finite&(proxy>=0)&(proxy<=7200)
    result=np.empty(len(proxy),float)
    result[regular]=ordinary[regular]
    result[finite&~regular]=finite_other[finite&~regular]
    result[~finite]=missing[~finite]
    if not np.isfinite(result).all():raise ValueError('Missing or nonfinite route output')
    return result,regular,finite&~regular,~finite


def declare():
    f1,f3=[common.read_json(WEIGHTS/f'{fold}_weights.json') for fold in ['F1','F3']]
    assert f1['experts']==f3['experts']
    original,weights=release_weights(f1['global'],f3['global'])
    active={name:float(weight) for name,weight in zip(f1['experts'],weights) if weight>0}
    assert set(active)==set(EXPERTS)
    record=dict(recipe='Best verified local268.662991 ensemble full-year release',
        source_sha256=common.sha256(__file__),local_development_rmse=268.662991126,
        preserved_v1_source_sha256=common.sha256(Path(__file__).with_name('compose_v1_frozen.py')),
        component_directories=dict(ordinary=str(ORDINARY),ple=str(PLE),missing=str(MISSING)),
        weight_sources={f:common.sha256(WEIGHTS/f'{f}_weights.json') for f in ['F1','F3']},
        experts=f1['experts'],original_weights=original.tolist(),weights=weights.tolist(),active_weights=active,
        weight_policy='Competition-season weighted existing tune-fitted F1/F3 globals; numerical values below1e-12 declared zero, renormalize once; no new weight fit or score/feedback optimization.',
        removed_total_mass=float(original[weights==0].sum()),
        routes=dict(ordinary='Finite proxy0..7200 inclusive: frozen weighted full-year experts withPLE387.',
                    finite_nonordinary='Full-year original225leaf63, including negative and>7200.',
                    missing='Original nested blends .75*(.75*V2+.25*ExtraTrees)+.25*normalized.'),
        training='All eligible2025 rows per existing routes, final int(median(F1,F3 tune-selected iterations)). Existing V2 missing base reused with exact native replay.',
        submission=dict(endpoint='https://s3.opensky-network.org',bucket='prc-2026-knowledgeable-helicopter',
                        key='knowledgeable-helicopter_v3.parquet',rows=344841,rounding='nearest-even int32, no clipping'),
        prerequisites='Every component complete and hash-bound; independent final verification before single upload.',
        authority='User explicitly requested fully train best model and make second daily submission.')
    OUT.mkdir(parents=True,exist_ok=True)
    file=OUT/'protocol.json'
    if file.exists():assert common.read_json(file)==record,'Frozen release protocol changed'
    else:common.write_json(file,record)
    return record


def checked_prediction(folder,name,expected_ids):
    marker=common.read_json(folder/'manifest.json')
    assert marker['status']=='complete'
    assert common.sha256(folder/name)==marker['outputs'][name]
    frame=pd.read_parquet(folder/name)
    assert frame[ID].is_unique and len(frame)==len(expected_ids)
    assert set(frame[ID])==set(expected_ids)
    values=frame.set_index(ID).loc[expected_ids,'prediction_sec'].to_numpy(float)
    assert np.isfinite(values).all()
    return values,dict(folder=str(folder),manifest_sha256=common.sha256(folder/'manifest.json'),
                       predictions_sha256=marker['outputs'][name])


def compose():
    protocol=declare()
    assert not (OUT/'submission_ready.json').exists(),'Preserve completed submission'
    ranking=ROOT/'private_runs/submission_v2/ranking_meta.parquet'
    binding=common.read_json(ranking.parent/'ranking_inputs.json')
    assert common.sha256(ranking)==binding['files'][ranking.name]
    meta=pd.read_parquet(ranking)
    finite=np.isfinite(meta.proxy_sec)
    finite_ids=meta.loc[finite,ID]
    missing_ids=meta.loc[~finite,ID]
    total=np.zeros(len(finite_ids),float)
    receipts={}
    for name,short in EXPERTS.items():
        folder=PLE if short=='ple387' else ORDINARY/short
        values,receipts[name]=checked_prediction(folder,'ranking_predictions.parquet',finite_ids)
        total+=protocol['active_weights'][name]*values
    other,receipts['finite_nonordinary']=checked_prediction(ORDINARY/'lgb225','ranking_predictions.parquet',finite_ids)
    missing,receipts['missing']=checked_prediction(MISSING,'missing_ranking.parquet',missing_ids)
    ordinary_all=np.full(len(meta),np.nan);ordinary_all[finite]=total
    other_all=np.full(len(meta),np.nan);other_all[finite]=other
    missing_all=np.full(len(meta),np.nan);missing_all[~finite]=missing
    prediction,regular,unusual,absent=route(meta.proxy_sec,ordinary_all,other_all,missing_all)
    result=meta[[ID,'proxy_sec']].assign(prediction_sec=prediction,
        route=np.select([regular,unusual,absent],['ordinary_ensemble','finite_nonordinary_lgb225','missing_blend'],default='invalid'))
    result.to_parquet(OUT/'ranking_predictions.parquet',index=False)
    template=ROOT/'data/09-15-2026-18-55-03_files_list/submitting.parquet'
    filename=OUT/protocol['submission']['key']
    report=build_submission(template,result[[ID,'prediction_sec']],filename)
    assert report['passed'] and report['rows']==344841
    common.write_json(OUT/'submission_ready.json',dict(status='ready_for_independent_verification',
        protocol_sha256=common.sha256(OUT/'protocol.json'),submission=report,components=receipts,
        route_counts=dict(ordinary=int(regular.sum()),finite_nonordinary=int(unusual.sum()),missing=int(absent.sum())),
        prediction_sha256=common.sha256(OUT/'ranking_predictions.parquet'),bytes=filename.stat().st_size,
        official_score=None,uploaded=False))
    print('SUBMISSION_READY',report,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--declare-only',action='store_true');args=parser.parse_args()
    if args.declare_only:print(declare())
    else:compose()
