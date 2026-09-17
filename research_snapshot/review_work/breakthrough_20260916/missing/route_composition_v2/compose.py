"""Declared disjoint composition of frozen observed-proxy branches; no fitting."""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
    os.environ[key]='1'
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pyarrow as pa
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'review_work/campaign_20260916'))
sys.path.insert(0,str(ROOT/'review_work/breakthrough_20260916/models/verification_id_template'))
import common
from audit import comparison,metric
from taxiout.metrics import evaluate,season_score
ID,TARGET,TIME=common.ID,common.TARGET,common.MOVEMENT
OUT=ROOT/'private_runs/breakthrough_20260916/missing/route_composition_v2'
BASE=ROOT/'private_runs/breakthrough_20260916'
pa.set_cpu_count(1)
pa.set_io_thread_count(1)


def registry(fold):
    return {'ordinary':(BASE/'models/deeper_audit/simplex_v3'/fold,'airport_shrunk_simplex.parquet'),
        'nonordinary':(BASE/'deeper_lgb/combined'/f'lightgbm_leaf63_aobt_allfinite_{fold}_s20260916','candidate.parquet'),
        'missing':(BASE/'models/missing_forest'/f'extratrees_missing_template_idcontext_{fold}_s20260916','blend25.parquet')}


def routes(proxy):
    proxy=np.asarray(proxy,float)
    finite=np.isfinite(proxy)
    ordinary=finite&(proxy>=0)&(proxy<=7200)
    return {'ordinary':ordinary,'nonordinary':finite&~ordinary,'missing':~finite}


def declaration():
    sources={}
    for fold in ['F1','F3']:
        sources[fold]={}
        for name,(directory,filename) in registry(fold).items():
            record=common.read_json(directory/'manifest.json')
            assert record['status']=='complete'
            if name!='ordinary':
                assert record['seed']==20260916
            sources[fold][name]=dict(directory=str(directory),filename=filename,
                manifest_sha256=common.sha256(directory/'manifest.json'),prediction_sha256=record['outputs'][filename])
    payload=dict(source_sha256=common.sha256(__file__),sources=sources,folds=['F1','F3'],seed=20260916,
        routing='Depends onlyon observedproxy: finite0..7200=>frozenV3airport_simplex; finitenegativeor>7200=>originalleaf63fullcandidate; nonfinite=>frozenET25percentreferenceblend.',
        weights='No weightsfittedorselected. Reuseallfrozencomponentweights; disjointexactrowreplacement.',
        floor='None. Mainpredictionunfloored; no clipping or supportprojection.',
        labels='Originalfullscorelabels/IDs retained unchanged; labelsonlyusedafterroutingforevaluation/sensitivity.',
        diagnostic='Single-dayremovalsvsV3/V2; top2missingroutegainrowsperfold removal; componentrouteSSE contributions.',
        selection='Explicitlyadaptiveafterexposeddevelopmentcomponentselection; nofreshholdout,noofficialscore,no230claim.',
        declaration_timing='This declarationiswrittenbeforethisprocessopensanyrowpredictionfile.')
    path=OUT/'protocol.json'
    if path.exists():
        assert common.read_json(path)['declaration']==payload
    else:
        common.write_json(path,dict(created_utc=common.utc_now(),declaration=payload))
    return payload


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--declare-only',action='store_true')
    args=parser.parse_args()
    declared=declaration()
    if args.declare_only:
        print('DECLARED_COMPOSITION_WITHOUT_ROW_READS',OUT/'protocol.json',flush=True)
        return
    if (OUT/'summary.json').exists():
        raise ValueError('Preserve completed composition')
    path=ROOT/'private_runs/screening_230/data/interim/audit/departures.parquet'
    assert common.sha256(path)==common.read_json(ROOT/'private_runs/screening_230/reports/data_audit.json')['artifacts'][path.name]
    meta=pd.read_parquet(path,columns=[ID,TARGET,TIME,'FLIGHT_ID_mvt','proxy_sec'])
    results={}
    for fold in ['F1','F3']:
        directory=OUT/fold
        directory.mkdir(exist_ok=False)
        reference,refrec=common.reference(fold)
        idx,split,_=common.fold_data(meta,fold,full=True)
        assert common.object_hash(split)==common.object_hash(refrec['split'])
        np.testing.assert_array_equal(reference[ID],meta.iloc[idx['score']][ID])
        np.testing.assert_array_equal(reference[TARGET],meta.iloc[idx['score']][TARGET])
        proxy=meta.iloc[idx['score']].proxy_sec.to_numpy(float)
        np.testing.assert_allclose(proxy,reference.proxy_sec.to_numpy(float),rtol=0,atol=0,equal_nan=True)
        masks=routes(proxy)
        np.testing.assert_array_equal(sum(mask.astype(int) for mask in masks.values()),1)
        components={}
        for name,receipt in declared['sources'][fold].items():
            source=Path(receipt['directory'])
            assert common.sha256(source/'manifest.json')==receipt['manifest_sha256']
            assert common.sha256(source/receipt['filename'])==receipt['prediction_sha256']
            frame=pd.read_parquet(source/receipt['filename'])
            np.testing.assert_array_equal(frame[ID],reference[ID])
            np.testing.assert_array_equal(frame[TARGET],reference[TARGET])
            components[name]=frame.prediction_sec.to_numpy(float)
        pred=np.empty(len(reference),float)
        route=np.empty(len(reference),object)
        for name,mask in masks.items():
            pred[mask]=components[name][mask]
            route[mask]=name
            np.testing.assert_array_equal(pred[mask],components[name][mask])
        assert np.isfinite(pred).all()
        base=reference.prediction_sec.to_numpy(float)
        v3=components['ordinary']
        np.testing.assert_array_equal(v3[~masks['ordinary']],base[~masks['ordinary']])
        np.testing.assert_array_equal(components['missing'][~masks['missing']],base[~masks['missing']])
        y=reference[TARGET].to_numpy(float)
        frame=reference.drop(columns=[TARGET,'error_sec','squared_error','label_bin','month','day'],errors='ignore').copy()
        frame['prediction_sec']=pred
        full,scored=evaluate(frame,reference[[ID,TARGET]])
        scored['composition_route']=route
        scored.to_parquet(directory/'candidate.parquet',index=False)
        day=reference.day.to_numpy()
        missing_gain=np.where(masks['missing'],(base-y)**2-(pred-y)**2,-np.inf)
        largest=np.argsort(missing_gain)[-2:]
        keep=~np.isin(np.arange(len(y)),largest)
        missing_total=float(missing_gain[masks['missing']].sum())
        contribution={}
        for name,mask in masks.items():
            a,b,c=metric(y,base,mask),metric(y,v3,mask),metric(y,pred,mask)
            contribution[name]=dict(n=int(mask.sum()),V2=a,V3=b,composed=c,
                gain_sse_vs_V2=a['sse']-c['sse'],gain_sse_vs_V3=b['sse']-c['sse'])
        contribution_total=sum(r['gain_sse_vs_V2'] for r in contribution.values())
        assert abs(contribution_total-(np.sum((base-y)**2)-np.sum((pred-y)**2)))<1e-4
        sensitivity=scored.iloc[largest][[ID,TARGET,'prediction_sec','composition_route','day']].copy()
        sensitivity['V2']=base[largest]
        sensitivity['V3']=v3[largest]
        sensitivity['gain_sse']=missing_gain[largest]
        sensitivity.to_parquet(directory/'missing_top2_gain.parquet',index=False)
        record=dict(status='complete',fold=fold,protocol_sha256=common.sha256(OUT/'protocol.json'),
            source_sha256=common.sha256(__file__),source_receipts=declared['sources'][fold],split=split,
            route_counts={name:int(mask.sum()) for name,mask in masks.items()},reports=dict(candidate=full),
            metrics={name:metric(y,v) for name,v in [('V2',base),('V3',v3),('composed',pred)]},
            comparisons={'vsV2':comparison(y,base,pred,day),'vsV3':comparison(y,v3,pred,day)},
            route_contributions=contribution,
            missing_top2=dict(ids=reference.iloc[largest][ID].tolist(),gain_sse=float(missing_gain[largest].sum()),missing_total_gain_sse=missing_total,
                share=float(missing_gain[largest].sum()/missing_total)),
            remove_missing_top2={name:metric(y,v,keep) for name,v in [('V2',base),('V3',v3),('composed',pred)]},
            no_floor=True,no_new_weight_fit=True,outputs={p.name:common.sha256(p) for p in directory.glob('*.parquet')})
        common.write_json(directory/'manifest.json',record)
        results[fold]=record
        print('COMPOSED',fold,record['metrics'],'routes',record['route_counts'],flush=True)
    seasonal={name:season_score(results['F1']['metrics'][name],results['F3']['metrics'][name]) for name in ['V2','V3','composed']}
    removed={name:season_score(results['F1']['remove_missing_top2'][name],results['F3']['remove_missing_top2'][name]) for name in ['V2','V3','composed']}
    common.write_json(OUT/'summary.json',dict(status='complete',protocol_sha256=common.sha256(OUT/'protocol.json'),seasonal_rmse=seasonal,
        remove_missing_top2_seasonal=removed,main_is_unfloored=True,limitation='Frozencomponentselectionalreadydevelopment-exposed,missinggainsheavilytaildependent; noofficial/holdout/230claim.'))
    print('COMPOSITION_SEASONAL',seasonal,'REMOVED_MISSING_TOP2',removed,flush=True)


if __name__=='__main__':
    main()
