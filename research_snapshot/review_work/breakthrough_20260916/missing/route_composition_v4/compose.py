"""Predeclared final9 route compositions; wait for independent stacker verification."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
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
BASE=ROOT/'private_runs/breakthrough_20260916'
OUT=BASE/'missing/route_composition_v4'
STACK=BASE/'models/context_gate/final_simplex9_v1'
PREVIOUS=BASE/'missing/route_composition_v3'
VARIANTS={'global9':'global4','airport9':'airport4'}
pa.set_cpu_count(1)
pa.set_io_thread_count(1)


def source_receipt(folder,filename):
    marker=common.read_json(folder/'manifest.json')
    assert marker['status']=='complete'
    return dict(directory=str(folder),filename=filename,manifest_sha256=common.sha256(folder/'manifest.json'),prediction_sha256=marker['outputs'][filename])


def declare():
    fixed={}
    for fold in ['F1','F3']:
        fixed[fold]={
            'nonordinary':source_receipt(BASE/'deeper_lgb/combined'/f'lightgbm_leaf63_aobt_allfinite_{fold}_s20260916','candidate.parquet'),
            'missing':source_receipt(BASE/'models/missing_forest'/f'extratrees_missing_template_idcontext_{fold}_s20260916','blend25.parquet'),
            **{variant:source_receipt(PREVIOUS/prior/fold,'candidate.parquet') for variant,prior in VARIANTS.items()}}
    payload=dict(source_sha256=common.sha256(__file__),variants=VARIANTS,folds=['F1','F3'],fixed_sources=fixed,
        final9_directory=str(STACK),final9_runner_sha256=common.sha256(ROOT/'review_work/breakthrough_20260916/models/context_gate/final_simplex9/run.py'),
        routing='Finite proxy0..7200inclusive: frozen correspondingglobal9/airport9. Finite negative/>7200: original225leaf63candidate. Nonfinite: fixedET25percentV2blend.',
        comparison='global9 versus frozen global4 routecomposition; airport9 versus frozen airport4 routecomposition. CommonV2 for overallgains. Reportboth,neverselectonebynewoutcome.',
        selection='Bothvariants,allbranches,thresholds,andreferences declared before final9scoreaccess. No new fit,weights,clipping,floor,projection,or sourcechange.',
        prerequisite='Do not open final9score predictions until independent_verification.json statuspassed and boundscore_summaryhashmatches. Allnineexpertsrequired by upstreamprotocol.',
        receipt_timing='Unknown future final9artifact hashes will be frozen in execution_inputs.json after independentverification and before rowpredictionreads.',
        diagnostics='Full originalscoreIDs/labels,all-dayremovals,top2missinggainsvsV2,top10gainsvsV2andmatchedprevious,weightederrorbudget,independentcoefficient/route replay.',
        availability='Adaptiveexposeddevelopment,retrospectivebatch inputs. Noofficial/unseenholdoutclaim. XGBcrossdevice strictCPUcanaryfailure remains separate fromproducerreload.',
        declared_without_row_prediction_reads=True)
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/'protocol.json'
    if path.exists():
        assert common.read_json(path)['declaration']==payload
    else:
        common.write_json(path,dict(created_utc=common.utc_now(),declaration=payload))
    return payload


def execution_inputs(declared):
    verification=common.read_json(STACK/'independent_verification.json')
    assert verification['status']=='passed'
    assert verification['score_summary_sha256']==common.sha256(STACK/'score_summary.json')
    upstream=common.read_json(STACK/'protocol.json')
    assert len(upstream['experts'])==9 and upstream['deadline_policy'].startswith('All9expertsrequired')
    sources={fold:{v:source_receipt(STACK/f'{fold}_score',v+'.parquet') for v in VARIANTS} for fold in ['F1','F3']}
    receipts={name:common.sha256(STACK/name) for name in ['protocol.json','preparation.json','scoring_protocol.json','score_summary.json',
        'independent_verification.json','F1_weights.json','F3_weights.json']}
    result=dict(predeclaration_sha256=common.sha256(OUT/'protocol.json'),sources=sources,stacker_receipts=receipts,
        created_utc=common.utc_now(),independent_verification_passed_before_score_assembly=True)
    target=OUT/'execution_inputs.json'
    assert not target.exists(),'Preserve attempted execution'
    common.write_json(target,result)
    return result


def load(receipt,reference):
    folder=Path(receipt['directory'])
    assert common.sha256(folder/'manifest.json')==receipt['manifest_sha256']
    assert common.sha256(folder/receipt['filename'])==receipt['prediction_sha256']
    frame=pd.read_parquet(folder/receipt['filename'])
    np.testing.assert_array_equal(frame[ID],reference[ID])
    np.testing.assert_array_equal(frame[TARGET],reference[TARGET])
    return frame.prediction_sec.to_numpy(float)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--declare-only',action='store_true')
    args=parser.parse_args()
    declared=declare()
    if args.declare_only:
        print('FINAL9_ROUTES_PREDECLARED_WITHOUT_SCORE_READS',flush=True)
        return
    incoming=execution_inputs(declared)
    path=ROOT/'private_runs/screening_230/data/interim/audit/departures.parquet'
    assert common.sha256(path)==common.read_json(ROOT/'private_runs/screening_230/reports/data_audit.json')['artifacts'][path.name]
    meta=pd.read_parquet(path,columns=[ID,TARGET,TIME,'FLIGHT_ID_mvt','proxy_sec'])
    results={v:{} for v in VARIANTS}
    for fold in ['F1','F3']:
        reference,refrec=common.reference(fold)
        index,split,_=common.fold_data(meta,fold,full=True)
        assert common.object_hash(split)==common.object_hash(refrec['split'])
        for key in [ID,TARGET]:
            np.testing.assert_array_equal(reference[key],meta.iloc[index['score']][key])
        proxy=meta.iloc[index['score']].proxy_sec.to_numpy(float)
        np.testing.assert_allclose(proxy,reference.proxy_sec.to_numpy(float),rtol=0,atol=0,equal_nan=True)
        finite=np.isfinite(proxy)
        ordinary=finite&(proxy>=0)&(proxy<=7200)
        nonordinary=finite&~ordinary
        missing=~finite
        fixed={name:load(receipt,reference) for name,receipt in declared['fixed_sources'][fold].items()}
        y,base,days=reference[TARGET].to_numpy(float),reference.prediction_sec.to_numpy(float),reference.day.to_numpy()
        for variant in VARIANTS:
            directory=OUT/variant/fold
            directory.mkdir(parents=True,exist_ok=False)
            incoming_prediction=load(incoming['sources'][fold][variant],reference)
            np.testing.assert_array_equal(incoming_prediction[~ordinary],base[~ordinary])
            pred=np.select([missing,nonordinary],[fixed['missing'],fixed['nonordinary']],default=incoming_prediction)
            np.testing.assert_array_equal(pred[~ordinary],fixed[variant][~ordinary])
            assert np.isfinite(pred).all()
            controls={'V2':base,'previous_composition':fixed[variant],'own_ordinary':incoming_prediction,'composed':pred}
            frame=reference.drop(columns=[TARGET,'error_sec','squared_error','label_bin','month','day'],errors='ignore').copy()
            frame['prediction_sec']=pred
            metrics,scored=evaluate(frame,reference[[ID,TARGET]])
            scored['composition_route']=np.select([missing,nonordinary],['missing','nonordinary'],default='ordinary')
            scored.to_parquet(directory/'candidate.parquet',index=False)
            gains=np.where(missing,(base-y)**2-(pred-y)**2,-np.inf)
            top2=np.argsort(gains)[-2:]
            keep=~np.isin(np.arange(len(y)),top2)
            top10={}
            for name in ['V2','previous_composition']:
                gain=(controls[name]-y)**2-(pred-y)**2
                indices=np.argsort(gain)[-10:]
                remaining=~np.isin(np.arange(len(y)),indices)
                a,b=metric(y,controls[name],remaining),metric(y,pred,remaining)
                top10[name]=dict(ids=reference.iloc[indices][ID].tolist(),control=a,composed=b,gain_seconds=a['rmse_sec']-b['rmse_sec'])
            record=dict(status='complete',variant=variant,fold=fold,source_sha256=common.sha256(__file__),
                protocol_sha256=common.sha256(OUT/'protocol.json'),execution_inputs_sha256=common.sha256(OUT/'execution_inputs.json'),split=split,
                source_receipts=dict(fixed=declared['fixed_sources'][fold],ordinary=incoming['sources'][fold][variant]),
                metrics={name:metric(y,p) for name,p in controls.items()},reports={'candidate':{'metrics':metrics}},
                comparisons={name:comparison(y,p,pred,days) for name,p in controls.items() if name!='composed'},
                missing_top2=dict(ids=reference.iloc[top2][ID].tolist(),share=float(gains[top2].sum()/gains[missing].sum())),
                remove_missing_top2={name:metric(y,p,keep) for name,p in controls.items()},remove_top10_gain=top10,
                route_counts={'ordinary':int(ordinary.sum()),'nonordinary':int(nonordinary.sum()),'missing':int(missing.sum())},
                no_floor=True,no_new_weight_fit=True,outputs={'candidate.parquet':common.sha256(directory/'candidate.parquet')})
            common.write_json(directory/'manifest.json',record)
            results[variant][fold]=record
            print('FINAL9_COMPOSED',variant,fold,record['metrics']['composed']['rmse_sec'],flush=True)
    summary={}
    for variant,folds in results.items():
        summary[variant]=dict(seasonal_rmse={name:season_score(folds['F1']['metrics'][name],folds['F3']['metrics'][name]) for name in folds['F1']['metrics']},
            remove_missing_top2_seasonal={name:season_score(folds['F1']['remove_missing_top2'][name],folds['F3']['remove_missing_top2'][name]) for name in folds['F1']['metrics']},
            every_day_removal_improves_vs_previous=all(r['comparisons']['previous_composition']['all_day_removals_improve'] for r in folds.values()))
    common.write_json(OUT/'summary.json',dict(status='complete',protocol_sha256=common.sha256(OUT/'protocol.json'),variants=summary,
        main_is_unfloored=True,no_variant_selected=True,predeclared_before_final9_score_access=True))
    print('FINAL9_ROUTE_SEASONAL',summary,flush=True)


if __name__=='__main__':
    main()
