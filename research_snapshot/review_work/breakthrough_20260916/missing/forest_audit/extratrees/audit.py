"""ExtraTrees saved-model replay, permitted priors, and tail/common-floor audit."""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
    os.environ[key]='1'
from pathlib import Path
import sys
import importlib.util
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[4]
spec=importlib.util.spec_from_file_location('original_forest_audit',HERE.parent/'audit.py')
previous=importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)
common=previous.common
missing=previous.missing
context=previous.context
forest=previous.forest
metric=previous.prioraudit.metric
comparison=previous.prioraudit.comparison
read_json,write_json,sha256,object_hash=previous.read_json,previous.write_json,previous.sha256,previous.object_hash
ID,TARGET,TIME=previous.ID,previous.TARGET,previous.MOVEMENT
OUT=ROOT/'private_runs/breakthrough_20260916/missing/forest_audit/extratrees'


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    write_json(OUT/'protocol.json',dict(source_sha256=sha256(__file__),parent_audit_source_sha256=sha256(HERE.parent/'audit.py'),
        policy='Read-only ExtraTrees audit. No ensemble fitting. One CPU thread. All source/protocol/output hashes preserved.',
        diagnostics='Original scores, unchangedfiniteNM, removeETtop2gains/removeETvstemplatetop2/remove>=1daylabels, postblendcommonfitfloor andguardreferencebeforefixed25percentblend.'))
    x,meta=missing.load_data()
    peers=pd.read_parquet(context.CACHE/'features.parquet').set_index(ID)
    assert sha256(context.CACHE/'features.parquet')==read_json(context.CACHE/'audit.json')['feature_sha256']
    np.testing.assert_array_equal(peers.index,meta[ID])
    times=meta.set_index(ID).loc[x.index,TIME]
    fx=pd.concat([x,context.id_context_features(x,peers.loc[x.index],times)],axis=1)
    fx[forest.TIME_COLUMN]=times
    folds={}
    for fold in ['F1','F3']:
        idx,split,_=common.fold_data(meta,fold,full=True)
        reference,refrec=common.reference(fold)
        assert object_hash(split)==object_hash(refrec['split'])
        np.testing.assert_array_equal(reference[ID],meta.iloc[idx['score']][ID])
        np.testing.assert_array_equal(reference[TARGET],meta.iloc[idx['score']][TARGET])
        unavailable=~np.isfinite(meta.iloc[idx['score']].proxy_sec.to_numpy(float))
        y=reference[TARGET].to_numpy(float)
        day=reference.day.to_numpy()
        pred={'V2':reference.prediction_sec.to_numpy(float)}
        manifests={}
        candidates={}
        for name,directory in [
            ('template',previous.TEMPLATES/f'historical_template_{fold}_s20260916'),
            ('RF',previous.FORESTS/f'randomforest_missing_template_idcontext_{fold}_s20260916'),
            ('ET',previous.FORESTS/f'extratrees_missing_template_idcontext_{fold}_s20260916')]:
            record=previous.checked(directory)
            assert object_hash(record['split'])==object_hash(split)
            for stage,positions in idx.items():
                rows=positions[~np.isfinite(meta.iloc[positions].proxy_sec.to_numpy(float))]
                assert record['fit_ids'][stage]==dict(n=len(rows),hash=object_hash(meta.iloc[rows][ID].tolist()))
            for variant in ['candidate','blend25']:
                frame=pd.read_parquet(directory/(variant+'.parquet'))
                np.testing.assert_array_equal(frame[ID],reference[ID])
                np.testing.assert_array_equal(frame[TARGET],reference[TARGET])
                values=frame.prediction_sec.to_numpy(float)
                np.testing.assert_array_equal(values[~unavailable],pred['V2'][~unavailable])
                assert abs(metric(y,values)['rmse_sec']-record['reports'][variant]['metrics']['overall']['rmse_sec'])<1e-10
                if variant=='blend25':
                    pred[name]=values
                else:
                    candidates[name]=values
            np.testing.assert_allclose(pred[name],pred['V2']+.25*(candidates[name]-pred['V2']),rtol=0,atol=1e-9)
            manifests[name]=dict(path=str(directory),sha256=sha256(directory/'manifest.json'),fit_ids=record['fit_ids'])
            if name=='ET':
                previous.check_sources(record)
                assert sha256(previous.FORESTS/'protocol_extratrees_s20260916.json')==record['protocol_sha256']
                replay=previous.replay(directory,record,fx,meta,idx)
                tune_et=pd.read_parquet(directory/'tune_predictions.parquet')
                assert replay['fit']['estimator_class']=='ExtraTreesRegressor'
        floor=float(meta.iloc[idx['fit']][TARGET].min())
        guarddir=ROOT/'private_runs/breakthrough_20260916/models/support_guard'/fold
        guardrecord=previous.checked(guarddir)
        assert guardrecord['fit_min_target_sec']==floor
        guarded=pd.read_parquet(guarddir/'fit_min_floor.parquet')
        np.testing.assert_array_equal(guarded.prediction_sec,np.maximum(pred['V2'],floor))
        floor_preds={name:np.maximum(values,floor) for name,values in pred.items()}
        guardfirst={'V2':floor_preds['V2']}
        for name,values in candidates.items():
            guardfirst[name]=floor_preds['V2']+.25*(values-floor_preds['V2'])
            np.testing.assert_array_equal(guardfirst[name][~unavailable],floor_preds['V2'][~unavailable])
        gain=(pred['V2']-y)**2-(pred['ET']-y)**2
        incremental=(pred['template']-y)**2-(pred['ET']-y)**2
        top2=np.argsort(gain)[-2:]
        incremental_top2=np.argsort(incremental)[-2:]
        keep=np.ones(len(y),bool)
        tail=unavailable&(y>=86400)
        scenarios={'all':keep,'remove_ETgain_top2':~np.isin(np.arange(len(y)),top2),
            'remove_ET_vs_template_gain_top2':~np.isin(np.arange(len(y)),incremental_top2),
            'remove_missing_dayplus':~tail,'remove_missing_dayplus_days':~np.isin(day,np.unique(day[tail])),
            'remove_V2_below_fitfloor':pred['V2']>=floor}
        metrics={scenario:{name:metric(y,values,mask) for name,values in pred.items()} for scenario,mask in scenarios.items()}
        sensitivity=reference[[ID,TARGET,'day']].copy()
        for name,values in pred.items():
            sensitivity[name]=values
        for name,values in candidates.items():
            sensitivity[name+'_candidate']=values
        sensitivity['ET_gain_vs_V2']=gain
        sensitivity['ET_gain_vs_template']=incremental
        sensitivity.iloc[np.unique(np.r_[top2,incremental_top2,np.flatnonzero(tail)])].to_parquet(OUT/(fold+'_sensitivity_rows.parquet'),index=False)
        tune_rows=idx['tune'][~np.isfinite(meta.iloc[idx['tune']].proxy_sec.to_numpy(float))]
        np.testing.assert_array_equal(tune_et[ID],meta.iloc[tune_rows][ID])
        tune={}
        for name in ['template','RF']:
            vals=pd.read_parquet(Path(manifests[name]['path'])/'tune_predictions.parquet')
            np.testing.assert_array_equal(vals[ID],tune_et[ID])
            tune[name+'_vs_ET']=comparison(meta.iloc[tune_rows][TARGET].to_numpy(float),vals.prediction_sec.to_numpy(float),tune_et.prediction_sec.to_numpy(float),
                pd.to_datetime(meta.iloc[tune_rows][TIME],utc=True).dt.strftime('%Y-%m-%d').to_numpy())
        folds[fold]=dict(replay=replay,manifests=manifests,original_purged_IDs_and_labels_exact=True,finite_routes_unchanged=True,
            metrics=metrics,fit_floor=floor,
            pairs={name+'_vs_ET':comparison(y,values,pred['ET'],day) for name,values in pred.items() if name!='ET'},
            common_postblend_floor_metrics={name:metric(y,values) for name,values in floor_preds.items()},
            common_postblend_floor_pairs={name+'_vs_ET':comparison(y,values,floor_preds['ET'],day) for name,values in floor_preds.items() if name!='ET'},
            guard_reference_beforeblend_metrics={name:metric(y,values) for name,values in guardfirst.items()},
            guard_reference_beforeblend_pairs={name+'_vs_ET':comparison(y,values,guardfirst['ET'],day) for name,values in guardfirst.items() if name!='ET'},
            top2=dict(ids=reference.iloc[top2][ID].tolist(),gain=float(gain[top2].sum()),total_gain=float(gain.sum()),share=float(gain[top2].sum()/gain.sum())),
            template_incremental_top2=dict(ids=reference.iloc[incremental_top2][ID].tolist(),gain=float(incremental[incremental_top2].sum()),total_gain=float(incremental.sum())),
            tune=tune)
        print('EXTRATREES_AUDIT',fold,'replay',replay,'top2share',folds[fold]['top2']['share'],flush=True)
    seasonal={scenario:{name:previous.season_score(folds['F1']['metrics'][scenario][name],folds['F3']['metrics'][scenario][name])
        for name in folds['F1']['metrics'][scenario]} for scenario in folds['F1']['metrics']}
    floor_season={style:{name:previous.season_score(folds['F1'][style][name],folds['F3'][style][name]) for name in folds['F1'][style]}
        for style in ['common_postblend_floor_metrics','guard_reference_beforeblend_metrics']}
    write_json(OUT/'audit.json',dict(status='passed',source_sha256=sha256(__file__),folds=folds,seasonal=seasonal,common_floor_seasonal=floor_season,
        labels='Originalrawlabels retained inallmodels andfullcohorts. Removal/floor diagnostics only; notnewrouting rules.',
        selection='Already-exposeddevelopment. No new fittedensemble ormodel. Leafchoice fromoriginaltune only; historicalpriors reconstructed from permittedfit/refitlabels.'))
    print('SEASONAL',seasonal,flush=True)
    print('COMMON_FLOOR',floor_season,flush=True)


if __name__=='__main__':
    main()
