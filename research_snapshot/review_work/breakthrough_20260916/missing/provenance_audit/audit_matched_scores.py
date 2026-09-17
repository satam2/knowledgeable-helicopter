"""Read-only score/cohort audit for prior-source-frequency matched experiment."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[k]='1'
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pyarrow as pa
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'review_work/campaign_20260916'))
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'broader_information_audit'))
import common
from audit import stats,days
pa.set_cpu_count(1)
pa.set_io_thread_count(1)
ID,TARGET,TIME=common.ID,common.TARGET,common.MOVEMENT
BASE=ROOT/'private_runs/breakthrough_20260916/missing/provenance_audit'
OUT=BASE/'matched_score_audit'


def main():
    OUT.mkdir(exist_ok=False)
    path=ROOT/'private_runs/screening_230/data/interim/audit/departures.parquet'
    assert common.sha256(path)==common.read_json(ROOT/'private_runs/screening_230/reports/data_audit.json')['artifacts'][path.name]
    meta=pd.read_parquet(path,columns=[ID,TARGET,TIME,'FLIGHT_ID_mvt','proxy_sec'])
    eligible=np.isfinite(meta.proxy_sec)&meta.proxy_sec.between(0,7200)
    cache=BASE/'context_cache'
    marker=common.read_json(cache/'manifest.json')
    verified=common.read_json(cache/'verification.json')
    assert marker['status']=='complete' and verified['status']=='passed'
    assert common.sha256(cache/'manifest.json')==verified['manifest_sha256']
    for name,digest in marker['outputs'].items():
        assert common.sha256(cache/name)==digest
    for name,digest in marker['source_hashes'].items():
        assert common.sha256(ROOT/name)==digest
    folds={}
    errors={}
    for fold in ['F1','F3']:
        idx,split,_=common.fold_data(meta,fold,full=True)
        reference,refrec=common.reference(fold)
        assert common.object_hash(split)==common.object_hash(refrec['split'])
        arms={}
        manifests={}
        for arm in ['control','context']:
            directory=BASE/'matched_models'/arm/f'lightgbm_ordinary_source_residual_{fold}_s20260916'
            record=common.read_json(directory/'manifest.json')
            assert record['status']=='complete'
            for name,digest in record['outputs'].items():
                assert common.sha256(directory/name)==digest
            assert common.object_hash(split)==common.object_hash(record['split'])
            for stage,positions in idx.items():
                selected=positions[eligible.iloc[positions].to_numpy()]
                assert record['fit_ids'][stage]==dict(n=len(selected),hash=common.object_hash(meta.iloc[selected][ID].tolist()))
            assert record['anchor']['context']['manifest_sha256']==verified['manifest_sha256']
            assert record['anchor']['context']['verification_sha256']==common.sha256(cache/'verification.json')
            assert record['anchor']['context']['feature_sha256']==marker['outputs']['training_features.parquet']
            conventions=ROOT/'private_runs/breakthrough_20260916/missing/source_conventions'
            assert record['anchor']['conventions']['manifest_sha256']==common.sha256(conventions/'manifest.json')
            assert record['anchor']['conventions']['feature_sha256']==common.sha256(conventions/'features.parquet')
            tune=pd.read_parquet(directory/'tune_predictions.parquet')
            np.testing.assert_array_equal(tune[ID],meta.iloc[idx['tune'][eligible.iloc[idx['tune']].to_numpy()]][ID])
            candidate=pd.read_parquet(directory/'candidate.parquet')
            np.testing.assert_array_equal(candidate[ID],reference[ID])
            np.testing.assert_array_equal(candidate[TARGET],meta.iloc[idx['score']][TARGET])
            unsupported=~eligible.iloc[idx['score']].to_numpy()
            np.testing.assert_array_equal(candidate.prediction_sec.to_numpy()[unsupported],reference.prediction_sec.to_numpy()[unsupported])
            squared=(candidate.prediction_sec.to_numpy()-candidate[TARGET].to_numpy())**2
            np.testing.assert_array_equal(squared,candidate.squared_error)
            arms[arm]=squared
            manifests[arm]=record
            errors[arm+'_'+fold]=float(squared.mean())
        control,context=manifests['control'],manifests['context']
        assert control['fit_ids']==context['fit_ids']
        assert control['fit']['params']==context['fit']['params']
        additions=[c for c in context['feature_columns'] if c not in control['feature_columns']]
        assert additions==marker['features']
        assert control['feature_columns']==[c for c in context['feature_columns'] if c not in additions]
        dates=pd.to_datetime(meta.iloc[idx['score']][TIME],utc=True).dt.strftime('%Y-%m-%d').to_numpy()
        a,b=arms['control'],arms['context']
        gain=a-b
        order=np.argsort(gain)[::-1]
        removed={}
        for n in [1,2,5,10]:
            keep=np.ones(len(a),bool)
            keep[order[:n]]=False
            removed[str(n)]=stats(a,b,keep)
        daily=pd.DataFrame({'day':dates,'gain':gain}).groupby('day').gain.sum()
        folds[fold]=dict(overall=stats(a,b,np.ones(len(a),bool)),day_removals=days(a,b,dates),
            improved_days=int((daily>0).sum()),days=len(daily),feature_additions=additions,
            control_features=len(control['feature_columns']),context_features=len(context['feature_columns']),
            fit_ids=context['fit_ids'],unsupported_unchanged=int(unsupported.sum()),
            remove_top_gains=removed,all_outputs_hashes_verified=True,
            control_manifest_sha256=common.sha256(BASE/'matched_models'/'control'/f'lightgbm_ordinary_source_residual_{fold}_s20260916'/'manifest.json'),
            context_manifest_sha256=common.sha256(BASE/'matched_models'/'context'/f'lightgbm_ordinary_source_residual_{fold}_s20260916'/'manifest.json'))
        print('MATCHED',fold,folds[fold]['overall'],folds[fold]['improved_days'],flush=True)
    seasonal={arm:float(np.sqrt((192122*errors[arm+'_F1']+152719*errors[arm+'_F3'])/344841)) for arm in ['control','context']}
    common.write_json(OUT/'audit.json',dict(status='passed',source_sha256=common.sha256(__file__),folds=folds,
        seasonal_rmse=seasonal,seasonal_gain=seasonal['control']-seasonal['context'],
        cache_verification_sha256=common.sha256(cache/'verification.json'),
        availability='Reuses already independently verified strict prior60min sameairport/month proxyfrequency cache; final NM metadata remains retrospective, not fallback provenance or proven realtime publication.',
        interpretation='Tiny exposed-development gain; no new fitting or cache producer changes. All finite ordinary labels retained; all other routes V2exact.'))
    print('SEASONAL',seasonal,flush=True)


if __name__=='__main__':
    main()
