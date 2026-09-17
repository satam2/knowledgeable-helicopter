"""Rebuild early-only calibration inputs, independently solve simplex and replay."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='2'
import lightgbm as lgb
import argparse
import gc
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds
import psutil
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits
import validate_candidate as v

ROOT,ID,TARGET,TIME=v.ROOT,v.ID,v.TARGET,v.common.MOVEMENT
BASE=ROOT/'private_runs/tail240_20260916/forensics/current_calibration/v1'
VALIDATION=ROOT/'private_runs/tail240_20260916/validation'


def guard():
    info=psutil.Process().memory_info();peak=max(info.rss,info.peak_wset)
    assert peak<4*1024**3 and psutil.virtual_memory().available>=8*1024**3
    return peak


def close(a,b):np.testing.assert_allclose(a,b,rtol=1e-11,atol=1e-8)


def check_metrics(y,p,record):
    e=p-y;assert record['rows']==len(y)
    for key,value in dict(rmse=np.sqrt(np.mean(e*e)),mse=np.mean(e*e),mae=np.abs(e).mean(),bias=e.mean(),p95=np.quantile(np.abs(e),.95)).items():close(value,record[key])


def check_comparison(y,p,ref,days,record):
    check_metrics(y,p,record['candidate']);check_metrics(y,ref,record['baseline'])
    a=(ref-y)**2;b=(p-y)**2;gain=a-b
    close(np.sqrt(a.mean())-np.sqrt(b.mean()),record['gain'])
    dates=pd.DatetimeIndex(pd.to_datetime(days,utc=True)).floor('D');unique=dates.unique().sort_values()
    counts=[];left=[];right=[];deletions=[]
    for day,item in zip(unique,record['day_removal']):
        assert str(day)==item['day'];mask=dates==day
        counts.append(mask.sum());left.append(a[mask].sum());right.append(b[mask].sum())
        value=np.sqrt(a[~mask].mean())-np.sqrt(b[~mask].mean());close(value,item['gain']);deletions.append(value)
    assert all(value>0 for value in deletions)==record['every_day_positive']
    order=np.argsort(gain,kind='stable')[::-1]
    for count in [1,2,5,10]:
        keep=np.ones(len(y),bool);keep[order[:count]]=False
        close(np.sqrt(a[keep].mean())-np.sqrt(b[keep].mean()),record['remove_largest_gain_rows'][str(count)])
    draws=np.random.default_rng(20260916).multinomial(len(unique),np.repeat(1/len(unique),len(unique)),size=300)
    n=draws@np.asarray(counts)
    boot=np.sqrt((draws@np.asarray(left))/n)-np.sqrt((draws@np.asarray(right))/n)
    close(np.quantile(boot,[.025,.975]),record['gain_ci95'])


def solve_simplex(y,p):
    design=p-p[:,:1];target=y-p[:,0];scale=max(float(np.mean(target**2)),1.)
    def objective(w):
        e=design@w-target
        return float(e@e)/(len(y)*scale),2*design.T@e/(len(y)*scale)
    solved=minimize(objective,np.full(p.shape[1],1/p.shape[1]),jac=True,method='SLSQP',
        bounds=[(0.,1.)]*p.shape[1],constraints={'type':'eq','fun':lambda w:w.sum()-1.,'jac':lambda w:np.ones_like(w)},
        options={'maxiter':200,'ftol':1e-12})
    assert solved.success,solved.message
    weights=np.maximum(solved.x,0.);weights/=weights.sum()
    return weights


def main(fold):
    assert psutil.virtual_memory().available>=12*1024**3
    folder=BASE/fold;out=VALIDATION/f'current_calibration_{fold}_v1';out.mkdir(parents=True,exist_ok=False)
    protocol=v.read_json(BASE/'protocol.json');manifest=v.read_json(folder/'manifest.json')
    assert manifest['status']=='complete' and manifest['protocol_sha256']==v.sha256(BASE/'protocol.json')
    assert manifest['source_sha256']==protocol['source_sha256']==v.sha256(ROOT/'review_work/tail240_20260916/forensics/current_calibration/run_v1.py')
    for path,digest in protocol['dependencies'].items():assert v.sha256(ROOT/path)==digest
    for name,digest in manifest['outputs'].items():assert v.sha256(folder/name)==digest
    preflight=v.read_json(folder/'preflight.json');assert v.sha256(folder/'preflight.json')==manifest['preflight_sha256']
    assert preflight['protocol_sha256']==manifest['protocol_sha256'] and preflight['inputs_sha256']==manifest['outputs']['inputs.parquet']
    verified=v.read_json(VALIDATION/'current_calibration_preflight_v1/receipt.json')
    assert verified['status']=='passed' and verified['protocol_sha256']==manifest['protocol_sha256']
    frame=pd.read_parquet(folder/'inputs.parquet');ids=pd.Index(frame[ID]);assert ids.is_unique
    metadata=ROOT/'private_runs/screening_230/data/interim/audit/departures.parquet';assert v.sha256(metadata)==protocol['meta_sha256']
    raw=ds.dataset(metadata,format='parquet').to_table(columns=[ID,TARGET,TIME,'FLIGHT_ID_mvt'],filter=ds.field(ID).isin(ids.to_numpy()),use_threads=False).to_pandas().set_index(ID).loc[ids]
    for name in [TARGET,TIME,'FLIGHT_ID_mvt']:np.testing.assert_array_equal(raw[name],frame[name])
    cutoff=frame.sort_values([TIME,ID])[TIME].iloc[len(frame)*3//5]
    early=frame[TIME]<cutoff;late=~early;lateflights=set(frame.loc[late,'FLIGHT_ID_mvt'].dropna())
    purged=early&frame.FLIGHT_ID_mvt.notna()&frame.FLIGHT_ID_mvt.isin(lateflights);fit=(early&~purged).to_numpy();late=late.to_numpy()
    for name,values in [('early',early),('late',late),('purged',purged),('fit',fit)]:np.testing.assert_array_equal(frame[name],values)
    assert v.object_hash(frame.loc[fit,ID].tolist())==preflight['fit_ids_hash']
    assert v.object_hash(frame.loc[late,ID].tolist())==preflight['late_ids_hash']
    encoder=v.read_json(folder/'encoder.json');columns=encoder['columns'];vocab=encoder['vocab'];basecolumns=protocol['columns']
    assert columns==basecolumns+protocol['contrast_columns'] and len(columns)==396
    assert encoder['fit_ids_hash']==preflight['fit_ids_hash']
    matrix=np.full((len(frame),396),np.nan,dtype='float32',order='F');counts=np.zeros(387,int);known={name:set() for name in vocab}
    expected_receipts=[]
    for source in preflight['source_receipts']:
        path=Path(source['path']);assert v.sha256(path)==source['sha256'];names=source['columns'];filled=source['fill']
        expected_fill='screening_230/data/interim/features' not in path.as_posix() and 'sequence_flatten' not in path.as_posix()
        assert filled==expected_fill
        seen=np.zeros(len(ids),bool)
        for batch in pq.ParquetFile(path).iter_batches(batch_size=8192,columns=[ID,*names],use_threads=False):
            data=batch.to_pandas();pos=ids.get_indexer(data[ID]);take=pos>=0;positions=pos[take]
            assert not seen[positions].any() and len(np.unique(positions))==len(positions);seen[positions]=True
            for name in names:
                values=data.loc[take,name]
                if name in vocab:
                    known[name].update(values.iloc[np.flatnonzero(fit[positions])].dropna().astype(str).tolist())
                    mapping={word:i+2 for i,word in enumerate(vocab[name])}
                    values=values.astype('string').map(mapping).fillna(1).where(values.notna(),0).to_numpy('float32')
                else:
                    values=pd.to_numeric(values).to_numpy(dtype='float32',na_value=np.nan);values[~np.isfinite(values)]=np.nan
                    if filled:values[values==-999999.]=np.nan
                    assert encoder['sentinel_to_nan'][name]==filled
                matrix[positions,basecolumns.index(name)]=values
        for name in names:counts[basecolumns.index(name)]+=int(seen.sum())
        expected_receipts.append(dict(path=str(path),sha256=source['sha256'],rows=int(seen.sum()),columns=names))
        guard()
    assert np.all(counts==len(ids)) and {name:sorted(values) for name,values in known.items()}==vocab
    assert expected_receipts==manifest['feature_receipts']
    p=frame[protocol['experts']].to_numpy(float);y=frame[TARGET].to_numpy(float)
    matrix[:,387:]=(p-p.mean(axis=1,keepdims=True)).astype('float32')
    weights=solve_simplex(y[fit],p[fit]);savedweights=v.read_json(folder/'weights.json')
    np.testing.assert_allclose(weights,savedweights['weights'],rtol=0,atol=1e-9)
    assert savedweights['experts']==protocol['experts'] and savedweights['fit_ids_hash']==preflight['fit_ids_hash']
    baseline=p@weights
    model=lgb.Booster(model_file=str(folder/'model.txt'));assert model.current_iteration()==manifest['steps']==200 and model.feature_name()==columns
    for name,value in protocol['params'].items():assert manifest['params'][name]==value
    predicted=baseline+model.predict(matrix,num_threads=2,num_iteration=200)
    saved=pd.read_parquet(folder/'predictions.parquet');np.testing.assert_array_equal(saved[ID],frame[ID]);np.testing.assert_array_equal(saved[TARGET],y)
    max_delta=float(np.max(np.abs(predicted-saved.prediction_sec.to_numpy())))
    np.testing.assert_allclose(saved.early_constant_sec,baseline,rtol=0,atol=1e-7)
    np.testing.assert_allclose(saved.prediction_sec,predicted,rtol=0,atol=1e-7)
    close(saved.correction_sec,predicted-baseline);close(saved.squared_error,(predicted-y)**2)
    metrics=v.read_json(folder/'metrics.json');assert metrics==manifest['metrics']
    check_comparison(y[late],predicted[late],baseline[late],frame.loc[late,TIME],metrics['primary'])
    check_comparison(y[late],predicted[late],frame.loc[late,'frozen_current_sec'].to_numpy(),frame.loc[late,TIME],metrics['frozen_current_diagnostic'])
    check_metrics(y[fit],baseline[fit],metrics['early_training']['constant']);check_metrics(y[fit],predicted[fit],metrics['early_training']['candidate'])
    result=dict(status='passed',source_sha256=v.sha256(__file__),producer_manifest_sha256=v.sha256(folder/'manifest.json'),protocol_sha256=manifest['protocol_sha256'],
        all387_original_inputs_and_early_vocabularies_rebuilt=True,source_specific_sentinels_exact=True,all9_contrasts_exact=True,
        independently_solved_simplex_max_weight_delta=float(np.max(np.abs(weights-np.asarray(savedweights['weights'])))),
        native_prediction_max_abs_delta=max_delta,original_labels_ids_partitions_exact=True,all_metrics_influences_and_day_bootstrap_exact=True,
        fit_rows=int(fit.sum()),late_rows=int(late.sum()),primary=metrics['primary'],frozen_current_diagnostic=metrics['frozen_current_diagnostic'],
        peak_bytes=guard(),no_gpu=True,no_model_training=True,limitation='Independent early-only simplex optimization and saved native model replay; upstream full-tune early stopping exposure remains. No score/refit/ranking or complete-score claim.')
    v.write_json(out/'receipt.json',result)
    print('VERIFIED_CALIBRATION',fold,'primary',metrics['primary']['gain'],'frozendiagnostic',metrics['frozen_current_diagnostic']['gain'],'native_delta',max_delta,'peak',result['peak_bytes'],flush=True)


def seasonal():
    protocol=v.read_json(BASE/'protocol.json');record=v.read_json(BASE/'assessment.json');reports={}
    for fold in ['F1','F3']:
        r=v.read_json(VALIDATION/f'current_calibration_{fold}_v1/receipt.json');assert r['status']=='passed' and r['protocol_sha256']==v.sha256(BASE/'protocol.json')
        assert r['producer_manifest_sha256']==record['folds'][fold]['manifest_sha256']==v.sha256(BASE/fold/'manifest.json')
        reports[fold]=r['primary']
    weights=protocol['seasonal_weights']
    baseline=np.sqrt(sum(weights[f]*reports[f]['baseline']['mse'] for f in weights));candidate=np.sqrt(sum(weights[f]*reports[f]['candidate']['mse'] for f in weights))
    close(baseline,record['late_ordinary_seasonal_constant']);close(candidate,record['late_ordinary_seasonal_candidate']);close(baseline-candidate,record['gain'])
    passed=bool(baseline-candidate>=2 and all(r['gain']>0 and r['every_day_positive'] for r in reports.values()));assert passed==record['primary_gate_passed']
    out=VALIDATION/'current_calibration_seasonal_v1';out.mkdir(exist_ok=False)
    v.write_json(out/'receipt.json',dict(status='passed',source_sha256=v.sha256(__file__),assessment_sha256=v.sha256(BASE/'assessment.json'),primary_gate_passed=passed,late_ordinary_gain=float(baseline-candidate),baseline=float(baseline),candidate=float(candidate),no_complete_score_claim=True))
    print('VERIFIED_SEASONAL',passed,baseline-candidate,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--fold',choices=['F1','F3']);parser.add_argument('--seasonal',action='store_true');args=parser.parse_args()
    pa.set_cpu_count(1);pa.set_io_thread_count(1)
    with threadpool_limits(2):
        if args.seasonal:seasonal()
        else:
            assert args.fold;main(args.fold)
