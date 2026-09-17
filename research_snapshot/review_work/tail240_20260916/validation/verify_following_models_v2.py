"""Independent native replay and fixed-global9 complement audit for following415."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import lightgbm as lgb
import argparse
import gc
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import validate_candidate as v

ROOT=v.ROOT
common=v.common
ID,TARGET=v.ID,v.TARGET
pa.set_cpu_count(1);pa.set_io_thread_count(1)
original_guard=v.guard

def guard():
    peak=original_guard()
    assert peak<2*1024**3,peak
    return peak

v.guard=guard


def main(fold,root):
    folder=ROOT/root
    out=ROOT/'private_runs/tail240_20260916/validation'/('following_models_'+folder.name+'_'+fold)
    out.mkdir(parents=True,exist_ok=False)
    protocol=v.read_json(folder/'protocol.json')
    binding=v.read_json(v.BINDING)['folds'][fold]
    metadata=ROOT/'private_runs/screening_230/data/interim/audit/departures.parquet'
    assert v.sha256(metadata)==v.read_json(ROOT/'private_runs/screening_230/reports/data_audit.json')['artifacts'][metadata.name]
    meta=pd.read_parquet(metadata,columns=[ID,'FLIGHT_ID_mvt',common.MOVEMENT,'ADEP_mvt',TARGET,'proxy_sec'])
    idx,split,_=common.fold_data(meta,fold,full=True)
    assert v.object_hash(split)==binding['split_hash']
    frames={stage:meta.iloc[idx[stage]].loc[lambda x:np.isfinite(x.proxy_sec)].copy() for stage in ['fit','tune']}
    cohorts={stage:{'n':len(frame),'hash':v.object_hash(frame[ID].tolist())} for stage,frame in frames.items()}
    assert cohorts=={stage:binding['cohorts']['finite_nm'][stage] for stage in cohorts}
    tune=frames['tune'];ids=pd.Index(tune[ID]);fitids=pd.Index(frames['fit'][ID])
    y=tune[TARGET].to_numpy(float);proxy=tune.proxy_sec.to_numpy(float)
    del meta,idx;gc.collect()
    predictions={};receipts={}
    for arm in ['control387','following415']:
        path=folder/fold/arm
        record=v.read_json(path/'manifest.json')
        assert record['status']=='complete' and record['ids']==cohorts
        assert v.object_hash(record['split'])==binding['split_hash']
        assert record['protocol_sha256']==v.sha256(folder/'protocol.json')
        assert record['params']==protocol['params']
        for filename,digest in record['outputs'].items():assert v.sha256(path/filename)==digest
        enc=v.read_json(path/'encoder.json');columns=enc['columns']
        expected=protocol['full_columns']+(protocol['extra_columns'] if arm=='following415' else [])
        assert columns==expected
        matrix=np.full((len(ids),len(columns)),np.nan,dtype='float32',order='F')
        counts=np.zeros(len(columns),dtype=int);vocab={key:set() for key in enc['vocab']}
        for source in record['feature_receipts']:
            sourcepath=Path(source['path']);names=source['columns']
            assert v.sha256(sourcepath)==source['sha256']
            seen=np.zeros(len(ids),bool)
            for batch in pq.ParquetFile(sourcepath).iter_batches(batch_size=8192,columns=[ID,*names],use_threads=False):
                frame=batch.to_pandas();positions=ids.get_indexer(frame[ID]);keep=positions>=0;positions=positions[keep]
                assert len(np.unique(positions))==len(positions) and not seen[positions].any()
                seen[positions]=True
                fitmask=fitids.get_indexer(frame[ID])>=0
                for name in names:
                    if name in vocab:
                        vocab[name].update(frame.loc[fitmask,name].dropna().astype(str).tolist())
                        values=frame.loc[keep,name]
                        mapping={value:i+2 for i,value in enumerate(enc['vocab'][name])}
                        values=values.astype('string').map(mapping).fillna(1).where(values.notna(),0).to_numpy('float32')
                    else:
                        values=pd.to_numeric(frame.loc[keep,name]).to_numpy(dtype='float32',na_value=np.nan)
                        values[~np.isfinite(values)]=np.nan
                        if 'screening_230/data/interim/features' not in sourcepath.as_posix() and 'sequence_flatten' not in str(sourcepath):
                            values[np.isnan(values)]=-999999.
                    matrix[positions,columns.index(name)]=values
            for name in names:counts[columns.index(name)]+=int(seen.sum())
            v.guard()
        assert np.all(counts==len(ids))
        assert {name:sorted(values) for name,values in vocab.items()}==enc['vocab']
        model=lgb.Booster(model_file=str(path/'model.txt'))
        assert model.feature_name()==columns
        prediction=model.predict(matrix,num_threads=1)+proxy
        saved=pd.read_parquet(path/'tune.parquet')
        np.testing.assert_array_equal(saved[ID],ids)
        np.testing.assert_array_equal(prediction,saved.prediction_sec)
        assert np.isfinite(prediction).all()
        predictions[arm]=prediction
        receipts[arm]={'rows':len(ids),'features':len(columns),'native_max_abs_delta_sec':0.,
            'fit_only_category_vocabs_exact':len(vocab),'manifest_sha256':v.sha256(path/'manifest.json')}
        print('MODEL_REPLAYED',fold,arm,len(ids),v.guard(),flush=True)
        del matrix,model;gc.collect()
    oldfolder=ROOT/'private_runs/breakthrough_20260916/deeper_context_union'/f'lightgbm_leaf63_sequence_aobt_allfinite_{fold}_s20260916'
    oldrecord=v.read_json(oldfolder/'manifest.json')
    assert v.sha256(oldfolder/'tune_predictions.parquet')==oldrecord['outputs']['tune_predictions.parquet']
    old=pd.read_parquet(oldfolder/'tune_predictions.parquet')
    np.testing.assert_array_equal(old[ID],ids)
    np.testing.assert_allclose(old.prediction_sec,predictions['control387'],atol=1e-7,rtol=0)
    dates=tune[common.MOVEMENT].dt.floor('D').to_numpy();ordinary=tune.proxy_sec.between(0,7200).to_numpy()
    comparisons={}
    for name,mask in [('all_finite',np.ones(len(y),bool)),('ordinary_proxy',ordinary)]:
        comparisons[name]=(y[mask],predictions['following415'][mask],predictions['control387'][mask],dates[mask])
    base=ROOT/'private_runs/breakthrough_20260916/models/context_gate/final_simplex9_v1'
    preparation=v.read_json(base/'preparation.json')['folds'][fold]
    aligned_path=base/(fold+'_aligned_tune.parquet');assert v.sha256(aligned_path)==preparation['aligned_tune_sha256']
    weights=v.read_json(base/(fold+'_weights.json'));assert weights==preparation['weights']
    aligned=pd.read_parquet(aligned_path)
    np.testing.assert_array_equal(aligned[ID],ids[ordinary]);np.testing.assert_array_equal(aligned[TARGET],y[ordinary])
    np.testing.assert_allclose(aligned.lgb63_union,predictions['control387'][ordinary],atol=1e-7,rtol=0)
    global9=aligned[weights['experts']].to_numpy(float)@np.asarray(weights['global'])
    blend=.75*global9+.25*predictions['following415'][ordinary]
    replaced=global9+weights['global'][weights['experts'].index('lgb63_union')]*(predictions['following415'][ordinary]-aligned.lgb63_union.to_numpy())
    comparisons['global9_fixed25']=(y[ordinary],blend,global9,dates[ordinary])
    comparisons['global9_component_replacement_diagnostic']=(y[ordinary],replaced,global9,dates[ordinary])
    metrics={}
    for name,(target,candidate,control,days) in comparisons.items():
        a,b=v.metrics(target,candidate),v.metrics(target,control)
        metrics[name]={'candidate':a,'control':b,'gain':b['rmse']-a['rmse'],'paired':v.paired(target,control,candidate,days)}
    v.write_json(out/'receipt.json',{'status':'passed','source_sha256':v.sha256(__file__),'fold':fold,'cohorts':cohorts,
        'models':receipts,'metrics':metrics,'peak_rss_bytes':v.guard(),
        'caveat':'Fixed global9 weights were learned on same tune labels. No weights changed. No score/refit predictions evaluated.'})
    print('COMPLETE',fold,{k:x['gain'] for k,x in metrics.items()},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--fold',required=True,choices=['F1','F3'])
    parser.add_argument('--root',default='private_runs/tail240_20260916/models/following_groups_tune_v1')
    args=parser.parse_args();main(args.fold,args.root)
