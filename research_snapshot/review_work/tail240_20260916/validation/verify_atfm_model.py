"""Independent full-cohort ATFM LightGBM replay and fixed ensemble comparisons."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='2'
import lightgbm as lgb
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import validate_candidate as v

ROOT=v.ROOT
ID,TARGET=v.ID,v.TARGET
pa.set_cpu_count(1);pa.set_io_thread_count(1)


def main(fold,producer_root):
    root=ROOT/producer_root;folder=root/fold
    out=ROOT/'private_runs/tail240_20260916/validation'/('atfm_model_'+root.name+'_'+fold)
    out.mkdir(parents=True,exist_ok=False)
    marker=v.read_json(folder/'manifest.json');protocol=v.read_json(root/'protocol.json')
    assert marker['status']=='complete' and marker['protocol_sha256']==v.sha256(root/'protocol.json')
    assert marker['source_sha256']==protocol['source_sha256']
    for name,digest in marker['outputs'].items():assert v.sha256(folder/name)==digest
    binding=v.read_json(v.BINDING)['folds'][fold]
    assert v.object_hash(marker['split'])==binding['split_hash']
    meta_path=ROOT/'private_runs/screening_230/data/interim/audit/departures.parquet'
    assert v.sha256(meta_path)==v.read_json(ROOT/'private_runs/screening_230/reports/data_audit.json')['artifacts'][meta_path.name]
    meta=pd.read_parquet(meta_path,columns=[ID,'FLIGHT_ID_mvt',v.common.MOVEMENT,TARGET,'proxy_sec'])
    idx,_,_=v.common.fold_data(meta,fold,full=True)
    frames={s:meta.iloc[idx[s]].loc[lambda x:np.isfinite(x.proxy_sec)].copy() for s in ['fit','tune']}
    cohorts={s:{'n':len(x),'hash':v.object_hash(x[ID].tolist())} for s,x in frames.items()}
    assert cohorts==marker['ids']=={s:binding['cohorts']['finite_nm'][s] for s in frames}
    tune=frames['tune'];ids=pd.Index(tune[ID]);fitids=pd.Index(frames['fit'][ID])
    enc=v.read_json(folder/'encoder.json');columns=enc['columns'];assert columns==protocol['columns'] and len(columns)==416
    matrix=np.full((len(ids),len(columns)),np.nan,dtype='float32',order='F');counts=np.zeros(len(columns),int)
    vocabs={name:set() for name in enc['vocab']}
    for source in marker['feature_receipts']:
        path=Path(source['path']);assert v.sha256(path)==source['sha256'];names=source['columns'];seen=np.zeros(len(ids),bool)
        for batch in pq.ParquetFile(path).iter_batches(batch_size=8192,columns=[ID,*names],use_threads=False):
            frame=batch.to_pandas();positions=ids.get_indexer(frame[ID]);keep=positions>=0;positions=positions[keep]
            assert len(np.unique(positions))==len(positions) and not seen[positions].any();seen[positions]=True
            fitmask=fitids.get_indexer(frame[ID])>=0
            for name in names:
                if name in vocabs:
                    vocabs[name].update(frame.loc[fitmask,name].dropna().astype(str).tolist())
                    values=frame.loc[keep,name];mapping={value:i+2 for i,value in enumerate(enc['vocab'][name])}
                    values=values.astype('string').map(mapping).fillna(1).where(values.notna(),0).to_numpy('float32')
                else:
                    values=pd.to_numeric(frame.loc[keep,name]).to_numpy(dtype='float32',na_value=np.nan);values[~np.isfinite(values)]=np.nan
                    fill='screening_230/data/interim/features' not in path.as_posix() and 'sequence_flatten' not in str(path) and '/forensics/atfm/' not in path.as_posix()
                    if fill:values[np.isnan(values)]=-999999.
                matrix[positions,columns.index(name)]=values
        for name in names:counts[columns.index(name)]+=int(seen.sum())
        v.guard()
    assert np.all(counts==len(ids));assert {name:sorted(values) for name,values in vocabs.items()}==enc['vocab']
    model=lgb.Booster(model_file=str(folder/'model.txt'));assert model.feature_name()==columns
    prediction=model.predict(matrix,num_threads=2)+tune.proxy_sec.to_numpy(float)
    saved=pd.read_parquet(folder/'tune.parquet');np.testing.assert_array_equal(saved[ID],ids);np.testing.assert_array_equal(saved.prediction_sec,prediction)
    reference=v.read_json(folder/'control_receipt.json');control_folder=Path(reference['control_path'])
    assert v.sha256(control_folder/'manifest.json')==reference['manifest_sha256']
    control_marker=v.read_json(control_folder/'manifest.json');assert control_marker['ids']==cohorts and control_marker['params']==marker['params']
    assert v.sha256(control_folder/'tune.parquet')==control_marker['outputs']['tune.parquet']
    control=pd.read_parquet(control_folder/'tune.parquet');np.testing.assert_array_equal(control[ID],ids)
    old=ROOT/'private_runs/breakthrough_20260916/deeper_context_union'/f'lightgbm_leaf63_sequence_aobt_allfinite_{fold}_s20260916'
    old_marker=v.read_json(old/'manifest.json');assert v.sha256(old/'tune_predictions.parquet')==old_marker['outputs']['tune_predictions.parquet']
    old_pred=pd.read_parquet(old/'tune_predictions.parquet');np.testing.assert_array_equal(old_pred[ID],ids)
    np.testing.assert_array_equal(old_pred.prediction_sec,control.prediction_sec)
    y=tune[TARGET].to_numpy(float);dates=tune[v.common.MOVEMENT].dt.floor('D').to_numpy();ordinary=tune.proxy_sec.between(0,7200).to_numpy()
    comparisons={'matched_all_finite':(y,prediction,control.prediction_sec.to_numpy(),dates),
        'matched_ordinary':(y[ordinary],prediction[ordinary],control.prediction_sec.to_numpy()[ordinary],dates[ordinary])}
    base=ROOT/'private_runs/breakthrough_20260916/models/context_gate/final_simplex9_v1'
    prep=v.read_json(base/'preparation.json')['folds'][fold];aligned_path=base/(fold+'_aligned_tune.parquet')
    assert v.sha256(aligned_path)==prep['aligned_tune_sha256'];weights=v.read_json(base/(fold+'_weights.json'));assert weights==prep['weights']
    aligned=pd.read_parquet(aligned_path);np.testing.assert_array_equal(aligned[ID],ids[ordinary]);np.testing.assert_array_equal(aligned[TARGET],y[ordinary])
    np.testing.assert_array_equal(aligned.lgb63_union,control.prediction_sec.to_numpy()[ordinary])
    baseline=aligned[weights['experts']].to_numpy(float)@np.asarray(weights['global'])
    blend=.75*baseline+.25*prediction[ordinary]
    replace=baseline+weights['global'][weights['experts'].index('lgb63_union')]*(prediction[ordinary]-aligned.lgb63_union.to_numpy())
    comparisons['ordinary_fixed25']=(y[ordinary],blend,baseline,dates[ordinary]);comparisons['ordinary_union_replacement']=(y[ordinary],replace,baseline,dates[ordinary])
    assessment=v.read_json(folder/'assessment.json');reports={}
    for name,(target,candidate,ref,days) in comparisons.items():
        a,b=v.metrics(target,candidate),v.metrics(target,ref);expected=assessment[name]
        assert abs(a['rmse']-expected['rmse'])<1e-10 and abs(b['rmse']-expected['reference_rmse'])<1e-10
        pair=v.paired(target,ref,candidate,days);assert pair['all_day_removals_improve']==expected['all_day_removals_improve']
        reports[name]={'candidate':a,'control':b,'gain':b['rmse']-a['rmse'],'paired':pair}
    gate=all(reports[n]['gain']>0 and reports[n]['paired']['all_day_removals_improve'] for n in ['matched_all_finite','matched_ordinary','ordinary_fixed25']) and reports['ordinary_fixed25']['gain']>2
    assert gate==assessment['fold_gate']
    v.write_json(out/'receipt.json',{'status':'passed','source_sha256':v.sha256(__file__),'producer_manifest_sha256':v.sha256(folder/'manifest.json'),
        'cohorts':cohorts,'full_tune_native_max_abs_delta_sec':0.,'fit_vocab_exact':True,'atfm_blanks_preserved':True,
        'reports':reports,'fold_gate':gate,'peak_rss_bytes':v.guard(),'scope':'No score or refit evaluation; fixed global9 weights retain same-tune adaptive caveat.'})
    print('COMPLETE',fold,{n:r['gain'] for n,r in reports.items()},'GATE',gate,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--fold',required=True,choices=['F1','F3']);parser.add_argument('--root',required=True)
    args=parser.parse_args();main(args.fold,args.root)
