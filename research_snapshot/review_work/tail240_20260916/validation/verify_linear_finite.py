"""Independent full387 fit scaler reconstruction, native pair and primary replay."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='2'
import lightgbm as lgb
import argparse
import gc
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import psutil
from threadpoolctl import threadpool_limits
import validate_candidate as v

ROOT,ID,TARGET,TIME=v.ROOT,v.ID,v.TARGET,v.common.MOVEMENT
BASE=ROOT/'private_runs/tail240_20260916/models/linear_finite_tune_v1'


def guard():
    m=psutil.Process().memory_info();peak=max(m.rss,m.peak_wset)
    assert peak<10*1024**3 and psutil.virtual_memory().available>=8*1024**3
    return peak


def check_result(y,pred,reference,days,record):
    error=(pred-y)**2;old=(reference-y)**2;gain=old-error
    unique,codes=np.unique(days,return_inverse=True)
    deletion=(gain.sum()-np.bincount(codes,weights=gain))/(len(y)-np.bincount(codes))
    expected=dict(mse_gain=float(gain.mean()),day_removal_min_gain=float(deletion.min()),
        rmse=float(np.sqrt(error.mean())),reference_rmse=float(np.sqrt(old.mean())),
        gain=float(np.sqrt(old.mean())-np.sqrt(error.mean())))
    for key,value in expected.items():np.testing.assert_allclose(value,record[key],rtol=1e-10,atol=1e-9)
    assert record['days']==len(unique) and record['all_day_removals_improve']==bool((deletion>0).all())
    return v.paired(y,reference,pred,days)


def inspect_native(model,linear,categories):
    total=coefficients=0
    for start in range(0,model.current_iteration(),50):
        dump=model.dump_model(start_iteration=start,num_iteration=min(50,model.current_iteration()-start))
        stack=[tree['tree_structure'] for tree in dump['tree_info']]
        while stack:
            node=stack.pop()
            if 'split_index' in node:
                stack.extend([node['left_child'],node['right_child']]);continue
            total+=1
            for key in ['leaf_value','leaf_const']:
                if key in node:assert np.isfinite(node[key])
            fields=node.get('leaf_features',[]);values=node.get('leaf_coeff',[])
            assert len(fields)==len(values) and not set(fields)&categories
            assert np.isfinite(values).all()
            coefficients+=len(values)
        del dump,stack
        guard()
    assert (coefficients>0) if linear else (coefficients==0)
    return dict(leaves=total,linear_coefficients=coefficients,all_coefficients_finite=True,no_categorical_linear_regressors=True)


def main(fold):
    assert psutil.virtual_memory().available>=18*1024**3
    folder=BASE/fold;out=ROOT/'private_runs/tail240_20260916/validation'/f'linear_finite_{fold}_v1'
    out.mkdir(parents=True,exist_ok=False)
    protocol=v.read_json(BASE/'protocol.json');manifest=v.read_json(folder/'manifest.json')
    assert manifest['status']=='complete' and manifest['protocol_sha256']==v.sha256(BASE/'protocol.json')
    assert manifest['source_sha256']==protocol['source_sha256']==v.sha256(ROOT/'review_work/tail240_20260916/models/linear_finite_tune.py')
    for name,digest in protocol['sources'].items():assert v.sha256(ROOT/name)==digest
    for name,digest in manifest['outputs'].items():assert v.sha256(folder/name)==digest
    arms={arm:v.read_json(folder/arm/'manifest.json') for arm in ['constant','linear']}
    receipts=arms['constant']['feature_receipts']
    assert receipts==arms['linear']['feature_receipts']
    path=ROOT/'private_runs/screening_230/data/interim/audit/departures.parquet'
    assert v.sha256(path)==v.read_json(ROOT/'private_runs/screening_230/reports/data_audit.json')['artifacts'][path.name]
    meta=pd.read_parquet(path,columns=[ID,'FLIGHT_ID_mvt',TIME,TARGET,'proxy_sec'])
    idx,split,_=v.common.fold_data(meta,fold,full=True)
    parts={s:meta.iloc[idx[s]].loc[lambda f:np.isfinite(f.proxy_sec)].copy() for s in ['fit','tune']}
    fit,tune=parts['fit'],parts['tune'];nfit=len(fit)
    ids=pd.Index(pd.concat([fit[ID],tune[ID]],ignore_index=True));assert ids.is_unique
    cohorts={s:dict(n=len(f),hash=v.object_hash(f[ID].tolist())) for s,f in parts.items()}
    for arm,record in arms.items():
        assert record['status']=='complete' and record['arm']==arm and record['params']==protocol['params'][arm]
        assert record['protocol_sha256']==v.sha256(BASE/'protocol.json') and record['encoder_sha256']==v.sha256(folder/'encoder.json')
        assert record['ids']==cohorts and v.object_hash(record['split'])==v.object_hash(split)
        for name,digest in record['outputs'].items():assert v.sha256(folder/arm/name)==digest
    encoder=v.read_json(folder/'encoder.json');columns=encoder['columns'];vocab=encoder['vocab']
    assert columns==protocol['columns'] and len(columns)==387
    del meta,idx,parts;gc.collect()
    matrix=np.memmap(out/'matrix.float32',mode='w+',dtype='float32',shape=(len(ids),len(columns)),order='F')
    counts=np.zeros(len(columns),int);rebuilt={name:set() for name in vocab};filled=set();source_flags=[]
    for receipt in receipts:
        path=Path(receipt['path']);assert v.sha256(path)==receipt['sha256']
        fill='screening_230/data/interim/features' not in path.as_posix() and 'sequence_flatten' not in path.as_posix()
        names=receipt['columns'];seen=np.zeros(len(ids),bool)
        source_flags.append(dict(path=str(path),fill=fill))
        if fill:filled.update(names)
        for batch in pq.ParquetFile(path).iter_batches(batch_size=8192,columns=[ID,*names],use_threads=False):
            frame=batch.to_pandas();pos=ids.get_indexer(frame[ID]);keep=pos>=0;where=pos[keep]
            assert not seen[where].any() and len(np.unique(where))==len(where)
            seen[where]=True;fitmask=(pos>=0)&(pos<nfit)
            for name in names:
                values=frame.loc[keep,name]
                if name in vocab:
                    rebuilt[name].update(frame.loc[fitmask,name].dropna().astype(str).tolist())
                    mapping={word:i+2 for i,word in enumerate(vocab[name])}
                    values=values.astype('string').map(mapping).fillna(1).where(values.notna(),0).to_numpy('float32')
                else:
                    values=pd.to_numeric(values).to_numpy(dtype='float32',na_value=np.nan)
                    values[~np.isfinite(values)]=np.nan
                    if fill:values[np.isnan(values)]=-999999.
                matrix[where,columns.index(name)]=values
        assert int(seen.sum())==receipt['rows']
        for name in names:counts[columns.index(name)]+=int(seen.sum())
        print('SOURCE',path.parent.name,len(names),guard(),flush=True)
    assert np.all(counts==len(ids)) and {name:sorted(words) for name,words in rebuilt.items()}==vocab
    scaler={};raw_sentinel_counts={}
    for j,name in enumerate(columns):
        if name in vocab:continue
        values=np.asarray(matrix[:,j],dtype=np.float64).copy();values[~np.isfinite(values)]=np.nan
        if name in filled:values[values==-999999.]=np.nan
        else:raw_sentinel_counts[name]=int((values==-999999.).sum())
        valid=np.isfinite(values[:nfit]);observed=values[:nfit][valid]
        mean=float(observed.mean()) if len(observed) else 0.
        scale=float(observed.std()) if len(observed) else 1.;scale=scale if scale>1e-12 else 1.
        state=dict(mean=mean,scale=scale,observed_fit=int(valid.sum()),sentinel_to_nan=name in filled)
        assert state==encoder['scaler'][name],name
        scaled=((values-mean)/scale).astype('float32')
        assert np.array_equal(np.isfinite(scaled),np.isfinite(values))
        matrix[:,j]=scaled;scaler[name]=state
        del values,observed,scaled,valid
        if j%75==0:print('SCALER',j,guard(),flush=True)
    assert set(scaler)==set(encoder['scaler'])
    matrix.flush()
    oldfolder=ROOT/'private_runs/breakthrough_20260916/deeper_context_union'/f'lightgbm_leaf63_sequence_aobt_allfinite_{fold}_s20260916'
    assert v.sha256(oldfolder/'manifest.json')==protocol['controls'][fold]
    oldmarker=v.read_json(oldfolder/'manifest.json')
    assert oldmarker['feature_columns']==columns
    original_sources=ROOT/'private_runs/tail240_20260916/models'/('following_groups_tune_v1' if fold=='F1' else 'following_groups_tune_v2')/fold/'control387/manifest.json'
    assert v.read_json(original_sources)['feature_receipts']==receipts
    assert v.sha256(oldfolder/'tune_predictions.parquet')==oldmarker['outputs']['tune_predictions.parquet']
    old=pd.read_parquet(oldfolder/'tune_predictions.parquet');np.testing.assert_array_equal(old[ID],tune[ID])
    ensemble=ROOT/'private_runs/breakthrough_20260916/models/context_gate/final_simplex9_v1'
    prep=v.read_json(ensemble/'preparation.json')['folds'][fold];weights=v.read_json(ensemble/f'{fold}_weights.json')
    assert weights==prep['weights'] and v.sha256(ensemble/f'{fold}_aligned_tune.parquet')==prep['aligned_tune_sha256']
    aligned=pd.read_parquet(ensemble/f'{fold}_aligned_tune.parquet');ordinary=tune.proxy_sec.between(0,7200).to_numpy()
    np.testing.assert_array_equal(aligned[ID],tune.loc[ordinary,ID]);np.testing.assert_array_equal(aligned[TARGET],tune.loc[ordinary,TARGET])
    np.testing.assert_array_equal(aligned.lgb63_union,old.prediction_sec.to_numpy()[ordinary])
    neuralroot=ROOT/'private_runs/tail240_20260916/state/neural_context'/('v1' if fold=='F1' else 'v3')/fold
    assert v.sha256(neuralroot/'manifest.json')==protocol['neural_controls'][fold]
    neuralmarker=v.read_json(neuralroot/'manifest.json');assert v.sha256(neuralroot/'tune_predictions.parquet')==neuralmarker['outputs']['tune_predictions.parquet']
    neural=pd.read_parquet(neuralroot/'tune_predictions.parquet').set_index(ID).loc[aligned[ID]]
    np.testing.assert_array_equal(neural[TARGET],aligned[TARGET])
    current=aligned[weights['experts']].to_numpy(float)@np.asarray(weights['global'])
    current+=weights['global'][weights['experts'].index('tabm_ple8')]*(neural.prediction_sec.to_numpy()-aligned.tabm_ple8.to_numpy())
    y=tune[TARGET].to_numpy(float);days=tune[TIME].dt.floor('D').to_numpy();predictions={};checks={}
    for arm,record in arms.items():
        model=lgb.Booster(model_file=str(folder/arm/'model.txt'))
        assert model.feature_name()==columns and model.current_iteration()==record['steps']
        assert bool(model.params['linear_tree'])==(arm=='linear') and model.params['linear_lambda']==5
        residual=model.predict(matrix[nfit:],num_threads=2);pred=residual+tune.proxy_sec.to_numpy(float)
        saved=pd.read_parquet(folder/arm/'tune.parquet');np.testing.assert_array_equal(saved[ID],tune[ID]);np.testing.assert_array_equal(saved.prediction_sec,pred)
        np.testing.assert_allclose(pred-y,residual-(y-tune.proxy_sec.to_numpy(float)),rtol=1e-10,atol=1e-8)
        structure=inspect_native(model,arm=='linear',{columns.index(c) for c in vocab})
        checks[arm]=dict(native_predictions_exact=True,structure=structure,
            original_global=check_result(y,pred,old.prediction_sec.to_numpy(),days,record['results']['original_global']),
            ordinary_fixed25=check_result(y[ordinary],.75*current+.25*pred[ordinary],current,days[ordinary],record['results']['ordinary_fixed25']))
        assert record['results']==manifest['arms'][arm]
        predictions[arm]=pred;del model,residual;gc.collect();guard()
        print('NATIVE_EXACT',arm,flush=True)
    matched=check_result(y,predictions['linear'],predictions['constant'],days,manifest['matched_linear_vs_constant'])
    replacement=None
    if fold=='F1':
        diagnostic=ROOT/'private_runs/tail240_20260916/models/linear_replacement_diagnostic_v1/receipt.json'
        dr=v.read_json(diagnostic)
        assert dr['source_sha256']==v.sha256(ROOT/'review_work/tail240_20260916/models/linear_replacement_diagnostic.py')
        for path,digest in dr['producer_sources'].items():assert v.sha256(path)==digest
        weight=weights['global'][weights['experts'].index('lgb63_union')]
        assert weight==dr['coefficient'] and dr['rows']==int(ordinary.sum()) and dr['id_hash']==v.object_hash(aligned[ID].tolist())
        replaced=current+weight*(predictions['linear'][ordinary]-old.prediction_sec.to_numpy()[ordinary])
        replacement=check_result(y[ordinary],replaced,current,days[ordinary],dr['result'])
        for count in [1,5,10]:np.testing.assert_allclose(-replacement['remove_top_gain_rows'][str(count)]['delta_rmse'],dr['result']['remove_best_rows_gain_sec'][str(count)],rtol=1e-10,atol=1e-9)
        replacement=dict(paired=replacement,receipt_sha256=v.sha256(diagnostic),weight=weight,posthoc_only=True,original_gate_unchanged=True)
    pd.DataFrame({ID:tune[ID],TARGET:y,'constant_sec':predictions['constant'],'linear_sec':predictions['linear']}).to_parquet(out/'predictions.parquet',index=False)
    record=dict(status='passed',source_sha256=v.sha256(__file__),producer_manifest_sha256=v.sha256(folder/'manifest.json'),protocol_sha256=v.sha256(BASE/'protocol.json'),cohorts=cohorts,raw_label_hashes={s:v.object_hash(f[TARGET].tolist()) for s,f in [('fit',fit),('tune',tune)]},all387_source_values_and_fit_vocabularies_rebuilt=True,all372_fit_scalers_exact=True,source_fill_flags=source_flags,legitimate_raw_sentinel_counts=raw_sentinel_counts,all_native_predictions_exact=True,current387_composition_exact=True,matched=matched,arms=checks,posthoc_component_replacement=replacement,peak_bytes=guard(),limitation='Full original finite fit/tune values and scalers independently rebuilt; both native models and declared current387 fixed25 arithmetic replayed. No fitting, score-stage or ranking predictions performed.')
    v.write_json(out/'receipt.json',record)
    del matrix;gc.collect();(out/'matrix.float32').unlink()
    print('VERIFIED',fold,'matchedgain',-matched['delta_rmse'],'primarygain',-checks['linear']['ordinary_fixed25']['delta_rmse'],'peak',record['peak_bytes'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--fold',choices=['F1','F3'],required=True)
    pa.set_cpu_count(1);pa.set_io_thread_count(1)
    with threadpool_limits(2):main(parser.parse_args().fold)
