"""Full-tune signed conditional residual replay and raw-metric validation."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='2'
import lightgbm as lgb
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import validate_candidate as validation
import replay_features

ROOT,ID,TARGET=validation.ROOT,validation.ID,validation.TARGET
BASE=ROOT/'private_runs/tail240_20260916/state/risk/sign_mixture_v1'
OUT=ROOT/'private_runs/tail240_20260916/validation/sign_mixture_v1'
read,sha,oh,write=validation.read_json,validation.sha256,validation.object_hash,validation.write_json
pa.set_cpu_count(1)
pa.set_io_thread_count(1)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--fold',choices=['F1','F3'],required=True)
    args=parser.parse_args()
    folder=BASE/args.fold
    record=read(folder/'manifest.json')
    protocol=read(BASE/'protocol.json')
    assert record['status']=='complete' and record['protocol_sha256']==sha(BASE/'protocol.json')
    assert record['source_sha256']==protocol['source_sha256']==sha(ROOT/'review_work/tail240_20260916/state/risk/run_sign_mixture.py')
    for name,digest in record['outputs'].items():
        assert sha(folder/name)==digest
    binding=read(validation.BINDING)['folds'][args.fold]
    assert oh(record['split'])==binding['split_hash']
    path=ROOT/'private_runs/screening_230/data/interim/audit/departures.parquet'
    assert sha(path)==read(ROOT/'private_runs/screening_230/reports/data_audit.json')['artifacts'][path.name]
    meta=pd.read_parquet(path,columns=[ID,'FLIGHT_ID_mvt','MVT_TIME_UTC_mvt','ADEP_mvt',TARGET,'proxy_sec'])
    idx,_,_=validation.common.fold_data(meta,args.fold,full=True)
    parts={stage:meta.iloc[idx[stage]].loc[lambda frame:np.isfinite(frame.proxy_sec)].copy() for stage in ('fit','tune')}
    for stage,frame in parts.items():
        assert {'n':len(frame),'hash':oh(frame[ID].tolist())}==binding['cohorts']['finite_nm'][stage]
        assert record[stage+'_ids_hash']==oh(frame[ID].tolist()) and record[stage+'_label_hash']==oh(frame[TARGET].tolist())
    fit,tune=parts['fit'],parts['tune']
    output=pd.read_parquet(folder/'tune_predictions.parquet')
    for column in [ID,'MVT_TIME_UTC_mvt','ADEP_mvt',TARGET,'proxy_sec']:
        np.testing.assert_array_equal(output[column],tune[column])
    encoder=read(folder/'encoder.json')
    assert encoder['columns']==protocol['feature_columns']
    binary_encoder=read(ROOT/f'private_runs/tail240_20260916/state/risk/v1/{args.fold}/own/encoder.json')
    assert encoder==binary_encoder
    matrix=replay_features.rebuild(tune[ID],encoder,record['feature_receipts'])
    gate=lgb.Booster(model_file=str(folder/'gate.txt'))
    p=gate.predict(matrix,num_threads=2)
    np.testing.assert_allclose(p.sum(axis=1),1,atol=1e-12,rtol=0)
    assert (p>=0).all() and np.isfinite(p).all()
    means=[]
    rf=(fit[TARGET]-fit.proxy_sec).to_numpy()
    rt=(tune[TARGET]-tune.proxy_sec).to_numpy()
    classify=lambda values:np.where(values < -1800,0,np.where(values>1800,2,1))
    cf,ct=classify(rf),classify(rt)
    for i,name in enumerate(('negative','ordinary','positive')):
        np.testing.assert_array_equal(output['probability_'+name],p[:,i])
        model=lgb.Booster(model_file=str(folder/(name+'.txt')))
        predicted=model.predict(matrix,num_threads=2)
        np.testing.assert_array_equal(output['residual_mean_'+name],predicted)
        means.append(predicted)
        training=record['training'][name]
        assert training['fit_rows']==int((cf==i).sum()) and training['tune_rows']==int((ct==i).sum())
        assert training['raw_residual_min']==float(rf[cf==i].min()) and training['raw_residual_max']==float(rf[cf==i].max())
        assert training['raw_residual_mean']==float(rf[cf==i].mean())
    means=np.column_stack(means)
    mixture=tune.proxy_sec.to_numpy()+np.sum(p*means,axis=1)
    single=tune.proxy_sec.to_numpy()+lgb.Booster(model_file=str(folder/'matched_single.txt')).predict(matrix,num_threads=2)
    np.testing.assert_array_equal(mixture,output.mixture)
    np.testing.assert_array_equal(single,output.matched_single)
    union_dir=ROOT/f'private_runs/breakthrough_20260916/deeper_context_union/lightgbm_leaf63_sequence_aobt_allfinite_{args.fold}_s20260916'
    assert sha(union_dir/'manifest.json')==record['union_manifest_sha256'] and sha(union_dir/'tune_predictions.parquet')==record['union_tune_sha256']
    union=pd.read_parquet(union_dir/'tune_predictions.parquet').set_index(ID).loc[tune[ID],'prediction_sec'].to_numpy()
    np.testing.assert_array_equal(union,output.union)
    np.testing.assert_array_equal(.25*mixture+.75*union,output.blend25_mixture_union)
    np.testing.assert_array_equal(.25*single+.75*union,output.blend25_single_union)
    reports=read(folder/'metrics.json')
    checked={}
    for name,mask in [('all_finite',np.ones(len(tune),bool)),('ordinary_proxy',tune.proxy_sec.between(0,7200).to_numpy())]:
        y=tune[TARGET].to_numpy()[mask]
        predictions={key:output[key].to_numpy()[mask] for key in ('mixture','matched_single','union','blend25_mixture_union','blend25_single_union')}
        errors={key:(value-y)**2 for key,value in predictions.items()}
        for key,e in errors.items():
            assert float(np.sqrt(e.mean()))==reports[name]['rmse'][key]
        paired={}
        for candidate,control in [('mixture','matched_single'),('blend25_mixture_union','union'),('blend25_single_union','union')]:
            gain=errors[control]-errors[candidate]
            reported=reports[name]['paired'][candidate+'_vs_'+control]
            assert float(gain.mean())==reported['mse_improvement']
            days=tune['MVT_TIME_UTC_mvt'].dt.strftime('%Y-%m-%d').to_numpy()[mask]
            sums=pd.DataFrame({'day':days,'c':errors[candidate],'b':errors[control]}).groupby('day').agg(c=('c','sum'),b=('b','sum'),n=('c','size'))
            removals=[float(np.sqrt((errors[candidate].sum()-row.c)/(len(y)-row.n))-np.sqrt((errors[control].sum()-row.b)/(len(y)-row.n))) for row in sums.itertuples()]
            ranked=np.argsort(gain)[::-1]
            influence={str(n):float(np.sqrt(errors[candidate][ranked[n:]].mean())-np.sqrt(errors[control][ranked[n:]].mean())) for n in (1,2,5,10)}
            paired[candidate+'_vs_'+control]={'delta_rmse':reports[name]['rmse'][candidate]-reports[name]['rmse'][control],
                'all_day_removals_improve':all(value<0 for value in removals),'day_removal_delta_range':[min(removals),max(removals)],'remove_top_benefit_rows_delta':influence}
        checked[name]={'rmse':reports[name]['rmse'],'paired':paired}
    dest=OUT/args.fold
    dest.mkdir(parents=True,exist_ok=False)
    write(dest/'receipt.json',{'status':'passed','source_sha256':sha(__file__),'reconstruction_source_sha256':sha(replay_features.__file__),
        'producer_manifest_sha256':sha(folder/'manifest.json'),'fold':args.fold,'full_tune_native_gate_head_single_replay_max_delta':0.,
        'checks':checked,'peak_rss_bytes':validation.guard(),'scope':'No score predictions. Full rawcohort and tune features rebuilt, gate+all3heads+matchedsingle native replay exact; learned probabilities compose predictions, no oracle class. Pointmetrics/dayremovals/topbenefitrow removals independently recomputed.'})
    print(args.fold,checked,flush=True)


if __name__=='__main__':
    main()
