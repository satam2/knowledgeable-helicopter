"""Independent coefficient, restricted raw-label and diagnostic replay."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import psutil

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'review_work/campaign_20260916'))
import common
ID,TIME,TARGET=common.ID,common.MOVEMENT,common.TARGET
BASE=ROOT/'private_runs/tail240_20260916/state/exact_arrival/v1'
OUT=ROOT/'private_runs/tail240_20260916/validation/exact_arrival_association_v1'
SEED=20260916
pa.set_cpu_count(1)
pa.set_io_thread_count(1)


def close(a,b):
    np.testing.assert_allclose(a,b,rtol=1e-11,atol=1e-10)


def score(y,p,record):
    e=p-y
    assert record['n']==len(y)
    for key,value in dict(sse=np.dot(e,e),rmse=np.sqrt(np.mean(e*e)),mae=np.mean(abs(e)),bias=np.mean(e)).items():close(value,record[key])


def paired(y,p,ref,days,record):
    a=(p-y)**2;b=(ref-y)**2
    score(y,p,record['candidate']);score(y,ref,record['reference'])
    close(np.sqrt(b.mean())-np.sqrt(a.mean()),record['gain'])
    close((b-a).mean(),record['mse_gain'])
    unique=sorted(set(days));counts=[];gains=[]
    for day in unique:
        mask=days==day
        counts.append(mask.sum());gains.append((b[mask]-a[mask]).sum())
    draws=np.random.default_rng(SEED).multinomial(len(unique),np.repeat(1/len(unique),len(unique)),size=300)
    ci=np.quantile((draws@np.array(gains))/(draws@np.array(counts)),[.025,.975])
    close(ci,record['mse_gain_ci95'])
    deletion=[]
    for day,item in zip(unique,record['day_removals']):
        assert item['day']==str(day)
        keep=days!=day
        gain=np.sqrt(b[keep].mean())-np.sqrt(a[keep].mean())
        close(gain,item['gain']);deletion.append(gain)
    assert all(g>0 for g in deletion)==record['all_day_removals_improve']
    order=np.argsort(-(b-a),kind='stable')
    for n in [1,5,10]:
        keep=np.ones(len(y),bool);keep[order[:min(n,len(y)-1)]]=False
        close(np.sqrt(b[keep].mean())-np.sqrt(a[keep].mean()),record['remove_top_beneficial_rows_gain'][str(n)])


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    protocol=common.read_json(BASE/'protocol.json')
    manifest=common.read_json(BASE/'join_manifest.json')
    report=common.read_json(BASE/'association.json')
    assert report['protocol_sha256']==common.sha256(BASE/'protocol.json')
    assert report['frozen_join_manifest_sha256']==common.sha256(BASE/'join_manifest.json')
    for name,digest in protocol['source_hashes'].items():assert common.sha256(ROOT/name)==digest
    for binding in [manifest['outputs'],report['outputs']]:
        for name,digest in binding.items():assert common.sha256(BASE/name)==digest
    meta_path=ROOT/'private_runs/screening_230/data/interim/audit/departures.parquet'
    assert common.sha256(meta_path)==protocol['metadata_sha256']
    metadata=ds.dataset(meta_path,format='parquet')
    frames={}
    for fold in ['F1','F3']:
        frame=pd.read_parquet(BASE/f'{fold}_join.parquet')
        labels=metadata.to_table(columns=[ID,TARGET],filter=ds.field(ID).isin(frame[ID].to_numpy()),use_threads=False).to_pandas().set_index(ID)
        assert labels.index.is_unique and set(labels.index)==set(frame[ID])
        frame['raw_target_sec']=labels.loc[frame[ID],TARGET].to_numpy(float)
        control=ROOT/'private_runs/breakthrough_20260916/deeper_context_union'/f'lightgbm_leaf63_sequence_aobt_allfinite_{fold}_s20260916'
        assert common.sha256(control/'tune_predictions.parquet')==protocol['controls'][fold]['predictions_sha256']
        predictions=pd.read_parquet(control/'tune_predictions.parquet').set_index(ID)
        frame['baseline_prediction_sec']=predictions.loc[frame[ID],'prediction_sec'].to_numpy(float)
        frames[fold]=frame
    calibration=frames['F1'].loc[frames['F1'][TIME]<pd.Timestamp('2025-06-16',tz='UTC')]
    linked=calibration.loc[calibration.linked].copy()
    linked['error']=linked.raw_target_sec-linked.baseline_prediction_sec
    means={a:(f.error.mean(),f.delta_sec.mean()) for a,f in linked.groupby('ADEP_mvt',observed=True)}
    global_error=linked.error.mean();global_delta=linked.delta_sec.mean()
    scale=np.std(linked.delta_sec.to_numpy(float)) or 1.
    x=(linked.delta_sec.to_numpy()-np.array([means[a][1] for a in linked.ADEP_mvt]))/scale
    y=linked.error.to_numpy()-np.array([means[a][0] for a in linked.ADEP_mvt])
    coefficient=np.dot(x,y)/(np.dot(x,x)+100.)
    fitted=common.read_json(BASE/'calibration.json')
    assert fitted==report['calibration']
    assert fitted['calibration_id_hash']==common.object_hash(calibration[ID].tolist())
    assert fitted['calibration_rows']==len(calibration) and fitted['linked_calibration_rows']==len(linked)
    for key,value in dict(coefficient=coefficient,coefficient_closed_form=coefficient,delta_scale=scale,delta_global_mean=global_delta,global_error_mean=global_error).items():close(value,fitted[key])
    for airport,(error,delta) in means.items():
        close(error,fitted['airport_error_mean'][airport]);close(delta,fitted['airport_delta_mean'][airport])
    checks={}
    for name,frame in [('June16_30',frames['F1'].loc[frames['F1'][TIME]>=pd.Timestamp('2025-06-16',tz='UTC')].copy()),('October',frames['F3'].copy())]:
        linked=frame.linked.to_numpy(bool);baseline=frame.baseline_prediction_sec.to_numpy(float)
        nuisance=np.array([means.get(a,(global_error,global_delta))[0] for a in frame.ADEP_mvt])
        centers=np.array([means.get(a,(global_error,global_delta))[1] for a in frame.ADEP_mvt])
        bias=baseline+np.where(linked,nuisance,0.)
        delta=frame.delta_sec.to_numpy(float)
        pred=bias+np.where(linked,coefficient*(delta-centers)/scale,0.)
        artifact=pd.read_parquet(BASE/f'{name}_diagnostic.parquet')
        pd.testing.assert_frame_equal(artifact[frame.columns].reset_index(drop=True),frame.reset_index(drop=True))
        close(artifact.airport_bias_prediction_sec,bias);close(artifact.delta_prediction_sec,pred)
        np.testing.assert_array_equal(pred[~linked],baseline[~linked])
        y=frame.raw_target_sec.to_numpy(float);days=frame[TIME].dt.floor('D').to_numpy()
        result=report['results'][name]
        for key,selected in [('full_ordinary',np.ones(len(y),bool)),('linked',linked)]:
            item=result[key]
            for p,k in [(baseline,'baseline'),(bias,'airport_bias'),(pred,'delta')]:score(y[selected],p[selected],item[k])
            paired(y[selected],pred[selected],bias[selected],days[selected],item['delta_vs_bias'])
            paired(y[selected],pred[selected],baseline[selected],days[selected],item['delta_vs_baseline'])
        perm=result['permutation_null'];rng=np.random.default_rng(SEED)
        subset=frame.loc[linked].copy();subset['day']=subset[TIME].dt.floor('D')
        groups=list(subset.groupby(['ADEP_mvt','ADES_mvt','AIRCRAFT_OPERATOR_flt','day'],observed=True,dropna=False,sort=True).indices.values())
        indices=np.flatnonzero(linked);singletons=sum(len(g)==1 for g in groups)
        assert perm['singleton_rows']==singletons and perm['linked_rows']==len(indices)
        close(perm['singleton_share'],singletons/len(indices))
        actual=np.mean((bias-y)**2-(pred-y)**2);close(actual,perm['actual_mse_gain_vs_bias'])
        ge=0
        for item in perm['records']:
            shuffled=delta.copy()
            for group in groups:
                positions=indices[group];shuffled[positions]=delta[rng.permutation(positions)]
            candidate=bias+np.where(linked,coefficient*(shuffled-centers)/scale,0.)
            changed=np.count_nonzero(shuffled[indices]!=delta[indices])
            assert changed==item['changed_linked_rows']
            close(1-changed/len(indices),item['unchanged_linked_share'])
            gain=np.mean((bias-y)**2-(candidate-y)**2)
            close(gain,item['mse_gain_vs_bias']);ge+=gain>=actual
            close(np.sqrt(np.mean((bias-y)**2))-np.sqrt(np.mean((candidate-y)**2)),item['rmse_gain_vs_bias'])
        assert ge==perm['permutation_gains_ge_actual']
        checks[name]=dict(rows=len(frame),linked=int(linked.sum()),delta_vs_bias_gain=result['full_ordinary']['delta_vs_bias']['gain'],all_outputs_and_uncertainty_replayed=True)
    passed=all(r['full_ordinary']['delta_vs_bias']['gain']>0 and r['full_ordinary']['delta_vs_bias']['all_day_removals_improve'] and r['full_ordinary']['delta_vs_bias']['mse_gain_ci95'][0]>0 for r in report['results'].values())
    assert passed==report['association_screen_passed']
    peak=max(psutil.Process().memory_info().rss,psutil.Process().memory_info().peak_wset)
    assert peak<2*1024**3 and psutil.virtual_memory().available>=8*1024**3
    common.write_json(OUT/'receipt.json',dict(status='passed',source_sha256=common.sha256(__file__),association_sha256=common.sha256(BASE/'association.json'),coefficient=coefficient,calibration_only_June1_15=True,query_labels_only=True,all_raw_labels_baselines_predictions_exact=True,all_statistics_permutations_replayed=True,results=checks,association_screen_passed=passed,no_promotion=True,peak_bytes=peak))
    print('VERIFIED_ASSOCIATION',passed,coefficient,checks,flush=True)


if __name__=='__main__':main()
