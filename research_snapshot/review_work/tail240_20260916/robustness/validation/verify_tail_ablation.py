"""Independently reconstruct fixed missing-route ablations and influence signs."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
import lightgbm
from materialize_cohorts import ROOT,sha,read,guard
import json
import numpy as np
import pandas as pd

BASE=ROOT/'private_runs/tail240_20260916/robustness/tail/v1'
REPLAY=ROOT/'private_runs/tail240_20260916/score_gap/validation'
DEST=ROOT/'private_runs/tail240_20260916/robustness/validation/tail_ablation_review.json'
ID,TARGET='MVT_ID_mvt','TAXITIME_SEC_mvt'
WEIGHTS={'F1':192122/344841,'F3':152719/344841}


def close(a,b):np.testing.assert_allclose(a,b,rtol=1e-12,atol=1e-8)


def main():
    protocol=read(BASE/'protocol.json');summary=read(BASE/'summary.json');influence=read(BASE/'component_influence.json')
    source=ROOT/'review_work/tail240_20260916/robustness/tail'
    assert summary['source_sha256']==protocol['declaration']['source_sha256']==sha(source/'ablate_v1.py')
    assert summary['protocol_sha256']==sha(BASE/'protocol.json')
    assert influence['source_sha256']==sha(source/'component_influence_v1.py')
    assert influence['main_source_sha256']==summary['source_sha256']
    assert influence['summary_sha256']==sha(BASE/'summary.json')
    binding=read(REPLAY/'receipt.json');seasonal={};removals={};daily={}
    for fold,w in WEIGHTS.items():
        path=REPLAY/f'{fold}_replay.parquet';assert sha(path)==binding['outputs'][path.name]
        original=pd.read_parquet(path)
        path=BASE/f'{fold}_replay.parquet';assert sha(path)==summary['replay_bindings'][fold]['sha256']
        frame=pd.read_parquet(path)
        for name in [ID,TARGET]:np.testing.assert_array_equal(frame[name],original[name])
        np.testing.assert_array_equal(frame.current,original.local_fold_weights_raw)
        np.testing.assert_array_equal(frame.v2,original.v2_raw)
        np.testing.assert_array_equal(frame.missing,~np.isfinite(original.proxy_sec))
        np.testing.assert_array_equal(frame.day,original.MVT_TIME_UTC_mvt.dt.strftime('%Y-%m-%d'))
        np.testing.assert_array_equal(frame.airport,original.ADEP_mvt.astype(str))
        sources={}
        for name,entry in protocol['declaration']['sources'][fold].items():
            assert sha(entry['path'])==entry['sha256']
            value=pd.read_parquet(entry['path'])
            np.testing.assert_array_equal(value[ID],frame[ID])
            sources[name]=value.prediction_sec.to_numpy(float)
        missing=frame.missing.to_numpy();y=frame[TARGET].to_numpy();cur=(frame.current-y)**2
        predictions={name:frame.current.to_numpy().copy() for name in protocol['declaration']['arms']}
        predictions['remove_normalized'][missing]=sources['forest_blend25'][missing]
        predictions['remove_ExtraTrees'][missing]=.75*frame.v2.to_numpy()[missing]+.25*sources['normalized'][missing]
        predictions['remove_both'][missing]=frame.v2.to_numpy()[missing]
        close(frame.current.to_numpy()[missing],.75*sources['forest_blend25'][missing]+.25*sources['normalized'][missing])
        for name,p in predictions.items():
            close(p,frame[name]);np.testing.assert_array_equal(p[~missing],frame.current.to_numpy()[~missing])
            other=(p-y)**2;benefit=other-cur
            seasonal[name]=seasonal.get(name,0)+w*float(other.mean())
            result=summary['folds'][fold][name]
            close(result['all']['mse'],other.mean())
            close(result['paired_rmse_gain'],np.sqrt(cur.mean())-np.sqrt(other.mean()))
            if name=='current':continue
            supplemental=influence['folds'][fold][name]
            close(supplemental['component_benefit_rmse'],np.sqrt(other.mean())-np.sqrt(cur.mean()))
            positive_order=frame.loc[missing].assign(benefit=benefit[missing]).sort_values('benefit',ascending=False,kind='stable').index
            daily_values=[]
            for rec in result['day_removals']:
                keep=frame.day.ne(rec['day']).to_numpy()
                value=np.sqrt(other[keep].mean())-np.sqrt(cur[keep].mean())
                close(value,-rec['paired_rmse_gain']);daily_values.append(float(value))
            close(supplemental['day_removal_component_benefit_range'],[min(daily_values),max(daily_values)])
            daily[fold+'/'+name]=[min(daily_values),max(daily_values)]
            for k in (1,2,5,10):
                chosen=positive_order[:k];keep=~frame.index.isin(chosen)
                rec=supplemental['remove_component_supporting_rows'][str(k)]
                np.testing.assert_array_equal(rec['removed_ids'],frame.loc[chosen,ID])
                close(rec['current_mse'],cur[keep].mean());close(rec['ablation_mse'],other[keep].mean())
                close(rec['removed_component_sse_benefit'],benefit.iloc[chosen].sum())
                close(rec['removed_share_of_net_benefit'],benefit.iloc[chosen].sum()/benefit.sum())
                value=removals.setdefault((name,str(k)),np.zeros(2))
                value+=w*np.array([cur[keep].mean(),other[keep].mean()])
        guard()
    for name,mse in seasonal.items():close(np.sqrt(mse),summary['seasonal'][name]['rmse'])
    for (name,k),values in removals.items():
        rec=influence['seasonal_remove_k_component_supporting_rows_per_fold'][name][k]
        close(rec['component_rmse_benefit'],np.sqrt(values[1])-np.sqrt(values[0]))
    result=dict(status='passed',source_sha256=sha(__file__),protocol_sha256=sha(BASE/'protocol.json'),summary_sha256=sha(BASE/'summary.json'),
        influence_sha256=sha(BASE/'component_influence.json'),all_predictions_reconstructed=True,finite_predictions_unchanged=True,
        independently_bound_original_labels_ids_routes=True,all_component_supporting_row_deletions_verified=True,
        all_daily_component_benefits_verified=True,seasonal_rmse={name:float(np.sqrt(mse)) for name,mse in seasonal.items()},
        day_removal_component_benefit_ranges=daily,
        limitation='Exposed development sensitivity only. Positive supplemental benefit favors retaining component; main ablation gains use the opposite sign. Row deletion does not alter canonical all-row endpoints.',
        peak_bytes=guard(),fit_used=False,ranking_labels_read=False)
    assert not DEST.exists();DEST.parent.mkdir(parents=True,exist_ok=True)
    DEST.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
