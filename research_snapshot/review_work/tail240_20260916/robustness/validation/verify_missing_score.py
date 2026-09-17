"""Post-score independent missing-route metrics, influences and freeze binding."""
from verify_missing import ROOT,BASE,OUT,COLS,ID,TIME,TARGET,sha,read,guard,check_protocol,stage_meta,stage_labels
from verify_ensemble import close,check_metrics
from taxiout.artifacts import object_hash
import json
import numpy as np
import pandas as pd


def check_pair(y,candidate,control,days,record):
    a=np.square(candidate-y);b=np.square(control-y);gain=b-a
    data=pd.DataFrame(dict(day=days,gain=gain)).groupby('day',sort=True).agg(total=('gain','sum'),n=('gain','size'))
    draws=np.random.default_rng(20260916).multinomial(len(data),np.full(len(data),1/len(data)),size=2000)
    ci=np.quantile((draws@data.total.to_numpy())/(draws@data.n.to_numpy()),[.025,.975])
    close(ci,record['mse_gain_ci95']);close(gain.mean(),record['mse_gain'])
    close(np.sqrt(b.mean())-np.sqrt(a.mean()),record['rmse_gain'])
    assert len(record['day_removals'])==len(data)
    for day,rec in zip(data.index,record['day_removals']):
        assert day==rec['day'];keep=days!=day
        assert rec['rows_removed']==int((~keep).sum())
        close(gain[keep].mean(),rec['mse_gain'])
        close(np.sqrt(b[keep].mean())-np.sqrt(a[keep].mean()),rec['rmse_gain'])
    order=pd.Series(gain).sort_values(ascending=False,kind='stable').index
    for k in [2,5,10]:
        rec=record['top_beneficial_deletions'][str(k)];chosen=order[:k]
        np.testing.assert_array_equal(rec['removed_positions'],chosen)
        keep=~np.isin(np.arange(len(y)),chosen)
        close(gain[keep].mean(),rec['mse_gain']);close(gain[chosen].sum(),rec['removed_sse_gain'])
        close(np.sqrt(b[keep].mean())-np.sqrt(a[keep].mean()),rec['rmse_gain'])


def main():
    protocol,cohorts=check_protocol();folder=BASE/'evaluation_v2'
    # No evaluation labels are loaded before this completed producer receipt exists.
    summary=read(folder/'summary.json');assert summary['status']=='complete'
    assert summary['protocol_sha256']==sha(BASE/'protocol.json')
    opened=read(folder/'evaluation_opened.json')
    source=ROOT/'review_work/tail240_20260916/robustness/tail/chronological'
    assert opened['source_sha256']==summary['source_sha256']==sha(source/'score_v2.py')
    assert opened['frozen_metric_source_sha256']==summary['metric_source_sha256']==sha(source/'run_v1.py')
    releasepath=BASE/'independent_evaluation_release.json'
    assert opened['release_sha256']==summary['independent_release_sha256']==sha(releasepath)
    release=read(releasepath);assert release['status']=='released'
    assert release['protocol_sha256']==summary['protocol_sha256']
    assert opened['previous_attempt_sha256']==sha(BASE/'evaluation/evaluation_opened.json')
    assert opened['protocol_sha256']==summary['protocol_sha256']
    assert opened['manifests']==summary['model_manifest_hashes']
    bound={}
    for fold in protocol['panels']:
        for arm in protocol['arms']:
            path=BASE/fold/arm/'manifest.json';marker=read(path)
            assert sha(path)==opened['manifests'][fold+'/'+arm]
            assert marker['status']=='complete_predictions_frozen' and marker['protocol_sha256']==summary['protocol_sha256']
            checked=read(OUT/f'{fold}_{arm}.json')
            assert checked['status']=='passed' and checked['expert_manifest_sha256']==sha(path)
            released=release['independent_receipts'][fold+'/'+arm]
            assert sha(released['path'])==released['sha256']==sha(OUT/f'{fold}_{arm}.json')
            assert pd.Timestamp(marker['target_access_events'][-1]['utc'])<=pd.Timestamp(opened['utc'])
            bound[fold+'/'+arm]=dict(manifest_sha256=sha(path),verification_sha256=sha(OUT/f'{fold}_{arm}.json'))
    reports={}
    for fold,months in protocol['panels'].items():
        reports[fold]={}
        for month in months:
            stage='evaluation_'+month;expected=stage_meta(cohorts,fold,stage)
            record=summary['folds'][fold][month];path=folder/(fold+'_'+month+'.parquet')
            assert sha(path)==record['replay_sha256'];replay=pd.read_parquet(path)
            pd.testing.assert_frame_equal(replay[COLS].reset_index(drop=True),expected,check_exact=True)
            y=stage_labels(cohorts,expected);np.testing.assert_array_equal(y,replay[TARGET])
            assert record['rows']==len(y) and record['id_hash']==object_hash(expected[ID].tolist())
            assert record['label_hash']==object_hash(y.tolist())
            for arm in protocol['arms']:
                model_root=BASE/fold/arm;marker=read(model_root/'manifest.json');predpath=model_root/(stage+'.parquet')
                assert sha(predpath)==marker['predictions'][stage]['sha256']
                predicted=pd.read_parquet(predpath)
                np.testing.assert_array_equal(predicted[ID],replay[ID]);np.testing.assert_array_equal(predicted.prediction_sec,replay[arm])
                check_metrics(y,replay[arm],record['metrics'][arm])
                for airport,metrics in record['airports'].items():
                    mask=expected.ADEP_mvt.astype(str).eq(airport).to_numpy()
                    check_metrics(y[mask],replay.loc[mask,arm],metrics[arm])
            assert set(record['airports'])==set(expected.ADEP_mvt.astype(str).unique())
            days=expected[TIME].dt.strftime('%Y-%m-%d').to_numpy()
            check_pair(y,replay.et_leaf20.to_numpy(),replay.et_leaf1.to_numpy(),days,record['et20_vs_et1'])
            reports[fold][month]=dict(rows=len(y),all_metrics_airports_and2000bootstrap_and_deletions_verified=True,
                et20_vs_et1=record['et20_vs_et1']['rmse_gain'])
            guard()
    result=dict(status='passed',source_sha256=sha(__file__),protocol_sha256=sha(BASE/'protocol.json'),
        producer_summary_sha256=sha(folder/'summary.json'),evaluation_opened_sha256=sha(folder/'evaluation_opened.json'),
        model_bindings=bound,panels=reports,labels_read_only_after_all_six_models_frozen_and_producer_score_complete=True,
        fit_used=False,gpu_used=False,peak_bytes=guard(),
        limitation='Exposed missing-route diagnostics only; December shared across origins and not independent replication. No full-pipeline or2026 generalization claim.')
    dest=OUT/'evaluation_v2.json';assert not dest.exists();dest.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
