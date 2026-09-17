"""Separate calibration-only and post-score independent verification entrypoints."""
from materialize_cohorts import ROOT,COLS,ID,TIME,sha,read,guard
from taxiout.artifacts import object_hash
from mixture_oracle import validate_state
import argparse
import json
import numpy as np
import pandas as pd

BASE=ROOT/'private_runs/tail240_20260916/robustness/chronological_v1'
OUT=ROOT/'private_runs/tail240_20260916/robustness/validation/ensemble'
COHORTS=ROOT/'private_runs/tail240_20260916/robustness/validation/cohorts_v1'
TARGET='TAXITIME_SEC_mvt'
FAMILIES=['lgb','ple','catboost']
VARIANTS=['equal','simplex','shrunk']


def close(a,b):np.testing.assert_allclose(a,b,rtol=1e-10,atol=1e-7)


def protocols():
    training=read(BASE/'protocol.json');protocol=read(BASE/'ensemble/protocol.json')
    assert protocol['training_protocol_sha256']==sha(BASE/'protocol.json')
    assert protocol['source_sha256']==sha(ROOT/'review_work/tail240_20260916/robustness/evaluate.py')
    assert protocol['mixture_sha256']==sha(ROOT/'review_work/tail240_20260916/robustness/mixture.py')
    for path,digest in training['sources'].items():assert sha(ROOT/path)==digest
    assert protocol['families']==FAMILIES and protocol['variants']==VARIANTS
    assert protocol['evaluation']==training['evaluation']
    assert training['cohort_manifest_sha256']==sha(COHORTS/'manifest.json')
    return protocol,read(COHORTS/'manifest.json')


def metadata(cohort,fold,stage,finite=True):
    entry=cohort['folds'][fold]['stages'][stage];path=ROOT/entry['path']
    assert sha(path)==entry['sha256']
    data=pd.read_parquet(path)
    return data.loc[np.isfinite(data.proxy_sec)].reset_index(drop=True) if finite else data


def labels(cohort,frame):
    path=ROOT/cohort['metadata_path'];assert sha(path)==cohort['metadata_sha256']
    data=pd.read_parquet(path,columns=[ID,TARGET],filters=[(TIME,'>=',frame[TIME].min()),(TIME,'<=',frame[TIME].max())]).set_index(ID)
    assert data.index.is_unique
    return data.loc[frame[ID],TARGET].to_numpy(float)


def write_result(fold,action,result):
    OUT.mkdir(parents=True,exist_ok=True);path=OUT/f'{fold}_{action}.json'
    assert not path.exists()
    result.update(status='passed',verifier_sha256=sha(__file__),simplex_oracle_sha256=sha(ROOT/'review_work/tail240_20260916/robustness/validation/mixture_oracle.py'),
        fold=fold,action=action,peak_bytes=guard(),fit_used=False)
    path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)


def calibrate(fold):
    protocol,cohort=protocols();folder=BASE/'ensemble'/fold/'calibration'
    manifest=read(folder/'manifest.json');assert manifest['status']=='frozen_before_evaluation'
    assert manifest['evaluation_labels_read'] is False
    assert manifest['protocol_sha256']==sha(BASE/'ensemble/protocol.json')
    assert manifest['weights_sha256']==sha(folder/'weights.json')
    state=read(folder/'weights.json');assert state['families']==FAMILIES
    stages=['calibration']+['evaluation_'+m for m in protocol['evaluation']['panels'][fold]]
    assert set(stages)==set(manifest['predictions'])
    checks={};oracle=None
    for stage in stages:
        entry=manifest['predictions'][stage];path=folder/(stage+'.parquet')
        assert sha(path)==entry['sha256']
        frame=pd.read_parquet(path);expected=metadata(cohort,fold,stage)
        assert TARGET not in frame
        pd.testing.assert_frame_equal(frame[COLS],expected,check_exact=True)
        arrays=[]
        for family in FAMILIES:
            root=BASE/fold/family/'refit';binding=entry['experts'][family]
            assert sha(root/'manifest.json')==binding['manifest_sha256']
            expert=read(root/'manifest.json')
            assert expert['status']=='complete' and expert['protocol_sha256']==protocol['training_protocol_sha256']
            assert sha(root/(stage+'.parquet'))==binding['prediction_sha256']==expert['predictions'][stage]['sha256']
            saved=pd.read_parquet(root/(stage+'.parquet'))
            pd.testing.assert_frame_equal(saved[COLS],expected,check_exact=True)
            np.testing.assert_array_equal(frame[family],saved.prediction_sec)
            arrays.append(saved.prediction_sec.to_numpy(float))
        p=np.column_stack(arrays)
        for variant in VARIANTS:close(frame[variant],p@np.array(state[variant]))
        if stage=='calibration':
            mask=expected.proxy_sec.between(0,7200).to_numpy();y=labels(cohort,expected.loc[mask])
            assert state['label_hash']==object_hash(y.tolist())
            assert state['id_hash']==object_hash(expected.loc[mask,ID].tolist())
            assert state['bindings']==entry['experts']
            oracle=validate_state(state,p[mask],y)
        checks[stage]=dict(rows=len(frame),sha256=entry['sha256'],all_expert_and_mixture_predictions_reproduced=True)
        guard()
    write_result(fold,'calibration',dict(protocol_sha256=sha(BASE/'ensemble/protocol.json'),
        calibration_manifest_sha256=sha(folder/'manifest.json'),oracle=oracle,predictions=checks,
        labels_read_only='ordinary calibration rows',evaluation_labels_read=False))


def measure(y,p):
    e=np.asarray(p)-np.asarray(y)
    return dict(rows=len(y),mse=float(e@e/len(e)),rmse=float(np.sqrt(e@e/len(e))),
        mae=float(np.sum(np.abs(e))/len(e)),bias=float(np.sum(e)/len(e)))


def check_metrics(y,p,record):
    for name,value in measure(y,p).items():close(value,record[name])


def check_pair(y,candidate,control,days,record):
    a=np.square(candidate-y);b=np.square(control-y)
    table=pd.DataFrame(dict(day=days,a=a,b=b)).groupby('day',sort=True).agg(a=('a','sum'),b=('b','sum'),n=('a','size'))
    draws=np.random.default_rng(20260916).multinomial(len(table),np.full(len(table),1/len(table)),size=2000)
    samples=(draws@(table.b-table.a).to_numpy())/(draws@table.n.to_numpy())
    close(np.quantile(samples,[.025,.975]),record['mse_gain_ci95'])
    close(np.sqrt(b.mean())-np.sqrt(a.mean()),record['gain_sec']);close((b-a).mean(),record['mse_gain'])
    removal=[]
    for day in table.index:
        keep=np.asarray(days)!=day
        removal.append(float(np.sqrt(b[keep].mean())-np.sqrt(a[keep].mean())))
    close(min(removal),record['day_removal_gain_min'])
    assert (min(removal)>0)==record['all_day_removals_positive']
    order=pd.Series(b-a).sort_values(ascending=False,kind='stable').index
    for k in [1,2,5,10]:
        keep=~np.isin(np.arange(len(y)),np.array(order[:k]))
        close(np.sqrt(b[keep].mean())-np.sqrt(a[keep].mean()),record['top_beneficial_deletion_gain'][str(k)])


def score(fold):
    protocol,cohort=protocols();folder=BASE/'ensemble'/fold/'evaluation'
    # This gate is checked before reading any evaluation labels.
    manifest=read(folder/'manifest.json');assert manifest['status']=='complete'
    assert manifest['protocol_sha256']==sha(BASE/'ensemble/protocol.json')
    calibration=BASE/'ensemble'/fold/'calibration';calmark=read(calibration/'manifest.json')
    assert manifest['calibration_manifest_sha256']==sha(calibration/'manifest.json')
    verified=read(OUT/f'{fold}_calibration.json')
    assert verified['status']=='passed' and verified['calibration_manifest_sha256']==sha(calibration/'manifest.json')
    for name,digest in manifest['outputs'].items():assert sha(folder/name)==digest
    opened=read(folder/'evaluation_opened.json')
    assert opened['calibration_manifest_sha256']==sha(calibration/'manifest.json')
    assert pd.Timestamp(opened['utc'])>=pd.Timestamp(calmark['completed_utc'])
    reports=read(folder/'metrics.json');checks={}
    for month in protocol['evaluation']['panels'][fold]:
        stage='evaluation_'+month;frame=pd.read_parquet(folder/(stage+'.parquet'))
        original=pd.read_parquet(calibration/(stage+'.parquet'))
        assert sha(calibration/(stage+'.parquet'))==calmark['predictions'][stage]['sha256']
        pd.testing.assert_frame_equal(frame.drop(columns=TARGET),original,check_exact=True)
        y=labels(cohort,frame);np.testing.assert_array_equal(y,frame[TARGET])
        report=reports[month]
        assert report['label_hash']==object_hash(y.tolist()) and report['rows']==len(frame)
        days=frame[TIME].dt.floor('D').to_numpy();ordinary=frame.proxy_sec.between(0,7200).to_numpy()
        for name,keep in [('ordinary',ordinary),('all_finite',np.ones(len(frame),bool))]:
            for variant in FAMILIES+VARIANTS:check_metrics(y[keep],frame[variant].to_numpy()[keep],report[name][variant])
            for variant in ['shrunk','simplex']:
                check_pair(y[keep],frame[variant].to_numpy()[keep],frame.equal.to_numpy()[keep],days[keep],report[name][variant+'_vs_equal'])
        if 'historical_hybrid' in report:
            hybrid=pd.read_parquet(folder/'historical_hybrid.parquet')
            expected=metadata(cohort,fold,stage,finite=False)
            pd.testing.assert_frame_equal(hybrid[COLS],expected,check_exact=True)
            localroot=ROOT/'private_runs/tail240_20260916/score_gap/validation'
            localmark=read(localroot/'receipt.json');path=localroot/f'{fold}_replay.parquet'
            assert sha(path)==localmark['outputs'][path.name]
            local=pd.read_parquet(path)
            np.testing.assert_array_equal(hybrid[ID],local[ID]);np.testing.assert_array_equal(hybrid[TARGET],local[TARGET])
            np.testing.assert_array_equal(hybrid.v3,local.local_fold_weights_raw)
            positions=pd.Index(hybrid[ID]).get_indexer(frame.loc[ordinary,ID]);assert (positions>=0).all()
            candidate=hybrid.v3.to_numpy().copy();candidate[positions]=frame.loc[ordinary,'shrunk']
            np.testing.assert_array_equal(candidate,hybrid.candidate)
            hy=hybrid[TARGET].to_numpy()
            check_metrics(hy,candidate,report['historical_hybrid']['candidate']);check_metrics(hy,hybrid.v3.to_numpy(),report['historical_hybrid']['v3'])
            check_pair(hy,candidate,hybrid.v3.to_numpy(),hybrid[TIME].dt.floor('D').to_numpy(),report['historical_hybrid']['paired'])
        checks[month]=dict(rows=len(frame),all_metrics_pairs_bootstraps_deletions_verified=True,
            shrunk_vs_equal_ordinary_gain=report['ordinary']['shrunk_vs_equal']['gain_sec'])
        guard()
    write_result(fold,'evaluation',dict(evaluation_manifest_sha256=sha(folder/'manifest.json'),
        protocol_sha256=sha(BASE/'ensemble/protocol.json'),months=checks,
        labels_opened_only_after_producer_score_completed=True,
        limitation='Exposed development sensitivity, not selection-adjusted or fresh temporal generalization; hybrid preserves historical nonordinary routes.'))


def summary(_fold=None):
    protocol,_=protocols();reports={};bindings={}
    for fold in ('F1','F3'):
        folder=BASE/'ensemble'/fold/'evaluation';manifest=read(folder/'manifest.json')
        assert manifest['status']=='complete' and manifest['protocol_sha256']==sha(BASE/'ensemble/protocol.json')
        verified_path=OUT/f'{fold}_evaluation.json';verified=read(verified_path)
        assert verified['status']=='passed' and verified['evaluation_manifest_sha256']==sha(folder/'manifest.json')
        for name,digest in manifest['outputs'].items():assert sha(folder/name)==digest
        reports[fold]=read(folder/'metrics.json')
        bindings[fold]=dict(evaluation_manifest_sha256=sha(folder/'manifest.json'),metrics_sha256=sha(folder/'metrics.json'),
            independent_evaluation_sha256=sha(verified_path))
    checks={}
    for fold,month in [('F1','2025-07'),('F3','2025-11')]:
        pair=reports[fold][month]['ordinary']['shrunk_vs_equal']
        checks[fold+'_primary']=bool(pair['gain_sec']>0 and pair['mse_gain_ci95'][0]>0 and
            pair['all_day_removals_positive'] and pair['top_beneficial_deletion_gain']['10']>0)
        checks[fold+'_December']=reports[fold]['2025-12']['ordinary']['shrunk_vs_equal']['mse_gain_ci95'][0]>=0
    summary_path=BASE/'ensemble/summary.json';result=read(summary_path)
    assert result['status']=='complete' and result['checks']==checks
    assert result['shrunk_vs_equal_gate_passed']==all(checks.values())
    weights={'F1':192122/344841,'F3':152719/344841};hybrid={}
    for name in ['candidate','v3']:
        hybrid[name]=float(np.sqrt(sum(weights[f]*reports[f][m]['historical_hybrid'][name]['mse'] for f,m in [('F1','2025-07'),('F3','2025-11')])))
        close(hybrid[name],result['historical_hybrid_rmse'][name])
    close(hybrid['v3']-hybrid['candidate'],result['historical_hybrid_gain_sec'])
    write_result('both','summary',dict(summary_sha256=sha(summary_path),protocol_sha256=sha(BASE/'ensemble/protocol.json'),
        evaluation_bindings=bindings,checks=checks,shrunk_vs_equal_gate_passed=all(checks.values()),historical_hybrid_rmse=hybrid,
        label_reads=False,limitation='Summary inputs independently hash-bound; all endpoints remain exposed development, shared December is not independent replication.'))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['calibrate','score','summary'])
    parser.add_argument('--fold',choices=['F1','F3']);args=parser.parse_args()
    assert args.action=='summary' or args.fold
    globals()[args.action](args.fold)
