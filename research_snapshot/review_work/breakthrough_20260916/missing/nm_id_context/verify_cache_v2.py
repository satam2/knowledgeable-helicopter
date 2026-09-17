"""Independent all-seven-field checks, plus IDs and original tune re-evaluation."""
import verify_cache as independent
import initial_control as controls
import build_cache_v2 as builder
import audit_ids as base
from deeper_audit import EDGES
import numpy as np
import pandas as pd


def main():
    directory=builder.CACHE
    output=directory/'verification.json'
    if output.exists():
        raise ValueError('Completed verification retained')
    marker=base.read_json(directory/'manifest.json')
    for name,digest in marker['outputs'].items():
        assert base.sha256(directory/name)==digest
    train=pd.read_parquet(directory/'training_features.parquet')
    rank=pd.read_parquet(directory/'ranking_features.parquet')
    frames=[]
    for filename in ['training_2025-06-01_2025-07-01.parquet','training_2025-07-01_2025-08-01.parquet']:
        frames.append(base.pq.read_table(base.common.RAW/filename,columns=base.COLS,use_threads=False).to_pandas())
    raw=pd.concat(frames,ignore_index=True)
    raw=raw.loc[pd.to_datetime(raw[base.MOVEMENT],utc=True).dt.month.eq(6)]
    checks=[dict(dataset='June2025_with_crosspack_context',**independent.check(raw,train,20260916))]
    raw=base.pq.read_table(base.common.RAW/'ranking.parquet',columns=base.COLS,use_threads=False).to_pandas()
    checks.append(dict(dataset='ranking',**independent.check(raw,rank,20260917)))
    np.testing.assert_array_equal(rank[base.ID],raw.loc[raw[base.PHASE].eq('DEP'),base.ID])
    x,meta=base.common.load_data()
    np.testing.assert_array_equal(train[base.ID],meta[base.ID])
    convdir=base.ROOT/'private_runs/breakthrough_20260916/missing/source_conventions'
    convmarker=base.read_json(convdir/'manifest.json')
    assert base.sha256(convdir/'features.parquet')==convmarker['feature_sha256']
    names=['conv_clock_rank_signature','conv_nm_est_gap_bucket']
    conv=pd.read_parquet(convdir/'features.parquet',columns=[base.ID,*names]).set_index(base.ID)
    frame=train.copy()
    proxy=meta.proxy_sec.to_numpy(float)
    for name in names:
        frame[name]=conv.loc[meta[base.ID],name].to_numpy()
    frame['airport']=meta.ADEP_mvt.astype(str).to_numpy()
    frame['proxy_bin']=np.floor(np.where(np.isfinite(proxy),proxy,0)/300)
    frame['residual']=meta[base.TARGET].to_numpy(float)-proxy
    own={'own_IOBT':'IOBT_flt','own_EOBT':'EOBT_1_flt','own_schedule':'SCHED_TIME_UTC_mvt'}
    for name,clock in own.items():
        other=x['takeoff_minus_'+clock].to_numpy(float)
        delta=proxy-other
        delta[~np.isfinite(other)|(other==-999999)]=np.nan
        frame[name]=pd.cut(delta,EDGES,labels=False)
    frame['peerNM']=pd.cut(frame.nmid_peer_nm_minus_own_nm,EDGES,labels=False)
    frame['peerEOBT']=pd.cut(frame.nmid_peer_eobt_minus_own_nm,EDGES,labels=False)
    frame=frame.fillna(-1)
    parent=['airport',*names,'proxy_bin']
    reports={}
    for fold in ['F1','F3']:
        idx,_,_=base.common.fold_data(meta,fold,full=True)
        eligible=np.isfinite(proxy)&(proxy>=0)&(proxy<=7200)
        a=idx['fit'][eligible[idx['fit']]]
        b=idx['tune'][eligible[idx['tune']]]
        fit,tune=frame.iloc[a],frame.iloc[b]
        f=base.grouped_predict(fit,fit,parent)
        t=base.grouped_predict(fit,tune,parent)
        reports[fold]={'base':base.stats(t-tune.residual.to_numpy())}
        for column in ['own_IOBT','own_EOBT','own_schedule','peerNM','peerEOBT']:
            a=controls.adjust(fit,fit,['airport',column],f)
            b=controls.adjust(fit,tune,['airport',column],f)
            f+=a
            t+=b
            reports[fold]['after_'+column]=base.stats(t-tune.residual.to_numpy())
    base.write_json(output,dict(status='passed',created_utc=base.utc_now(),manifest_sha256=base.sha256(directory/'manifest.json'),
        verifier_sha256=base.sha256(__file__),independent_verifier_sha256=base.sha256(independent.__file__),checks=checks,
        training_id_order_verified=True,ranking_id_order_verified=True,tune_predictions=reports,
        score_labels_used=False,method='Explicitlower/higher NMIDselection anddatetime medians; wholecache hashes/IDorders; finalcache tune reevaluation.'))
    print('VERIFIED_V2',checks,{fold:{k:v['rmse'] for k,v in methods.items()} for fold,methods in reports.items()},flush=True)


if __name__=='__main__':
    main()
