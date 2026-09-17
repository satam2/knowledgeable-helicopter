"""Control every own planned-clock difference before approving an NMID cache."""
import audit_ids as base
from deeper_audit import EDGES
import numpy as np
import pandas as pd


def adjust(fit,query,keys,fitprediction):
    leftover=fit.copy()
    leftover['residual']=fit.residual.to_numpy()-fitprediction
    grouped=leftover.groupby(keys,observed=True,dropna=False).residual.agg(['mean','size']).reset_index()
    matched=query[keys].merge(grouped,on=keys,how='left',validate='many_to_one')
    count=matched['size'].fillna(0).to_numpy()
    return matched['mean'].fillna(0).to_numpy()*count/(count+50.)


def main():
    out=base.OUT/'initial_control.json'
    if out.exists():
        raise ValueError('Completed owninitial-clock control retained')
    protocol=dict(created_utc=base.utc_now(),script_sha256=base.sha256(__file__),
        dependencies={name:base.sha256(base.HERE/name) for name in ['audit_ids.py','deeper_audit.py']},
        design='Fixedfit-only additive13bucket rawresidualmeans. Baseline ownorder+NM/EOBTgap+ownproxy5min. Thenown IOBT-N,EOBT-N,schedule-N; thenpeerN-N/peerEOBT-N.',
        labels='Original fit/tune only; no scorelabels, no clipping.',
        fixed_order=['own_IOBT','own_EOBT','own_schedule','peerNM','peerEOBT'],shrinkage=50)
    base.write_json(base.OUT/'initial_control_protocol.json',protocol)
    frozen=base.read_json(base.ROOT/'private_runs/submission_v2/protocol.json')
    frames=[]
    for path in sorted(base.common.RAW.glob('training*.parquet'))[:10]:
        assert base.sha256(path)==frozen['raw_hashes'][path.name]
        raw=base.pq.read_table(path,columns=base.COLS,use_threads=False).to_pandas()
        dep=raw.loc[raw[base.PHASE].eq('DEP')].copy()
        peers=base.neighbors(dep,raw)
        nm=base.seconds(dep.AOBT_3_flt)
        values={'peerNM':peers.nmid_peer_nm_seconds.to_numpy()-nm,
            'peerEOBT':peers.nmid_peer_initial_seconds.to_numpy()-nm,
            'own_IOBT':base.seconds(dep.IOBT_flt)-nm,
            'own_EOBT':base.seconds(dep.EOBT_1_flt)-nm,
            'own_schedule':base.seconds(dep.SCHED_TIME_UTC_mvt)-nm}
        frame=pd.DataFrame({base.ID:dep[base.ID].to_numpy()})
        for name,value in values.items():
            frame[name]=pd.cut(value,EDGES,labels=False)
        frames.append(frame)
    audited=base.read_json(base.ROOT/'private_runs/screening_230/reports/data_audit.json')
    metadata=base.ROOT/'private_runs/screening_230/data/interim/audit/departures.parquet'
    assert base.sha256(metadata)==audited['artifacts'][metadata.name]
    meta=pd.read_parquet(metadata)
    directory=base.ROOT/'private_runs/breakthrough_20260916/missing/source_conventions'
    marker=base.read_json(directory/'manifest.json')
    assert base.sha256(directory/'features.parquet')==marker['feature_sha256']
    names=['conv_clock_rank_signature','conv_nm_est_gap_bucket']
    conv=pd.read_parquet(directory/'features.parquet',columns=[base.ID,*names]).set_index(base.ID)
    context=pd.concat(frames,ignore_index=True).set_index(base.ID).reindex(meta[base.ID]).fillna(-1)
    for col in names:
        context[col]=conv.loc[meta[base.ID],col].to_numpy()
    proxy=meta.proxy_sec.to_numpy(float)
    context['airport']=meta.ADEP_mvt.astype(str).to_numpy()
    context['proxy_bin']=np.floor(np.where(np.isfinite(proxy),proxy,0)/300)
    context['residual']=meta[base.TARGET].to_numpy(float)-proxy
    parent=['airport',*names,'proxy_bin']
    reports={}
    for fold in ['F1','F3']:
        idx,_,_=base.common.fold_data(meta,fold,full=True)
        eligible=np.isfinite(proxy)&(proxy>=0)&(proxy<=7200)
        a=idx['fit'][eligible[idx['fit']]]
        b=idx['tune'][eligible[idx['tune']]]
        fit,tune=context.iloc[a],context.iloc[b]
        f=base.grouped_predict(fit,fit,parent)
        t=base.grouped_predict(fit,tune,parent)
        reports[fold]={'base':base.stats(t-tune.residual.to_numpy())}
        for column in protocol['fixed_order']:
            correction_f=adjust(fit,fit,['airport',column],f)
            correction_t=adjust(fit,tune,['airport',column],f)
            f+=correction_f
            t+=correction_t
            reports[fold]['after_'+column]=base.stats(t-tune.residual.to_numpy())
    base.write_json(out,dict(created_utc=base.utc_now(),protocol_sha256=base.sha256(base.OUT/'initial_control_protocol.json'),
        tune_predictions=reports,score_labels_used=False,feature_cache_written=False))
    print({fold:{name:value['rmse'] for name,value in models.items()} for fold,models in reports.items()},flush=True)


if __name__=='__main__':
    main()
