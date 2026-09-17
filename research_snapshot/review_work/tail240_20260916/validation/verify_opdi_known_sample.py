"""Independent existing-known-sample eligibility and direct-offset diagnostic."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
import pandas as pd
import validate_candidate as v

ROOT=v.ROOT
BASE=ROOT/'private_runs/tail240_20260916/forensics/opdi_known_sample_v1'
OUT=ROOT/'private_runs/tail240_20260916/validation/opdi_known_sample_v1'


def distribution(values):
    a=np.asarray(values,float)
    if not len(a):return {'n':0}
    return {'n':len(a),'mean':float(a.mean()),'quantiles':{str(q):float(np.quantile(a,q)) for q in [0,.01,.1,.5,.9,.99,1]}}


def close(left,right):
    if isinstance(left,dict):
        assert set(left)==set(right)
        for key in left:close(left[key],right[key])
    elif left is None:assert right is None
    else:np.testing.assert_allclose(left,right,rtol=1e-12,atol=1e-10)


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    summary=v.read_json(BASE/'summary.json');protocol=v.read_json(BASE/'protocol.json')
    assert summary['status']=='complete' and summary['protocol_sha256']==v.sha256(BASE/'protocol.json')
    assert protocol['source_sha256']==summary['source_sha256']==v.sha256(ROOT/'review_work/tail240_20260916/forensics/opdi_known_sample_v1.py')
    for name,digest in summary['outputs'].items():assert v.sha256(BASE/name)==digest
    rows=pd.read_parquet(BASE/'eligible_known_rows.parquet');witness=pd.read_parquet(BASE/'eligible_exact_witnesses.parquet')
    assert len(rows)==5000 and rows[v.ID].is_unique
    metadata=ROOT/'private_runs/screening_230/data/interim/audit/departures.parquet'
    assert v.sha256(metadata)==v.read_json(ROOT/'private_runs/screening_230/reports/data_audit.json')['artifacts'][metadata.name]
    meta=pd.read_parquet(metadata,columns=[v.ID,'FLIGHT_ID_mvt',v.common.MOVEMENT,'ADEP_mvt','proxy_sec'])
    meta[v.TARGET]=np.nan
    allowed=set();checks={}
    for fold in ['F1','F3']:
        idx,split,_=v.common.fold_data(meta,fold,full=True)
        assert v.object_hash(split)==v.object_hash(summary['splits'][fold])
        for stage in ['fit','tune']:
            selected=rows.loc[rows[v.ID].isin(meta.iloc[idx[stage]][v.ID])]
            allowed.update(selected[v.ID]);entry=summary['results'][fold+'_'+stage]
            checks[fold+'_'+stage]={}
            for slice_name,query in [('all_finite',selected),('ordinary',selected.loc[selected.proxy_sec.between(0,7200)])]:
                checks[fold+'_'+stage][slice_name]={}
                for mode in ['raw','learned','lexical']:
                    prefix='opdi_'+mode+'_day1_';result=entry[slice_name][mode]
                    assert result['queries']==len(query)
                    assert result['coverage_count_gt0']==int(query[prefix+'count_30min'].gt(0).sum())
                    assert result['resolved_count']==int(query[prefix+'offset_sec'].notna().sum())
                    assert result['ambiguous_candidates']==int(query[prefix+'count_30min'].gt(1).sum())
                    for group in ['resolved','unique']:
                        q=query.loc[query[prefix+'offset_sec'].notna()]
                        if group=='unique':q=q.loc[q[prefix+'count_30min'].eq(1)]
                        offset=q[prefix+'offset_sec'].to_numpy(float);y=q[v.TARGET].to_numpy(float);proxy=q.proxy_sec.to_numpy(float)
                        nm=y-proxy;opdi=y-offset;gap=np.abs(nm)>1800
                        expected={'n':len(q),'fraction':len(q)/len(query),'offset_sec':distribution(offset),
                            'nm_proxy_sec':distribution(proxy),'raw_y_sec':distribution(y),'y_minus_nm':distribution(nm),
                            'y_minus_opdi_offset':distribution(opdi),'nm_rmse':float(np.sqrt(np.mean(nm**2))) if len(q) else None,
                            'opdi_offset_rmse':float(np.sqrt(np.mean(opdi**2))) if len(q) else None,
                            'nm_mae':float(np.mean(np.abs(nm))) if len(q) else None,'opdi_offset_mae':float(np.mean(np.abs(opdi))) if len(q) else None,
                            'first_seen_more_than600sec_before_takeoff':int((offset>600).sum()),
                            'first_seen_more_than600sec_after_takeoff':int((offset< -600).sum()),'offset_within600sec':int((np.abs(offset)<=600).sum()),
                            'offset_gt600_and_before_nm_aobt':int(((offset>600)&(offset>proxy)).sum()),
                            'opdi_offset_closer_to_y':int((np.abs(opdi)<np.abs(nm)).sum()),'raw_nm_gap_gt1800':int(gap.sum()),
                            'gap_and_opdi_closer':int((gap&(np.abs(opdi)<np.abs(nm))).sum()),'duration_sec':distribution(q[prefix+'duration_sec'])}
                        close(expected,result['groups'][group])
                        w=witness.loc[witness['mode'].eq(mode)&witness[v.ID].isin(q[v.ID])].set_index(v.ID).loc[q[v.ID]]
                        assert w.public_id_exact.notna().all()
                        np.testing.assert_array_equal(w.offset_sec,offset)
                        np.testing.assert_array_equal((q[v.common.MOVEMENT].to_numpy()-w.first_seen.to_numpy())/np.timedelta64(1,'s'),offset)
                    checks[fold+'_'+stage][slice_name][mode]={'queries':len(query),'resolved':result['resolved_count'],'all_reported_statistics_exact':True}
    assert allowed==set(rows[v.ID])
    labels=pd.read_parquet(metadata,columns=[v.ID,v.TARGET,'proxy_sec'],filters=[(v.ID,'in',sorted(allowed))]).set_index(v.ID).loc[rows[v.ID]]
    np.testing.assert_array_equal(labels[v.TARGET],rows[v.TARGET]);np.testing.assert_array_equal(labels.proxy_sec,rows.proxy_sec)
    v.write_json(OUT/'receipt.json',{'status':'passed','source_sha256':v.sha256(__file__),'producer_summary_sha256':v.sha256(BASE/'summary.json'),
        'checks':checks,'unique_eligible_ids':len(allowed),'raw_targets_verified_only_for_eligible_ids':True,'no_score_stage_evaluation':True,
        'scope':'All reported matched-group points/distributions and exact-witness offsets verified. Existing exact public IDs rely on prior v3 raw oracle. Small deterministic known sample, not a full-population coverage estimate or ground/offblock timestamp validation.',
        'peak_rss_bytes':v.guard()})
    print('PASSED',checks,flush=True)


if __name__=='__main__':main()
