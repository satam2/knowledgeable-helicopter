"""Independent fixed fit-only surface residual arithmetic and truth binding."""
import numpy as np
import pandas as pd
import verify_surface_join as prior

v,ID,TIME=prior.v,prior.ID,prior.TIME
TARGET=v.TARGET
BASE=prior.BASE
OUT=v.ROOT/'private_runs/tail240_20260916/validation/surface_dual_diagnostic_v2'


def distribution(values):
    a=np.asarray(values,float)
    a=a[np.isfinite(a)]
    if not len(a):return {'n':0}
    qs=[0,.01,.1,.5,.9,.99,1]
    return dict(n=len(a),mean=float(np.mean(a)),rmse=float(np.linalg.norm(a)/np.sqrt(len(a))),mae=float(np.mean(abs(a))),quantiles={str(q):float(np.quantile(a,q)) for q in qs})


def summarize(x):
    finite=np.isfinite(x.proxy_sec.to_numpy(float))
    unique=x.unique_exit_offset_sec.notna().to_numpy()
    paired=x.loc[finite&unique]
    nerror=paired[TARGET].to_numpy()-paired.proxy_sec.to_numpy()
    eerror=paired[TARGET].to_numpy()-paired.unique_exit_offset_sec.to_numpy()
    many=x.loc[x.eligible_parking_exit_count>1]
    return dict(queries=len(x),finite_nm=int(finite.sum()),unique_exit=int(unique.sum()),multiple_exits=len(many),raw_y=distribution(x[TARGET]),nm_residual_allfinite=distribution(x.loc[finite,TARGET]-x.loc[finite,'proxy_sec']),unique_event_offset=distribution(x.loc[unique,'unique_exit_offset_sec']),unique_event_residual=distribution(x.loc[unique,TARGET]-x.loc[unique,'unique_exit_offset_sec']),paired_count=len(paired),paired_nm_residual=distribution(nerror),paired_event_residual=distribution(eerror),paired_event_closer=int((abs(eerror)<abs(nerror)).sum()),paired_event_within60=int((abs(eerror)<=60).sum()),paired_event_within300=int((abs(eerror)<=300).sum()),paired_nm_within60=int((abs(nerror)<=60).sum()),paired_event_minus_nm=distribution(paired.unique_exit_offset_sec-paired.proxy_sec),paired_unique_firstseen_offset=distribution(paired.opdi_first_seen_offset_sec),raw_y_within_multiple_exit_range=int(((many[TARGET]>=many.exit_offset_min_sec)&(many[TARGET]<=many.exit_offset_max_sec)).sum()),multiple_exit_range_width=distribution(many.exit_offset_max_sec-many.exit_offset_min_sec),no_multiple_exit_choice=True)


def exact(left,right):
    if isinstance(left,dict):
        assert set(left)==set(right)
        for key in left:exact(left[key],right[key])
    else:np.testing.assert_allclose(left,right,atol=1e-10,rtol=1e-12)


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    folder=BASE/'dual_diagnostic_v2'
    protocol=v.read_json(folder/'protocol.json')
    produced=v.read_json(folder/'summary.json')
    assert produced['status']=='complete' and produced['protocol_sha256']==v.sha256(folder/'protocol.json')
    assert produced['source_sha256']==protocol['source_sha256']==v.sha256(v.ROOT/'review_work/tail240_20260916/forensics/surface_dual_diagnostic_v2.py')
    assert protocol['metric_source_sha256']==v.sha256(v.ROOT/'review_work/tail240_20260916/forensics/surface_diagnostic_v1.py')
    for name,digest in produced['outputs'].items():assert v.sha256(folder/name)==digest
    meta_path=v.ROOT/'private_runs/screening_230/data/interim/audit/departures.parquet'
    assert v.sha256(meta_path)==v.read_json(v.ROOT/'private_runs/screening_230/reports/data_audit.json')['artifacts'][meta_path.name]
    meta=pd.read_parquet(meta_path,columns=[ID,'FLIGHT_ID_mvt',TIME])
    meta[TARGET]=np.nan
    indices,split,_=v.common.fold_data(meta,'F1',full=True)
    assert v.object_hash(split)==v.object_hash(produced['original_f1_split'])
    query=pd.read_parquet(BASE/'join_v2/queries.parquet',columns=[ID,TIME])
    assert query[ID].isin(meta.iloc[indices['fit']][ID]).all()
    truth=pd.read_parquet(meta_path,columns=[ID,TARGET,'proxy_sec'],filters=[(ID,'in',query[ID].tolist())]).set_index(ID)
    results={}
    frames={}
    for arm,name in [('first_seen','join_v2'),('takeoff_event','takeoff_join_v2')]:
        assert v.sha256(BASE/name/'manifest.json')==protocol['arm_manifests'][arm]
        f=pd.read_parquet(BASE/name/'features.parquet')
        np.testing.assert_array_equal(f[ID],query[ID])
        for column in [TARGET,'proxy_sec']:f[column]=truth.loc[f[ID],column].to_numpy()
        saved=pd.read_parquet(folder/(arm+'_rows.parquet'))
        pd.testing.assert_frame_equal(f[saved.columns],saved)
        frames[arm]=f
        result=dict(all=summarize(f),airports={},match_modes={},nm_status={})
        for key,g in f.groupby('ADEP_mvt'):result['airports'][key]=summarize(g)
        for key,g in f.groupby('match_mode'):result['match_modes'][key]=summarize(g)
        for key,mask in [('finite',np.isfinite(f.proxy_sec)),('missing',~np.isfinite(f.proxy_sec))]:result['nm_status'][key]=summarize(f.loc[mask])
        exact(result,produced['results'][arm])
        results[arm]=result['all']
    unique={arm:set(f.loc[f.unique_exit_offset_sec.notna(),ID]) for arm,f in frames.items()}
    for arm,f in frames.items():
        other='takeoff_event' if arm=='first_seen' else 'first_seen'
        shared=unique[arm]&unique[other]
        expected=dict(shared_unique_exits=summarize(f.loc[f[ID].isin(shared)]),new_unique_exits=summarize(f.loc[f[ID].isin(unique[arm]-unique[other])]))
        exact(expected,produced['comparisons'][arm])
    receipt=dict(status='passed',source_sha256=v.sha256(__file__),protocol_sha256=v.sha256(folder/'protocol.json'),summary_sha256=v.sha256(folder/'summary.json'),all_original_f1_fit_ids=True,all_raw_labels_and_proxy_exact=True,all_group_metrics_and_intersections_exact=True,no_model=True,no_score=True,results=results,peak_bytes=prior.guard())
    v.write_json(OUT/'receipt.json',receipt)
    print({k:value for k,value in receipt.items() if k!='results'},flush=True)
    print({a:{k:d[k] for k in ['paired_count','paired_nm_residual','paired_event_residual']} for a,d in results.items()},flush=True)


if __name__=='__main__':main()
