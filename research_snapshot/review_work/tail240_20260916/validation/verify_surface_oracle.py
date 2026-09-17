"""Independent aggregate oracle budget on the frozen surface pilot."""
import numpy as np
import pandas as pd
import verify_surface_diagnostic as diagnostic

v,BASE,ID,TIME,TARGET=diagnostic.v,diagnostic.BASE,diagnostic.ID,diagnostic.TIME,diagnostic.TARGET
OUT=v.ROOT/'private_runs/tail240_20260916/validation/surface_oracle_v1'


def exact(a,b):
    if isinstance(a,dict):
        assert set(a)==set(b)
        for key in a:exact(a[key],b[key])
    elif a is None:assert b is None
    else:np.testing.assert_allclose(a,b,rtol=1e-12,atol=1e-10)


def summary(frame):
    finite=np.isfinite(frame.proxy_sec.to_numpy())
    covered=frame.unique_exit_offset_sec.notna().to_numpy()
    subset=frame.loc[finite]
    error=subset[TARGET].to_numpy()-subset.proxy_sec.to_numpy()
    perfect=np.where(covered[finite],0.,error)
    sse=float(error@error);after=float(perfect@perfect);n=len(subset)
    before=float(np.sqrt(sse/n)) if n else None
    result=float(np.sqrt(after/n)) if n else None
    return dict(queries=len(frame),finite_nm=n,missing_nm=int((~finite).sum()),covered_finite=int((covered&finite).sum()),covered_missing=int((covered&~finite).sum()),nm_sse=sse,removable_nm_sse=sse-after,removable_fraction=(sse-after)/sse if sse else None,nm_rmse=before,oracle_rmse=result,oracle_gain=before-result if n else None)


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    source=BASE/'oracle_budget_v1/receipt.json'
    r=v.read_json(source)
    assert r['status']=='complete' and r['source_sha256']==v.sha256(v.ROOT/'review_work/tail240_20260916/forensics/surface_oracle_budget_v1.py')
    assert r['input_summary_sha256']==v.sha256(BASE/'dual_diagnostic_v2/summary.json')
    assert v.read_json(diagnostic.OUT/'receipt.json')['status']=='passed'
    querypath=BASE/'join_v2/queries.parquet'
    assert v.sha256(querypath)==r['query_sha256']
    query=pd.read_parquet(querypath,columns=[ID,TIME])
    results={}
    for arm in ['first_seen','takeoff_event']:
        name=arm+'_rows.parquet';path=BASE/'dual_diagnostic_v2'/name
        assert v.sha256(path)==r['inputs'][name]
        frame=pd.read_parquet(path).merge(query,on=ID,validate='one_to_one')
        frame['utc_day']=frame[TIME].dt.strftime('%Y-%m-%d')
        expected=dict(all=summary(frame),airports={a:summary(g) for a,g in frame.groupby('ADEP_mvt')},days={d:summary(g) for d,g in frame.groupby('utc_day')},airport_days={a+'|'+d:summary(g) for (a,d),g in frame.groupby(['ADEP_mvt','utc_day'])})
        for key,value in expected.items():exact(value,r['results'][arm][key])
        missing=frame.loc[frame.proxy_sec.isna()&frame.unique_exit_offset_sec.notna(),[ID,'ADEP_mvt','utc_day',TARGET,'unique_exit_offset_sec']].to_dict('records')
        assert missing==r['results'][arm]['missing_covered']
        results[arm]=expected['all']
    receipt=dict(status='passed',source_sha256=v.sha256(__file__),producer_receipt_sha256=v.sha256(source),all_day_airport_day_arithmetic_exact=True,covered_missing_rows_exact=True,results=results,limitation='Fixed pilot and raw-NM comparator only; not a current-model bound or future-sample performance limit.')
    v.write_json(OUT/'receipt.json',receipt)
    print(results,flush=True)


if __name__=='__main__':main()
