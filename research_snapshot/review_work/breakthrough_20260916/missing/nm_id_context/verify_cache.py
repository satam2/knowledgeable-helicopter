"""Independent explicit NMID neighbor selection on sampled raw observations."""
import audit_ids as base
import numpy as np
import pandas as pd


def check(raw,cache,seed):
    raw=raw.copy()
    raw['month']=pd.to_datetime(raw[base.MOVEMENT],utc=True).dt.strftime('%Y-%m')
    queries=raw.loc[raw[base.PHASE].eq('DEP')].sample(n=60,random_state=seed)
    missing=raw.loc[raw[base.PHASE].eq('DEP') & raw[base.FLIGHT_ID].isna()].head(5)
    queries=pd.concat([queries,missing]).drop_duplicates(base.ID)
    cache=cache.set_index(base.ID)
    maximum=0.
    for _,query in queries.iterrows():
        context=raw.loc[raw.month.eq(query.month) & raw[base.FLIGHT_ID].notna() & raw[base.FLIGHT_ID].ne(query[base.FLIGHT_ID])]
        known=np.sort(context[base.FLIGHT_ID].unique())
        if pd.notna(query[base.FLIGHT_ID]):
            lower=known[known<query[base.FLIGHT_ID]][-8:]
            upper=known[known>query[base.FLIGHT_ID]][:8]
            selected=np.r_[lower,upper]
        else:
            selected=[]
        values={}
        for clock,name in [('AOBT_3_flt','nm'),('EOBT_1_flt','eobt')]:
            perflight=[]
            for flight in selected:
                timestamps=pd.to_datetime(context.loc[context[base.FLIGHT_ID].eq(flight),clock],utc=True).dropna()
                if len(timestamps):
                    perflight.append(float(np.median([stamp.timestamp() for stamp in timestamps])))
            peer=float(np.median(perflight)) if perflight else np.nan
            own=pd.Timestamp(query.AOBT_3_flt).timestamp() if pd.notna(query.AOBT_3_flt) else np.nan
            takeoff=pd.Timestamp(query[base.MOVEMENT]).timestamp()
            values['nmid_peer_'+name+'_minus_own_nm']=peer-own
            values['nmid_peer_'+name+'_minus_takeoff']=peer-takeoff
            values['nmid_peer_'+name+'_count']=len(perflight)
        values['nmid_source_id_present']=float(pd.notna(query[base.FLIGHT_ID]))
        actual=cache.loc[query[base.ID]]
        for name,value in values.items():
            expected=np.float32(value)
            if np.isnan(expected):
                assert np.isnan(actual[name]),name
            else:
                maximum=max(maximum,abs(float(actual[name])-float(expected)))
                assert actual[name]==expected,(query[base.ID],name,actual[name],expected)
    return dict(queries=len(queries),columns=7,max_abs_float32_delta=maximum)


def main():
    directory=base.OUT/'cache_v1'
    output=directory/'verification.json'
    if output.exists():
        raise ValueError('Completed independent verification preserved')
    marker=base.read_json(directory/'manifest.json')
    for name,digest in marker['outputs'].items():
        assert base.sha256(directory/name)==digest
    checked=[]
    for filename,cache,seed in [('training_2025-06-01_2025-07-01.parquet','training_features.parquet',20260916),
        ('ranking.parquet','ranking_features.parquet',20260917)]:
        raw=base.pq.read_table(base.common.RAW/filename,columns=base.COLS,use_threads=False).to_pandas()
        frame=pd.read_parquet(directory/cache)
        checked.append(dict(file=filename,**check(raw,frame,seed)))
    base.write_json(output,dict(status='passed',created_utc=base.utc_now(),manifest_sha256=base.sha256(directory/'manifest.json'),
        verifier_sha256=base.sha256(__file__),checks=checked,no_labels_read=True,method='Explicit lower/higher ID selection, perflight datetime medians; no producer feature function called.'))
    print('VERIFIED',checked,flush=True)


if __name__=='__main__':
    main()
