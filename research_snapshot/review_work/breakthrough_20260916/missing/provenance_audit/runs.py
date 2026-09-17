"""Clock proxy atom/run diagnostics on original fit/tune periods only."""
import audit as core
import numpy as np
import pandas as pd


def lengths(values):
    starts=np.r_[0,np.flatnonzero(values[1:]!=values[:-1])+1,len(values)]
    return np.diff(starts)


def report(values):
    run=lengths(values)
    return dict(rows=len(values),runs=len(run),mean_length=float(run.mean()),maximum=int(run.max()),
        rows_in_runs_ge3=int(run[run>=3].sum()),rows_in_runs_ge5=int(run[run>=5].sum()))


def main():
    out=core.OUT/'runs.json'
    if out.exists():
        raise ValueError('Completedruns retained')
    x,meta=core.common.load_data()
    del x
    proxy=meta.proxy_sec.to_numpy(float)
    finite=np.isfinite(proxy)&(proxy>=0)&(proxy<=7200)
    times=pd.to_datetime(meta[core.MOVEMENT],utc=True)
    records=[]
    for fold in ['F1','F3']:
        idx,_,_=core.common.fold_data(meta,fold,full=True)
        for stage in ['fit','tune']:
            selected=idx[stage][finite[idx[stage]]]
            data=meta.iloc[selected][['ADEP_mvt',core.MOVEMENT]].copy()
            data['proxy']=proxy[selected]
            data['day']=times.iloc[selected].dt.strftime('%Y-%m-%d')
            for airport,group in data.groupby('ADEP_mvt',observed=True):
                group=group.sort_values(core.MOVEMENT)
                values=group.proxy.to_numpy(float)
                rounded=np.floor((values+30)/60)
                counts=pd.Series(values).value_counts().head(8)
                rng=np.random.default_rng(20260916)
                shuffled=rounded.copy()
                for positions in group.groupby('day').indices.values():
                    shuffled[positions]=rng.permutation(shuffled[positions])
                exact=report(values)
                observed=report(rounded)
                null=report(shuffled)
                records.append(dict(fold=fold,stage=stage,airport=str(airport),exact_run=exact,rounded_run=observed,
                    within_day_permuted_run=null,exact_atoms=[dict(proxy_seconds=float(value),n=int(count),share=float(count/len(values))) for value,count in counts.items()],
                    exact_mod60_zero_share=float(np.mean(np.mod(values,60)==0))))
    core.write_json(out,dict(created_utc=core.utc_now(),source_sha256=core.sha256(__file__),records=records,
        scope='Diagnostic single withinairportday permutation; no inferentialpvalues, provenanceclassification,scorelabels or model.'))
    print('RUNS_DONE',len(records),flush=True)


if __name__=='__main__':
    main()
