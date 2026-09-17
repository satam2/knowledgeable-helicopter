"""Diagnostic-only bounds from supplied clocks; labels never define deployed routing."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'campaign_20260916'))
from common import WORKSPACE, RAW, ID, TARGET, reference, write_json, sha256, utc_now, external_path
from taxiout.schema import CLOCKS, MOVEMENT, PHASE, BLOCK
import numpy as np
import pandas as pd

OUT=external_path(WORKSPACE/'private_runs/mechanism_20260916')
W={'F1':192122/344841,'F3':152719/344841}


def main():
    refs={f:reference(f)[0] for f in W}
    ids=set(np.concatenate([v[ID].to_numpy() for v in refs.values()]))
    pieces=[]
    for p in sorted(RAW.glob('training_*.parquet')):
        raw=pd.read_parquet(p,columns=[ID,PHASE,MOVEMENT,*CLOCKS])
        raw=raw.loc[raw[ID].isin(ids)]
        if len(raw):
            frame=raw[[ID]].copy()
            for c in CLOCKS:
                frame[c]=(pd.to_datetime(raw[MOVEMENT],utc=True)-pd.to_datetime(raw[c],utc=True)).dt.total_seconds()
            pieces.append(frame)
    clocks=pd.concat(pieces).set_index(ID)
    report={'folds':{},'seasonal_mse':{},'interpretation':'Label-aware nearest-clock/convex-envelope bounds are diagnostic, not achievable scores, proposed rules, or noise floors.'}
    for fold,ref in refs.items():
        x=clocks.loc[ref[ID],CLOCKS].to_numpy(float)
        y=ref[TARGET].to_numpy(float)
        original=ref.prediction_sec.to_numpy()
        finite=np.isfinite(x)
        rawerrors=np.where(finite,(x-y[:,None])**2,np.inf)
        nearest=np.min(rawerrors,axis=1)
        candidates=np.column_stack([x,original])
        errors=np.where(np.isfinite(candidates),(candidates-y[:,None])**2,np.inf)
        selected=np.argmin(errors,axis=1)
        best=candidates[np.arange(len(y)),selected]
        names=[*CLOCKS,'reference']
        low=np.nanmin(candidates,axis=1)
        high=np.nanmax(candidates,axis=1)
        convex=np.maximum(low,np.minimum(y,high))
        masks={'all':np.ones(len(y),bool),'missing_clock':~np.isfinite(x[:,0]),
            'finite_gap_over_30min':np.isfinite(x[:,0])&(np.abs(y-x[:,0])>1800),
            'finite_gap_at_most_30min':np.isfinite(x[:,0])&(np.abs(y-x[:,0])<=1800)}
        metrics={}
        for group,m in masks.items():
            before=float(np.sum((original[m]-y[m])**2))
            after=float(np.sum((best[m]-y[m])**2))
            metrics[group]={'n':int(m.sum()),'reference_rmse':float(np.sqrt(before/m.sum())),
                'nearest_clock_or_reference_oracle_rmse':float(np.sqrt(after/m.sum())),
                'oracle_sse_reduction_pct':float(100*(before-after)/before),
                'any_clock_within_60sec_pct':float(100*np.mean(nearest[m]<=60**2)),
                'oracle_chosen_clock_counts':{name:int(np.sum(selected[m]==i)) for i,name in enumerate(names)}}
        for name,values in [('reference',original),('nearest_clock_or_reference_oracle',best),('clock_reference_convex_envelope_oracle',convex)]:
            report['seasonal_mse'][name]=report['seasonal_mse'].get(name,0.)+W[fold]*float(np.mean((values-y)**2))
        metrics['each_observed_clock']={c:{'available_pct':float(100*finite[:,i].mean()),
            'match_within_60_sec_among_available_pct':float(100*np.mean(rawerrors[finite[:,i],i]<=3600)),
            'rmse_on_available':float(np.sqrt(np.mean(rawerrors[finite[:,i],i])))} for i,c in enumerate(CLOCKS)}
        report['folds'][fold]=metrics
    report['seasonal_rmse']={k:float(np.sqrt(v)) for k,v in report['seasonal_mse'].items()}
    report['created_utc']=utc_now()
    report['script_sha256']=sha256(__file__)
    write_json(OUT/'clock_candidate_oracles.json',report)
    print(report['seasonal_rmse'],flush=True)
    for fold in W:
        print(fold,report['folds'][fold]['missing_clock'],report['folds'][fold]['finite_gap_over_30min'],flush=True)


if __name__=='__main__':
    main()
