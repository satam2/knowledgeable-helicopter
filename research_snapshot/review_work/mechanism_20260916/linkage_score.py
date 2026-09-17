"""Score fixed retrospective exact record links; no fitting or score-based matching."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'campaign_20260916'))
from common import WORKSPACE, ID, TARGET, reference, write_json, read_json, sha256, external_path, utc_now
from taxiout.metrics import evaluate, paired_stability
import numpy as np
import pandas as pd

OUT=external_path(WORKSPACE/'private_runs/mechanism_20260916/linkage_score')
W={'F1':192122/344841,'F3':152719/344841}


if __name__=='__main__':
    results={}
    for fold,w in W.items():
        ref,_=reference(fold)
        path=WORKSPACE/'private_runs/mechanism_20260916/information'/f'{fold}_exact_missing_links.parquet'
        linked=pd.read_parquet(path)
        assert linked.MVT_ID_mvt_dep.is_unique
        rows=pd.Index(ref[ID]).get_indexer(linked.MVT_ID_mvt_dep)
        assert (rows>=0).all() and ref.iloc[rows].proxy_status.eq('missing').all()
        for variant,weight in [('recovered_proxy',1.),('recovered_blend25',.25)]:
            pred=ref.prediction_sec.to_numpy().copy()
            pred[rows]+=weight*(linked.candidate_proxy_sec.to_numpy()-pred[rows])
            f=ref.drop(columns=[TARGET,'error_sec','squared_error','label_bin','month','day'],errors='ignore').copy()
            f['prediction_sec']=pred
            metrics,frame=evaluate(f,ref[[ID,TARGET]])
            folder=OUT/fold
            folder.mkdir(parents=True,exist_ok=True)
            frame.to_parquet(folder/f'{variant}.parquet',index=False)
            r=results.setdefault(variant,{'weighted_mse':0.,'folds':{}})
            r['weighted_mse']+=w*metrics['overall']['rmse_sec']**2
            r['folds'][fold]={'metrics':metrics,'stability':paired_stability(ref,frame,repetitions=500),
                'recovered_rows':len(rows),'match_receipt_sha256':sha256(path),
                'on_recovered':{'proxy_rmse':float(np.sqrt(np.mean((linked.candidate_proxy_sec.to_numpy()-ref.iloc[rows][TARGET].to_numpy())**2))),
                    'reference_rmse':float(np.sqrt(ref.iloc[rows].squared_error.mean()))}}
    for r in results.values():
        r['seasonal_rmse']=float(np.sqrt(r['weighted_mse']))
    write_json(OUT/'summary.json',dict(created_utc=utc_now(),results=results,
        policy='Declared retrospective same-flight destination-arrival record linkage; later observations are used. Existing causal-policy comparator unchanged.',
        matching='Exact flight number+route, adjacent UTCday, unique plausible0..7200 AOBT-derived proxy; no hidden labels used.',
        script_sha256=sha256(__file__)))
    print({k:r['seasonal_rmse'] for k,r in results.items()},flush=True)
