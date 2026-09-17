"""Diagnose arbitrary label-regime boundaries using saved private endpoint tables."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "knowledgeable-helicopter-screening/src"))
from taxiout.artifacts import write_json, sha256, utc_now
from taxiout.schema import TARGET

OUT = ROOT / "private_runs/mechanism_20260916/validation"


def main():
    results = {}
    for fold in ("F1", "F3"):
        results[fold] = {}
        for stage in ("tune", "score"):
            path = OUT / f"missing_endpoints_{fold}_{stage}.parquet"
            frame = pd.read_parquet(path)
            y, schedule, g, b, p = [frame[c].to_numpy(float) for c in (TARGET, "schedule_sec", "good_expert_sec", "bad_expert_sec", "probability")]
            gap = np.abs(y-schedule)
            hard = np.where(gap <= 60, g, b)
            actual = b + p*(g-b)
            d = g-b
            w = np.clip(np.divide(y-b,d,out=np.zeros(len(y)),where=np.abs(d)>1e-12),0,1)
            bound = b+w*d
            hard_sse = (hard-y)**2
            oracle_sse = (bound-y)**2
            wrong_z = (gap>60) & ((g-y)**2 < (b-y)**2)
            group = pd.cut(gap, [-np.inf,60,300,1800,np.inf],labels=["0_to_60","60_to_300","300_to_1800","over_1800"],right=True)
            bands = {}
            for name in group.categories:
                keep = np.asarray(group == name)
                bands[str(name)] = {"n":int(keep.sum()), "hard_regime_sse_share":float(hard_sse[keep].sum()/hard_sse.sum()),
                    "actual_mixture_sse_share":float(((actual[keep]-y[keep])**2).sum()/((actual-y)**2).sum()),
                    "hard_regime_rmse":float(np.sqrt(np.mean(hard_sse[keep]))) if keep.any() else None,
                    "schedule_rmse":float(np.sqrt(np.mean((schedule[keep]-y[keep])**2))) if keep.any() else None,
                    "convex_bound_rmse":float(np.sqrt(np.mean(oracle_sse[keep]))) if keep.any() else None,
                    "target_over7200_n":int((y[keep]>7200).sum())}
            results[fold][stage] = {"source_sha256":sha256(path),"bands":bands,
                "inconsistent_label_but_good_endpoint_closer_n":int(wrong_z.sum()),
                "inconsistent_label_but_good_endpoint_closer_hard_sse_share":float(hard_sse[wrong_z].sum()/hard_sse.sum()),
                "inconsistent_label_but_good_endpoint_closer_reducible_hard_sse_share":float((hard_sse[wrong_z]-oracle_sse[wrong_z]).sum()/(hard_sse-oracle_sse).sum())}
    write_json(OUT / "gate_threshold_audit.json", {"created_utc":utc_now(),"results":results,"script_sha256":sha256(__file__),
        "scope":"Label-based diagnosis only; no inference route, training or threshold optimization."})
    print(results,flush=True)


if __name__ == "__main__":
    main()
