"""Verify tune-only anchor result and localize its fixed comparison."""
from pathlib import Path
import numpy as np
import pandas as pd
from run_anchor_mixture_v3 import ROOT,OUT,LEXICAL,ID,TARGET
from taxiout.artifacts import read_json,write_json,sha256,utc_now


def main():
    dest=ROOT / "private_runs/tail240_20260916/forensics/models/anchor_audit_v2"
    dest.mkdir(parents=True,exist_ok=False)
    summary=read_json(OUT / "summary.json")
    assert summary["status"]=="complete"
    metadata=ROOT / "private_runs/screening_230/data/interim/audit/departures.parquet"
    assert sha256(metadata)==read_json(ROOT / "private_runs/screening_230/reports/data_audit.json")["artifacts"][metadata.name]
    meta=pd.read_parquet(metadata,columns=[ID,TARGET,"ADEP_mvt"]).set_index(ID)
    metrics=[]
    gates=[]
    for fold in ["F1","F3"]:
        folder=OUT / fold
        record=read_json(folder / "manifest.json")
        for filename,digest in record["outputs"].items():
            assert sha256(folder / filename)==digest
        preds={}
        for arm in ["three_anchor","dst_four_anchor"]:
            best=record["reports"][arm]["best"]
            p=pd.read_parquet(folder / f"{arm}_{best['trees']}_tune.parquet").set_index(ID)
            np.testing.assert_array_equal(p[TARGET],meta.loc[p.index,TARGET])
            assert np.isfinite(p.prediction_sec).all()
            preds[arm]=p.prediction_sec
            np.testing.assert_allclose(float(np.mean((p.prediction_sec-p[TARGET])**2)),best["mse"],rtol=1e-12)
        previous=LEXICAL / "lexical/models" / f"historical_template_{fold}_s20260916/tune_predictions.parquet"
        assert sha256(previous)==read_json(previous.parent / "manifest.json")["outputs"][previous.name]
        preds["previous_lexical"]=pd.read_parquet(previous).set_index(ID).prediction_sec
        joined=pd.DataFrame(preds)
        assert joined.notna().all().all()
        joined=joined.join(meta,validate="one_to_one")
        joined["fold"]=fold
        joined.reset_index().to_parquet(dest / f"{fold}_tune_comparison.parquet",index=False)
        for airport,rows in [("ALL",joined),*list(joined.groupby("ADEP_mvt",sort=True,observed=True))]:
            for arm in preds:
                metrics.append({"fold":fold,"airport":airport,"arm":arm,"n":len(rows),"rmse":float(np.sqrt(np.mean((rows[arm]-rows[TARGET])**2)))})
        gate=record["reports"]["dst_four_anchor"]["best"]["mse"] < min(record["reports"]["three_anchor"]["best"]["mse"],record["previous_lexical_mse"])
        assert gate==record["gate_passed"]
        gates.append(gate)
    assert all(gates)==summary["gate_passed"]
    results=pd.DataFrame(metrics)
    results.to_csv(dest / "airport_metrics.csv",index=False)
    write_json(dest / "summary.json",{"status":"passed","created_utc":utc_now(),"source_sha256":sha256(__file__),
        "source_summary_sha256":sha256(OUT / "summary.json"),"gate_passed":all(gates),"metrics":metrics,
        "score_predictions_computed":False,"conclusion":"Gate failure prohibits advancing to score predictions under this declaration." if not all(gates) else "Gate passed; later score work requires separate declaration."})
    print(results.to_string(index=False))


if __name__=="__main__":
    main()
