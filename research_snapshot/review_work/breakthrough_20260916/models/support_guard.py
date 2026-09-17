"""Fit-period lower-support guard versus saved route-appropriate fallback."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT / "review_work/campaign_20260916"))
import common
from taxiout.artifacts import read_json,write_json,sha256,utc_now,object_hash
from taxiout.paths import external_path
from taxiout.schema import ID,TARGET
from taxiout.metrics import evaluate,paired_stability,season_score

OUT=external_path(ROOT / "private_runs/breakthrough_20260916/models/support_guard")


def main():
    declaration={"created_utc":utc_now(),"script_sha256":sha256(__file__),
        "hypothesis":"Rare grossly negative schedule-mixture extrapolations may be safer routed to an existing supervised expert.",
        "floor":"Minimum original departure target over that fold's purged FIT rows only; not zero, no upper bound.",
        "variants":["fit_min_floor","fit_min_route_fallback"],
        "fallback":"If prediction is below fit minimum: missing-NM uses saved missing.cbm; finite-NM uses saved direct.cbm. No additional clipping.",
        "label_policy":"No scoring label accessed for routing or threshold; all original labels/rows preserved.",
        "development_notice":"Triggered after five negative F1 predictions were observed; exposed-fold hypothesis diagnostic, not fresh validation.",
        "official_limit":"Final submitted ranking predictions were all positive; this guard cannot explain or improve that submission's official score."}
    protocol=OUT / "protocol.json"
    if protocol.exists():
        raise ValueError("Support-guard evaluation already declared; preserve prior artifact")
    write_json(protocol,declaration)
    x,meta=common.load_data()
    results={}
    for fold in ("F1","F3","F2","G1"):
        idx,split,_=common.fold_data(meta,fold,full=True)
        reference,manifest=common.reference(fold)
        assert object_hash(split)==object_hash(manifest["split"])
        assert np.array_equal(reference[ID],meta.iloc[idx["score"]][ID])
        floor=float(meta.iloc[idx["fit"]][TARGET].min())
        original=reference.prediction_sec.to_numpy()
        changed=original<floor
        missing=~np.isfinite(reference.proxy_sec.to_numpy())
        old=ROOT / "private_runs/screening_230/models" / manifest["reference_run"]
        bundle=read_json(old / "bundle.json")
        old_manifest=read_json(old / "manifest.json")
        assert sha256(old / "bundle.json")==old_manifest["outputs"]["bundle.json"]
        fallback=original.copy()
        model_sources={}
        for route,use in (("missing",changed&missing),("direct",changed&~missing)):
            file=old / f"{route}.cbm"
            assert sha256(file)==bundle["models"][route]
            model_sources[route]={"path":str(file),"sha256":sha256(file)}
            if use.any():
                model=CatBoostRegressor().load_model(str(file))
                fx=x.iloc[idx["score"]].loc[use,bundle["columns"]]
                fallback[use]=model.predict(fx,thread_count=2)
                replay=CatBoostRegressor().load_model(str(file)).predict(fx,thread_count=2)
                assert np.array_equal(fallback[use],replay)
        variants={"fit_min_floor":np.maximum(original,floor),"fit_min_route_fallback":fallback}
        folder=OUT / fold
        folder.mkdir(parents=True)
        reports={}
        for name,predictions in variants.items():
            assert np.array_equal(predictions[~changed],original[~changed])
            assert np.isfinite(predictions).all()
            frame=reference.drop(columns=[TARGET,"error_sec","squared_error","label_bin","month","day"],errors="ignore").copy()
            frame["prediction_sec"]=predictions
            metric,errors=evaluate(frame,meta.iloc[idx["score"]][[ID,TARGET]])
            errors.to_parquet(folder / f"{name}.parquet",index=False)
            reports[name]={"metrics":metric,"stability":paired_stability(reference,errors,repetitions=1000)}
        labels=reference[TARGET].to_numpy()
        row_report=pd.DataFrame({ID:reference.loc[changed,ID],TARGET:labels[changed],"reference_sec":original[changed],
            "fit_min_floor_sec":variants["fit_min_floor"][changed],"fallback_sec":fallback[changed],
            "route":reference.loc[changed,"route"].to_numpy(),"day":reference.loc[changed,"day"].to_numpy()})
        row_report.to_parquet(folder / "guarded_rows.parquet",index=False)
        record={"status":"complete","fold":fold,"fit_min_target_sec":floor,"fit_max_target_sec":float(meta.iloc[idx["fit"]][TARGET].max()),
            "fit_ids_hash":object_hash(meta.iloc[idx["fit"]][ID].tolist()),"split":split,"changed_rows":int(changed.sum()),
            "changed_missing_rows":int((changed&missing).sum()),"scoring_labels_below_fit_min":int((labels<floor).sum()),
            "scoring_negative_labels":int((labels<0).sum()),"minimum_reference_prediction":float(original.min()),
            "fallback_below_fit_min_rows":int((fallback<floor).sum()),"reports":reports,
            "model_sources":model_sources,"reference_manifest_hash":object_hash(manifest),
            "outputs":{p.name:sha256(p) for p in folder.iterdir() if p.is_file()},"script_sha256":sha256(__file__)}
        write_json(folder / "manifest.json",record)
        results[fold]=record
        print("GUARD",fold,"floor",floor,"changed",int(changed.sum()),{k:v["metrics"]["overall"]["rmse_sec"] for k,v in reports.items()},flush=True)
    submission=ROOT / "private_runs/submission_v2"
    ready=read_json(submission / "submission_ready.json")
    ranking_file=submission / "ranking_predictions.parquet"
    assert sha256(ranking_file)==ready["ranking_predictions_sha256"]
    ranking=pd.read_parquet(ranking_file,columns=["prediction_sec"])
    global_min=float(meta[TARGET].min())
    summary={"created_utc":utc_now(),"seasonal_rmse":{variant:season_score(results["F1"]["reports"][variant]["metrics"]["overall"],results["F3"]["reports"][variant]["metrics"]["overall"]) for variant in variants},
        "folds":{f:{"changed_rows":r["changed_rows"],"fit_min_target_sec":r["fit_min_target_sec"],"score_labels_below_fit_min":r["scoring_labels_below_fit_min"]} for f,r in results.items()},
        "ranking_check":{"prediction_sha256":sha256(ranking_file),"rows":len(ranking),"minimum_prediction_sec":float(ranking.prediction_sec.min()),
            "full2025_min_label_sec":global_min,"changed_rows_by_full_training_min":int((ranking.prediction_sec<global_min).sum()),
            "changed_rows_by_zero":int((ranking.prediction_sec<0).sum())},
        "risk":"Empirical training minimum is not a proven physical bound. Future valid labels may fall below it; fallback has its own estimation error. Do not present score-triggered safeguards as independent validation.",
        "script_sha256":sha256(__file__)}
    write_json(OUT / "summary.json",summary)
    print("GUARD SUMMARY",summary,flush=True)


if __name__=="__main__":
    main()
