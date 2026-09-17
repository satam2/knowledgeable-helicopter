"""No-training loss alignment and label-aware endpoint diagnostics for missing clocks."""

import sys
from pathlib import Path

import lightgbm
import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HERE = Path(__file__).resolve()
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "review_work/campaign_20260916"))
import common
from aviation import run_missing_mixture_original_executed as mixture
from taxiout.artifacts import read_json, write_json, sha256, utc_now
from taxiout.metrics import scores
from taxiout.paths import external_path
from taxiout.schema import ID, TARGET

OUT = external_path(ROOT / "private_runs/mechanism_20260916/validation")
W = {"F1": 192122 / 344841, "F3": 152719 / 344841}


def oracle(good, bad, y):
    gap = good - bad
    p = np.clip(np.divide(y - bad, gap, out=np.zeros(len(y)), where=np.abs(gap) > 1e-12), 0, 1)
    return p, bad + p * gap


def endpoint_values(model, x, schedule):
    classifier = model["classifier"]
    if "constant" in classifier:
        probability = np.full(len(x), classifier["constant"])
    else:
        classifier["model"].set_params(n_jobs=2)
        probability = classifier["model"].predict_proba(classifier["encoder"].transform(x), num_iteration=classifier["steps"])[:, 1]
    good = schedule + mixture._predict_head(model["good"], x)
    bad = mixture._predict_head(model["bad"], x)
    return probability, good, bad


def diagnose(y, schedule, probability, good, bad, airports):
    gap = good - bad
    weight = gap**2
    z = (np.abs(y - schedule) <= 60).astype(float)
    clipped_probability = np.clip(probability, 1e-12, 1 - 1e-12)
    pstar, bound = oracle(good, bad, y)
    variants = {"actual_mixture": bad + probability * gap, "label_regime_hard_oracle": bad + z * gap,
                "best_convex_endpoint_oracle": bound, "good_only": good, "bad_only": bad, "schedule_only": schedule}
    metrics = {name: scores(y, value) for name, value in variants.items()}
    signed_excess = (variants["actual_mixture"] - y)**2 - (bound - y)**2
    order = np.argsort(signed_excess)[::-1]
    logloss = -(z * np.log(clipped_probability) + (1-z) * np.log(1-clipped_probability))
    wrong_threshold = (probability >= .5) != (z >= .5)
    return {"metrics": metrics, "n": len(y), "consistent_rows": int(z.sum()),
        "unweighted_brier": float(np.mean((probability-z)**2)),
        "unweighted_logloss": float(logloss.mean()),
        "unweighted_classifier_error_fraction": float(wrong_threshold.mean()),
        "gap_squared_weighted_brier": float(np.average((probability-z)**2, weights=weight)),
        "gap_squared_weighted_classifier_error_fraction": float(np.average(wrong_threshold, weights=weight)),
        "optimal_convex_weight_differs_from_label_by_gt0_25_fraction": float((np.abs(pstar-z) > .25).mean()),
        "optimal_convex_interior_fraction": float(((pstar > .01) & (pstar < .99)).mean()),
        "target_outside_endpoint_interval_fraction": float(((y < np.minimum(good,bad)) | (y > np.maximum(good,bad))).mean()),
        "mixture_sse_above_label_aware_convex_bound": float(signed_excess.sum()),
        "top10_rows_fraction_of_excess_sse": float(signed_excess[order[:10]].sum()/signed_excess.sum()),
        "top1pct_rows_fraction_of_excess_sse": float(signed_excess[order[:max(1,int(np.ceil(len(y)*.01)))]].sum()/signed_excess.sum()),
        "group_by_airport": {str(a): {"n": int(np.sum(airports==a)),
            "actual_rmse": scores(y[airports==a], variants["actual_mixture"][airports==a])["rmse_sec"],
            "convex_bound_rmse": scores(y[airports==a], bound[airports==a])["rmse_sec"],
            "excess_sse_fraction": float(signed_excess[airports==a].sum()/signed_excess.sum())}
            for a in np.unique(airports)}}, variants


def main():
    x, meta = common.load_data()
    eligible = ~np.isfinite(meta.proxy_sec.to_numpy(float)) & np.isfinite(meta.schedule_sec.to_numpy(float))
    x = x.loc[meta.loc[eligible, ID]].copy()
    folder = ROOT / "private_runs/next_230/features"
    marker = read_json(folder / "manifest.json")
    assert sha256(folder / "extensions.parquet") == marker["sha256"]
    columns = [c for c in pq.read_schema(folder / "extensions.parquet").names if not c.startswith("rw_")]
    ext = pd.read_parquet(folder / "extensions.parquet", columns=columns).set_index(ID).loc[x.index]
    x = pd.concat([x, ext], axis=1)
    audit, aggregate, provenance = {}, {}, {}
    for fold in W:
        idx, _, _ = common.fold_data(meta, fold, full=True)
        directory = ROOT / "private_runs/campaign_20260916/missing_mixture/models" / f"lightgbm_soft_missing_{fold}_s20260916"
        manifest = read_json(directory / "manifest.json")
        provenance[fold] = {"manifest_sha256": sha256(directory / "manifest.json")}
        audit[fold] = {}
        for stage, filename in [("tune", "fit_model.joblib"), ("score", "model.joblib")]:
            assert sha256(directory / filename) == manifest["outputs"][filename]
            positions = idx[stage][eligible[idx[stage]]]
            rows = meta.iloc[positions]
            sx = x.loc[rows[ID]]
            model = joblib.load(directory / filename)
            y, schedule = rows[TARGET].to_numpy(float), rows.schedule_sec.to_numpy(float)
            probability, good, bad = endpoint_values(model, sx, schedule)
            result, variants = diagnose(y, schedule, probability, good, bad, rows.ADEP_mvt.astype(str).to_numpy())
            result["experts_training_period"] = "fit only" if stage == "tune" else "refit"
            audit[fold][stage] = result
            if stage == "score":
                reference, _ = common.reference(fold)
                change = eligible[idx["score"]]
                assert np.array_equal(reference.loc[change, ID], rows[ID])
                physical_path = ROOT / "private_runs/campaign_20260916/models" / f"lightgbm_direct_physical_s200k_{fold}_s20260916/candidate.parquet"
                physical = pd.read_parquet(physical_path)
                assert np.array_equal(physical[ID], reference[ID])
                pairs = {"schedule_vs_reference_convex_oracle": (schedule, reference.loc[change, "prediction_sec"].to_numpy()),
                         "schedule_vs_physical_convex_oracle": (schedule, physical.loc[change,"prediction_sec"].to_numpy())}
                for name, (a,b) in pairs.items():
                    _, value = oracle(a,b,y)
                    variants[name] = value
                    result["metrics"][name] = scores(y, value)
                result["metrics"]["submitted_reference_missing"] = scores(y, reference.loc[change,"prediction_sec"])
                expected = pd.read_parquet(directory / "candidate.parquet")
                if not np.allclose(expected.loc[change,"prediction_sec"], variants["actual_mixture"], rtol=0, atol=1e-9):
                    raise AssertionError("Existing mixture replay failed")
                for name, value in variants.items():
                    pred = reference.prediction_sec.to_numpy().copy()
                    pred[change] = value
                    metric = scores(reference[TARGET], pred)
                    aggregate.setdefault(name, {})[fold] = metric
                table = pd.DataFrame({ID: rows[ID].to_numpy(), TARGET:y, "schedule_sec":schedule,
                    "good_expert_sec":good, "bad_expert_sec":bad,"probability":probability,
                    "reference_sec":reference.loc[change,"prediction_sec"].to_numpy(),
                    "physical_sec":physical.loc[change,"prediction_sec"].to_numpy()})
            else:
                table = pd.DataFrame({ID:rows[ID].to_numpy(), TARGET:y,"schedule_sec":schedule,
                    "good_expert_sec":good,"bad_expert_sec":bad,"probability":probability})
            path = OUT / f"missing_endpoints_{fold}_{stage}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            table.to_parquet(path, index=False)
            result["private_endpoint_cache_sha256"] = sha256(path)
    for name, folds in list(aggregate.items()):
        aggregate[name] = {"seasonal_complete_cohort_rmse": float(np.sqrt(sum(W[f]*m["sse"]/m["n"] for f,m in folds.items()))), "folds":folds}
    write_json(OUT / "missing_gate_audit.json", {"created_utc":utc_now(),"folds":audit,"complete_cohort_counterfactuals":aggregate,
        "provenance":provenance,"script_sha256":sha256(__file__),
        "loss_identity": "For d=g-b, prediction=b+p*d: (y-pred)^2=d^2*(p-(y-b)/d)^2. Fit OOS soft target t without clipping, sample_weight=d^2, clip only inferred p to [0,1]. d approximately0 needs fallback/no information.",
        "limits":["Every oracle uses the hidden scoring label and is strictly a diagnostic endpoint bound, never an achievable accuracy claim.","Raw gate target is not the 60-second regime indicator; weighted classification is not exactly the mixture MSE objective.","Tune examples are fit-only expert predictions, but head iteration selection has already used tune labels; stricter calibration requires a separate chronological calibration layer.","The schedule-vs-reference oracle includes submitted reference as an expert, so its gain bound does not establish a new independent physical expert."]})
    print("MISSING GATE COUNTERFACTUALS", {k:v["seasonal_complete_cohort_rmse"] for k,v in aggregate.items()}, flush=True)


if __name__ == "__main__":
    main()
