"""Declared tune-only simplex applied to frozen independently-refit predictions."""
import argparse
import shutil
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "review_work/campaign_20260916"))
import common
import simplex
import audit_tuning
from taxiout.artifacts import read_json, write_json, sha256, object_hash, utc_now
from taxiout.metrics import evaluate, paired_stability, season_score
from taxiout.schema import ID, TARGET

OUT = ROOT / "private_runs/breakthrough_20260916/models/stacking/simplex_v1"
EXPERTS = ("tabm_source", "lgb_allinfo")
VARIANTS = ("global_simplex", "airport_shrunk_simplex")


def day_sensitivity(first, second):
    np.testing.assert_array_equal(first[ID], second[ID])
    daily = pd.DataFrame({"day": first.day, "first": first.squared_error, "second": second.squared_error}).groupby("day").agg(first=("first", "sum"), second=("second", "sum"), n=("first", "size"))
    result = []
    for day, values in daily.iterrows():
        n = len(first) - values.n
        delta = np.sqrt((second.squared_error.sum() - values.second) / n) - np.sqrt((first.squared_error.sum() - values.first) / n)
        result.append({"removed_day": day, "delta_rmse": float(delta), "removed_day_sse_gain": float(values.first - values.second)})
    return result


def sources(fold):
    receipts = {}
    for expert in EXPERTS:
        directory = audit_tuning.registry(fold)[expert]
        record = read_json(directory / "manifest.json")
        if record["status"] != "complete":
            raise ValueError("Stacking needs complete fit/refit experts")
        receipts[expert] = {"directory": str(directory), "manifest_sha256": sha256(directory / "manifest.json"),
                            "tune_sha256": record["outputs"]["tune_predictions.parquet"],
                            "score_sha256": record["outputs"]["candidate.parquet"],
                            "refit_model_sha256": record["outputs"]["model.joblib"],
                            "fit_model_sha256": record["outputs"]["fit_model.joblib"], "split": record["split"],
                            "fit_ids": record["fit_ids"], "replay_delta": record["reload_max_abs_delta"]}
    fallback = ROOT / "private_runs/next_230/models" / f"clock_and_rome_ensemble_{fold}_s20260910"
    record = read_json(fallback / "manifest.json")
    receipts["reference"] = {"directory": str(fallback), "manifest_sha256": sha256(fallback / "manifest.json"),
                              "score_sha256": record["outputs"]["score_predictions.parquet"]}
    return receipts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", nargs="+", default=["F1", "F3"])
    parser.add_argument("--declare-only", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    receipts = {fold: sources(fold) for fold in args.folds}
    audit = read_json(audit_tuning.OUT / "tuning_audit.json")
    protocol = {"folds": args.folds, "experts": list(EXPERTS), "variants": list(VARIANTS),
                "source_receipts": receipts, "source_hashes": {path.name: sha256(path) for path in [Path(__file__), Path(simplex.__file__), Path(audit_tuning.__file__)]},
                "tuning_audit_sha256": sha256(audit_tuning.OUT / "tuning_audit.json"),
                "cohort": "Originalpurged ordinaryfiniteNMproxy0..7200;allother scorerows retainreferenceexactly",
                "global_objective": "Minimize originaltune sum(Y-TabM-alpha*(LightGBM-TabM))^2 subjectto0<=alpha<=1",
                "airport_objective": "Same localrawSSE+1000*globalmean(disagreement^2)*(alpha_airport-alpha_global)^2, subjectto0<=alpha<=1",
                "shrinkage_rows": simplex.SHRINKAGE_ROWS, "unknown_airport": "Globalcoefficient",
                "no_intercept": True, "no_target_clipping": True, "no_score_weight_refit": True,
                "tune_caveat": "Originaltune also used for baseearlystopping;early-lateaudit not untouchedvalidation",
                "controls": "Two components applied ordinaryonly, submittedreference elsewhere; referencecompletecohort",
                "declaration_timing": "Protocol written before any scoreprediction file is opened by this process"}
    path = OUT / "protocol.json"
    if path.exists():
        if read_json(path)["declaration"] != protocol:
            raise ValueError("Frozen simplex declaration differs")
    else:
        write_json(path, {"created_utc": utc_now(), "declaration": protocol})
    for src in [Path(__file__), Path(simplex.__file__), Path(audit_tuning.__file__)]:
        shutil.copyfile(src, OUT / src.name)
    if args.declare_only:
        print("DECLARED", path, flush=True)
        return
    meta_path = ROOT / "private_runs/screening_230/data/interim/audit/departures.parquet"
    frozen = read_json(ROOT / "private_runs/screening_230/reports/data_audit.json")
    assert sha256(meta_path) == frozen["artifacts"][meta_path.name]
    meta = pd.read_parquet(meta_path, columns=[ID, "proxy_sec", "ADEP_mvt"])
    meta = meta.set_index(ID)
    results = {}
    for fold in args.folds:
        dest = OUT / fold
        if dest.exists():
            raise ValueError("Prior simplex result retained; do not overwrite")
        dest.mkdir()
        tune = pd.read_parquet(audit_tuning.OUT / f"{fold}_aligned_tune.parquet")
        assert object_hash(tune[ID].tolist()) == audit["folds"][fold]["ordinary_ids_sha256"]
        for expert in EXPERTS:
            receipt = receipts[fold][expert]
            directory = Path(receipt["directory"])
            assert sha256(directory / "manifest.json") == receipt["manifest_sha256"]
            assert sha256(directory / "tune_predictions.parquet") == receipt["tune_sha256"]
            raw = pd.read_parquet(directory / "tune_predictions.parquet").set_index(ID)
            np.testing.assert_array_equal(tune[expert], raw.loc[tune[ID], "prediction_sec"])
        weights = simplex.solve(tune[TARGET], tune[EXPERTS[0]], tune[EXPERTS[1]], tune.ADEP_mvt)
        write_json(dest / "weights.json", weights)
        reference, refrecord = common.reference(fold)
        assert sha256(Path(receipts[fold]["reference"]["directory"]) / "manifest.json") == receipts[fold]["reference"]["manifest_sha256"]
        allinfo = meta.loc[reference[ID]]
        eligible = np.isfinite(allinfo.proxy_sec.to_numpy()) & allinfo.proxy_sec.between(0, 7200).to_numpy()
        components = {}
        for expert in EXPERTS:
            receipt = receipts[fold][expert]
            assert object_hash(receipt["split"]) == object_hash(refrecord["split"])
            scorepath = Path(receipt["directory"]) / "candidate.parquet"
            assert sha256(scorepath) == receipt["score_sha256"]
            component = pd.read_parquet(scorepath)
            np.testing.assert_array_equal(reference[ID], component[ID])
            np.testing.assert_array_equal(reference[TARGET], component[TARGET])
            components[expert] = component.prediction_sec.to_numpy()
        complete = {}
        for expert in EXPERTS:
            values = reference.prediction_sec.to_numpy().copy()
            values[eligible] = components[expert][eligible]
            complete[expert] = values
        replay = read_json(dest / "weights.json")
        for variant, conditioned in zip(VARIANTS, [False, True]):
            values = reference.prediction_sec.to_numpy().copy()
            values[eligible] = simplex.predict(weights, components[EXPERTS[0]][eligible], components[EXPERTS[1]][eligible], allinfo.ADEP_mvt.to_numpy()[eligible], conditioned)
            repeated = simplex.predict(replay, components[EXPERTS[0]][eligible], components[EXPERTS[1]][eligible], allinfo.ADEP_mvt.to_numpy()[eligible], conditioned)
            np.testing.assert_array_equal(values[eligible], repeated)
            np.testing.assert_array_equal(values[~eligible], reference.prediction_sec.to_numpy()[~eligible])
            complete[variant] = values
        errors = {"reference": reference}
        metrics = {}
        for name, values in complete.items():
            frame = reference.drop(columns=[TARGET, "error_sec", "squared_error", "label_bin", "month", "day"], errors="ignore").copy()
            frame["prediction_sec"] = values
            metric, scored = evaluate(frame, reference[[ID, TARGET]])
            scored.to_parquet(dest / f"{name}.parquet", index=False)
            errors[name], metrics[name] = scored, metric
        comparisons = {}
        for variant in VARIANTS:
            comparisons[variant] = {control: {"stability": paired_stability(errors[control], errors[variant], repetitions=1000),
                                              "day_removals": day_sensitivity(errors[control], errors[variant])}
                                    for control in ["reference", *EXPERTS]}
        record = {"status": "complete", "completed_utc": utc_now(), "fold": fold, "protocol_sha256": sha256(path),
                  "ordinary_score_rows": int(eligible.sum()), "protected_rows": int((~eligible).sum()),
                  "weights": weights, "metrics": metrics, "comparisons": comparisons,
                  "saved_weight_replay_max_delta": 0., "sources": receipts[fold]}
        record["outputs"] = {file.name: sha256(file) for file in dest.iterdir() if file.is_file()}
        write_json(dest / "manifest.json", record)
        results[fold] = record
        print("RESULT", fold, {name: m["overall"]["rmse_sec"] for name, m in metrics.items()}, flush=True)
    summary = {"created_utc": utc_now(), "protocol_sha256": sha256(path)}
    if set(results) == {"F1", "F3"}:
        summary["seasonal_rmse"] = {name: season_score(results["F1"]["metrics"][name]["overall"], results["F3"]["metrics"][name]["overall"])
                                    for name in [*EXPERTS, *VARIANTS]}
    write_json(OUT / "summary.json", summary)
    print("SUMMARY", summary, flush=True)


if __name__ == "__main__":
    main()
