"""Recover completed artifacts after nonfinite metadata serialization failure."""

import sys
import time
from pathlib import Path

import lightgbm
import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))
from common import OUT, WORKSPACE, ID, TARGET, load_data, fold_data, reference
from common import read_json, write_json, sha256, object_hash, utc_now
from taxiout.metrics import evaluate, paired_stability
from xgb_adapter import predict


def main():
    name = "xgboost_correction_base_s200k_F1_s20260916"
    dest = OUT / "models" / name
    original = read_json(dest / "manifest.json")
    if original["status"] == "complete":
        raise ValueError("Recovery was already completed; preserve completed manifest")
    archive = OUT / "validation" / "xgb_f1_pre_recovery_manifest.json"
    if archive.exists():
        if read_json(archive) != original:
            raise ValueError("Recovery original changed")
    else:
        write_json(archive, original)
    old_source = HERE.with_name("xgb_adapter_before_json_fix.py")
    if sha256(old_source) != original["code_hashes"]["xgb_adapter.py"]:
        raise AssertionError("Preserved original XGBoost source does not match run")
    expected_outputs = ["fit_model.joblib", "model.joblib", "tune_predictions.parquet", "candidate.parquet", "blend25.parquet"]
    original_outputs = {name: sha256(dest / name) for name in expected_outputs}
    x, meta = load_data()
    idx, split, sampled = fold_data(meta, "F1")
    ref, refrec = reference("F1")
    if object_hash(split) != object_hash(original["split"]) or sampled != original["sampled"]:
        raise AssertionError("Fold or sampled cohorts differ")
    proxy = meta.proxy_sec.to_numpy(float)
    ordinary = np.isfinite(proxy) & (proxy >= 0) & (proxy <= 7200)
    rows = {stage: (positions[ordinary[positions]] if stage != "score" else positions)
            for stage, positions in idx.items()}
    fit_model = joblib.load(dest / "fit_model.joblib")
    refit_model = joblib.load(dest / "model.joblib")
    if fit_model["steps"] != refit_model["steps"]:
        raise AssertionError("Refit rounds differ from selected tuning rounds")
    tune = pd.read_parquet(dest / "tune_predictions.parquet")
    if not np.array_equal(tune[ID], meta.iloc[rows["tune"]][ID]):
        raise AssertionError("Tuning cohort changed")
    start = time.monotonic()
    tune_replayed = predict(fit_model, x.iloc[rows["tune"]]) + proxy[rows["tune"]]
    tune_delta = float(np.max(np.abs(tune_replayed - tune.prediction_sec)))
    score_replayed = predict(refit_model, x.iloc[rows["score"]])
    inference_seconds = time.monotonic() - start
    combined = ref.prediction_sec.to_numpy().copy()
    changed = ordinary[rows["score"]]
    combined[changed] = proxy[rows["score"]][changed] + score_replayed[changed]
    expected = {"candidate": combined, "blend25": .75 * ref.prediction_sec.to_numpy() + .25 * combined}
    reports, deltas = {}, {}
    for variant, values in expected.items():
        frame = pd.read_parquet(dest / f"{variant}.parquet")
        if not np.array_equal(frame[ID], ref[ID]) or not np.array_equal(frame[TARGET], ref[TARGET]):
            raise AssertionError("Recovered prediction ID/label mismatch")
        delta = float(np.max(np.abs(frame.prediction_sec.to_numpy() - values)))
        if delta > 1e-9:
            raise AssertionError("Saved model and stored predictions do not replay")
        deltas[variant] = delta
        clean = frame.drop(columns=[TARGET, "error_sec", "squared_error", "label_bin", "month", "day"], errors="ignore")
        metric, errors = evaluate(clean, meta.iloc[rows["score"]][[ID, TARGET]])
        reports[variant] = {"metrics": metric, "stability": paired_stability(ref, errors, repetitions=500)}
    if tune_delta > 1e-9:
        raise AssertionError("Saved fit model and tune predictions do not replay")
    record = dict(original)
    recovered = {"recovered_utc": utc_now(), "script_sha256": sha256(__file__),
        "original_manifest_sha256": sha256(archive), "original_adapter_sha256": sha256(old_source),
        "current_adapter_sha256": sha256(HERE.parents[1] / "xgb_adapter.py"),
        "reason": "All training artifacts existed; final manifest serialization failed on XGBoost params missing=np.nan.",
        "behavior_change": "None: evidence-only NaN parameter sanitization.",
        "tune_replay_max_abs_delta": tune_delta, "score_replay_max_abs_delta": deltas,
        "verification_inference_seconds": inference_seconds,
        "original_timing_policy": "Original tuning/refit/inference/runtime and peak RSS were not durably written; null, not inferred."}
    record.update(status="complete", completed_utc=utc_now(), recovery=recovered,
        tune={"steps": int(fit_model["steps"]), "recovered": True},
        refit={"steps": int(refit_model["steps"]), "recovered": True},
        tuning_runtime_sec=None, refit_runtime_sec=None, inference_runtime_sec=None,
        runtime_sec=None, peak_rss_bytes=None, reload_max_abs_delta=max(deltas.values()),
        features_used=list(x), reports=reports,
        fit_ids={stage: {"n": len(positions), "hash": object_hash(meta.iloc[positions][ID].tolist())}
                 for stage, positions in rows.items()}, outputs=original_outputs)
    if original_outputs != {name: sha256(dest / name) for name in expected_outputs}:
        raise AssertionError("A saved artifact changed during recovery")
    write_json(dest / "manifest.json", record)
    write_json(OUT / "validation/xgb_f1_recovery.json", recovered)
    print("RECOVERED", {v: r["metrics"]["overall"]["rmse_sec"] for v, r in reports.items()}, flush=True)


if __name__ == "__main__":
    main()
