"""Independently replay only the inherited V2 missing-clock predictions."""
from preflight import ROOT, OUT, V2, ID, read, sha
import json
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor


def main():
    dest = OUT / "v2_missing_native_receipt.json"
    assert not dest.exists()
    prior = read(OUT / "preflight_receipt.json")
    assert prior["status"] == "passed"
    inputs = read(V2 / "ranking_inputs.json")
    for name, digest in inputs["files"].items():
        assert sha(V2 / name) == digest
    meta = pd.read_parquet(V2 / "ranking_meta.parquet")
    missing = ~np.isfinite(meta.proxy_sec.to_numpy(float))
    positions = np.flatnonzero(missing)
    base = pd.read_parquet(V2 / "ranking_base.parquet").iloc[positions]
    schedule = pd.read_parquet(V2 / "ranking_schedule.parquet").iloc[positions]
    local = meta.iloc[positions]
    np.testing.assert_array_equal(base.index, local[ID])
    np.testing.assert_array_equal(schedule.index, local[ID])
    record = prior["models"]["missing"]
    path = V2 / "components/missing/model.cbm"
    assert sha(path) == record["model_sha256"]
    model = CatBoostRegressor().load_model(str(path))
    values = model.predict(base, thread_count=1)
    rome = local.ADEP_mvt.eq("LIRF").to_numpy() & np.isfinite(local.schedule_sec.to_numpy())
    means = read(V2 / "rome_means.json")
    members = []
    for seed in (20260910, 20260911, 20260912):
        models = []
        for part, klass in (("classifier", CatBoostClassifier), ("regressor", CatBoostRegressor)):
            name = f"rome_{seed}_{part}"
            path = V2 / "components" / name / "model.cbm"
            assert sha(path) == prior["models"][name]["model_sha256"]
            models.append(klass().load_model(str(path)))
        probability = models[0].predict_proba(schedule.iloc[np.flatnonzero(rome)], thread_count=1)[:, 1]
        residual = models[1].predict(schedule.iloc[np.flatnonzero(rome)], thread_count=1)
        members.append(local.schedule_sec.to_numpy()[rome] + probability * means[str(seed)] + (1 - probability) * residual)
    values[rome] = np.mean(members, axis=0)
    expected = pd.read_parquet(V2 / "ranking_predictions.parquet").set_index(ID).loc[local[ID], "prediction_sec"].to_numpy(float)
    np.testing.assert_array_equal(values, expected)
    receipt = dict(status="passed", rows=len(local), rome_schedule_rows=int(rome.sum()),
                   max_abs_delta_sec=float(np.max(np.abs(values - expected))),
                   preflight_sha256=sha(OUT / "preflight_receipt.json"),
                   means_sha256=sha(V2 / "rome_means.json"), verifier_sha256=sha(__file__),
                   prediction_sha256=sha(V2 / "ranking_predictions.parquet"),
                   fitting_used=False, gpu_used=False, native_threads=1)
    dest.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
