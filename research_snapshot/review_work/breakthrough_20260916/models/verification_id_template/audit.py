"""Independent saved-prediction sensitivity and historical-prior isolation audit."""
import argparse
import gc
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "review_work/breakthrough_20260916/missing"))
import run_missing_models as base
from taxiout.metrics import season_score
from taxiout.artifacts import read_json, write_json, sha256, object_hash, utc_now
from taxiout.schema import ID, TARGET, MOVEMENT

OUT = ROOT / "private_runs/breakthrough_20260916/models/verification_id_template"
MODELROOT = ROOT / "private_runs/breakthrough_20260916/missing"


def verified(directory, filename, record=None):
    record = record or read_json(directory / "manifest.json")
    if record["status"] != "complete" or sha256(directory / filename) != record["outputs"][filename]:
        raise ValueError(f"Incomplete or changed artifact: {directory / filename}")
    return pd.read_parquet(directory / filename)


def metric(y, pred, mask=None):
    use = np.ones(len(y), bool) if mask is None else mask
    return {"n": int(use.sum()), "sse": float(np.square(pred[use] - y[use]).sum()),
            "rmse_sec": float(np.sqrt(np.square(pred[use] - y[use]).mean()))}


def comparison(y, left, right, day):
    ae, be = (left - y) ** 2, (right - y) ** 2
    daily = pd.DataFrame({"day": day, "a": ae, "b": be}).groupby("day").agg(a=("a", "sum"), b=("b", "sum"), n=("a", "size"))
    leave = []
    for name, row in daily.iterrows():
        n = len(y) - row.n
        delta = float(np.sqrt((be.sum() - row.b) / n) - np.sqrt((ae.sum() - row.a) / n))
        leave.append({"removed_day": str(name), "remaining_n": int(n), "delta_rmse": delta,
                      "removed_day_sse_gain": float(row.a - row.b)})
    return {"delta_rmse": float(np.sqrt(be.mean()) - np.sqrt(ae.mean())),
            "sse_gain": float(ae.sum() - be.sum()), "improved_days": int((daily.b < daily.a).sum()),
            "total_days": len(daily), "leave_one_day_out_delta_range": [min(r["delta_rmse"] for r in leave), max(r["delta_rmse"] for r in leave)],
            "all_day_removals_improve": all(r["delta_rmse"] < 0 for r in leave), "day_removals": leave}


def prior_audit(fold, x, meta, idx, manifest, directory):
    missing = ~np.isfinite(meta.proxy_sec.to_numpy(float))
    use = {stage: rows[missing[rows]] for stage, rows in idx.items()}
    for stage, rows in use.items():
        assert manifest["fit_ids"][stage] == {"n": len(rows), "hash": object_hash(meta.iloc[rows][ID].tolist())}
    checks = {}
    for stage, filename, query in [("fit", "fit_model.joblib", "tune"), ("refit", "model.joblib", "score")]:
        assert sha256(directory / filename) == manifest["outputs"][filename]
        saved = joblib.load(directory / filename)
        prior = saved["prior"]
        rows = use[stage]
        times = pd.to_datetime(meta.iloc[rows][MOVEMENT], utc=True)
        qtimes = pd.to_datetime(meta.iloc[use[query]][MOVEMENT], utc=True)
        expected = base.HistoricalTemplate().fit(x.loc[meta.iloc[rows][ID]], meta.iloc[rows][TARGET].to_numpy(float), times)
        assert prior.history_n == len(rows)
        assert prior.last_fit == times.max() and prior.last_fit < qtimes.min()
        assert prior.global_mean == expected.global_mean
        assert len(prior.tables) == len(expected.tables)
        for (keys, table), (ekeys, etable) in zip(prior.tables, expected.tables):
            assert keys == ekeys
            pd.testing.assert_frame_equal(table, etable)
        checks[stage] = {"rows": len(rows), "last_prior_time": str(prior.last_fit),
                         "first_query_time": str(qtimes.min()), "all_saved_group_tables_match_exact_permitted_labels": True,
                         "ids_hash": manifest["fit_ids"][stage]["hash"]}
        del saved
        gc.collect()
    rows = use["fit"]
    sx = x.loc[meta.iloc[rows][ID]]
    times = pd.Series(pd.to_datetime(meta.iloc[rows][MOVEMENT], utc=True).to_numpy(), index=sx.index)
    labels = meta.iloc[rows][TARGET].to_numpy(float)
    original = base.crossfit_templates(sx, labels, times)
    months = pd.to_datetime(times, utc=True).dt.strftime("%Y-%m")
    pivot = sorted(months.unique())[len(months.unique()) // 2]
    mutated = labels.copy()
    mutated[months.eq(pivot).to_numpy()] += 1000000
    changed = base.crossfit_templates(sx, mutated, times)
    unchanged = months.le(pivot).to_numpy()
    pd.testing.assert_frame_equal(original.loc[unchanged], changed.loc[unchanged])
    checks["crossfit_mutation"] = {"mutated_month": pivot, "current_and_earlier_months_unchanged": True,
                                    "checked_rows": int(unchanged.sum()), "label_perturbation_sec": 1000000}
    return checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", nargs="+", default=["F1", "F3", "F2", "G1"])
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    x, meta = base.load_data()
    allfolds = {}
    for fold in args.folds:
        directory = MODELROOT / "id_context_v1/models" / f"historical_template_{fold}_s20260916"
        manifest = read_json(directory / "manifest.json")
        if manifest["status"] != "complete":
            raise ValueError(f"Awaiting completed model for {fold}")
        reference, refrecord = base.common.reference(fold)
        idx, split, _ = base.common.fold_data(meta, fold, full=True)
        assert object_hash(split) == object_hash(manifest["split"]) == object_hash(refrecord["split"])
        assert np.array_equal(reference[ID], meta.iloc[idx["score"]][ID])
        y = reference[TARGET].to_numpy(float)
        day = reference.day.to_numpy()
        candidate = verified(directory, "candidate.parquet", manifest)
        blend = verified(directory, "blend25.parquet", manifest)
        guard = verified(ROOT / "private_runs/breakthrough_20260916/models/support_guard" / fold, "fit_min_floor.parquet")
        for frame in [candidate, blend, guard]:
            assert np.array_equal(reference[ID], frame[ID]) and np.array_equal(reference[TARGET], frame[TARGET])
        missing = ~np.isfinite(meta.iloc[idx["score"]].proxy_sec.to_numpy(float))
        pred = {"reference": reference.prediction_sec.to_numpy(), "guarded_reference": guard.prediction_sec.to_numpy(),
                "id_candidate": candidate.prediction_sec.to_numpy(), "id_blend25": blend.prediction_sec.to_numpy()}
        assert np.array_equal(pred["id_candidate"][~missing], pred["reference"][~missing])
        assert np.allclose(pred["id_blend25"], .75 * pred["reference"] + .25 * pred["id_candidate"], atol=1e-9, rtol=0)
        pred["id_blend25_guarded_reference"] = pred["guarded_reference"].copy()
        pred["id_blend25_guarded_reference"][missing] = .75 * pred["guarded_reference"][missing] + .25 * pred["id_candidate"][missing]
        baseroot = MODELROOT / "models" / f"historical_template_{fold}_s20260916"
        if (baseroot / "manifest.json").exists():
            rawbase = verified(baseroot, "candidate.parquet")
            baseblend = verified(baseroot, "blend25.parquet")
            assert np.array_equal(reference[ID], rawbase[ID]) and np.array_equal(reference[ID], baseblend[ID])
            pred["base_candidate"] = rawbase.prediction_sec.to_numpy()
            pred["base_blend25"] = baseblend.prediction_sec.to_numpy()
            pred["base_blend25_guarded_reference"] = pred["guarded_reference"].copy()
            pred["base_blend25_guarded_reference"][missing] = .75 * pred["guarded_reference"][missing] + .25 * pred["base_candidate"][missing]
        pairs = [("reference", "id_candidate"), ("reference", "id_blend25"),
                 ("guarded_reference", "id_blend25"), ("guarded_reference", "id_blend25_guarded_reference")]
        if "base_blend25" in pred:
            pairs.extend([("base_candidate", "id_candidate"), ("base_blend25", "id_blend25"),
                          ("base_blend25_guarded_reference", "id_blend25_guarded_reference")])
        tail = missing & (y >= 86400)
        tail_days = np.isin(day, np.unique(day[tail]))
        sensitivity = {}
        for scenario, keep in [("all", np.ones(len(y), bool)), ("remove_missing_dayplus_rows", ~tail),
                                ("remove_missing_dayplus_days", ~tail_days)]:
            sensitivity[scenario] = {"rows_removed": int((~keep).sum()), "metrics": {name: metric(y, values, keep) for name, values in pred.items()}}
        details = reference.loc[tail, [ID, TARGET, "day", "ADEP_mvt"]].copy()
        for name, values in pred.items():
            details[name] = values[tail]
        details.to_parquet(OUT / f"{fold}_dayplus_rows.parquet", index=False)
        frame = reference[[ID, TARGET, "day", "ADEP_mvt"]].copy()
        for name, values in pred.items():
            frame[name] = values
        frame.to_parquet(OUT / f"{fold}_comparison.parquet", index=False)
        allfolds[fold] = {"missing_rows": int(missing.sum()), "tail_rows": int(tail.sum()), "tail_days": np.unique(day[tail]).tolist(),
                          "metrics": {name: metric(y, values) for name, values in pred.items()},
                          "comparisons": {f"{right}_vs_{left}": comparison(y, pred[left], pred[right], day) for left, right in pairs},
                          "sensitivity": sensitivity, "prior_isolation": prior_audit(fold, x, meta, idx, manifest, directory),
                          "manifest_sha256": sha256(directory / "manifest.json"), "protocol_sha256": sha256(MODELROOT / "id_context_v1/protocol.json")}
        print("VERIFIED", fold, {name: values["rmse_sec"] for name, values in allfolds[fold]["metrics"].items()}, flush=True)
    seasonal = {}
    if "F1" in allfolds and "F3" in allfolds:
        for scenario in ["all", "remove_missing_dayplus_rows", "remove_missing_dayplus_days"]:
            summer = allfolds["F1"]["sensitivity"][scenario]["metrics"]
            winter = allfolds["F3"]["sensitivity"][scenario]["metrics"]
            seasonal[scenario] = {name: season_score(summer[name], winter[name]) for name in summer.keys() & winter.keys()}
    result = {"created_utc": utc_now(), "folds": allfolds, "seasonal": seasonal, "script_sha256": sha256(__file__),
              "caveat": "Removing rows/days is sensitivity analysis only; every scored artifact retains all original labels. Folds exposed and adaptively reused."}
    write_json(OUT / "audit.json", result)
    print("SEASONAL", seasonal, flush=True)


if __name__ == "__main__":
    main()
