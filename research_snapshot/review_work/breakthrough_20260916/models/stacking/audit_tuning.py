"""Read-only expert cohort/complementarity audit using original tune labels only."""
import sys
from pathlib import Path
import itertools
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "review_work/campaign_20260916"))
import common
from taxiout.artifacts import read_json, write_json, sha256, object_hash, utc_now
from taxiout.schema import ID, TARGET, MOVEMENT

OUT = ROOT / "private_runs/breakthrough_20260916/models/stacking"
BREAK = ROOT / "private_runs/breakthrough_20260916"


def registry(fold):
    suffix = f"aobt_allfinite_{fold}_s20260916"
    entries = {
        "cb_global_base": BREAK / "missing/conventions_models/models" / f"global_base_{fold}_s20260916",
        "tabm_base": BREAK / "models" / f"tabm_{suffix}",
        "tabm_source": BREAK / "models/augmented/source_past__source_twosided" / f"tabm_{suffix}",
        "tabm_ple8": BREAK / "models/tabm_ple/base" / f"tabm_ple8_{suffix}",
        "lgb_base": BREAK / "cpu/base" / f"lightgbm_{suffix}",
        "lgb_conventions": BREAK / "information_models/conventions" / f"lightgbm_{suffix}",
        "lgb_allinfo": BREAK / "information_models/conventions__geometry__source_past__source_twosided__surface_T__trajectory__weather_T" / f"lightgbm_{suffix}",
        "old_direct": ROOT / "private_runs/next_230/clock_cpu_experts" / fold,
        "old_residual": ROOT / "private_runs/next_230/models" / f"capacity_d8_5000_{fold}_s20260910",
    }
    return entries


def read_expert(name, directory, split, tune, ordinary_ids, ordinary_fit_ids, ordinary_refit_ids):
    marker = "expert.json" if name == "old_direct" else "manifest.json"
    record = read_json(directory / marker)
    if record["status"] != "complete":
        return None, {"status": "pending_complete_expert", "directory": str(directory)}
    assert object_hash(record["split"]) == object_hash(split)
    filename = "tune_direct.parquet" if name == "old_direct" else "tune_predictions.parquet"
    path = directory / filename
    digest = sha256(path)
    assert digest == record["outputs"][filename]
    frame = pd.read_parquet(path).set_index(ID)
    if not frame.index.is_unique or not frame.index.isin(tune.index).all():
        raise ValueError("Tuning prediction IDs duplicate or extend beyond original tune month")
    expected = ordinary_ids if name in ["old_direct", "old_residual", "cb_global_base"] else tune.index[np.isfinite(tune.proxy_sec)]
    np.testing.assert_array_equal(frame.index, expected)
    if name.startswith("old_"):
        assert record["fit_id_hash"] == object_hash(ordinary_fit_ids.tolist())
        assert record["refit_id_hash"] == object_hash(ordinary_refit_ids.tolist())
    else:
        assert record["fit_ids"]["tune"]["hash"] == object_hash(frame.index.tolist())
    if name == "old_direct":
        np.testing.assert_array_equal(frame[TARGET], tune.loc[frame.index, TARGET])
        pred = frame.direct
    elif name == "old_residual":
        offset = tune.loc[frame.index, "proxy_sec"]
        np.testing.assert_array_equal(frame.target, tune.loc[frame.index, TARGET] - offset)
        pred = frame.prediction + offset
    else:
        pred = frame.prediction_sec
    if not np.isfinite(pred).all():
        raise ValueError("Nonfinite saved tune predictions")
    return pred.rename(name), {"path": str(path), "sha256": digest, "manifest_sha256": sha256(directory / marker),
                               "rows": len(frame), "ordinary_common_rows": len(ordinary_ids),
                               "excluded_finite_nonordinary_rows": len(frame) - len(ordinary_ids),
                               "original_split_verified": True, "fit_only_prediction_source": True}


def rmse(y, p):
    return float(np.sqrt(np.mean(np.square(np.asarray(p) - np.asarray(y)))))


def pair_weight(y, first, second):
    difference = second - first
    denominator = float(difference @ difference)
    return 0. if denominator == 0 else float(np.clip(difference @ (y - first) / denominator, 0., 1.))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    declaration = {"created_utc": utc_now(), "folds": ["F1", "F3"], "score_predictions_read": False,
                   "labels": "Only original tune UTCmonth read through parquet timestamp filters",
                   "cohort": "Intersection of complete ordinary finiteNMproxy0..7200 tune rows",
                   "experts": list(registry("F1")), "diagnostics": "IndividualRMSE,errorcorrelation,pairconvexfit on tune,earlydays1..20 pairfit thenlatedays21+ evaluation",
                   "warning": "Tune used for baseearlystopping; neither pairfit nor within-tune split is freshunbiasedvalidation",
                   "no_stacking_scores": "No score predictions, refit predictions or learnedV2gate outputs read or generated",
                   "source_sha256": sha256(__file__)}
    write_json(OUT / "audit_protocol.json", declaration)
    meta_path = ROOT / "private_runs/screening_230/data/interim/audit/departures.parquet"
    audit = read_json(ROOT / "private_runs/screening_230/reports/data_audit.json")
    assert sha256(meta_path) == audit["artifacts"][meta_path.name]
    meta = pd.read_parquet(meta_path, columns=[ID, MOVEMENT, "ADEP_mvt", "proxy_sec", "FLIGHT_ID_mvt"])
    results = {}
    for fold in ["F1", "F3"]:
        idx, split = common.make_fold(meta, common.load_config("configs/folds.yaml")[fold])
        tune = meta.iloc[idx["tune"]].set_index(ID)
        start, end = split["spec"]["tune"]
        labels = pd.read_parquet(meta_path, columns=[ID, TARGET],
                                 filters=[(MOVEMENT, ">=", pd.Timestamp(start, tz="UTC")), (MOVEMENT, "<", pd.Timestamp(end, tz="UTC"))]).set_index(ID)
        tune[TARGET] = labels.loc[tune.index, TARGET]
        ordinary = np.isfinite(meta.proxy_sec) & meta.proxy_sec.between(0, 7200)
        stage_ids = {stage: pd.Index(meta.iloc[positions[ordinary.iloc[positions].to_numpy()]][ID]) for stage, positions in idx.items()}
        frames, receipts = [], {}
        for name, directory in registry(fold).items():
            frame, receipt = read_expert(name, directory, split, tune, stage_ids["tune"], stage_ids["fit"], stage_ids["refit"])
            receipts[name] = receipt
            if frame is not None:
                frames.append(frame)
        joined = pd.concat(frames, axis=1, join="inner").loc[stage_ids["tune"]]
        assert len(joined) == len(stage_ids["tune"]) and not joined.isna().any().any()
        selected = tune.loc[joined.index]
        y = selected[TARGET].to_numpy(float)
        early = pd.to_datetime(selected[MOVEMENT], utc=True).dt.day.le(20).to_numpy()
        individual = {name: {"all_tune_rmse": rmse(y, joined[name]), "early_rmse": rmse(y[early], joined[name].to_numpy()[early]),
                              "late_rmse": rmse(y[~early], joined[name].to_numpy()[~early])} for name in joined}
        residuals = joined.subtract(y, axis=0)
        pairs = []
        for first, second in itertools.combinations(joined.columns, 2):
            a, b = joined[first].to_numpy(), joined[second].to_numpy()
            weight = pair_weight(y, a, b)
            early_weight = pair_weight(y[early], a[early], b[early])
            prediction = a + weight * (b - a)
            late_prediction = a[~early] + early_weight * (b[~early] - a[~early])
            pairs.append({"first": first, "second": second, "fitted_second_weight": weight,
                          "all_tune_fit_rmse": rmse(y, prediction),
                          "in_sample_gain_vs_best_component": min(individual[first]["all_tune_rmse"], individual[second]["all_tune_rmse"]) - rmse(y, prediction),
                          "early_fitted_second_weight": early_weight, "late_rmse": rmse(y[~early], late_prediction),
                          "late_gain_vs_best_late_component": min(individual[first]["late_rmse"], individual[second]["late_rmse"]) - rmse(y[~early], late_prediction)})
        airports = {}
        for airport, positions in selected.groupby("ADEP_mvt", observed=True).indices.items():
            airports[str(airport)] = {"rows": len(positions), "individual_rmse": {name: rmse(y[positions], joined[name].to_numpy()[positions]) for name in joined}}
        joined.assign(**{TARGET: y, MOVEMENT: selected[MOVEMENT], "ADEP_mvt": selected.ADEP_mvt}).reset_index().to_parquet(OUT / f"{fold}_aligned_tune.parquet", index=False)
        results[fold] = {"common_rows": len(joined), "ordinary_ids_sha256": object_hash(joined.index.tolist()),
                         "all_tune_rows": len(tune), "early_rows": int(early.sum()), "late_rows": int((~early).sum()),
                         "receipts": receipts, "individual": individual, "error_correlations": residuals.corr().to_dict(),
                         "pairs": sorted(pairs, key=lambda value: value["all_tune_fit_rmse"]), "airports": airports}
        print("TUNE", fold, len(joined), individual, flush=True)
        print("BEST_PAIRS", fold, results[fold]["pairs"][:4], flush=True)
    write_json(OUT / "tuning_audit.json", {"created_utc": utc_now(), "folds": results, "declaration_sha256": sha256(OUT / "audit_protocol.json"),
                                           "caveat": declaration["warning"]})


if __name__ == "__main__":
    main()
