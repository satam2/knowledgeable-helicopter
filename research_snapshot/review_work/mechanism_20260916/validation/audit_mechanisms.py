"""Independent mechanism and evaluation audit using existing local evidence only."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "knowledgeable-helicopter-screening/src"))
from taxiout.artifacts import read_json, write_json, sha256, object_hash, utc_now
from taxiout.config import load_config
from taxiout.metrics import scores
from taxiout.paths import external_path
from taxiout.schema import ID, TARGET, MOVEMENT
from taxiout.splits import make_fold

OUT = external_path(ROOT / "private_runs/mechanism_20260916/validation")
W = {"F1": 192122 / 344841, "F3": 152719 / 344841}


def masks(frame):
    y, proxy, schedule = [frame[col].to_numpy(float) for col in (TARGET, "proxy_sec", "schedule_sec")]
    missing = ~np.isfinite(proxy)
    gap = np.isfinite(proxy) & (np.abs(y - proxy) > 1800)
    return {"missing": missing, "finite_source_gap_gt1800": gap,
            "finite_source_gap_le1800": ~(missing | gap), "missing_or_gap": missing | gap,
            "ordinary_input_proxy": np.isfinite(proxy) & (proxy >= 0) & (proxy <= 7200),
            "invalid_finite_proxy": np.isfinite(proxy) & ((proxy < 0) | (proxy > 7200)),
            "target_gt7200": y > 7200, "target_lt0": y < 0,
            "missing_finite_schedule": missing & np.isfinite(schedule),
            "missing_schedule_consistent60": missing & np.isfinite(schedule) & (np.abs(y - schedule) <= 60),
            "missing_schedule_inconsistent60": missing & np.isfinite(schedule) & (np.abs(y - schedule) > 60)}


def group_profile(frame):
    y = frame[TARGET].to_numpy(float)
    proxy = frame.proxy_sec.to_numpy(float)
    correction = y - proxy
    records = {}
    for name, keep in masks(frame).items():
        cy = y[keep]
        cr = correction[keep & np.isfinite(correction)]
        records[name] = {"n": int(keep.sum()), "row_share": float(keep.mean()),
            "label_mean": float(cy.mean()) if len(cy) else None,
            "label_p99": float(np.quantile(cy, .99)) if len(cy) else None,
            "label_gt7200": int((cy > 7200).sum()),
            "correction_energy_sec2": float(np.dot(cr, cr)),
            "airports": {str(airport): int(n) for airport, n in frame.loc[keep].groupby("ADEP_mvt", observed=True).size().items()}}
    return records


def aggregate_budget(frames):
    budget = {}
    mse = sum(W[f] * frame.squared_error.mean() for f, frame in frames.items())
    for name in masks(next(iter(frames.values()))):
        energy = mass = 0.0
        by_fold = {}
        for fold, frame in frames.items():
            keep = masks(frame)[name]
            selected = frame.loc[keep]
            contribution = W[fold] * selected.squared_error.sum() / len(frame)
            energy += contribution
            mass += W[fold] * keep.mean()
            daily = selected.groupby("day", observed=True).squared_error.sum().sort_values(ascending=False)
            by_fold[fold] = {**scores(selected[TARGET], selected.prediction_sec),
                "weighted_mse_contribution": float(contribution),
                "top_day_group_sse_share": float(daily.iloc[0] / daily.sum()) if len(daily) and daily.sum() else None,
                "top_three_day_group_sse_share": float(daily.head(3).sum() / daily.sum()) if len(daily) and daily.sum() else None}
        budget[name] = {"weighted_rows_fraction": float(mass), "weighted_mse_contribution": float(energy),
                        "mse_fraction": float(energy / mse), "conditional_weighted_rmse": float(np.sqrt(energy / mass)),
                        "perfect_group_counterfactual_rmse": float(np.sqrt(max(0, mse - energy))), "folds": by_fold}
    return float(mse), budget


def main():
    references, provenance = {}, {}
    for fold in W:
        folder = ROOT / "private_runs/next_230/models" / f"clock_and_rome_ensemble_{fold}_s20260910"
        record = read_json(folder / "manifest.json")
        path = folder / "score_predictions.parquet"
        assert sha256(path) == record["outputs"][path.name]
        references[fold] = pd.read_parquet(path)
        provenance[f"reference_{fold}"] = {"manifest_sha256": sha256(folder / "manifest.json"), "predictions_sha256": sha256(path)}
    baseline_mse, budget = aggregate_budget(references)
    objectives = {}
    for desired in (243.5805, 230.0):
        reduction = baseline_mse - desired**2
        actions = {}
        for name in ("missing", "finite_source_gap_gt1800", "missing_or_gap", "finite_source_gap_le1800", "ordinary_input_proxy"):
            energy = budget[name]["weighted_mse_contribution"]
            fraction = reduction / energy
            actions[name] = {"group_sse_reduction_fraction_needed_with_others_fixed": float(fraction),
                             "mathematically_sufficient_if_perfect": bool(fraction <= 1),
                             "group_error_amplitude_multiplier_needed": float(np.sqrt(max(0, 1 - fraction))) if fraction <= 1 else None}
        objectives[str(desired)] = {"target_rmse": desired, "required_weighted_mse_reduction": reduction,
            "required_total_sse_reduction_fraction": reduction / baseline_mse,
            "relative_rmse_reduction_fraction": 1 - desired / np.sqrt(baseline_mse), "group_only_scenarios": actions}
    scenarios = []
    for exceptional_reduction in (0, .25, .5, .75, 1.):
        for ordinary_reduction in (0, .10, .20, .30):
            remaining = baseline_mse - exceptional_reduction * budget["missing_or_gap"]["weighted_mse_contribution"] - ordinary_reduction * budget["finite_source_gap_le1800"]["weighted_mse_contribution"]
            scenarios.append({"exceptional_sse_reduction": exceptional_reduction, "ordinary_sse_reduction": ordinary_reduction, "counterfactual_rmse": float(np.sqrt(remaining))})

    old = ROOT / "private_runs/screening_230"
    metadata = old / "data/interim/audit/departures.parquet"
    report = read_json(old / "reports/data_audit.json")
    assert sha256(metadata) == report["artifacts"][metadata.name]
    meta = pd.read_parquet(metadata)
    sampled_profiles = {}
    for fold in W:
        idx, split = make_fold(meta.drop(columns=TARGET), load_config("configs/folds.yaml")[fold])
        profiles = {}
        for stage in ("fit", "tune", "refit", "score"):
            positions = idx[stage]
            frame = meta.iloc[positions]
            profiles[stage] = {"rows": len(frame), "id_hash": object_hash(frame[ID].tolist()), "groups": group_profile(frame)}
            if stage in ("fit", "refit"):
                sample = np.sort(np.random.default_rng(20260916).choice(positions, 200000, replace=False))
                selected = meta.iloc[sample]
                profiles[stage + "_sample200k"] = {"rows": len(selected), "id_hash": object_hash(selected[ID].tolist()), "groups": group_profile(selected)}
                profiles[stage + "_sample_retention"] = {
                    name: {"count_fraction": profiles[stage + "_sample200k"]["groups"][name]["n"] / max(1, record["n"]),
                           "correction_energy_fraction": profiles[stage + "_sample200k"]["groups"][name]["correction_energy_sec2"] / record["correction_energy_sec2"] if record["correction_energy_sec2"] else None}
                    for name, record in profiles[stage]["groups"].items()}
        sampled_profiles[fold] = {"split_hash": split["split_hash"], "profiles": profiles}

    comparisons = {}
    run_roots = [ROOT / "private_runs/campaign_20260916/models", ROOT / "private_runs/campaign_20260916/missing_mixture/models"]
    caps = []
    for run_root in run_roots:
        for manifest in sorted(run_root.glob("*/manifest.json")):
            record = read_json(manifest)
            if record["status"] != "complete":
                continue
            fold = record["name"].split("_")[-2]
            if fold not in W:
                continue
            key = record["name"].replace("_" + fold + "_", "_SCREEN_")
            reference = references[fold]
            for variant in ("candidate", "blend25"):
                path = manifest.parent / f"{variant}.parquet"
                assert sha256(path) == record["outputs"][path.name]
                pred = pd.read_parquet(path)
                assert np.array_equal(pred[ID], reference[ID]) and np.array_equal(pred[TARGET], reference[TARGET])
                item = comparisons.setdefault(key, {}).setdefault(variant, {"folds": {}, "group_delta_weighted_mse": {name: 0. for name in budget}})
                item["folds"][fold] = scores(pred[TARGET], pred.prediction_sec)
                for name, keep in masks(reference).items():
                    delta = W[fold] * (pred.loc[keep, "squared_error"].sum() - reference.loc[keep, "squared_error"].sum()) / len(pred)
                    item["group_delta_weighted_mse"][name] += float(delta)
            steps = record["tune"]["steps"]
            if isinstance(steps, int) and record["family"] in ("catboost", "lightgbm", "xgboost"):
                caps.append({"name": record["name"], "selected_steps": steps, "cap": 600, "within_last_5_percent": steps >= 570})
    for records in comparisons.values():
        for record in records.values():
            if set(record["folds"]) == set(W):
                record["seasonal_rmse"] = float(np.sqrt(sum(W[f] * m["sse"] / m["n"] for f, m in record["folds"].items())))

    result = {"created_utc": utc_now(), "baseline_weighted_mse": baseline_mse,
        "baseline_seasonal_rmse": float(np.sqrt(baseline_mse)), "objectives": objectives,
        "group_budget": budget, "counterfactual_scenarios": scenarios,
        "sample_profiles": sampled_profiles, "measured_model_group_comparisons": comparisons,
        "iteration_caps": caps, "provenance": provenance, "metadata_sha256": sha256(metadata),
        "script_sha256": sha256(__file__),
        "limits": ["Counterfactual perfect prediction is label-aware bookkeeping, not an attainable model or a noise floor.",
                   "243.5805 is a historical official ranking-population score, not a proven local-fold equivalent.",
                   "Groups using target-clock disagreement or target duration are diagnostic only and cannot define inference routes.",
                   "Sampling reduction is expected representation loss; it does not prove rare labels are learnable or a sample was biased.",
                   "Selection at the iteration cap suggests a truncated search, not proven underfitting or guaranteed benefit from larger capacity."]}
    write_json(OUT / "mechanism_audit.json", result)
    print("AUDIT", {"baseline": result["baseline_seasonal_rmse"], "required_sse_reductions": {k: v["required_total_sse_reduction_fraction"] for k, v in objectives.items()}, "models": len(comparisons)}, flush=True)


if __name__ == "__main__":
    main()
