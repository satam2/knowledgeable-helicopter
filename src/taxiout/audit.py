from collections import Counter

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from taxiout.artifacts import environment, object_hash, sha256, utc_now, write_json
from taxiout.availability import assert_observations, make_observations
from taxiout.config import ROOT
from taxiout.paths import artifact_path, raw_root
from taxiout.io import concat_frames, read_raw, training_paths
from taxiout.schema import BLOCK, CATEGORIES, FLIGHT_ID, ID, MOVEMENT, PHASE, TARGET, duration, target_identity, unique_ids, utc_period_strings


def audit(config):
    paths = training_paths(config)
    boundaries = pd.date_range("2025-01-01", "2026-01-01", freq="MS")
    expected_files = {f"training_{start:%Y-%m-%d}_{end:%Y-%m-%d}.parquet" for start, end in zip(boundaries[:-1], boundaries[1:])}
    if {p.name for p in paths} != expected_files:
        raise ValueError("Expected the twelve official 2025 training partitions; verify a changed pack before proceeding")
    root = raw_root(config)
    artifact_path("reports").mkdir(parents=True, exist_ok=True)
    manifest = {"created_utc": utc_now(), "files": []}
    phases, missing, departure_missing, vocabulary = Counter(), Counter(), Counter(), {c: set() for c in CATEGORIES}
    frames, all_ids, anomalies = [], [], []
    unlabelled_arrivals = 0
    file_boundary_spillovers = []
    for path in [*paths, root / "ranking.parquet", root / "submitting.parquet"]:
        schema = pq.read_schema(path)
        manifest["files"].append({"file": path.name, "sha256": sha256(path), "bytes": path.stat().st_size,
                                  "rows": pq.read_metadata(path).num_rows, "schema": str(schema.remove_metadata())})
        if not path.name.startswith("training_"):
            continue
        raw = read_raw(path)
        start, end = [pd.Timestamp(s, tz="UTC") for s in path.stem.split("_")[1:3]]
        spill = ~(raw[MOVEMENT].ge(start) & raw[MOVEMENT].lt(end))
        for record in raw.loc[spill, [ID, PHASE, MOVEMENT]].to_dict("records"):
            file_boundary_spillovers.append({"file": path.name, **record})
        unlabelled_arrivals += target_identity(raw)["unlabelled_arrival_context"]
        phases.update(raw[PHASE].value_counts().to_dict())
        missing.update(raw.isna().sum().to_dict())
        all_ids.append(raw[[ID]])
        obs, labels, _ = make_observations(raw)
        assert_observations(obs)
        dep = raw.loc[raw[PHASE].eq("DEP")].copy()
        departure_missing.update(dep.isna().sum().to_dict())
        dep["proxy_sec"] = duration(dep[MOVEMENT], dep["AOBT_3_flt"])
        dep["schedule_sec"] = duration(dep[MOVEMENT], dep["SCHED_TIME_UTC_mvt"])
        for col in vocabulary:
            vocabulary[col].update(dep[col].dropna().astype(str))
        suspect = dep[TARGET].le(0) | dep[TARGET].gt(7200) | dep["proxy_sec"].lt(0) | dep["proxy_sec"].gt(7200)
        flagged = dep.loc[suspect, [ID, FLIGHT_ID, "ADEP_mvt", MOVEMENT, BLOCK, TARGET, "AOBT_3_flt", "proxy_sec"]].copy()
        flagged["nonpositive_label"] = flagged[TARGET].le(0)
        flagged["over_two_hours"] = flagged[TARGET].gt(7200)
        flagged["nm_missing"] = flagged["AOBT_3_flt"].isna()
        anomalies.append(flagged)
        frames.append(dep[[ID, FLIGHT_ID, "ADEP_mvt", MOVEMENT, TARGET, "proxy_sec", "schedule_sec", "STAND_mvt", "RUNWAY_mvt"]])
        print(f"Audited {path.name}: {len(raw):,} movements", flush=True)
    unique_ids(pd.concat(all_ids, ignore_index=True))
    dep = concat_frames(frames)
    unique_ids(dep)
    ranking = read_raw(root / "ranking.parquet")
    observations, _, _ = make_observations(ranking)
    assert_observations(observations)
    rd = ranking.loc[ranking[PHASE].eq("DEP")].copy()
    ra = ranking.loc[ranking[PHASE].eq("ARR")]
    template = read_raw(root / "submitting.parquet")
    unique_ids(template)
    if list(template) != [ID, TARGET] or len(template) != len(rd) or not template[ID].isin(rd[ID]).all():
        raise ValueError("Template does not match ranking departures")
    if dep[ID].isin(ranking[ID]).any():
        raise ValueError("Training/ranking movement ID overlap")
    if rd[[BLOCK, TARGET]].notna().any().any():
        raise ValueError("Ranking contains unexpected departure labels; investigate updated contract")
    if len(ranking.columns) != len(read_raw(paths[0]).columns):
        raise ValueError("Training/ranking schema columns differ")
    missing_proxy = dep["proxy_sec"].isna()
    unseen = {c: {"new_values": len(set(rd[c].dropna().astype(str)) - vocabulary[c]),
                   "row_pct": float((~rd[c].astype(str).isin(vocabulary[c]) & rd[c].notna()).mean() * 100)} for c in vocabulary}
    pairs = set(zip(dep["ADEP_mvt"].astype(str), dep["STAND_mvt"].astype(str)))
    unknown_pairs = np.array([p not in pairs for p in zip(rd["ADEP_mvt"].astype(str), rd["STAND_mvt"].astype(str))])
    result = {
        "environment": environment(), "training_rows": sum(phases.values()), "phases": dict(phases),
        "training_departures": len(dep), "ranking_rows": len(ranking), "ranking_departures": len(rd), "template_rows": len(template),
        "airports": sorted(dep["ADEP_mvt"].unique().tolist()), "raw_columns": len(ranking.columns),
        "ranking_departure_months": pd.Series(utc_period_strings(rd[MOVEMENT], "M")).value_counts().sort_index().to_dict(),
        "target": {"mean": float(dep[TARGET].mean()), "median": float(dep[TARGET].median()),
                   "p95": float(dep[TARGET].quantile(.95)), "p99": float(dep[TARGET].quantile(.99)),
                   "max": int(dep[TARGET].max()), "negative": int(dep[TARGET].lt(0).sum()),
                   "zero": int(dep[TARGET].eq(0).sum()), "over_two_hours": int(dep[TARGET].gt(7200).sum())},
        "missing_proxy": int(missing_proxy.sum()), "missing_proxy_over_two_hours": int((missing_proxy & dep[TARGET].gt(7200)).sum()),
        "proxy_coverage_pct": float((~missing_proxy).mean() * 100), "ranking_proxy_coverage_pct": float(rd["AOBT_3_flt"].notna().mean() * 100),
        "departure_identity_max_abs_sec": 0.0, "arrival_identity_max_abs_sec": 0.0,
        "unlabelled_arrival_context": unlabelled_arrivals,
        "file_boundary_spillovers": file_boundary_spillovers,
        "template_ids_unique_and_match": True, "unseen_ranking_categories": unseen,
        "unseen_ranking_airport_stand_pct": float(unknown_pairs.mean() * 100),
        "raw_missing_counts": dict(missing), "discrepancies": ["Official prose refers to 11 airports; actual table/data have 10."],
    }
    expected = {"training_rows": 4167797, "training_departures": 2085047, "ranking_rows": 689534, "ranking_departures": 344841}
    result["prior_count_comparison"] = {k: {"observed": result[k], "prior": v, "matches": result[k] == v} for k, v in expected.items()}
    pd.DataFrame([{"field": col, "dtype": str(ranking[col].dtype),
                   "training_departure_nonnull_pct": (1 - departure_missing[col] / len(dep)) * 100,
                   "ranking_departure_nonnull_pct": float(rd[col].notna().mean() * 100),
                   "ranking_arrival_nonnull_pct": float(ra[col].notna().mean() * 100),
                   "training_category_unique": len(vocabulary[col]) if col in vocabulary else None}
                  for col in ranking]).to_csv(artifact_path("reports/field_inventory.csv"), index=False)
    out = artifact_path("data/interim/audit")
    out.mkdir(parents=True, exist_ok=True)
    pd.concat(anomalies, ignore_index=True).to_parquet(out / "anomalies.parquet", index=False)
    dep.drop(columns=["STAND_mvt", "RUNWAY_mvt"]).to_parquet(out / "departures.parquet", index=False)
    manifest["template_rows"] = len(template)
    manifest["ranking_departure_months"] = result["ranking_departure_months"]
    result["input_manifest_hash"] = object_hash({k: v for k, v in manifest.items() if k != "created_utc"})
    result["artifacts"] = {p.name: sha256(p) for p in [out / "departures.parquet", out / "anomalies.parquet"]}
    write_json(artifact_path("reports/input_manifest.json"), manifest)
    write_json(artifact_path("reports/data_audit.json"), result)
    return result
