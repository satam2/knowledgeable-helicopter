"""Frozen-composition tail diagnosis; labels are diagnostic, never feature inputs."""
import os
for variable in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[variable] = "1"
import gc
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
import psutil
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "review_work/campaign_20260916"))
import common
from taxiout.artifacts import read_json, write_json, sha256, object_hash, utc_now
from taxiout.paths import external_path

OUT = external_path(ROOT / "private_runs/tail240_20260916/forensics/initial_v1")
RAW = common.RAW
ID, TARGET, TIME = common.ID, common.TARGET, common.MOVEMENT
BASE = ROOT / "private_runs/breakthrough_20260916/missing/route_composition_v4"
FIELDS = [ID, "PHASE_mvt", TIME, "SCHED_TIME_UTC_mvt", "FLIGHT_mvt", "ADEP_mvt", "ADES_mvt",
          "STAND_mvt", "RUNWAY_mvt", "AIRCRAFT_TYPE_mvt", "FLIGHT_RULE_mvt", "FLIGHT_ID_mvt",
          "AOBT_3_flt", "IOBT_flt", "LOBT_flt", "EOBT_1_flt", "CALLSIGN_flt", "ADEP_flt", "ADES_flt"]
pa.set_cpu_count(1)
pa.set_io_thread_count(1)

def normalized(series):
    return series.fillna("").astype("string[pyarrow]").str.upper().str.replace(r"\s+", "", regex=True)

def times(series):
    return pd.to_datetime(series, utc=True, errors="coerce").astype("datetime64[ns, UTC]")

def check_resources():
    mem = psutil.Process().memory_info()
    peak = getattr(mem, "peak_wset", mem.rss)
    assert peak < 6 * 1024**3 and psutil.virtual_memory().available > 8 * 1024**3
    return peak

def main():
    began = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=False)
    protocol = {"source_sha256": sha256(__file__), "created_utc": utc_now(),
        "baseline": str(BASE / "global9"), "scope": "Posthoc exposed-score diagnostic only; no prediction experiment or routing rule.",
        "selection": "Top25 squared errors per fold and NM-missing/present route, unioned; all-year labels only for mechanism descriptions.",
        "controls": "Same airport/route, excluding query; observable dissimilarity only, pick30nearest before inspecting labels. Also label-selected abs-error<=120 controls explicitly diagnostic.",
        "distance": "Flight mismatch4; destination mismatch2; stand mismatch1; aircraft mismatch1; runway mismatch.5; schedule gap log-distance + circular scheduled-hour distance/12 + takeoff-hour distance/12.",
        "resources": "1Arrow/BLAS thread;2CPU budget;6GiB peak cap/8GiB available reserve."}
    write_json(OUT / "protocol.json", protocol)
    frozen = read_json(ROOT / "private_runs/submission_v2/protocol.json")
    meta_path = ROOT / "private_runs/screening_230/data/interim/audit/departures.parquet"
    expected_meta = read_json(ROOT / "private_runs/screening_230/reports/data_audit.json")["artifacts"][meta_path.name]
    assert sha256(meta_path) == expected_meta
    meta = pd.read_parquet(meta_path, columns=[ID, TARGET, TIME, "FLIGHT_ID_mvt", "proxy_sec"])
    verification = read_json(BASE / "verification.json")
    assert verification["status"] == "passed"
    score_parts = []
    for fold in ["F1", "F3"]:
        folder = BASE / "global9" / fold
        manifest = read_json(folder / "manifest.json")
        assert sha256(folder / "manifest.json") == verification["folds"][fold]["global9"]["manifest_sha256"]
        assert sha256(folder / "candidate.parquet") == manifest["outputs"]["candidate.parquet"]
        score = pd.read_parquet(folder / "candidate.parquet")
        indices, split, _ = common.fold_data(meta, fold, full=True)
        assert object_hash(split) == object_hash(manifest["split"])
        np.testing.assert_array_equal(score[ID], meta.iloc[indices["score"]][ID])
        np.testing.assert_array_equal(score[TARGET], meta.iloc[indices["score"]][TARGET])
        score_parts.append(score[[ID, TARGET, "prediction_sec"]].assign(fold=fold))
    scores = pd.concat(score_parts, ignore_index=True).set_index(ID)
    frames = []
    raw_receipts = {}
    for path in sorted(RAW.glob("training_*.parquet")):
        assert sha256(path) == frozen["raw_hashes"][path.name]
        raw = pd.read_parquet(path, columns=FIELDS, filters=[("PHASE_mvt", "=", "DEP")], dtype_backend="pyarrow")
        raw_receipts[path.name] = frozen["raw_hashes"][path.name]
        frames.append(raw.drop(columns="PHASE_mvt"))
        check_resources()
    raw = pd.concat(frames, ignore_index=True).set_index(ID)
    del frames
    assert raw.index.is_unique
    np.testing.assert_array_equal(raw.index, meta[ID])
    raw[TARGET] = meta[TARGET].to_numpy(float)
    raw["proxy_sec"] = meta.proxy_sec.to_numpy(float)
    raw["missing_nm"] = ~np.isfinite(raw.proxy_sec)
    raw["flight_normalized"] = normalized(raw.FLIGHT_mvt)
    raw["month"] = times(raw[TIME]).dt.strftime("%Y-%m")
    raw["movement_time"] = times(raw[TIME])
    raw["schedule_time"] = times(raw.SCHED_TIME_UTC_mvt)
    raw["schedule_gap_sec"] = (raw.movement_time - raw.schedule_time).dt.total_seconds()
    raw["schedule_gap_log"] = np.sign(raw.schedule_gap_sec) * np.log1p(raw.schedule_gap_sec.abs())
    raw["schedule_hour"] = raw.schedule_time.dt.hour + raw.schedule_time.dt.minute / 60
    raw["movement_hour"] = raw.movement_time.dt.hour + raw.movement_time.dt.minute / 60
    raw["schedule_date_delta"] = (raw.movement_time.dt.normalize() - raw.schedule_time.dt.normalize()).dt.total_seconds() / 86400
    # Label-derived fields remain only in diagnostic outputs.
    raw["hidden_block_diagnostic"] = raw.movement_time - pd.to_timedelta(raw[TARGET], unit="s")
    raw["hidden_block_minus_schedule_diagnostic_sec"] = (raw.hidden_block_diagnostic - raw.schedule_time).dt.total_seconds()
    raw["target_day_count_diagnostic"] = np.floor(raw[TARGET] / 86400)
    raw["target_mod_day_diagnostic_sec"] = raw[TARGET] % 86400
    scored = raw.loc[scores.index].join(scores.drop(columns=TARGET), validate="one_to_one")
    scored["error_sec"] = scored.prediction_sec - scored[TARGET]
    scored["squared_error"] = scored.error_sec**2
    picks = []
    for _, group in scored.groupby(["fold", "missing_nm"], observed=True):
        picks.extend(group.nlargest(25, "squared_error").index)
    cases = scored.loc[picks].sort_values("squared_error", ascending=False)
    cases.reset_index().to_parquet(OUT / "cases.parquet", index=False)
    cases.reset_index().drop(columns=[c for c in cases if c.endswith("_flt")], errors="ignore").to_csv(OUT / "cases.csv", index=False)
    keys = set(zip(cases.ADEP_mvt, cases.flight_normalized))
    history = raw.loc[[key in keys for key in zip(raw.ADEP_mvt, raw.flight_normalized)]].copy()
    history.reset_index().to_parquet(OUT / "flight_history.parquet", index=False)
    tails = raw.loc[raw[TARGET].abs().gt(7200)].copy()
    tails.reset_index().to_parquet(OUT / "all_year_tail_diagnostics.parquet", index=False)
    neighbors = []
    for identity, case in cases.iterrows():
        pool = scored.loc[scored.ADEP_mvt.eq(case.ADEP_mvt) & scored.missing_nm.eq(case.missing_nm) & scored.index.to_series().ne(identity)].copy()
        distance = pd.Series(0., index=pool.index)
        for field, weight in [("flight_normalized", 4), ("ADES_mvt", 2), ("STAND_mvt", 1), ("AIRCRAFT_TYPE_mvt", 1), ("RUNWAY_mvt", .5)]:
            a = normalized(pool[field])
            b = "" if pd.isna(case[field]) else str(case[field]).upper().replace(" ", "")
            distance += a.ne(b).astype(float) * weight
        distance += (pool.schedule_gap_log - case.schedule_gap_log).abs().fillna(2)
        for field in ["schedule_hour", "movement_hour"]:
            delta = (pool[field] - case[field]).abs()
            distance += np.minimum(delta, 24 - delta).fillna(12) / 12
        pool["observable_distance"] = distance
        for kind, selected in [("observable_nearest", pool.nsmallest(30, "observable_distance")),
                               ("correct_prediction_diagnostic", pool.loc[pool.error_sec.abs().le(120)].nsmallest(10, "observable_distance"))]:
            selected = selected.copy()
            selected["case_id"] = identity
            selected["control_kind"] = kind
            neighbors.append(selected)
    pd.concat(neighbors).reset_index().to_parquet(OUT / "matched_neighbors.parquet", index=False)
    monthly = []
    for (month, airport, missing), group in raw.groupby(["month", "ADEP_mvt", "missing_nm"], observed=True):
        monthly.append({"month": month, "airport": airport, "missing_nm": bool(missing), "n": len(group),
            "over2h": int(group[TARGET].gt(7200).sum()), "over12h": int(group[TARGET].gt(43200).sum()),
            "over24h": int(group[TARGET].gt(86400).sum()), "negative": int(group[TARGET].lt(0).sum()),
            "over24h_mod_short": int((group[TARGET].gt(86400) & group.target_mod_day_diagnostic_sec.lt(3600)).sum()),
            "schedule_day_delta": {str(k): int(v) for k, v in group.schedule_date_delta.value_counts(dropna=False).items()}})
    write_json(OUT / "monthly_tail_counts.json", monthly)
    write_json(OUT / "manifest.json", {"status": "complete", "source_sha256": sha256(__file__),
        "protocol_sha256": sha256(OUT / "protocol.json"), "raw_hashes": raw_receipts, "metadata_sha256": expected_meta,
        "baseline_verification_sha256": sha256(BASE / "verification.json"), "case_rows": len(cases),
        "history_rows": len(history), "tail_rows": len(tails), "peak_bytes": check_resources(),
        "elapsed_sec": time.monotonic() - began, "labels_diagnostic_only": True,
        "outputs": {path.name: sha256(path) for path in OUT.iterdir() if path.is_file() and path.name != "manifest.json"}})
    print("FORENSIC_CASES", len(cases), "HISTORY", len(history), "TAILS", len(tails), flush=True)
    print(cases[["fold", "ADEP_mvt", "FLIGHT_mvt", "missing_nm", TARGET, "prediction_sec", "schedule_gap_sec", "schedule_date_delta", "target_mod_day_diagnostic_sec"]].head(20).to_string(), flush=True)

if __name__ == "__main__":
    main()
