"""Private aggregate schema/source diagnostics; raw files remain in place."""
import hashlib
import json
from pathlib import Path
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "knowledgeable-helicopter-screening/src"))
from taxiout.paths import external_path
from taxiout.schema import ID, FLIGHT_ID, MOVEMENT, BLOCK, TARGET, PHASE

OUT = external_path(ROOT / "private_runs/mechanism_20260916/information")
RAW = external_path(ROOT / "data/09-15-2026-18-55-03_files_list")
USED = {"ADEP_mvt", "RUNWAY_mvt", "STAND_mvt", "ADES_mvt", "AIRCRAFT_TYPE_mvt", "AIRCRAFT_OPERATOR_flt", "WK_TBL_CAT_flt", "MARKET_SEGMENT_flt", "FLIGHT_TYPE_flt", MOVEMENT, "AOBT_3_flt", "EOBT_1_flt", "IOBT_flt", "LOBT_flt", "SCHED_TIME_UTC_mvt", PHASE}


def seconds(series):
    return pd.to_datetime(series, utc=True).dt.as_unit("ns").astype("int64").to_numpy(dtype=float) / 1e9


def stats(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"n": 0}
    return {"n": len(values), "mean": float(values.mean()), "rmse": float(np.sqrt(np.mean(values**2))),
        "mae": float(np.abs(values).mean()), "median": float(np.median(values)),
        "abs_q50": float(np.quantile(np.abs(values), .5)), "abs_q90": float(np.quantile(np.abs(values), .9)),
        "abs_q99": float(np.quantile(np.abs(values), .99)), "exact_pct": float(100*np.mean(values == 0)),
        "within60_pct": float(100*np.mean(np.abs(values) <= 60))}


def delta(a, b):
    return (a-b).dt.total_seconds().to_numpy(float)


def metric_subset(predicted, actual, mask):
    return stats(np.asarray(predicted)[mask]-np.asarray(actual)[mask])


def finite_json(value):
    if isinstance(value, dict):
        return {k: finite_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [finite_json(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def id_diagnostics(raw):
    airport = raw.ADEP_mvt.where(raw[PHASE].eq("DEP"), raw.ADES_mvt)
    results = []
    for airport_code, group in raw.groupby(airport, observed=True):
        for month, part in group.groupby(group[MOVEMENT].dt.strftime("%Y-%m"), observed=True):
            dep = part.loc[part[PHASE].eq("DEP")].copy()
            arr = part.loc[part[PHASE].eq("ARR") & part[BLOCK].notna()].copy()
            record = {"airport": str(airport_code), "month": str(month), "departures": len(dep), "arrivals": len(arr)}
            if not len(dep):
                continue
            for identity in (ID, FLIGHT_ID):
                fit = dep.loc[dep[identity].notna() & dep[BLOCK].notna()]
                if len(fit) > 2:
                    record[identity + "_spearman_takeoff"] = float(fit[identity].corr(pd.Series(seconds(fit[MOVEMENT]), index=fit.index), method="spearman"))
                    record[identity + "_spearman_block"] = float(fit[identity].corr(pd.Series(seconds(fit[BLOCK]), index=fit.index), method="spearman"))
                    ordered = fit.sort_values(identity)
                    db = np.diff(seconds(ordered[BLOCK]))
                    dt = np.diff(seconds(ordered[MOVEMENT]))
                    record[identity + "_adjacent_block_reverse_pct"] = float(100*np.mean(db < 0))
                    record[identity + "_adjacent_takeoff_reverse_pct"] = float(100*np.mean(dt < 0))
                    record[identity + "_adjacent_block_abs_median_sec"] = float(np.median(np.abs(db)))
                    record[identity + "_adjacent_takeoff_abs_median_sec"] = float(np.median(np.abs(dt)))
            arr = arr.sort_values(ID).drop_duplicates(ID)
            if len(arr) >= 2:
                aids = arr[ID].to_numpy(float)
                ablock = seconds(arr[BLOCK])
                dids = dep[ID].to_numpy(float)
                loc = np.searchsorted(aids, dids)
                inside = (loc > 0) & (loc < len(arr))
                positions = np.flatnonzero(inside)
                if not len(positions):
                    record["arrival_id_brackets"] = {"rows": 0, "coverage_pct": 0.0}
                    results.append(record)
                    continue
                lo, hi = loc[inside]-1, loc[inside]
                fraction = (dids[inside]-aids[lo])/(aids[hi]-aids[lo])
                prediction = ablock[lo] + fraction*(ablock[hi]-ablock[lo])
                takeoff = seconds(dep[MOVEMENT])[inside]
                have_label = dep[BLOCK].notna().to_numpy()[inside]
                block = seconds(dep[BLOCK])[inside]
                record["arrival_id_brackets"] = {"rows": len(positions), "coverage_pct": float(100*inside.mean()),
                    "forward_block_order_pct": float(100*np.mean(ablock[hi] >= ablock[lo])),
                    "both_arrivals_complete_before_takeoff_pct": float(100*np.mean(np.maximum(ablock[lo], ablock[hi]) < takeoff)),
                    "block_span_sec": stats(ablock[hi]-ablock[lo]),
                    "diagnostic_interpolated_block_error_sec": metric_subset(prediction, block, have_label),
                    "bracket_contains_hidden_block_pct": float(100*np.mean((block[have_label] >= np.minimum(ablock[lo], ablock[hi])[have_label]) & (block[have_label] <= np.maximum(ablock[lo], ablock[hi])[have_label]))) if have_label.any() else None}
                slices = {}
                for bound in (300, 900, 1800, 3600):
                    usable = (ablock[hi] >= ablock[lo]) & (ablock[hi]-ablock[lo] <= bound)
                    slices[str(bound)] = {"rows": int(usable.sum()), "coverage_pct_all_departures": float(100*usable.sum()/len(dep)),
                        "diagnostic_block_error": metric_subset(prediction, block, usable & have_label)}
                record["arrival_id_brackets"]["fixed_forward_span_limits_sec"] = slices
            results.append(record)
    return results


def other_diagnostics(raw):
    dep = raw.loc[raw[PHASE].eq("DEP")].copy()
    missing = dep.AOBT_3_flt.isna()
    result = {"departures": len(dep), "missing_aobt": int(missing.sum()), "missing_nm_fields": {col: int(dep.loc[missing, col].notna().sum()) for col in dep if col.endswith("_flt") or col in [FLIGHT_ID, "FLIGHT_mvt", "SCHED_TIME_UTC_mvt"]}}
    result["diverted_filed_destination"] = int((dep.ADES_FILED_flt.notna() & dep.ADES_flt.notna() & dep.ADES_FILED_flt.ne(dep.ADES_flt)).sum())
    result["arvt3_minus_arvt1_sec"] = stats(delta(dep.ARVT_3_flt, dep.ARVT_1_flt))
    result["arvt3_after_takeoff_pct"] = float(100*(dep.ARVT_3_flt > dep[MOVEMENT]).sum()/max(1, dep.ARVT_3_flt.notna().sum()))
    result["actual_minus_planned_duration_sec"] = stats(delta(dep.ARVT_3_flt, dep.AOBT_3_flt)-delta(dep.ARVT_1_flt, dep.EOBT_1_flt))
    for col, pattern in [("FLIGHT_mvt", r"^([A-Z]{2,3})(?=[0-9])"), ("CALLSIGN_flt", r"^([A-Z]{3})(?=[0-9])")]:
        text = dep[col].astype("string")
        prefix = text.str.extract(pattern, expand=False)
        result[col + "_parse"] = {"present": int(text.notna().sum()), "unique": int(text.nunique()), "prefix_parsed": int(prefix.notna().sum()), "prefix_unique": int(prefix.nunique()), "prefix_parsed_missing_aobt": int(prefix[missing].notna().sum())}
    arr = raw.loc[raw[PHASE].eq("ARR") & raw[FLIGHT_ID].notna()]
    unique_arr = arr.drop_duplicates(FLIGHT_ID, keep=False)
    matched = dep.loc[dep[FLIGHT_ID].notna()].merge(unique_arr[[FLIGHT_ID, ID, MOVEMENT, BLOCK, "ARVT_3_flt", "AOBT_3_flt", "ADES_mvt"]], on=FLIGHT_ID, suffixes=("_dep", "_arr"), how="inner", validate="many_to_one")
    result["same_nm_flight_arrival"] = {"rows": len(matched), "departure_coverage_pct": float(100*len(matched)/len(dep)),
        "join_scope": "within supplied file; unique arrival FLIGHT_ID only; no aircraft-rotation claim", "duplicate_arrival_flight_rows_excluded": len(arr)-len(unique_arr)}
    if len(matched):
        result["same_nm_flight_arrival"].update({
            "arvt3_minus_landing_sec": stats(delta(matched.ARVT_3_flt_dep, matched[MOVEMENT+"_arr"])),
            "arvt3_minus_inblock_sec": stats(delta(matched.ARVT_3_flt_dep, matched[BLOCK+"_arr"])),
            "aobt_dep_arr_equal_pct": float(100*matched.AOBT_3_flt_dep.eq(matched.AOBT_3_flt_arr).mean()),
            "arrival_after_departure_takeoff_pct": float(100*matched[MOVEMENT+"_arr"].gt(matched[MOVEMENT+"_dep"]).mean()),
            "missing_aobt_recovered": int((matched.AOBT_3_flt_dep.isna() & matched.AOBT_3_flt_arr.notna()).sum()),
            "destination_matches_pct": float(100*matched.ADES_mvt_dep.eq(matched.ADES_mvt_arr).mean())})
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((ROOT/"private_runs/submission_v2/protocol.json").read_text())
    receipts = json.loads((ROOT/"output/research_20260916/sources.json").read_text())
    summary = {"created_utc": datetime.now(timezone.utc).isoformat(), "read_only_raw": True, "network_requests": 0,
        "source_receipts_reused": {k: receipts[k] for k in ["competition_data", "competition_eligibility"]},
        "scope": "Information availability and label-aware forensic diagnostics only; no model or score-cohort improvement claim.",
        "base_raw_fields_used": sorted(USED), "packs": []}
    for path in sorted(RAW.glob("*.parquet")):
        if path.name == "submitting.parquet":
            continue
        start = time.monotonic()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == protocol["raw_hashes"][path.name]
        raw = pq.read_table(path, use_threads=False).to_pandas()
        for col in raw:
            if pd.api.types.is_datetime64_any_dtype(raw[col]):
                raw[col] = pd.to_datetime(raw[col], utc=True)
        record = {"file": path.name, "sha256": digest, "rows": len(raw), "schema": {str(field.name): str(field.type) for field in pq.read_schema(path)}, "columns": {}, "by_phase": {}}
        for phase in ("DEP", "ARR"):
            part = raw.loc[raw[PHASE].eq(phase)]
            record["by_phase"][phase] = {"rows": len(part), "columns": {col: {"present": int(part[col].notna().sum()), "unique": int(part[col].nunique())} for col in raw}}
        record["diagnostics"] = other_diagnostics(raw)
        record["id_order"] = id_diagnostics(raw)
        record["elapsed_sec"] = time.monotonic()-start
        summary["packs"].append(record)
        (OUT/"information_audit.json").write_text(json.dumps(finite_json(summary), indent=2, allow_nan=False), encoding="utf-8")
        print(path.name, record["diagnostics"]["departures"], round(record["elapsed_sec"], 2), flush=True)
    summary["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (OUT/"information_audit.json").write_text(json.dumps(finite_json(summary), indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
