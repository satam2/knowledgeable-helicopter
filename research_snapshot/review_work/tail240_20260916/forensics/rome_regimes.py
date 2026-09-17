"""Diagnostic-only Rome missing-source phenotype and flight morphology census."""
import os
for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[key] = "1"
import re
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pyarrow as pa
import psutil

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "review_work/campaign_20260916"))
import common
from taxiout.artifacts import read_json, write_json, sha256, utc_now
from taxiout.paths import external_path
OUT = external_path(ROOT / "private_runs/tail240_20260916/forensics/rome_regimes_v1")
pa.set_cpu_count(1)
pa.set_io_thread_count(1)
ID, TARGET, TIME = common.ID, common.TARGET, common.MOVEMENT

def t(series):
    return pd.to_datetime(series, utc=True, errors="coerce").astype("datetime64[ns, UTC]")

def main():
    OUT.mkdir(parents=True, exist_ok=False)
    write_json(OUT / "protocol.json", {"created_utc": utc_now(), "source_sha256": sha256(__file__),
        "scope": "Label-defined phenotype exploration, no prediction/routing claims.",
        "fixed_phenotypes": "near_schedule=abs(Y-(T-S))<=60; ordinary=0<=Y<=3600; day_plus_short=Y>=86400 and0<=Ymod86400<=3600; other retained.",
        "observables": "Flight prefix morphology, missing metadata, stand/runway, schedule date/gap, same-flight schedule collisions, nearby row source-date order.",
        "comparisons": "All twelve months separately; never fit a score-label rule."})
    frozen = read_json(ROOT / "private_runs/submission_v2/protocol.json")
    metadata = ROOT / "private_runs/screening_230/data/interim/audit/departures.parquet"
    assert sha256(metadata) == read_json(ROOT / "private_runs/screening_230/reports/data_audit.json")["artifacts"][metadata.name]
    meta = pd.read_parquet(metadata, columns=[ID, TARGET])
    rows = []
    fields = [ID, TIME, "SCHED_TIME_UTC_mvt", "FLIGHT_mvt", "ADEP_mvt", "ADES_mvt", "STAND_mvt", "RUNWAY_mvt", "AIRCRAFT_TYPE_mvt", "FLIGHT_RULE_mvt", "AOBT_3_flt", "FLIGHT_ID_mvt"]
    for path in sorted(common.RAW.glob("training_*.parquet")):
        assert sha256(path) == frozen["raw_hashes"][path.name]
        part = pd.read_parquet(path, columns=fields, filters=[("PHASE_mvt", "=", "DEP"), ("ADEP_mvt", "=", "LIRF")], dtype_backend="pyarrow")
        rows.append(part)
    data = pd.concat(rows, ignore_index=True).set_index(ID)
    data[TARGET] = meta.set_index(ID).loc[data.index, TARGET]
    data["month"] = t(data[TIME]).dt.strftime("%Y-%m")
    data["missing_nm"] = t(data.AOBT_3_flt).isna()
    data["takeoff"] = t(data[TIME])
    data["scheduled"] = t(data.SCHED_TIME_UTC_mvt)
    data["schedule_gap_sec"] = (data.takeoff - data.scheduled).dt.total_seconds()
    data["schedule_day_delta"] = (data.takeoff.dt.normalize() - data.scheduled.dt.normalize()).dt.total_seconds() / 86400
    data["schedule_hhmm"] = data.scheduled.dt.hour * 60 + data.scheduled.dt.minute
    data["flight"] = data.FLIGHT_mvt.fillna("").astype("string").str.upper().str.replace(r"\s+", "", regex=True)
    data["prefix_length"] = data.flight.str.extract(r"^([A-Z]*)", expand=False).str.len()
    data["prefix3"] = data.flight.str[:3]
    data["prefix_extra_last1"] = data.flight.str.len().ge(4) & data.flight.str[3].eq(data.flight.str[2])
    data["prefix_extra_last2"] = data.flight.str.len().ge(5) & data.flight.str[3:5].eq(data.flight.str[1:3])
    data["aircraft_missing"] = data.AIRCRAFT_TYPE_mvt.isna()
    data["stand_missing"] = data.STAND_mvt.isna()
    data["near_schedule_diagnostic"] = (data[TARGET] - data.schedule_gap_sec).abs().le(60)
    data["ordinary_diagnostic"] = data[TARGET].between(0, 3600)
    data["day_plus_short_diagnostic"] = data[TARGET].ge(86400) & data[TARGET].mod(86400).between(0, 3600)
    data["schedule_bin"] = pd.cut(data.schedule_gap_sec, [-np.inf,0,3600,7200,14400,43200,72000,100800,np.inf], right=False).astype(str)
    data["schedule_collision_count"] = data.groupby(["month", "flight", "scheduled"], dropna=False)["flight"].transform("size")
    data["day_flight_count"] = data.groupby(["month", "flight", data.takeoff.dt.date], dropna=False)["flight"].transform("size")
    missing = data.loc[data.missing_nm].copy()
    missing.reset_index().to_parquet(OUT / "rome_missing.parquet", index=False)
    diagnostics = []
    groupings = [["month"], ["month", "aircraft_missing"], ["month", "prefix_length"],
                 ["month", "prefix_extra_last1"], ["month", "prefix_extra_last2"],
                 ["month", "schedule_bin"], ["month", "RUNWAY_mvt"],
                 ["month", "aircraft_missing", "schedule_bin"], ["month", "schedule_collision_count"]]
    for groups in groupings:
        for keys, subset in missing.groupby(groups, dropna=False, observed=True):
            if not isinstance(keys, tuple):
                keys = (keys,)
            diagnostics.append({"grouping": "|".join(groups), **{name: str(value) for name,value in zip(groups,keys)},
                "n": len(subset), "near_schedule": int(subset.near_schedule_diagnostic.sum()),
                "ordinary": int(subset.ordinary_diagnostic.sum()), "day_plus_short": int(subset.day_plus_short_diagnostic.sum()),
                "mean_target": float(subset[TARGET].mean()), "median_target": float(subset[TARGET].median())})
    pd.DataFrame(diagnostics).to_csv(OUT / "regime_counts.csv", index=False)
    special = missing.loc[missing.day_plus_short_diagnostic | missing[TARGET].ge(20000)].copy()
    special.reset_index().to_csv(OUT / "large_cases.csv", index=False)
    write_json(OUT / "manifest.json", {"status": "complete", "source_sha256": sha256(__file__),
        "all_rome_rows": len(data), "missing_rome_rows": len(missing),
        "raw_hashes": frozen["raw_hashes"], "labels_diagnostic_only": True,
        "peak_bytes": getattr(psutil.Process().memory_info(), "peak_wset", psutil.Process().memory_info().rss),
        "outputs": {p.name:sha256(p) for p in OUT.iterdir() if p.is_file() and p.name != "manifest.json"}})
    print(pd.DataFrame(diagnostics).query("grouping == 'month'").to_string(index=False))
    print("LARGE CASES", special[["month", "flight", "aircraft_missing", "prefix_length", "prefix_extra_last1", "prefix_extra_last2", "schedule_gap_sec", TARGET, "schedule_collision_count"]].to_string())

if __name__ == "__main__":
    main()
