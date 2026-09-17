"""Outcome-blind public flight enumeration for Rome missing-clock diagnostics."""
import os
for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[key] = "1"
import sys
import gc
from pathlib import Path
import pandas as pd
import numpy as np
import pyarrow as pa
import psutil
from run_lexical import lexical_features, decode_flight_tokens

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "review_work/campaign_20260916"))
import common
from taxiout.artifacts import read_json, write_json, sha256, utc_now
OUT = ROOT / "private_runs/tail240_20260916/forensics/opdi_days_v1"
INPUT = ROOT / "private_runs/tail240_20260916/forensics/rome_regimes_v1"
PUBLIC = ROOT / "private_runs/breakthrough_20260916/missing/opdi_rotation"
ID, TARGET, TIME = common.ID, common.TARGET, common.MOVEMENT
pa.set_cpu_count(1)
pa.set_io_thread_count(1)


def norm(s):
    return s.fillna("").astype("string").str.upper().str.replace(r"\s+", "", regex=True)


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    write_json(OUT / "protocol.json", {"created_utc": utc_now(), "source_sha256": sha256(__file__),
        "scope": "Enumerate every public flight candidate matching hypothetical canonical code and both airports within plus/minus30h. Own movement calendar month only. No label enters enumeration or selection.",
        "selection": "No winner selected; retain all candidates and ambiguity. Public first_seen is not a gate clock or definitive takeoff timestamp.",
        "diagnostic": "Read outcomes only after immutable outcome-blind candidate table is written. Compare near-zero and plus/minus1day matches at fixed600sec tolerance.",
        "resources": "1CPU,6GiB process cap,8GiB reserve."})
    path = INPUT / "rome_missing.parquet"
    assert sha256(path) == read_json(INPUT / "manifest.json")["outputs"][path.name]
    q = pd.read_parquet(path, columns=[ID,TIME,"SCHED_TIME_UTC_mvt","FLIGHT_mvt","ADEP_mvt","ADES_mvt"])
    q[TIME] = pd.to_datetime(q[TIME], utc=True)
    q["SCHED_TIME_UTC_mvt"] = pd.to_datetime(q.SCHED_TIME_UTC_mvt, utc=True)
    q["canonical"] = decode_flight_tokens(lexical_features(q.FLIGHT_mvt).lex_canonical_flight)
    q["month"] = q[TIME].dt.strftime("%Y%m")
    sources = {r["month"]: r for r in read_json(PUBLIC / "acquisition.json")["sources"]}
    records, coverage = [], []
    for month, rows in q.groupby("month", sort=True):
        item = sources[month]
        public_path = Path(item["local_path"])
        assert sha256(public_path) == item["sha256"]
        p = pd.read_parquet(public_path, columns=["id","icao24","flt_id","adep","ades","first_seen","last_seen"], dtype_backend="pyarrow")
        p = p.loc[norm(p.adep).eq("LIRF")].copy()
        for col in ["first_seen","last_seen"]:
            p[col] = pd.to_datetime(p[col], utc=True).astype("datetime64[ns, UTC]")
        p = p.loc[p.first_seen.dt.strftime("%Y%m").eq(month)].copy()
        p["canonical"] = decode_flight_tokens(lexical_features(p.flt_id).lex_canonical_flight)
        p["dest"] = norm(p.ades)
        groups = {key: value for key,value in p.groupby(["canonical","dest"], sort=False)}
        before = len(records)
        for row in rows.itertuples(index=False):
            f = row._asdict()
            key = (f["canonical"], "" if pd.isna(f["ADES_mvt"]) else str(f["ADES_mvt"]).upper())
            if not all(key) or key not in groups:
                continue
            candidates = groups[key]
            delta = (f[TIME] - candidates.first_seen).dt.total_seconds()
            candidates = candidates.loc[delta.abs().le(30*3600)]
            for _, candidate in candidates.iterrows():
                records.append({ID:f[ID],"month":month,"query_flight":f["FLIGHT_mvt"],"canonical":f["canonical"],
                    "query_time":f[TIME],"query_schedule":f["SCHED_TIME_UTC_mvt"],"destination":key[1],
                    "public_id":candidate.id,"public_flight":candidate.flt_id,"public_icao24":candidate.icao24,
                    "first_seen":candidate.first_seen,"last_seen":candidate.last_seen,
                    "movement_minus_first_sec":(f[TIME]-candidate.first_seen).total_seconds(),
                    "schedule_minus_first_sec":(f["SCHED_TIME_UTC_mvt"]-candidate.first_seen).total_seconds(),
                    "candidate_count":len(candidates)})
        coverage.append({"month":month,"queries":len(rows),"candidates":len(records)-before})
        peak = getattr(psutil.Process().memory_info(), "peak_wset", psutil.Process().memory_info().rss)
        assert peak < 6*1024**3 and psutil.virtual_memory().available > 8*1024**3
        print("ENUMERATED", coverage[-1], flush=True)
        del p, groups
        gc.collect()
    candidates = pd.DataFrame(records)
    candidates.to_parquet(OUT / "outcome_blind_candidates.parquet", index=False)
    write_json(OUT / "enumeration_receipt.json", {"completed_utc": utc_now(), "sha256":sha256(OUT / "outcome_blind_candidates.parquet"), "coverage":coverage, "target_read":False})
    labels = pd.read_parquet(path, columns=[ID,TARGET,"schedule_gap_sec","near_schedule_diagnostic","ordinary_diagnostic","day_plus_short_diagnostic"])
    diagnostic = candidates.merge(labels,on=ID,how="left",validate="many_to_one")
    diagnostic["near_supplied_movement"] = diagnostic.movement_minus_first_sec.abs().le(600)
    diagnostic["near_previous_day"] = diagnostic.movement_minus_first_sec.sub(86400).abs().le(600)
    diagnostic["near_next_day"] = diagnostic.movement_minus_first_sec.add(86400).abs().le(600)
    diagnostic.to_parquet(OUT / "candidate_diagnostics.parquet",index=False)
    diagnostic.loc[diagnostic.day_plus_short_diagnostic].to_csv(OUT / "day_plus_cases.csv",index=False)
    grouped = diagnostic.groupby(ID).agg(candidate_count=("public_id","size"),same_day=("near_supplied_movement","sum"),previous_day=("near_previous_day","sum"),next_day=("near_next_day","sum"))
    table = labels.merge(grouped,on=ID,how="left").fillna({"candidate_count":0,"same_day":0,"previous_day":0,"next_day":0})
    table.to_csv(OUT / "query_diagnostics.csv",index=False)
    write_json(OUT / "manifest.json", {"status":"complete","source_sha256":sha256(__file__),"peak_bytes":peak,
        "queries":len(q),"candidate_rows":len(candidates),"matched_queries":candidates[ID].nunique(),
        "outputs":{p.name:sha256(p) for p in OUT.iterdir() if p.is_file()}})
    print(table.groupby(["day_plus_short_diagnostic","near_schedule_diagnostic"])[["candidate_count","same_day","previous_day","next_day"]].agg(["count","sum"]).to_string())
    print(diagnostic.loc[diagnostic.day_plus_short_diagnostic,[ID,"query_flight",TARGET,"schedule_gap_sec","movement_minus_first_sec","schedule_minus_first_sec","candidate_count"]].to_string(index=False))


if __name__ == "__main__":
    main()
