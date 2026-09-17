"""Strict local identity linkage; targets and private block times are never read."""
import os
for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[key] = "1"
import gc
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
import psutil
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "knowledgeable-helicopter-screening/src"))
from taxiout.paths import external_path
OUT = external_path(ROOT / "private_runs/breakthrough_20260916/missing/opdi_rotation")
RAW = external_path(ROOT / "data/09-15-2026-18-55-03_files_list")
ID = "MVT_ID_mvt"
TIME = "MVT_TIME_UTC_mvt"
FEATURES = ["opdi_match", "opdi_match_ambiguous", "opdi_match_offset_sec",
            "opdi_previous_leg_available", "opdi_previous_same_airport",
            "opdi_ground_interval_sec", "opdi_previous_leg_duration_sec",
            "opdi_previous_leg_age_at_takeoff_sec"]
PUBLIC_COLS = ["id", "icao24", "flt_id", "adep", "ades", "first_seen", "last_seen"]
QUERY_COLS = [ID, TIME, "PHASE_mvt", "FLIGHT_mvt", "ADEP_mvt", "ADES_mvt", "AOBT_3_flt"]
pa.set_cpu_count(1)
pa.set_io_thread_count(1)

def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()

def normalize(series):
    return series.fillna("").astype("string[pyarrow]").str.upper().str.replace(r"\s+", "", regex=True)

def timestamps(series):
    return pd.to_datetime(series, utc=True, errors="coerce").astype("datetime64[ns, UTC]")

def build_month(query, public, month):
    query = query.copy()
    public = public.copy()
    query[TIME] = timestamps(query[TIME])
    public["first_seen"] = timestamps(public["first_seen"])
    public["last_seen"] = timestamps(public["last_seen"])
    query["cs"] = normalize(query["FLIGHT_mvt"])
    public["cs"] = normalize(public["flt_id"])
    for name in ["adep", "ades", "icao24"]:
        public[name] = normalize(public[name])
    for name in ["ADEP_mvt", "ADES_mvt"]:
        query[name] = normalize(query[name])
    # Only the declared calendar month may contribute any leg.
    public = public.loc[public["first_seen"].dt.strftime("%Y%m").eq(month)].copy()
    result = pd.DataFrame(np.nan, index=pd.Index(query[ID], name=ID), columns=FEATURES, dtype=np.float32)
    result[[FEATURES[0], FEATURES[1], FEATURES[3], FEATURES[4]]] = np.float32(0)
    candidates = public.loc[public["cs"].isin(query["cs"]) & public["cs"].ne("") & public["icao24"].ne("")]
    groups = {key: group for key, group in candidates.groupby(["cs", "adep", "ades"], sort=False)}
    matches = []
    for row in query.itertuples(index=False):
        fields = row._asdict()
        key = (fields["cs"], fields["ADEP_mvt"], fields["ADES_mvt"])
        if not all(key) or fields[TIME].strftime("%Y%m") != month:
            continue
        group = groups.get(key)
        if group is None:
            continue
        delta = (fields[TIME] - group["first_seen"]).dt.total_seconds()
        hits = group.loc[delta.abs().le(600)]
        if len(hits) > 1:
            result.loc[fields[ID], "opdi_match_ambiguous"] = 1
        elif len(hits) == 1:
            hit = hits.iloc[0]
            result.loc[fields[ID], "opdi_match"] = 1
            result.loc[fields[ID], "opdi_match_offset_sec"] = (fields[TIME] - hit["first_seen"]).total_seconds()
            matches.append((fields, hit))
    tails = {hit["icao24"] for _, hit in matches}
    legs = public.loc[public["icao24"].isin(tails)]
    by_tail = {key: group for key, group in legs.groupby("icao24", sort=False)}
    for fields, hit in matches:
        all_legs = by_tail[hit["icao24"]]
        before = all_legs.loc[all_legs["first_seen"].lt(hit["first_seen"])]
        if before["last_seen"].ge(hit["first_seen"]).any():
            continue
        completed = before.loc[before["last_seen"].lt(hit["first_seen"]) & before["last_seen"].gt(before["first_seen"])]
        if completed.empty:
            continue
        latest = completed.loc[completed["last_seen"].eq(completed["last_seen"].max())]
        if len(latest) != 1:
            continue
        previous = latest.iloc[0]
        result.loc[fields[ID], "opdi_previous_leg_available"] = 1
        if previous["ades"] != fields["ADEP_mvt"]:
            continue
        result.loc[fields[ID], "opdi_previous_same_airport"] = 1
        result.loc[fields[ID], "opdi_ground_interval_sec"] = (hit["first_seen"] - previous["last_seen"]).total_seconds()
        result.loc[fields[ID], "opdi_previous_leg_duration_sec"] = (previous["last_seen"] - previous["first_seen"]).total_seconds()
        result.loc[fields[ID], "opdi_previous_leg_age_at_takeoff_sec"] = (fields[TIME] - previous["last_seen"]).total_seconds()
    return result

def limits():
    memory = psutil.Process().memory_info()
    peak = getattr(memory, "peak_wset", memory.rss)
    assert peak < 2 * 1024**3, f"Memory exceeded2GiB: {peak}"
    assert psutil.virtual_memory().available >= 8 * 1024**3, "Host reserve below8GiB"
    return peak

def main():
    started = time.time()
    if (OUT / "manifest.json").exists():
        raise RuntimeError("Refusing to overwrite previous build")
    acquisition = json.loads((OUT / "acquisition.json").read_text())
    assert acquisition["status"] == "complete"
    metadata = ROOT / "private_runs/screening_230/data/interim/audit/departures.parquet"
    meta = pd.read_parquet(metadata, columns=[ID, "proxy_sec"])
    training_ids = pd.Index(meta.loc[~np.isfinite(meta["proxy_sec"]), ID], name=ID)
    assert len(training_ids) == 22470 and training_ids.is_unique
    del meta
    queries = []
    raw_receipts = []
    for path in sorted(RAW.glob("training_*.parquet")) + [RAW / "ranking.parquet"]:
        frame = pd.read_parquet(path, columns=QUERY_COLS, dtype_backend="pyarrow")
        frame = frame.loc[frame["PHASE_mvt"].eq("DEP") & timestamps(frame["AOBT_3_flt"]).isna()].copy()
        frame[TIME] = timestamps(frame[TIME])
        frame["split"] = "ranking" if path.name == "ranking.parquet" else "training"
        queries.append(frame.drop(columns=["AOBT_3_flt", "PHASE_mvt"]))
        raw_receipts.append({"path": str(path), "sha256": sha(path), "query_count": len(frame)})
        limits()
    query = pd.concat(queries, ignore_index=True)
    assert query[ID].is_unique
    assert set(query.loc[query["split"].eq("training"), ID]) == set(training_ids)
    ranking_ids = pd.Index(query.loc[query["split"].eq("ranking"), ID], name=ID)
    query["month"] = query[TIME].dt.strftime("%Y%m")
    chunks = []
    coverage = []
    peak = limits()
    sources = {item["month"]: item for item in acquisition["sources"]}
    for month, part in query.groupby("month", sort=True):
        assert month in sources, f"Unexpected query month {month}"
        path = Path(sources[month]["local_path"])
        assert sha(path) == sources[month]["sha256"]
        public = pd.read_parquet(path, columns=PUBLIC_COLS, dtype_backend="pyarrow")
        features = build_month(part, public, month)
        chunks.append(features)
        stats = {"month": month, "queries": len(part), "public_rows": len(public),
                 "matched": int(features["opdi_match"].sum()),
                 "ambiguous": int(features["opdi_match_ambiguous"].sum()),
                 "previous_same_airport": int(features["opdi_previous_same_airport"].sum()),
                 "airport": []}
        for airport, subset in part.groupby("ADEP_mvt", sort=True):
            ext = features.loc[subset[ID]]
            stats["airport"].append({"airport": airport, "queries": len(subset),
                                     "matched": int(ext["opdi_match"].sum()),
                                     "rotation": int(ext["opdi_previous_same_airport"].sum())})
        coverage.append(stats)
        peak = max(peak, limits())
        print("OPDI_FEATURES", json.dumps(stats), flush=True)
        del public, features
        gc.collect()
    all_features = pd.concat(chunks)
    assert all_features.index.is_unique and set(all_features.index) == set(query[ID])
    outputs = {}
    for name, ids in [("training_features.parquet", training_ids), ("ranking_features.parquet", ranking_ids)]:
        frame = all_features.loc[ids].astype(np.float32).reset_index()
        frame.to_parquet(OUT / name, index=False)
        outputs[name] = sha(OUT / name)
    manifest = {"status": "complete", "features": FEATURES, "outputs": outputs,
                "source_sha256": sha(__file__), "acquisition_sha256": sha(OUT / "acquisition.json"),
                "metadata_sha256": sha(metadata), "raw_inputs": raw_receipts, "coverage": coverage,
                "training_rows": len(training_ids), "ranking_rows": len(ranking_ids),
                "peak_bytes": peak, "elapsed_sec": time.time() - started,
                "availability": "Explicit retrospective public monthly flight lists. Month-isolated previous completed leg; no private targets or block times read.",
                "license_scope": "Local noncommercial attributed research; competition prize eligibility not established."}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("COMPLETE", len(training_ids), len(ranking_ids), peak, flush=True)

if __name__ == "__main__":
    main()
