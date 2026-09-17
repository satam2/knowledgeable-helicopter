"""Earlier-month physical taxi references for source-offset diagnostics only."""
import os
for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[key] = "1"
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import psutil
from run_lexical import lexical_features, decode_flight_tokens

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "review_work/campaign_20260916"))
import common
from taxiout.artifacts import read_json, write_json, sha256, utc_now, object_hash
ID, TARGET, TIME = common.ID, common.TARGET, common.MOVEMENT
OUT = ROOT / "private_runs/tail240_20260916/forensics/dst_v1"
INPUT = ROOT / "private_runs/tail240_20260916/forensics/rome_regimes_v1"
pa.set_cpu_count(1)
pa.set_io_thread_count(1)


def reference(history, query):
    if len(history) and history[TIME].max() >= query[TIME].min():
        raise ValueError("Reference history must precede every query")
    global_median = float(history[TARGET].median()) if len(history) else 900.
    result = pd.DataFrame({"physical_reference":global_median,"reference_n":len(history),"reference_level":"airport" if len(history) else "empty900"},index=query.index)
    for keys, minimum, level in [(["RUNWAY_mvt"],100,"runway"),(["STAND_mvt","RUNWAY_mvt"],20,"stand_runway")]:
        table = history.groupby(keys,dropna=False)[TARGET].agg(["median","size"]).reset_index()
        joined = query[keys].reset_index().merge(table,on=keys,how="left",validate="many_to_one").set_index(ID).loc[query.index]
        enough = joined["size"].ge(minimum)
        result.loc[enough,"physical_reference"] = joined.loc[enough,"median"]
        result.loc[enough,"reference_n"] = joined.loc[enough,"size"]
        result.loc[enough,"reference_level"] = level
    return result


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    write_json(OUT / "protocol.json", {"created_utc":utc_now(),"source_sha256":sha256(__file__),
        "scope":"Diagnostic source-offset classes only; no prediction rule, no score outcomes.",
        "reference":"Original purged fold fit IDs, known finite NM proxy, Rome DEP. Strictly earlier UTC months. Median rawY hierarchical airport, runway atn>=100, stand+runway atn>=20. January900 fallback. No label trimming.",
        "offset":"Europe/Rome local UTC offset at observed movement time, 3600winter/7200summer; no inferred dates.",
        "diagnostic_classes":"absolute(Y-reference)<=600; absolute(Y-reference-localoffset)<=600; absolute(Y-reference-86400)<=600; absolute(Y-schedulegap)<=60. Classes overlap. Keep residuals and every counterexample.",
        "fit_tune":"F1/F3 fit and tune only; reference never uses query-month or tune labels.",
        "resources":"1CPU,6GiB process cap,8GiB reserve."})
    metadata = ROOT / "private_runs/screening_230/data/interim/audit/departures.parquet"
    assert sha256(metadata) == read_json(ROOT / "private_runs/screening_230/reports/data_audit.json")["artifacts"][metadata.name]
    meta = pd.read_parquet(metadata)
    path = INPUT / "rome_missing.parquet"
    assert sha256(path) == read_json(INPUT / "manifest.json")["outputs"][path.name]
    missing = pd.read_parquet(path).set_index(ID)
    missing[TIME] = pd.to_datetime(missing[TIME],utc=True)
    lex = lexical_features(missing.FLIGHT_mvt)
    missing["lex_kind"] = decode_flight_tokens(lex.lex_kind)
    frozen = read_json(ROOT / "private_runs/submission_v2/protocol.json")
    parts = []
    for raw in sorted(common.RAW.glob("training_*.parquet")):
        assert sha256(raw) == frozen["raw_hashes"][raw.name]
        part = pd.read_parquet(raw,columns=[ID,"STAND_mvt","RUNWAY_mvt"],filters=[("PHASE_mvt","=","DEP"),("ADEP_mvt","=","LIRF")])
        parts.append(part)
    details = pd.concat(parts,ignore_index=True).set_index(ID)
    known = meta.loc[meta.ADEP_mvt.eq("LIRF") & np.isfinite(meta.proxy_sec)].set_index(ID).join(details,validate="one_to_one")
    known[TIME] = pd.to_datetime(known[TIME],utc=True)
    for frame in [known,missing]:
        for column in ["STAND_mvt","RUNWAY_mvt"]:
            frame[column] = frame[column].astype("string").fillna("<MISSING>")
    outputs, receipts = [], {}
    for fold in ["F1","F3"]:
        indices, split, _ = common.fold_data(meta,fold,full=True)
        allowed = known.loc[known.index.isin(meta.iloc[indices["fit"]][ID])]
        receipts[fold] = {"split_hash":object_hash(split),"reference_known_n":len(allowed)}
        for stage in ["fit","tune"]:
            query = missing.loc[missing.index.isin(meta.iloc[indices[stage]][ID])].copy()
            query["period"] = fold+"_"+stage
            for month, rows in query.groupby(query[TIME].dt.strftime("%Y-%m")):
                start = pd.Timestamp(month+"-01",tz="UTC")
                history = allowed.loc[allowed[TIME].lt(start)]
                rows = rows.join(reference(history,rows))
                local = rows[TIME].dt.tz_convert("Europe/Rome").dt.tz_localize(None)
                rows["local_offset_sec"] = (local-rows[TIME].dt.tz_localize(None)).dt.total_seconds()
                rows["ordinary_distance_sec"] = rows[TARGET]-rows.physical_reference
                rows["dst_distance_sec"] = rows.ordinary_distance_sec-rows.local_offset_sec
                rows["day_distance_sec"] = rows.ordinary_distance_sec-86400
                rows["ordinary_reference_diagnostic"] = rows.ordinary_distance_sec.abs().le(600)
                rows["dst_reference_diagnostic"] = rows.dst_distance_sec.abs().le(600)
                rows["day_reference_diagnostic"] = rows.day_distance_sec.abs().le(600)
                outputs.append(rows)
    data = pd.concat(outputs)
    data.reset_index().to_parquet(OUT / "earlier_diagnostics.parquet",index=False)
    summaries = []
    for groups in [["period"],["period","local_offset_sec"],["period","local_offset_sec","lex_kind"],["period","local_offset_sec","schedule_bin"],["period","flight"]]:
        for keys,rows in data.groupby(groups,dropna=False,observed=True):
            if not isinstance(keys,tuple): keys=(keys,)
            summaries.append({"grouping":"|".join(groups),**dict(zip(groups,keys)),"n":len(rows),
                "ordinary_reference_n":int(rows.ordinary_reference_diagnostic.sum()),"dst_reference_n":int(rows.dst_reference_diagnostic.sum()),
                "day_reference_n":int(rows.day_reference_diagnostic.sum()),"near_schedule_n":int(rows.near_schedule_diagnostic.sum()),
                "mean_y":float(rows[TARGET].mean()),"median_y":float(rows[TARGET].median()),
                "median_ref":float(rows.physical_reference.median()),"median_dst_residual":float(rows.dst_distance_sec.median())})
    pd.DataFrame(summaries).to_csv(OUT / "conditional_counts.csv",index=False)
    same_flights = data.loc[data.period.eq("F3_fit")].groupby("flight").local_offset_sec.nunique()
    data.loc[data.flight.isin(same_flights[same_flights>1].index)].reset_index().to_csv(OUT / "cross_dst_same_flight.csv",index=False)
    peak = getattr(psutil.Process().memory_info(),"peak_wset",psutil.Process().memory_info().rss)
    assert peak<6*1024**3 and psutil.virtual_memory().available>8*1024**3
    write_json(OUT / "manifest.json",{"status":"complete","source_sha256":sha256(__file__),"split_receipts":receipts,"peak_bytes":peak,
        "outputs":{p.name:sha256(p) for p in OUT.iterdir() if p.is_file()}})
    print(pd.DataFrame(summaries).query("grouping=='period|local_offset_sec|lex_kind'").to_string(index=False))


if __name__ == "__main__":
    main()
