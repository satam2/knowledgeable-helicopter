"""Aggregate saved schema audit and test phase-ID ordering without training."""
import json
from pathlib import Path
import sys
from collections import defaultdict
from datetime import datetime, timezone
import hashlib

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from audit_information import ROOT, RAW, OUT, ID, FLIGHT_ID, MOVEMENT, BLOCK, PHASE, stats, seconds, finite_json


def main():
    audit = json.loads((OUT/"information_audit.json").read_text())
    result = {"created_utc": datetime.now(timezone.utc).isoformat(), "packs": [], "coverage": {}, "ordering": []}
    for cohort, packs in [("training", [p for p in audit["packs"] if p["file"].startswith("training")]), ("ranking", [p for p in audit["packs"] if p["file"] == "ranking.parquet"])]:
        n = sum(p["diagnostics"]["departures"] for p in packs)
        missing = sum(p["diagnostics"]["missing_aobt"] for p in packs)
        nm = {c: sum(p["diagnostics"]["missing_nm_fields"].get(c, 0) for p in packs) for c in packs[0]["diagnostics"]["missing_nm_fields"]}
        cols = {c: {"present": sum(p["by_phase"]["DEP"]["columns"][c]["present"] for p in packs), "coverage_pct": 100*sum(p["by_phase"]["DEP"]["columns"][c]["present"] for p in packs)/n} for c in packs[0]["by_phase"]["DEP"]["columns"]}
        linked = sum(p["diagnostics"]["same_nm_flight_arrival"]["rows"] for p in packs)
        result["coverage"][cohort] = {"departure_rows": n, "missing_aobt": missing, "missing_aobt_pct": 100*missing/n,
            "fields_present_among_missing_aobt": nm, "columns": cols, "same_flight_destination_arrival_rows": linked,
            "same_flight_destination_arrival_coverage_pct": 100*linked/n,
            "same_flight_missing_aobt_recovered": sum(p["diagnostics"]["same_nm_flight_arrival"].get("missing_aobt_recovered",0) for p in packs),
            "arrival_id_bracket_rows": sum(r.get("arrival_id_brackets",{}).get("rows",0) for p in packs for r in p["id_order"]),
            "diverted_filed_destination_rows": sum(p["diagnostics"]["diverted_filed_destination"] for p in packs)}
    for path in sorted(RAW.glob("*.parquet")):
        if path.name == "submitting.parquet":
            continue
        data = pq.read_table(path, columns=[ID, FLIGHT_ID, PHASE, MOVEMENT, BLOCK, "ADEP_mvt", "ADES_mvt", "FLIGHT_mvt", "CALLSIGN_flt"], use_threads=False).to_pandas()
        raw_receipt = {"file": path.name, "phase_id_ranges": {}}
        for phase in ("DEP", "ARR"):
            z = data.loc[data[PHASE].eq(phase)]
            raw_receipt["phase_id_ranges"][phase] = {"min": float(z[ID].min()), "max": float(z[ID].max()), "integer_pct": float(100*np.mean(z[ID] == np.floor(z[ID])))}
        result["packs"].append(raw_receipt)
        dep = data.loc[data[PHASE].eq("DEP") & data[BLOCK].notna()].copy()
        dep["day"] = dep[MOVEMENT].dt.strftime("%Y-%m-%d")
        for airport, part in dep.groupby("ADEP_mvt", observed=True):
            block_diffs, movement_diffs, within_block_corr, within_takeoff_corr = [], [], [], []
            for day, day_part in part.groupby("day", observed=True):
                day_part = day_part.sort_values(ID)
                if len(day_part) < 10:
                    continue
                b, t = seconds(day_part[BLOCK]), seconds(day_part[MOVEMENT])
                block_diffs.extend(np.diff(b).tolist())
                movement_diffs.extend(np.diff(t).tolist())
                within_block_corr.append(float(day_part[ID].corr(pd.Series(b,index=day_part.index),method="spearman")))
                within_takeoff_corr.append(float(day_part[ID].corr(pd.Series(t,index=day_part.index),method="spearman")))
            if block_diffs:
                b,t=np.array(block_diffs),np.array(movement_diffs)
                result["ordering"].append({"file":path.name,"airport":str(airport),"days":len(within_block_corr),
                    "within_day_spearman_block_mean":float(np.mean(within_block_corr)),
                    "within_day_spearman_takeoff_mean":float(np.mean(within_takeoff_corr)),
                    "id_adjacent_block_reverse_pct":float(100*np.mean(b<0)),
                    "id_adjacent_takeoff_reverse_pct":float(100*np.mean(t<0)),
                    "id_adjacent_block_gap":stats(b),"id_adjacent_takeoff_gap":stats(t)})
        print(path.name,"phase ranges + within-day order",flush=True)
    result["script_sha256"]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (OUT/"information_summary.json").write_text(json.dumps(finite_json(result),indent=2,allow_nan=False),encoding="utf-8")


if __name__=="__main__":
    main()
