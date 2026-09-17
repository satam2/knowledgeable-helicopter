"""Earlier-period flight morphology census; outcome fields are diagnostic only."""
import os
for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[key] = "1"
import re
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import psutil

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "review_work/campaign_20260916"))
import common
from taxiout.artifacts import read_json, write_json, sha256, utc_now, object_hash
from taxiout.paths import external_path
OUT = external_path(ROOT / "private_runs/tail240_20260916/forensics/lexical_v1")
SOURCE = ROOT / "private_runs/tail240_20260916/forensics/rome_regimes_v1"
ID, TARGET, TIME = common.ID, common.TARGET, common.MOVEMENT


def morphology(flight):
    match = re.fullmatch(r"([A-Z]*)([0-9]+)([A-Z]*)", flight)
    if not match:
        return ["empty" if not flight else "unparsed", 0, "", 0, False, False, ""]
    prefix, number, suffix = match.groups()
    core, extra = prefix[:3], prefix[3:]
    kind = "standard" if len(prefix) in (2, 3) else "other_prefix"
    if len(prefix) > 3:
        kind = "repeated_suffix" if extra and any(extra == core[-n:] for n in (1, 2, 3)) else "long_other"
    canonical = core + number + suffix if kind == "repeated_suffix" else flight
    return [kind, len(prefix), core, len(suffix), number.startswith("0"), len(set(suffix)) == 1 and len(suffix) > 1, canonical]


def summarize(group):
    y = group[TARGET]
    return {"n": len(group), "ordinary_n": int(group.ordinary_diagnostic.sum()),
        "near_schedule_n": int(group.near_schedule_diagnostic.sum()),
        "day_plus_short_n": int(group.day_plus_short_diagnostic.sum()),
        "negative_n": int(y.lt(0).sum()), "over_7200_n": int(y.gt(7200).sum()),
        "ordinary_rate": float(group.ordinary_diagnostic.mean()),
        "near_schedule_rate": float(group.near_schedule_diagnostic.mean()),
        "day_plus_short_rate": float(group.day_plus_short_diagnostic.mean()),
        "mean_y": float(y.mean()), "median_y": float(y.median()),
        "mean_gap": float(group.schedule_gap_sec.mean()),
        "schedule_error_rmse": float(np.sqrt(np.mean((y - group.schedule_gap_sec) ** 2)))}


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    write_json(OUT / "protocol.json", {"created_utc": utc_now(), "source_sha256": sha256(__file__),
        "scope": "Earlier fit/tune diagnostic cross-tabs only. No fitted rule or score prediction.",
        "lexical_definition": "Parse letters,digits,letters. Repeated suffix means letters after first three equal last one,two,or three of first three. Canonical spelling is a hypothesis only.",
        "conditioning": "Retain ordinary and negative labels; stratify aircraft missing and schedule gap before comparing lexical groups. Phenotypes overlap.",
        "periods": "Original purged F1 and F3 fit/tune IDs; no score outcomes in these aggregates.",
        "resources": "2 CPU, 6 GiB peak cap; at least 8 GiB available host reserve."})
    manifest = read_json(SOURCE / "manifest.json")
    assert sha256(SOURCE / "rome_missing.parquet") == manifest["outputs"]["rome_missing.parquet"]
    data = pd.read_parquet(SOURCE / "rome_missing.parquet").set_index(ID)
    fields = ["lexical_kind", "alpha_prefix_len", "prefix_core", "alpha_suffix_len", "number_leading_zero", "suffix_repeated", "canonical_hypothesis"]
    lexical = pd.DataFrame([morphology(f) for f in data.flight], index=data.index, columns=fields)
    data = data.join(lexical)
    metadata = ROOT / "private_runs/screening_230/data/interim/audit/departures.parquet"
    assert sha256(metadata) == read_json(ROOT / "private_runs/screening_230/reports/data_audit.json")["artifacts"][metadata.name]
    meta = pd.read_parquet(metadata, columns=[ID, TARGET, TIME, "FLIGHT_ID_mvt", "proxy_sec"])
    aggregates, periods, splits = [], [], {}
    groupings = [[], ["aircraft_missing"], ["lexical_kind"], ["alpha_prefix_len"], ["alpha_suffix_len"],
        ["number_leading_zero"], ["suffix_repeated"], ["aircraft_missing", "lexical_kind"],
        ["aircraft_missing", "schedule_bin", "lexical_kind"], ["prefix_core", "lexical_kind"],
        ["RUNWAY_mvt", "lexical_kind"]]
    for fold in ["F1", "F3"]:
        indices, split, _ = common.fold_data(meta, fold, full=True)
        splits[fold] = {"split_hash": object_hash(split), "stages": {}}
        for stage in ["fit", "tune"]:
            ids = meta.iloc[indices[stage]][ID]
            rows = data.loc[data.index.isin(ids)].copy()
            rows["period"] = fold + "_" + stage
            periods.append(rows)
            splits[fold]["stages"][stage] = {"all_ids_hash": object_hash(ids.tolist()), "rome_missing_n": len(rows)}
            for columns in groupings:
                grouped = rows.groupby(columns, dropna=False, observed=True) if columns else [((), rows)]
                for keys, group in grouped:
                    if not isinstance(keys, tuple):
                        keys = (keys,)
                    aggregates.append({"period": fold + "_" + stage, "grouping": "|".join(columns) or "all",
                        **dict(zip(columns, keys)), **summarize(group)})
    results = pd.DataFrame(aggregates)
    results.to_csv(OUT / "conditional_counts.csv", index=False)
    earlier = pd.concat(periods).reset_index()
    earlier.to_parquet(OUT / "earlier_rows.parquet", index=False)
    earlier.loc[earlier.lexical_kind.eq("repeated_suffix")].to_csv(OUT / "repeated_prefix_rows.csv", index=False)
    peak = getattr(psutil.Process().memory_info(), "peak_wset", psutil.Process().memory_info().rss)
    assert peak < 6 * 1024**3 and psutil.virtual_memory().available > 8 * 1024**3
    write_json(OUT / "manifest.json", {"status": "complete", "source_sha256": sha256(__file__),
        "input_sha256": sha256(SOURCE / "rome_missing.parquet"), "splits": splits, "peak_bytes": peak,
        "labels_diagnostic_only": True, "outputs": {p.name: sha256(p) for p in OUT.iterdir() if p.is_file()}})
    for grouping in ["all", "aircraft_missing", "lexical_kind", "aircraft_missing|lexical_kind", "aircraft_missing|schedule_bin|lexical_kind"]:
        print("GROUPING", grouping)
        subset = results.loc[results.grouping.eq(grouping)]
        columns = ["period"] + ([] if grouping == "all" else grouping.split("|")) + ["n", "ordinary_n", "near_schedule_n", "day_plus_short_n", "mean_y", "mean_gap"]
        print(subset[columns].to_string(index=False))


if __name__ == "__main__":
    main()
