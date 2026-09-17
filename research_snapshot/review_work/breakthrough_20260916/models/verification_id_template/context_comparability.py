"""Label-free ranking record-order distribution audit and exact sensitivity summary."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "review_work/campaign_20260916"))
import common
from taxiout.artifacts import read_json, write_json, sha256, utc_now
from taxiout.schema import ID, MOVEMENT, TARGET, PHASE
from taxiout.metrics import season_score

OUT = ROOT / "private_runs/breakthrough_20260916/models/verification_id_template"


def stats(values):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    return {"n": len(values), "q05_q25_q50_q75_q95": np.quantile(values, [.05, .25, .5, .75, .95]).tolist() if len(values) else [],
            "over12h_earlier_fraction": float(np.mean(values < -43200)) if len(values) else None}


def features(frame):
    times = pd.to_datetime(frame[MOVEMENT], utc=True).dt.as_unit("ns").astype("int64").to_numpy() / 1e9
    result = pd.DataFrame(index=frame.index)
    frame = frame.assign(month=pd.to_datetime(frame[MOVEMENT], utc=True).dt.strftime("%Y-%m"))
    for width in [2, 8, 32]:
        estimate = np.full(len(frame), np.nan)
        dispersion = np.full(len(frame), np.nan)
        for _, positions in frame.groupby(["ADEP_mvt", "month"], observed=True).indices.items():
            positions = positions[np.argsort(frame.iloc[positions][ID].to_numpy())]
            padded = np.pad(times[positions], (width, width), constant_values=np.nan)
            windows = np.lib.stride_tricks.sliding_window_view(padded, 2 * width + 1)
            peers = np.concatenate([windows[:, :width], windows[:, width + 1:]], axis=1)
            estimate[positions] = np.nanmedian(peers, axis=1)
            dispersion[positions] = np.nanquantile(peers, .9, axis=1) - np.nanquantile(peers, .1, axis=1)
        result[f"id_peer_median_time_minus_query_w{width}"] = estimate - times
        result[f"id_peer_time_spread_w{width}"] = dispersion
    return result


def metric(frame, name, keep=None):
    if keep is not None:
        frame = frame.loc[keep]
    return {"n": len(frame), "sse": float(np.square(frame[name].to_numpy() - frame[TARGET].to_numpy()).sum())}


if __name__ == "__main__":
    frozen = read_json(ROOT / "private_runs/submission_v2/protocol.json")
    path = ROOT / "data/09-15-2026-18-55-03_files_list/ranking.parquet"
    assert sha256(path) == frozen["raw_hashes"][path.name]
    ranking = pq.read_table(path, columns=[ID, PHASE, MOVEMENT, "ADEP_mvt", "AOBT_3_flt"], use_threads=False).to_pandas()
    ranking = ranking.loc[ranking[PHASE].eq("DEP")].reset_index(drop=True)
    audit = read_json(ROOT / "private_runs/screening_230/reports/data_audit.json")
    mpath = ROOT / "private_runs/screening_230/data/interim/audit/departures.parquet"
    assert sha256(mpath) == audit["artifacts"][mpath.name]
    meta = pd.read_parquet(mpath, columns=[ID, MOVEMENT, "ADEP_mvt", "proxy_sec"])
    trainpeer = pd.read_parquet(ROOT / "private_runs/mechanism_20260916/id_neighbors/features.parquet").set_index(ID)
    assert np.array_equal(trainpeer.index, meta[ID])
    rankpeer = features(ranking)
    periods = pd.to_datetime(meta[MOVEMENT], utc=True).dt.strftime("%Y-%m")
    selected = meta.loc[periods.isin(["2025-01", "2025-07", "2025-11"])].reset_index(drop=True)
    recomputed = features(selected)
    expected = trainpeer.loc[selected[ID]].reset_index(drop=True)
    assert np.allclose(recomputed.to_numpy(), expected.to_numpy(), equal_nan=True)
    groups = []
    for kind, frame, peer, missing in [("training", meta, trainpeer.reset_index(drop=True), ~np.isfinite(meta.proxy_sec)),
                                        ("ranking", ranking, rankpeer, ranking.AOBT_3_flt.isna())]:
        months = pd.to_datetime(frame[MOVEMENT], utc=True).dt.strftime("%Y-%m")
        for (airport, month), positions in frame.assign(month=months).groupby(["ADEP_mvt", "month"], observed=True).indices.items():
            if kind == "training" and month not in ["2025-01", "2025-07", "2025-11"]:
                continue
            ids = np.sort(frame.iloc[positions][ID].to_numpy())
            cohort = positions[np.asarray(missing)[positions]]
            groups.append({"kind": kind, "airport": airport, "month": month, "rows": len(positions), "missing_rows": len(cohort),
                           "id_step_one_fraction": float(np.mean(np.diff(ids) == 1)), "median_id_step": float(np.median(np.diff(ids))),
                           "all_w8_delta": stats(peer.iloc[positions]["id_peer_median_time_minus_query_w8"]),
                           "missing_w8_delta": stats(peer.iloc[cohort]["id_peer_median_time_minus_query_w8"]),
                           "missing_w8_spread": stats(peer.iloc[cohort]["id_peer_time_spread_w8"])})
    f1 = pd.read_parquet(OUT / "F1_comparison.parquet")
    f3 = pd.read_parquet(OUT / "F3_comparison.parquet")
    names = [name for name in f1 if name not in [ID, TARGET, "day", "ADEP_mvt"]]
    tail = f3[TARGET] >= 86400
    assert tail.sum() == 2
    scenarios = {"all": np.ones(len(f3), bool), "remove_two_november_dayplus_rows": ~tail.to_numpy(),
                 "remove_two_november_dayplus_days": ~f3.day.isin(f3.loc[tail, "day"]).to_numpy()}
    for ident in f3.loc[tail, ID]:
        scenarios[f"remove_november_row_{int(ident)}"] = ~f3[ID].eq(ident).to_numpy()
    sensitivity = {scenario: {name: season_score(metric(f1, name), metric(f3, name, mask)) for name in names}
                   for scenario, mask in scenarios.items()}
    seasonal_removals = []
    for fold, frame in [("F1", f1), ("F3", f3)]:
        for day in sorted(frame.day.unique()):
            keep = ~frame.day.eq(day).to_numpy()
            scores = {name: season_score(metric(f1, name, keep if fold == "F1" else None),
                                         metric(f3, name, keep if fold == "F3" else None)) for name in names}
            seasonal_removals.append({"fold": fold, "removed_day": day, "scores": scores})
    result = {"created_utc": utc_now(), "ranking_label_free_rows": len(ranking),
              "reproduced_training_neighbor_features": True, "reproduced_training_rows": len(selected),
              "ranking_sha256": sha256(path), "script_sha256": sha256(__file__), "groups": groups,
              "november_sensitivity_seasonal": sensitivity, "seasonal_day_removals": seasonal_removals,
              "limitations": ["Ranking features here are distribution diagnostics, not a frozen production feature cache or validated prediction artifact.",
                              "Sorted-ID neighbors are source-record/extract-order context, not authenticated operational events or aircraft rotations.",
                              "No raw absolute ID is used as a model feature, but the neighborhood ordering still depends on ID allocation and extract policy.",
                              "Identical operator and similar marginal distributions cannot prove conditional target behavior transfers from 2025 to2026.",
                              "Each feature uses the same airport and UTC movement month; missing/partial source batches or changes in ID allocation can change nearest records.",
                              "Future movement rows are allowed by this explicitly retrospective supplied-batch contract."]}
    write_json(OUT / "context_comparability.json", result)
    print("NOVEMBER_SENSITIVITY", sensitivity, flush=True)
    print("RANKING_LIRF", [g for g in groups if g["airport"] == "LIRF"], flush=True)
