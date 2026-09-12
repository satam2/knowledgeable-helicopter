import numpy as np
import pandas as pd

from taxiout.artifacts import object_hash, write_json
from taxiout.config import ROOT, load_config
from taxiout.io import audited_departures
from taxiout.schema import FLIGHT_ID, ID, MOVEMENT, unique_ids, utc


def interval_mask(times, bounds):
    start, end = [pd.Timestamp(str(value), tz="UTC") for value in bounds]
    if start >= end:
        raise ValueError("Empty or reversed split interval")
    times = utc(times, required=True)
    return times.ge(start) & times.lt(end)


def make_fold(meta, spec):
    unique_ids(meta)
    masks = {stage: interval_mask(meta[MOVEMENT], spec[stage]).to_numpy() for stage in ["fit", "tune", "refit", "score"]}
    if any((masks[a] & masks[b]).any() for a, b in [("fit", "tune"), ("refit", "score"), ("tune", "score")]):
        raise ValueError("Fit, tuning and scoring ranges overlap")
    if not np.array_equal(masks["fit"] | masks["tune"], masks["refit"]):
        raise ValueError("Refit must equal fit plus tuning")
    purge = {}
    for stage, against in [("fit", "score"), ("tune", "score"), ("refit", "score"), ("fit", "tune")]:
        flights = meta.loc[masks[against], FLIGHT_ID].dropna().unique()
        related = meta[FLIGHT_ID].notna().to_numpy() & meta[FLIGHT_ID].isin(flights).to_numpy()
        remove = masks[stage] & related
        purge[f"{stage}_against_{against}"] = int(remove.sum())
        masks[stage] &= ~remove
    positions = {stage: np.flatnonzero(mask) for stage, mask in masks.items()}
    if any(len(rows) == 0 for rows in positions.values()):
        raise ValueError("Fold contains an empty stage")
    evidence = {"spec": spec, "purged_related_departures": purge,
                "stages": {stage: {"n": len(rows), "id_hash": object_hash(meta.iloc[rows][ID].tolist())} for stage, rows in positions.items()}}
    evidence["split_hash"] = object_hash(evidence)
    return positions, evidence


def make_splits(path="configs/folds.yaml"):
    meta = audited_departures(columns=[ID, FLIGHT_ID, MOVEMENT])
    reports = {name: make_fold(meta, spec)[1] for name, spec in load_config(path).items()}
    write_json(ROOT / "reports/splits.json", reports)
    return reports
