import numpy as np
import pandas as pd

ID = "MVT_ID_mvt"
TARGET = "TAXITIME_SEC_mvt"
MOVEMENT = "MVT_TIME_UTC_mvt"
BLOCK = "BLOCK_TIME_UTC_mvt"
PHASE = "PHASE_mvt"
FLIGHT_ID = "FLIGHT_ID_mvt"
CATEGORIES = ["ADEP_mvt", "RUNWAY_mvt", "STAND_mvt", "ADES_mvt", "AIRCRAFT_TYPE_mvt",
              "AIRCRAFT_OPERATOR_flt", "WK_TBL_CAT_flt", "MARKET_SEGMENT_flt", "FLIGHT_TYPE_flt"]
CLOCKS = ["AOBT_3_flt", "EOBT_1_flt", "IOBT_flt", "LOBT_flt", "SCHED_TIME_UTC_mvt"]
FEATURE_INPUTS = [ID, FLIGHT_ID, PHASE, MOVEMENT, *CATEGORIES, *CLOCKS,
                  "ADEP_flt", "ADES_flt", "AIRCRAFT_TYPE_flt", "ARVT_1_flt"]


def unique_ids(frame):
    if ID not in frame or frame[ID].isna().any() or not frame[ID].is_unique:
        raise ValueError("Movement IDs must be present, nonnull and unique")
    if not np.isfinite(frame[ID].to_numpy(dtype=float)).all():
        raise ValueError("Movement IDs must be finite")


def align(left, right, columns=None):
    unique_ids(left)
    unique_ids(right)
    if len(left) != len(right) or not left[ID].isin(right[ID]).all():
        raise ValueError("ID cohorts differ: missing or extra predictions/labels")
    right = right if columns is None else right[[ID, *columns]]
    result = left.merge(right, on=ID, how="left", sort=False, validate="one_to_one")
    if not np.array_equal(result[ID].to_numpy(), left[ID].to_numpy()):
        raise ValueError("Join changed ID order")
    return result


def utc(series, required=False):
    converted = pd.to_datetime(series, utc=True, errors="coerce")
    if required and converted.isna().any():
        raise ValueError("Required movement timestamp missing or invalid")
    return converted


def duration(end, start):
    return (utc(end) - utc(start)).dt.total_seconds()


def utc_period_strings(times, unit):
    if unit not in {"M", "D"}:
        raise ValueError("Expected UTC month or day unit")
    values = utc(times, required=True).dt.tz_convert(None).to_numpy()
    return values.astype(f"datetime64[{unit}]").astype(str)


def target_identity(raw):
    unique_ids(raw)
    if not raw[PHASE].isin(["DEP", "ARR"]).all():
        raise ValueError("Unexpected movement phase")
    utc(raw[MOVEMENT], required=True)
    delta = duration(raw[MOVEMENT], raw[BLOCK])
    expected = delta.where(raw[PHASE].eq("DEP"), -delta)
    labels = pd.to_numeric(raw[TARGET], errors="coerce")
    missing_arrival = raw[PHASE].eq("ARR") & labels.isna() & expected.isna()
    checked = ~missing_arrival
    if not np.isfinite(labels[checked]).all() or not np.isfinite(expected[checked]).all() or not np.array_equal(expected[checked].to_numpy(), labels[checked].to_numpy()):
        raise ValueError("Raw phase-specific target identity failed")
    return {"DEP_max_abs_error_sec": 0.0, "ARR_max_abs_error_sec": 0.0, "unlabelled_arrival_context": int(missing_arrival.sum())}
