import numpy as np
import pandas as pd

from taxiout.availability import assert_observations
from taxiout.schema import ID, MOVEMENT, PHASE, utc


def nanoseconds(times):
    return utc(times, required=True).dt.as_unit("ns").astype("int64").to_numpy()


def prior_counts(event_times, query_times, minutes):
    events = np.sort(nanoseconds(event_times))
    query = nanoseconds(query_times)
    width = pd.Timedelta(minutes=minutes).value
    return np.searchsorted(events, query, side="left") - np.searchsorted(events, query - width, side="left")


def traffic_features(departures, context, month_isolated=True):
    assert_observations(context)
    result = pd.DataFrame(index=departures.index)
    event_airport = np.where(context[PHASE].eq("DEP"), context["ADEP_mvt"], context["ADES_mvt"])
    event_ns, query_ns = nanoseconds(context[MOVEMENT]), nanoseconds(departures[MOVEMENT])
    event_keys = pd.DataFrame({"airport": pd.Categorical(event_airport), "phase": context[PHASE].to_numpy(),
                               "month": (context[MOVEMENT].dt.year * 12 + context[MOVEMENT].dt.month).to_numpy() if month_isolated else 0})
    query_keys = pd.DataFrame({"airport": pd.Categorical(departures["ADEP_mvt"]),
                               "month": (departures[MOVEMENT].dt.year * 12 + departures[MOVEMENT].dt.month).to_numpy() if month_isolated else 0})
    # Group once: repeated object-string scans of the full year dominate runtime.
    event_groups = {key: np.sort(event_ns[positions]) for key, positions in event_keys.groupby(["airport", "phase", "month"], observed=True).indices.items()}
    query_groups = query_keys.groupby(["airport", "month"], observed=True).indices
    for phase in ["DEP", "ARR"]:
        for minutes in [15, 60]:
            counts = np.zeros(len(departures), dtype="float32")
            width = pd.Timedelta(minutes=minutes).value
            for (airport, month), positions in query_groups.items():
                times = event_groups.get((airport, phase, month), np.empty(0, dtype=np.int64))
                query = query_ns[positions]
                counts[positions] = np.searchsorted(times, query, side="left") - np.searchsorted(times, query - width, side="left")
            result[f"{phase.lower()}_prior_{minutes}m"] = counts
    if month_isolated:
        ts = departures[MOVEMENT]
        start = pd.Series(pd.to_datetime(ts.dt.tz_convert(None).to_numpy().astype("datetime64[M]"), utc=True), index=ts.index)
        age = (ts - start).dt.total_seconds()
        for minutes in [15, 60]:
            result[f"observed_window_{minutes}m_sec"] = np.minimum(age, minutes * 60).astype("float32")
        result["month_edge"] = age.lt(3600).astype("float32")
    return result
