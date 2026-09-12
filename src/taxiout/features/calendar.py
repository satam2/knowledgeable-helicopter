import numpy as np
import pandas as pd

TIMEZONES = {"EDDF": "Europe/Berlin", "EDDM": "Europe/Berlin", "EGLL": "Europe/London",
             "EHAM": "Europe/Amsterdam", "LEBL": "Europe/Madrid", "LEMD": "Europe/Madrid",
             "LFPG": "Europe/Paris", "LIRF": "Europe/Rome", "LSZH": "Europe/Zurich", "LTFM": "Europe/Istanbul"}


def calendar_features(ts, airports, local=False):
    result = pd.DataFrame({"utc_hour": ts.dt.hour, "weekday": ts.dt.dayofweek, "month": ts.dt.month, "minute": ts.dt.minute}, index=ts.index)
    if local:
        for name in ["local_hour", "local_weekday", "utc_offset_hours", "dst"]:
            result[name] = np.nan
        for airport in airports.unique():
            if airport not in TIMEZONES:
                raise ValueError(f"Unknown airport timezone: {airport}")
            idx = airports.eq(airport)
            times = ts.loc[idx].dt.tz_convert(TIMEZONES[airport])
            result.loc[idx, "local_hour"] = times.dt.hour
            result.loc[idx, "local_weekday"] = times.dt.dayofweek
            result.loc[idx, "utc_offset_hours"] = (times.dt.tz_localize(None) - ts.loc[idx].dt.tz_localize(None)).dt.total_seconds() / 3600
            # ZoneInfo supplies historical DST, including Istanbul's permanent offset.
            result.loc[idx, "dst"] = [float(t.dst().total_seconds() != 0) for t in times]
    return result.astype("float32")
