import pandas as pd

from taxiout.schema import CLOCKS, MOVEMENT, duration, utc


def clock_features(dep, quality=False):
    result = pd.DataFrame(index=dep.index)
    for col in CLOCKS:
        result["takeoff_minus_" + col] = duration(dep[MOVEMENT], dep[col])
    result["nm_actual_minus_estimated"] = duration(dep["AOBT_3_flt"], dep["EOBT_1_flt"])
    result["last_minus_initial"] = duration(dep["LOBT_flt"], dep["IOBT_flt"])
    result["proxy_missing"] = result["takeoff_minus_AOBT_3_flt"].isna().astype(float)
    if quality:
        for col in CLOCKS:
            clock = utc(dep[col])
            result[col + "_missing"] = clock.isna().astype(float)
            result[col + "_invalid_parse"] = (dep[col].notna() & clock.isna()).astype(float)
            result[col + "_second"] = clock.dt.second
            result[col + "_after_takeoff"] = result["takeoff_minus_" + col].lt(0).astype(float)
        for col in ["ADEP", "ADES", "AIRCRAFT_TYPE"]:
            a, b = dep[col + "_mvt"], dep[col + "_flt"]
            result[col + "_source_missing"] = b.isna().astype(float)
            result[col + "_source_disagrees"] = (a.notna() & b.notna() & a.astype(str).ne(b.astype(str))).astype(float)
        result["planned_duration_sec"] = duration(dep["ARVT_1_flt"], dep["EOBT_1_flt"])
        result["proxy_negative"] = result["takeoff_minus_AOBT_3_flt"].lt(0).astype(float)
        result["proxy_over_two_hours"] = result["takeoff_minus_AOBT_3_flt"].gt(7200).astype(float)
    return result.astype("float32")
