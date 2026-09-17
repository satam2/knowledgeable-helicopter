import numpy as np
import pandas as pd

from run_transfer import movement_only


def test_mask_removes_all_nm_inputs_but_keeps_schedule_and_location():
    frame = pd.DataFrame({"ADEP_mvt": pd.Categorical(["s4:LIRF", "s4:EDDF"]),
                          "AIRCRAFT_OPERATOR_flt": pd.Categorical(["s1:A", "m:"]),
                          "takeoff_minus_AOBT_3_flt": np.array([960, -999999], dtype="float32"),
                          "nm_actual_minus_estimated": np.array([20, -999999], dtype="float32"),
                          "last_minus_initial": np.array([40, -999999], dtype="float32"),
                          "proxy_missing": np.array([0, 1], dtype="float32"),
                          "takeoff_minus_SCHED_TIME_UTC_mvt": np.array([1800, 1900], dtype="float32")})
    masked = movement_only(frame)
    assert masked.AIRCRAFT_OPERATOR_flt.astype(str).tolist() == ["m:", "m:"]
    assert masked.takeoff_minus_AOBT_3_flt.tolist() == [-999999, -999999]
    assert masked.nm_actual_minus_estimated.tolist() == [-999999, -999999]
    assert masked.last_minus_initial.tolist() == [-999999, -999999]
    assert masked.proxy_missing.tolist() == [1, 1]
    assert masked.takeoff_minus_SCHED_TIME_UTC_mvt.tolist() == [1800, 1900]
    assert masked.ADEP_mvt.astype(str).tolist() == ["s4:LIRF", "s4:EDDF"]
    assert frame.takeoff_minus_AOBT_3_flt.tolist() == [960, -999999]
    assert frame.dtypes.astype(str).to_dict() == masked.dtypes.astype(str).to_dict()
