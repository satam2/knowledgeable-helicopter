import numpy as np
import pandas as pd
import pytest

from taxiout.config import load_config
from taxiout.schema import BLOCK, CATEGORIES, CLOCKS, FLIGHT_ID, ID, MOVEMENT, PHASE, TARGET


@pytest.fixture
def config():
    return load_config("configs/base.yaml")


@pytest.fixture
def movements():
    n = 48
    ts = pd.Series(pd.date_range("2025-03-30 00:00", periods=n, freq="5min", tz="UTC"))
    df = pd.DataFrame({ID: np.arange(n, dtype=float), FLIGHT_ID: np.arange(n, dtype=float), MOVEMENT: ts, PHASE: ["DEP"] * n})
    for col in CATEGORIES:
        df[col] = "A"
    df["ADEP_mvt"] = "EDDF"
    df["ADES_mvt"] = "EGLL"
    df["STAND_mvt"] = ["A" if i % 2 else "B" for i in range(n)]
    for col in CLOCKS:
        df[col] = ts - pd.Timedelta(minutes=16)
    df[BLOCK] = ts - pd.Timedelta(minutes=15)
    df[TARGET] = 900 + np.arange(n) % 5
    df[BLOCK] = ts - pd.to_timedelta(df[TARGET], unit="s")
    for col in ["ADEP", "ADES", "AIRCRAFT_TYPE"]:
        df[col + "_flt"] = df[col + "_mvt"]
    df["ARVT_1_flt"] = ts + pd.Timedelta(hours=1)
    return df
