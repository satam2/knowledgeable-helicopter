"""Independent brute-force oracle for randomized rolling arrival features."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "knowledgeable-helicopter-screening/src"))
sys.path.insert(0, str(HERE.parents[1] / "aviation"))
from arrival_features import aviation_features, extract_completed_arrivals
from taxiout.artifacts import sha256, utc_now, write_json
from taxiout.availability import make_observations
from taxiout.schema import BLOCK, CLOCKS, FLIGHT_ID, ID, MOVEMENT, PHASE, TARGET


def main():
    rng = np.random.default_rng(483)
    n_arr, n_dep = 160, 40
    landing = pd.Timestamp("2025-06-30 23:00", tz="UTC") + pd.to_timedelta(rng.integers(0, 12000, n_arr), unit="s")
    duration = rng.integers(30, 8000, n_arr)
    takeoff = pd.Timestamp("2025-06-30 23:00", tz="UTC") + pd.to_timedelta(rng.integers(600, 14400, n_dep), unit="s")
    phase = ["ARR"] * n_arr + ["DEP"] * n_dep
    movement = pd.DatetimeIndex(list(landing) + list(takeoff))
    block = pd.DatetimeIndex(list(landing + pd.to_timedelta(duration, unit="s")) + list(takeoff - pd.Timedelta(minutes=10)))
    raw = pd.DataFrame({ID: np.arange(n_arr + n_dep),
        FLIGHT_ID: rng.integers(1, 130, n_arr + n_dep), PHASE: phase, MOVEMENT: movement, BLOCK: block,
        "ADEP_mvt": ["X"] * n_arr + ["AAA"] * n_dep,
        "ADES_mvt": ["AAA"] * n_arr + ["X"] * n_dep,
        "RUNWAY_mvt": rng.choice(["01", "02"], n_arr + n_dep),
        "STAND_mvt": rng.choice(["A1", "A2"], n_arr + n_dep),
        "WK_TBL_CAT_flt": rng.choice(["H", "M", "J"], n_arr + n_dep),
        TARGET: np.r_[duration, np.full(n_dep, 600)]})
    for clock in CLOCKS:
        raw[clock] = raw[MOVEMENT] - pd.Timedelta(minutes=20)
    obs, _, _ = make_observations(raw)
    dep = obs.loc[obs[PHASE].eq("DEP")]
    arrivals = extract_completed_arrivals(raw)
    result = aviation_features(dep, obs, arrivals)
    checks = 0
    for _, query in dep.iterrows():
        qtime = query[MOVEMENT]
        month = qtime.year * 100 + qtime.month
        same = arrivals.loc[(arrivals.airport == query.ADEP_mvt) & (arrivals.month == month)
                            & (arrivals.flight != str(query[FLIGHT_ID]))]
        for minutes in (15, 30, 60):
            selected = same.loc[(same.completion < qtime) & (same.completion >= qtime - pd.Timedelta(minutes=minutes))
                                & (same.duration >= 0)]
            count = result.loc[query[ID], f"arr_airport_count_{minutes}m"]
            mean = result.loc[query[ID], f"arr_airport_mean_sec_{minutes}m"]
            if count != len(selected):
                raise AssertionError("Prefix rolling count differs from independent scan")
            expected = selected.duration.mean() if len(selected) else -999999.
            if not np.isclose(mean, expected, rtol=1e-6, atol=1e-3):
                raise AssertionError("Prefix rolling mean differs from independent scan")
            checks += 2
        valid = same.completion.notna() & same.duration.ge(0) & np.isfinite(same.duration)
        ended = same.landing + pd.Timedelta(minutes=120)
        ended.loc[valid] = pd.concat([ended.loc[valid], same.loc[valid, "completion"]], axis=1).min(axis=1)
        expected_open = ((same.landing < qtime) & (ended >= qtime)).sum()
        if result.loc[query[ID], "surface_arrivals_open_120m"] != expected_open:
            raise AssertionError("Bounded inventory differs from independent interval scan")
        checks += 1
    write_json(ROOT / "private_runs/campaign_20260916/validation/aviation_independent_oracle.json", {
        "status": "passed", "created_utc": utc_now(), "seed": 483,
        "invented_arrivals": n_arr, "invented_departures": n_dep, "comparisons": checks,
        "checks": ["Three duration windows count/mean", "bounded interval inventory", "same-flight exclusion", "landing-month isolation"],
        "module_sha256": sha256(HERE.parents[1] / "aviation/arrival_features.py"),
        "script_sha256": sha256(__file__),
        "limits": "Uses synthetic observations only; final NM publication availability remains unproven."})
    print(f"PASS {checks} independent rolling/inventory comparisons", flush=True)


if __name__ == "__main__":
    main()
