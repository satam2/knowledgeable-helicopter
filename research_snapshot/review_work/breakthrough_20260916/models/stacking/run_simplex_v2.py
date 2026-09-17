"""Reporting-only repair; identical frozen simplex formulas and experts."""
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import run_simplex as frozen


def day_sensitivity(first, second):
    np.testing.assert_array_equal(first[frozen.ID], second[frozen.ID])
    daily = pd.DataFrame({"day": first.day, "first": first.squared_error, "second": second.squared_error}).groupby("day").agg(first=("first", "sum"), second=("second", "sum"), n=("first", "size"))
    result = []
    for day, values in daily.iterrows():
        n = len(first) - values["n"]
        delta = np.sqrt((second.squared_error.sum() - values["second"]) / n) - np.sqrt((first.squared_error.sum() - values["first"]) / n)
        result.append({"removed_day": day, "delta_rmse": float(delta), "removed_day_sse_gain": float(values["first"] - values["second"])})
    return result


if __name__ == "__main__":
    previous = frozen.OUT
    frozen.OUT = previous.with_name("simplex_v2")
    frozen.OUT.mkdir(parents=True, exist_ok=True)
    receipt = {"created_utc": frozen.utc_now(), "wrapper_sha256": frozen.sha256(__file__),
               "original_protocol_sha256": frozen.sha256(previous / "protocol.json"),
               "original_runner_sha256": frozen.sha256(frozen.__file__),
               "repair": "PandasSeries field access first/second uses bracket indexing in reporting only",
               "objective_parameters_expert_selection_unchanged": True,
               "prior_partial_outputs": {str(path.relative_to(previous)): frozen.sha256(path) for path in previous.rglob("*") if path.is_file()},
               "tune_aligned_sha256": {fold: frozen.sha256(frozen.audit_tuning.OUT / f"{fold}_aligned_tune.parquet") for fold in ["F1", "F3"]}}
    frozen.write_json(frozen.OUT / "reporting_repair.json", receipt)
    shutil.copyfile(__file__, frozen.OUT / Path(__file__).name)
    frozen.day_sensitivity = day_sensitivity
    frozen.main()
    for path in (previous / "F1").glob("*.parquet"):
        before = pd.read_parquet(path)
        after = pd.read_parquet(frozen.OUT / "F1" / path.name)
        pd.testing.assert_frame_equal(before, after)
    assert frozen.read_json(previous / "F1/weights.json") == frozen.read_json(frozen.OUT / "F1/weights.json")
    print("REPORTING_REPAIR_IDENTICAL_F1_WEIGHTS_AND_PREDICTIONS", flush=True)
