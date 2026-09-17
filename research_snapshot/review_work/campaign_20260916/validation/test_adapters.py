"""Tiny serialization and fit-only unknown-category checks, no private data."""

import sys
import json
import hashlib
from pathlib import Path

# This Windows runtime needs LightGBM's DLL loaded before pandas/Arrow.
import lightgbm
import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))
import lgbm_adapter
import xgb_adapter


def main():
    rng = np.random.default_rng(17)
    x = pd.DataFrame({"number": rng.normal(size=500),
                      "category": pd.Categorical(np.tile(["A", "B", None, "C", "A"], 100))})
    y = x.number.to_numpy() * 40 + np.tile([20, 10, -5, -10, 20], 100)
    train, tune = x.iloc[:400].copy(), x.iloc[400:].copy()
    tune["category"] = tune.category.astype("string")
    tune.loc[tune.index[:10], "category"] = "UNSEEN"
    output = HERE.parents[3] / "private_runs/campaign_20260916/validation"
    output.mkdir(parents=True, exist_ok=True)
    results = {}
    for adapter in (lgbm_adapter, xgb_adapter):
        model, evidence = adapter.fit(train, y[:400], (tune, y[400:]), threads=2)
        assert "UNSEEN" not in model["encoder"].maps["category"]
        transformed = model["encoder"].transform(tune)
        assert (transformed.category.iloc[:10].astype(int) == 1).all()
        pred = adapter.predict(model, tune)
        assert pred.shape == (100,) and np.isfinite(pred).all()
        file = output / f"{adapter.__name__}_synthetic.joblib"
        joblib.dump(model, file)
        repeated = adapter.predict(joblib.load(file), tune)
        assert np.array_equal(pred, repeated)
        refit, final = adapter.fit(x, y, steps=evidence["steps"], threads=2)
        assert np.isfinite(adapter.predict(refit, tune)).all()
        assert final["steps"] == evidence["steps"]
        results[adapter.__name__] = {"selected_steps": evidence["steps"],
            "unknown_category_not_fit": True, "unknown_code": 1,
            "finite_predictions": True, "joblib_replay_exact": True,
            "refit_same_steps": True,
            "module_sha256": hashlib.sha256(Path(adapter.__file__).read_bytes()).hexdigest()}
        print(f"PASS {adapter.__name__}: steps={evidence['steps']}; finite unknowns; exact saved-model replay; refit")
    (output / "adapter_smoke_tests.json").write_text(json.dumps({
        "status": "passed", "synthetic_rows": 500, "threads": 2,
        "command": ".review-venv/Scripts/python.exe -B -u review_work/campaign_20260916/validation/test_adapters.py",
        "runtime_workaround": "Preload lightgbm before pandas/Arrow to avoid Windows native DLL access violation",
        "results": results}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
