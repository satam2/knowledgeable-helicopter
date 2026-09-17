from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from taxiout.artifacts import Run, write_json
from taxiout.availability import make_observations
from taxiout.config import ROOT
from taxiout.features.pipeline import FeaturePipeline
from taxiout.io import training_paths
from taxiout.predict import predict_features
from taxiout.schema import ID


def test_private_json_cannot_be_written_inside_checkout():
    with pytest.raises(ValueError, match="repository"):
        write_json(ROOT / "privacy-probe.json", {"synthetic": True})


def test_run_requires_external_artifact_root(monkeypatch, config):
    monkeypatch.delenv("TAXIOUT_ARTIFACT_ROOT", raising=False)
    with pytest.raises(ValueError, match="TAXIOUT_ARTIFACT_ROOT"):
        Run("synthetic", config, {})


def test_run_writes_only_to_external_root(tmp_path, monkeypatch, config):
    monkeypatch.setenv("TAXIOUT_ARTIFACT_ROOT", str(tmp_path))
    run = Run("synthetic", config, {})
    assert run.path.parent == tmp_path / "models"
    assert (run.path / "manifest.json").exists()


def test_raw_directory_inside_repository_rejected(config, monkeypatch):
    monkeypatch.delenv("TAXIOUT_RAW_DIR", raising=False)
    with pytest.raises(ValueError, match="repository"):
        training_paths({**config, "raw_dir": str(ROOT / "data")})


def test_prefix_uses_observed_text_and_training_only_support(config, movements):
    config = {**config, "features": {**config["features"], "flight_prefix": True}}
    movements = movements.iloc[:5].copy()
    movements["FLIGHT_mvt"] = ["AB123", "ABC4", None, "123AB", "ZZZ9"]
    obs, _, _ = make_observations(movements)
    pipeline = FeaturePipeline(config)
    x = pipeline.transform(obs)
    assert "flight_prefix" in x
    assert x.flight_prefix.astype(str).tolist() == ["s2:AB", "s3:ABC", "m:", "m:", "s3:ZZZ"]
    assert x.flight_prefix_unparsed.tolist() == [0, 0, 1, 1, 0]
    pipeline.fit_features(x.iloc[:4])
    assert pipeline.new_category(x).tolist() == [False, False, False, False, True]


def test_rome_schedule_route_is_observation_only_and_keeps_fallbacks(config, movements):
    class Constant:
        def __init__(self, value):
            self.value = value

        def predict(self, x, **kwargs):
            return np.full(len(x), self.value, dtype=float)

    obs, _, _ = make_observations(movements.iloc[:5])
    pipeline = FeaturePipeline(config)
    x = pipeline.transform(obs)
    pipeline.fit_features(x)
    meta = pd.DataFrame({ID: x.index, "ADEP_mvt": ["LIRF", "EDDF", "LIRF", "LIRF", "LIRF"],
                         "proxy_sec": [np.nan, np.nan, np.nan, 960, np.nan],
                         "schedule_sec": [10000, 10000, np.nan, 10000, -100]})
    models = {"direct": Constant(1000), "residual": Constant(-60),
              "missing": Constant(800), "rome_schedule": Constant(-20)}
    enabled = {**config, "rome_schedule_residual": True}
    result = predict_features(pipeline, models, enabled, x, meta)
    assert result.prediction_sec.tolist() == [9980, 800, 800, 900, -120]
    assert result.route.tolist() == ["rome_schedule_residual", "specialist_missing", "specialist_missing",
                                     "residual", "rome_schedule_residual"]
    disabled = predict_features(pipeline, models, config, x, meta)
    assert disabled.prediction_sec.tolist() == [800, 800, 800, 900, 800]
