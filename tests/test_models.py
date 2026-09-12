import numpy as np
import pandas as pd
import pytest

from taxiout.artifacts import Run
from taxiout.availability import make_observations
from taxiout.features.pipeline import FeaturePipeline
from taxiout.models.baselines import HierarchicalMean
from taxiout.models.direct import fit_model
from taxiout.models.residual import combine, proxy_status, residual_target
from taxiout.predict import load_bundle, predict_features, save_bundle
from taxiout.schema import BLOCK, ID, TARGET


def test_residual_sign_and_fallback(config):
    assert residual_target([900], [960]).tolist() == [-60]
    proxy = [960., np.nan, -100., 99999.]
    status = proxy_status(proxy, config)
    prediction, routes = combine(proxy, [-60] * 4, [1000] * 4, status)
    assert prediction.tolist() == [900, 1000, 1000, 1000]
    assert routes.tolist() == ["residual", "direct_missing", "direct_invalid", "direct_invalid"]


def test_combined_route_names_and_quality_slices(movements, config):
    class Constant:
        def __init__(self, value):
            self.value = value

        def predict(self, x, **kwargs):
            return np.full(len(x), self.value, dtype=float)

    obs, _, _ = make_observations(movements.iloc[:3])
    pipeline = FeaturePipeline(config)
    x = pipeline.transform(obs)
    pipeline.fit_features(x)
    meta = obs[[ID, "ADEP_mvt"]].copy()
    meta["proxy_sec"] = [960., np.nan, 87000.]
    models = {"direct": Constant(1000), "residual": Constant(-60), "missing": Constant(800)}
    output = predict_features(pipeline, models, config, x, meta)
    assert output.prediction_sec.tolist() == [900., 800., 1000.]
    assert output.route.tolist() == ["residual", "specialist_missing", "direct_invalid"]
    relaxed = {**config, "proxy_max": 1e12}
    output = predict_features(pipeline, models, relaxed, x, meta)
    assert output.prediction_sec.tolist() == [900., 800., 86940.]
    assert output.proxy_status.tolist() == ["present", "missing", "invalid"]
    assert output.routing_status.tolist() == ["present", "missing", "present"]
    output = predict_features(pipeline, models, {**config, "long_proxy_correction": True}, x, meta)
    assert output.prediction_sec.tolist() == [900., 800., 86940.]
    assert output.route.tolist() == ["residual", "specialist_missing", "residual_long_proxy"]


def test_prior_oof_no_own_label_and_unknown_backoff():
    x = pd.DataFrame({ID: [1., 2., 3., 4.], "ADEP_mvt": ["A"] * 4, "RUNWAY_mvt": ["R"] * 4, "STAND_mvt": ["S"] * 4}).set_index(ID)
    labels = pd.DataFrame({ID: x.index, TARGET: [10., 20., 30., 40.]})
    first = HierarchicalMean().fit_transform_oof(x, labels, folds=4)
    labels.loc[0, TARGET] = 1000000
    second = HierarchicalMean().fit_transform_oof(x, labels, folds=4)
    assert first[0] == second[0] == 30
    estimator = HierarchicalMean().fit(x, labels)
    unknown = x.iloc[:1].assign(ADEP_mvt="new", RUNWAY_mvt="new", STAND_mvt="new")
    assert estimator.predict(unknown)[0] == labels[TARGET].mean()


def test_model_bundle_smoke_and_prediction_leakage(tmp_path, movements, config):
    config = {**config, "iterations": 8, "depth": 3, "threads": 1}
    obs, labels, _ = make_observations(movements)
    pipeline = FeaturePipeline(config)
    x = pipeline.transform(obs)
    pipeline.fit_features(x.iloc[:32])
    model, _ = fit_model(x.iloc[:32], labels.iloc[:32][TARGET].to_numpy(float), config)
    meta = movements[[ID, "ADEP_mvt"]].copy()
    meta["proxy_sec"] = 960.
    before = predict_features(pipeline, {"direct": model}, config, x.iloc[32:], meta.iloc[32:])
    movements[TARGET] = 1e9
    movements[BLOCK] = pd.Timestamp("1900-01-01", tz="UTC")
    altered, _, _ = make_observations(movements)
    after = predict_features(pipeline, {"direct": model}, config, pipeline.transform(altered).iloc[32:], meta.iloc[32:])
    np.testing.assert_array_equal(before.prediction_sec, after.prediction_sec)
    from taxiout.artifacts import write_json
    write_json(tmp_path / "manifest.json", {"status": "incomplete"})
    save_bundle(tmp_path, pipeline, {"direct": model}, config)
    reloaded, models, state = load_bundle(tmp_path, allow_incomplete=True)
    repeated = predict_features(reloaded, models, state, x.iloc[32:], meta.iloc[32:])
    np.testing.assert_allclose(before.prediction_sec, repeated.prediction_sec, rtol=0, atol=1e-9)
    unknown = altered.copy()
    unknown.loc[32:, "STAND_mvt"] = "NEVER_OBSERVED"
    unknown_x = reloaded.transform(unknown).iloc[32:]
    unknown_prediction = predict_features(reloaded, models, state, unknown_x, meta.iloc[32:])
    assert unknown_prediction.new_category.all()
    assert np.isfinite(unknown_prediction.prediction_sec).all()
    with pytest.raises(ValueError, match="incomplete"):
        load_bundle(tmp_path)
    with pytest.raises(ValueError, match="names/dtypes"):
        predict_features(reloaded, models, state, x.iloc[32:].drop(columns="utc_hour"), meta.iloc[32:])
