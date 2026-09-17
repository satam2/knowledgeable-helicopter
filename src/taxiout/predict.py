import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from taxiout.artifacts import inference_source_hashes, object_hash, read_json, sha256, write_json
from taxiout.availability import POLICY_VERSION
from taxiout.cache import ranking_features
from taxiout.config import ROOT
from taxiout.paths import external_path
from taxiout.features.pipeline import FeaturePipeline
from taxiout.models.residual import combine, proxy_status
from taxiout.schema import ID


def save_bundle(path, pipeline, models, config):
    path = external_path(path)
    for name, model in models.items():
        model.save_model(str(path / (name + ".cbm")))
    state = {"policy": POLICY_VERSION, "config": config, "columns": pipeline.columns,
             "inference_source_hashes": inference_source_hashes(),
             "dtypes": pipeline.dtypes, "vocabulary": {k: sorted(v) for k, v in pipeline.vocabulary.items()},
             "models": {name: sha256(path / (name + ".cbm")) for name in models}}
    state["schema_hash"] = object_hash({"columns": state["columns"], "dtypes": state["dtypes"]})
    write_json(path / "bundle.json", state)


def load_bundle(path, allow_incomplete=False):
    path = external_path(path)
    manifest = read_json(path / "manifest.json")
    if not allow_incomplete and manifest["status"] != "complete":
        raise ValueError("Refusing incomplete model run")
    state = read_json(path / "bundle.json")
    if state["policy"] != POLICY_VERSION:
        raise ValueError("Observation policy mismatch")
    if state["inference_source_hashes"] != inference_source_hashes():
        raise ValueError("Inference source differs from serialized feature policy")
    if state["schema_hash"] != object_hash({"columns": state["columns"], "dtypes": state["dtypes"]}):
        raise ValueError("Corrupt feature schema")
    if not allow_incomplete and manifest["outputs"].get("bundle.json") != sha256(path / "bundle.json"):
        raise ValueError("Corrupt preprocessing state")
    pipeline = FeaturePipeline(state["config"])
    pipeline.columns, pipeline.dtypes = state["columns"], state["dtypes"]
    pipeline.vocabulary = {k: set(v) for k, v in state["vocabulary"].items()}
    models = {}
    for name, digest in state["models"].items():
        file = path / (name + ".cbm")
        if sha256(file) != digest:
            raise ValueError("Model checksum mismatch")
        model = CatBoostRegressor()
        model.load_model(str(file))
        models[name] = model
    if "direct" not in models:
        raise ValueError("Complete direct fallback model is required")
    return pipeline, models, state["config"]


def predict_features(pipeline, models, config, x, meta):
    if list(x) != pipeline.columns or {c: str(x[c].dtype) for c in x} != pipeline.dtypes:
        raise ValueError("Model feature names/dtypes differ")
    if not np.array_equal(x.index, meta[ID]):
        raise ValueError("Prediction metadata is not ID aligned")
    status = proxy_status(meta["proxy_sec"], config)
    values = models["direct"].predict(x, thread_count=config["threads"])
    route = np.char.add("direct_", status).astype("U32")
    if "residual" in models:
        correction = models["residual"].predict(x, thread_count=config["threads"])
        values, route = combine(meta["proxy_sec"], correction, values, status)
        route = route.astype("U32")
        if config.get("long_proxy_correction", False):
            long_proxy = np.isfinite(meta["proxy_sec"]) & meta["proxy_sec"].gt(config["proxy_max"])
            values[long_proxy] = meta.loc[long_proxy, "proxy_sec"].to_numpy() + correction[long_proxy]
            route[long_proxy] = "residual_long_proxy"
    if "missing" in models:
        mask = status == "missing"
        if mask.any():
            values[mask] = models["missing"].predict(x.iloc[np.flatnonzero(mask)], thread_count=config["threads"])
            route[mask] = "specialist_missing"
    if config.get("rome_schedule_residual", False) and "rome_schedule" in models:
        mask = ((status == "missing") & meta["ADEP_mvt"].eq("LIRF").to_numpy()
                & np.isfinite(meta["schedule_sec"].to_numpy()))
        if mask.any():
            values[mask] = (meta.loc[mask, "schedule_sec"].to_numpy()
                            + models["rome_schedule"].predict(x.iloc[np.flatnonzero(mask)], thread_count=config["threads"]))
            route[mask] = "rome_schedule_residual"
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite model prediction")
    output = meta.copy()
    output["prediction_sec"] = values
    output["route"] = route
    output["proxy_status"] = proxy_status(meta["proxy_sec"], {"proxy_min": 0, "proxy_max": 7200})
    output["routing_status"] = status
    output["new_category"] = pipeline.new_category(x)
    return output


def predict_candidate(bundle, input_path, output=None):
    pipeline, models, config = load_bundle(bundle)
    x, meta = ranking_features(config, external_path(input_path), pipeline)
    predictions = predict_features(pipeline, models, config, x, meta)
    if output:
        path = external_path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_parquet(path, index=False)
    return predictions
