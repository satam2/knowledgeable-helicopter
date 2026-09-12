import pandas as pd
from importlib.metadata import version

from taxiout.artifacts import object_hash, read_json, sha256, write_json
from taxiout.availability import POLICY_VERSION, make_observations
from taxiout.config import ROOT
from taxiout.features.pipeline import FeaturePipeline
from taxiout.io import audited_departures, concat_frames, read_raw, training_paths
from taxiout.schema import FEATURE_INPUTS, FLIGHT_ID, ID, MOVEMENT, PHASE, TARGET, duration, unique_ids


def cache_identity(config, manifest):
    source = [*ROOT.glob("src/taxiout/features/*.py"), ROOT / "src/taxiout/availability.py", ROOT / "src/taxiout/schema.py", ROOT / "src/taxiout/cache.py"]
    return {"policy": POLICY_VERSION, "features": config["features"], "coverage": config["coverage"],
            "libraries": {name: version(name) for name in ["numpy", "pandas", "pyarrow", "pytz", "tzdata"]},
            "inputs": {f["file"]: f["sha256"] for f in manifest["files"]},
            "source": {p.name: sha256(p) for p in source}, "fit_ids": None, "split_ids": None,
            "encoder_state": "none: deterministic observation-only features; safe across folds"}


def prepare(config, manifest):
    identity = cache_identity(config, manifest)
    out = ROOT / "data/interim/features" / object_hash(identity)[:16]
    out.mkdir(parents=True, exist_ok=True)
    paths = training_paths(config)
    pipeline = FeaturePipeline(config)
    pending = []
    for path in paths:
        cache = out / path.name
        marker = cache.with_suffix(".json")
        if not marker.exists():
            pending.append(path)
            continue
        record = read_json(marker)
        if record["identity"] != identity or not cache.exists() or record["sha256"] != sha256(cache):
            raise ValueError(f"Incompatible or corrupt feature cache: {cache}")
    if not pending:
        return out
    context = None
    if config["features"]["traffic"]:
        context = make_observations(concat_frames([read_raw(p, columns=[ID, PHASE, MOVEMENT, "ADEP_mvt", "ADES_mvt"]) for p in paths]))[0]
        context_month = context[MOVEMENT].dt.year * 12 + context[MOVEMENT].dt.month
    for path in pending:
        cache = out / path.name
        marker = cache.with_suffix(".json")
        obs, _, _ = make_observations(read_raw(path, FEATURE_INPUTS))
        local_context = context
        if context is not None and config["coverage"] == "month-isolated":
            months = (obs[MOVEMENT].dt.year * 12 + obs[MOVEMENT].dt.month).unique()
            local_context = context.loc[context_month.isin(months)]
        x = pipeline.transform(obs, local_context)
        x.reset_index().to_parquet(cache, index=False)
        write_json(marker, {"identity": identity, "sha256": sha256(cache), "rows": len(x)})
        print(f"Cached {path.name}: {len(x):,} departures, {len(x.columns)} features", flush=True)
    return out


def load_training(config, manifest):
    out = prepare(config, manifest)
    x = concat_frames([pd.read_parquet(out / p.name) for p in training_paths(config)])
    unique_ids(x)
    meta = audited_departures(manifest)
    if not x[ID].equals(meta[ID]):
        raise ValueError("Cached features and audited labels have different IDs/order")
    labels = meta[[ID, TARGET]].copy()
    meta = meta.drop(columns=TARGET)
    return x.set_index(ID), meta, labels


def ranking_features(config, path, pipeline):
    obs, _, _ = make_observations(read_raw(path, FEATURE_INPUTS))
    x = pipeline.transform(obs)
    dep = obs.loc[obs[PHASE].eq("DEP")]
    meta = dep[[ID, FLIGHT_ID, "ADEP_mvt", MOVEMENT]].copy().reset_index(drop=True)
    meta["proxy_sec"] = duration(dep[MOVEMENT], dep["AOBT_3_flt"]).to_numpy()
    return x, meta
