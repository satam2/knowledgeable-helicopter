from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from pandas.api.types import union_categoricals

from taxiout.artifacts import object_hash, read_json, sha256
from taxiout.config import ROOT
from taxiout.paths import artifact_path, raw_root


def read_raw(path, columns=None):
    return pq.read_table(path, columns=columns).to_pandas(strings_to_categorical=True)


def concat_frames(frames):
    if not frames:
        raise ValueError("No input frames")
    data = {}
    for col in frames[0]:
        if all(isinstance(f[col].dtype, pd.CategoricalDtype) for f in frames):
            data[col] = union_categoricals([f[col].array for f in frames])
        else:
            data[col] = pd.concat([f[col] for f in frames], ignore_index=True)
    return pd.DataFrame(data)


def training_paths(config):
    return sorted(raw_root(config).glob("training_*.parquet"))


def audited_departures(manifest=None, columns=None):
    report = read_json(artifact_path("reports/data_audit.json"))
    path = artifact_path("data/interim/audit/departures.parquet")
    if report.get("artifacts", {}).get(path.name) != sha256(path):
        raise ValueError("Audited departure cache missing/corrupt; rerun audit")
    if manifest is not None and report["input_manifest_hash"] != object_hash(manifest):
        raise ValueError("Audited labels belong to a different input pack")
    return pd.read_parquet(path, columns=columns)


def verified_manifest(config):
    manifest = read_json(artifact_path("reports/input_manifest.json"))
    current = [*training_paths(config), raw_root(config) / "ranking.parquet", raw_root(config) / "submitting.parquet"]
    actual = {p.name: sha256(p) for p in current}
    expected = {p["file"]: p["sha256"] for p in manifest["files"]}
    if actual != expected:
        raise ValueError("Input pack changed; rerun audit and invalidate dependent runs")
    return {k: v for k, v in manifest.items() if k != "created_utc"}
