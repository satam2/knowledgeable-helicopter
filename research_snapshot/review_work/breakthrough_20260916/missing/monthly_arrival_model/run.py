"""Matched missing-clock template CatBoost plus fifteen retrospective ARR fields."""
import os
for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[key] = "1"
import argparse
import gc
from pathlib import Path
import shutil
import sys
import numpy as np
import pandas as pd
import psutil

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "review_work/breakthrough_20260916/missing"))
import run_missing_models as base
import run_id_context as context

CACHE = ROOT / "private_runs/breakthrough_20260916/retrospective_research/monthly_arrival"
OUT = base.external_path(ROOT / "private_runs/breakthrough_20260916/missing/monthly_arrival_model")
CONTROL = ROOT / "private_runs/breakthrough_20260916/missing/id_context_v1/models"
COLUMNS = [f"monthly_arr_{group}_{stat}" for group in ["airport", "stand", "runway"]
           for stat in ["count", "mean_sec", "median_sec", "q10_sec", "q90_sec"]]

def declare():
    sources = [Path(__file__), Path(base.__file__), Path(context.__file__), Path(base.common.__file__)]
    payload = {"source_hashes": {str(path.relative_to(ROOT)): base.sha256(path) for path in sources},
        "folds": ["F1", "F3"], "seed": 20260916, "threads": 2,
        "arm": "historical_template", "added_features": COLUMNS, "parameters": base.PARAMS,
        "control": "Original id_context_v1 historical_template, same full missing cohorts and raw labels.",
        "fit": "Unchanged chronological template crossfit and CatBoost raw residual, original tune stopping and fresh refit.",
        "variants": ["candidate", "blend25"], "blend_weight": .25,
        "availability": "Explicit retrospective final monthly ARR durations, with future/tied completions allowed; same movement/flight exclusions and deduplication inherited from verified cache. Not real-time.",
        "scope": "Adaptive information follow-up excluded from final9. No coefficient grid, clipping, changed score labels or row removal.",
        "cache_manifest_sha256": base.sha256(CACHE / "manifest.json"),
        "cache_verification_sha256": base.sha256(CACHE / "verification.json"),
        "cache_independent_oracle_sha256": base.sha256(CACHE / "independent_oracle.json"),
        "controls": {fold: base.sha256(CONTROL / f"historical_template_{fold}_s20260916/manifest.json") for fold in ["F1", "F3"]}}
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "protocol.json"
    if path.exists():
        assert base.object_hash(base.read_json(path)["declaration"]) == base.object_hash(payload)
    else:
        base.write_json(path, {"created_utc": base.utc_now(), "declaration": payload})
        snapshot = OUT / "source_snapshots"
        snapshot.mkdir()
        for source in sources:
            shutil.copyfile(source, snapshot / source.name)
    return payload

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--declare-only", action="store_true")
    args = parser.parse_args()
    payload = declare()
    if args.declare_only:
        print("MONTHLY_ARR_MISSING_DECLARED_NO_SCORE_READS", flush=True)
        return
    assert psutil.virtual_memory().available >= 12 * 1024**3
    manifest = base.read_json(CACHE / "manifest.json")
    verification = base.read_json(CACHE / "verification.json")
    oracle = base.read_json(CACHE / "independent_oracle.json")
    assert manifest["status"] == "complete" and verification["status"] == oracle["status"] == "passed"
    assert verification["manifest_sha256"] == oracle["manifest_sha256"] == base.sha256(CACHE / "manifest.json")
    assert manifest["features"] == COLUMNS
    path = CACHE / "training_features.parquet"
    assert base.sha256(path) == manifest["outputs"][path.name]
    x, meta = base.load_data()
    audit = base.read_json(context.CACHE / "audit.json")
    peerpath = context.CACHE / "features.parquet"
    assert base.sha256(peerpath) == audit["feature_sha256"]
    peer = pd.read_parquet(peerpath).set_index(base.ID)
    np.testing.assert_array_equal(peer.index, meta[base.ID])
    timestamps = meta.set_index(base.ID).loc[x.index, base.MOVEMENT]
    x = pd.concat([x, context.id_context_features(x, peer.loc[x.index], timestamps)], axis=1)
    del peer
    gc.collect()
    controls = {fold: base.read_json(CONTROL / f"historical_template_{fold}_s20260916/manifest.json") for fold in ["F1", "F3"]}
    for control in controls.values():
        assert control["status"] == "complete" and control["features_used"] == list(x)
        assert control["threads"] == 2 and control["seed"] == 20260916
    added = pd.read_parquet(path).set_index(base.ID)
    assert added.index.is_unique and list(added) == COLUMNS
    np.testing.assert_array_equal(added.index, meta[base.ID])
    added = added.loc[x.index]
    assert not set(added).intersection(x)
    x = pd.concat([x, added.replace([np.inf, -np.inf], np.nan).fillna(-999999).astype("float32")], axis=1)
    del added
    gc.collect()
    receipts = {"manifest_sha256": base.sha256(CACHE / "manifest.json"),
        "verification_sha256": base.sha256(CACHE / "verification.json"),
        "independent_oracle_sha256": base.sha256(CACHE / "independent_oracle.json"),
        "training_features_sha256": base.sha256(path), "context_sha256": audit["feature_sha256"]}
    execution = OUT / "execution_inputs.json"
    assert not execution.exists(), "Preserve prior executed attempt"
    base.write_json(execution, {"created_utc": base.utc_now(), "receipts": receipts, "features": list(x)})
    original_write = base.write_json
    def write_bound(path, value):
        if Path(path).name == "manifest.json":
            value.update(features="Original airport and ID-context fields plus fifteen monthly ARR fields",
                source_hashes=payload["source_hashes"], anchor={"monthly_arrival": receipts},
                formulation="missing_template_idcontext_monthly_arrival", feature_columns=list(x))
        original_write(path, value)
    base.OUT = OUT
    base.write_json = write_bound
    for fold in ["F1", "F3"]:
        index, split, _ = base.common.fold_data(meta, fold, full=True)
        eligible = ~np.isfinite(meta.proxy_sec.to_numpy(float))
        identities = {stage: {"n": int(eligible[rows].sum()),
            "hash": base.object_hash(meta.iloc[rows[eligible[rows]]][base.ID].tolist())} for stage, rows in index.items()}
        assert identities == controls[fold]["fit_ids"]
        assert base.object_hash(split) == base.object_hash(controls[fold]["split"])
        assert psutil.virtual_memory().available >= 8 * 1024**3
        base.run_arm("historical_template", fold, x, meta, 20260916, 2)
    records = {fold: base.read_json(OUT / f"models/historical_template_{fold}_s20260916/manifest.json") for fold in ["F1", "F3"]}
    scores = {variant: float(np.sqrt(sum(weight * records[fold]["reports"][variant]["metrics"]["overall"]["rmse_sec"]**2
        for fold, weight in [("F1", 192122/344841), ("F3", 152719/344841)]))) for variant in ["candidate", "blend25"]}
    base.write_json(OUT / "summary.json", {"seasonal_rmse": scores, "execution_inputs_sha256": base.sha256(execution)})
    print("MONTHLY_ARR_MISSING_SEASONAL", scores, flush=True)

if __name__ == "__main__":
    main()
