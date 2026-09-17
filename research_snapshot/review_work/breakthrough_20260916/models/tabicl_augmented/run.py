"""Coordinator-launched augmented-context TabICL missing-record comparison."""
import torch
import argparse
import gc
import importlib
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import run_full as core
import adapter
import run_id_context as context
from taxiout.metrics import paired_stability


def compare_template(record, fold):
    directory = core.ROOT / "private_runs/breakthrough_20260916/missing/id_context_v1/models" / f"historical_template_{fold}_s20260916"
    manifest = core.read_json(directory / "manifest.json")
    assert manifest["status"] == "complete" and manifest["fit_ids"] == record["fit_ids"]
    assert core.object_hash(manifest["split"]) == core.object_hash(record["split"])
    comparisons = {}
    for variant in ("candidate", "blend25"):
        path = directory / f"{variant}.parquet"
        assert core.sha256(path) == manifest["outputs"][path.name]
        before = pd.read_parquet(path)
        after = pd.read_parquet(core.OUT / record["name"] / path.name)
        assert np.array_equal(before[core.ID], after[core.ID]) and np.array_equal(before[core.TARGET], after[core.TARGET])
        comparisons[variant] = paired_stability(before, after, repetitions=1000)
    return {"control_manifest_sha256": core.sha256(directory / "manifest.json"), "comparisons": comparisons,
            "caveat": "Same raw feature information, different encoders and model families; CatBoostfits template residual whileTabICLfits rawY"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", nargs="+", default=["F1", "F3"])
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--declare-only", action="store_true")
    args = parser.parse_args()
    args.family = "tabicl_augmented"
    args.formulation = "missing_direct"
    core.OUT = core.external_path(core.OUT / "tabicl_augmented")
    core.OUT.mkdir(parents=True, exist_ok=True)
    original_hashes = core.source_hashes
    paths = [Path(__file__), Path(adapter.__file__), Path(adapter.frozen.__file__), Path(context.__file__), Path(adapter.missing.__file__)]
    core.source_hashes = lambda: {**original_hashes(), **{str(path.relative_to(core.ROOT)): core.sha256(path) for path in paths}}
    core.importlib = SimpleNamespace(import_module=lambda name: adapter if name == "tabm_gpu" else importlib.import_module(name))
    x, meta = adapter.missing.load_data()
    audit = core.read_json(context.CACHE / "audit.json")
    peerpath = context.CACHE / "features.parquet"
    assert core.sha256(peerpath) == audit["feature_sha256"]
    peer = pd.read_parquet(peerpath).set_index(core.ID)
    assert np.array_equal(peer.index, meta[core.ID])
    times = meta.set_index(core.ID).loc[x.index, adapter.missing.MOVEMENT]
    x = pd.concat([x, context.id_context_features(x, peer.loc[x.index], times)], axis=1)
    feature_control = core.read_json(core.ROOT / "private_runs/breakthrough_20260916/missing/id_context_v1/feature_manifest.json")
    assert list(x) == feature_control["columns"]
    x[adapter.TIME_COLUMN] = times
    forest_protocol = core.read_json(core.ROOT / "private_runs/breakthrough_20260916/models/missing_forest/protocol_extratrees_s20260916.json")
    assert list(x) == forest_protocol["features"]
    x = x.reindex(pd.Index(meta[core.ID], name=core.ID))
    eligible = ~np.isfinite(meta.proxy_sec.to_numpy(float))
    assert core.sha256(adapter.frozen.CHECKPOINT) == adapter.frozen.EXPECTED_SHA256
    protocol = core.OUT / f"protocol_s{args.seed}.json"
    payload = {"family": args.family, "seed": args.seed, "threads": args.threads, "folds": args.folds,
               "features": list(x), "source_hashes": core.source_hashes(), "context_sha256": audit["feature_sha256"],
               "checkpoint_sha256": adapter.frozen.EXPECTED_SHA256, "forest_feature_columns_exact_equal": True,
               "catboost_feature_columns_exact_equal_without_time_metadata": True,
               "template": "Earliermonth crossfit train features; fitprior for tune; freshrefitprior for score",
               "labels": "Direct unmodified original rawY, affine scaling insideTabICL; allmissingcontext retained",
               "parameters": {"estimators": 4, "batch_size": 1, "kv_cache": "repr", "offload": "cpu", "amp": True,
                              "fa3": False, "query_chunk": 512, "allow_auto_download": False},
               "mean_limit": "Arithmeticmean of999 predicted quantiles; no explicit tailintegration or guaranteed rawMSEoptimality",
               "network": "Offline localpinnedcheckpoint;HF_HUB_OFFLINE=1,TRANSFORMERS_OFFLINE=1,telemetrydisabled",
               "variants": ["candidate", "blend25"], "blend_weight": .25,
               "control": "Saved CatBoosthistoricaltemplate+IDcontext candidate/blend on identicalfolds and rawfeatureinformation",
               "selection": "Authorized adaptively exposeddevelopment comparison; no freshholdout claim",
               "raw_hashes": core.read_json(core.ROOT / "private_runs/submission_v2/protocol.json")["raw_hashes"]}
    if protocol.exists() and core.read_json(protocol) != payload:
        raise ValueError("Augmented TabICL protocol changed")
    core.write_json(protocol, payload)
    snapshot = core.OUT / "source_snapshots"
    snapshot.mkdir(exist_ok=True)
    for path in paths + [Path(core.__file__)]:
        shutil.copyfile(path, snapshot / path.name)
    if args.declare_only:
        print("DECLARED", protocol, "input_columns", len(x.columns), "model_columns", len(x.columns) - 1 + 5, flush=True)
        return
    info = {"rule": "All originalmissingNM contexts;rawY mean;finiteNM scoreroute preserved",
            "features": "Airport+IDcontext+earliermonth historicaltemplate", "context_sha256": audit["feature_sha256"]}
    records = {}
    comparison = {}
    for fold in args.folds:
        records[fold] = core.run(args, fold, x, meta, np.zeros(len(meta)), eligible, info, protocol)
        comparison[fold] = compare_template(records[fold], fold)
        gc.collect()
    summary = {"created_utc": core.utc_now(), "catboost_comparison": comparison}
    if set(records) == {"F1", "F3"}:
        summary["seasonal_rmse"] = {variant: core.season_score(records["F1"]["reports"][variant]["metrics"]["overall"],
                                                                records["F3"]["reports"][variant]["metrics"]["overall"])
                                     for variant in ("candidate", "blend25")}
    core.write_json(core.OUT / f"summary_s{args.seed}.json", summary)
    print("SUMMARY", summary, flush=True)


if __name__ == "__main__":
    main()
