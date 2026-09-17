"""Coordinator-launched exact residual moments on every finite NM clock."""
import lightgbm
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
sys.path.insert(0, str(HERE.parents[1]))
import run_full as core
import run_information
import adapter
from taxiout.metrics import paired_stability


def compare_control(record, fold, args):
    directory = core.ROOT / "private_runs/breakthrough_20260916/cpu/base" / f"lightgbm_aobt_allfinite_{fold}_s{args.seed}"
    if not (directory / "manifest.json").exists():
        return {"status": "control_not_available", "expected": str(directory)}
    control = core.read_json(directory / "manifest.json")
    if control["status"] != "complete":
        return {"status": "control_incomplete", "expected": str(directory)}
    assert core.object_hash(control["split"]) == core.object_hash(record["split"])
    assert control["fit_ids"] == record["fit_ids"]
    current = core.OUT / record["name"]
    comparisons = {}
    for variant in ("candidate", "blend25"):
        path = directory / f"{variant}.parquet"
        if core.sha256(path) != control["outputs"][path.name]:
            raise ValueError("Frozen LightGBM control artifact changed")
        before = pd.read_parquet(path)
        after = pd.read_parquet(current / f"{variant}.parquet")
        assert np.array_equal(before[core.ID], after[core.ID]) and np.array_equal(before[core.TARGET], after[core.TARGET])
        comparisons[variant] = paired_stability(before, after, repetitions=1000)
    return {"status": "compared", "control_manifest_sha256": core.sha256(directory / "manifest.json"),
            "comparisons": comparisons, "caveat": "Three heads increase capacity and compute; conventions arm also changes features"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", choices=["base", "conventions"], default="base")
    parser.add_argument("--folds", nargs="+", default=["F1", "F3"])
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--declare-only", action="store_true")
    args = parser.parse_args()
    args.family = "lightgbm_moments300"
    args.formulation = "aobt_allfinite"
    core.OUT = core.external_path(core.OUT / "residual_moments" / args.features)
    core.OUT.mkdir(parents=True, exist_ok=True)
    original_hashes = core.source_hashes
    paths = [Path(__file__), Path(adapter.__file__), Path(run_information.__file__),
             core.ROOT / "review_work/campaign_20260916/lgbm_adapter.py"]
    core.source_hashes = lambda: {**original_hashes(), **{str(path.relative_to(core.ROOT)): core.sha256(path) for path in paths}}
    core.importlib = SimpleNamespace(import_module=lambda name: adapter if name == "tabm_gpu" else importlib.import_module(name))
    x, meta = core.common.load_data()
    receipts = []
    if args.features == "conventions":
        x, receipts = run_information.additional(x, ["conventions"])
    x, offset, eligible, info = core.anchors(x, meta, args.formulation)
    info.update(feature_receipts=receipts, residual_decomposition="Exact center+upper+lower q300seconds")
    protocol = core.OUT / f"protocol_s{args.seed}.json"
    counts = {}
    for fold in args.folds:
        idx, split, _ = core.common.fold_data(meta, fold, full=True)
        counts[fold] = {stage: {"rows": int(eligible[positions].sum()),
                               "id_hash": core.object_hash(meta.iloc[positions[eligible[positions]]][core.ID].tolist())}
                        for stage, positions in idx.items()}
    payload = {"family": args.family, "formulation": args.formulation, "seed": args.seed, "threads": args.threads,
               "folds": args.folds, "features": args.features, "feature_columns": list(x), "feature_receipts": receipts,
               "source_hashes": core.source_hashes(), "counts": counts, "q_seconds": adapter.Q_SECONDS,
               "parts": {"center": "clip(r,-300,300)", "upper": "max(r-300,0)", "lower": "min(r+300,0)"},
               "identity": "center+upper+lower=r exactly; every finiteNM rawY residual retained including negative/>7200proxy",
               "parameters": adapter.parameters(args.seed, args.threads), "max_trees_per_head": adapter.MAX_TREES,
               "selection": "First minimizing commonprefix1..600 on original reconstructed residual tuneMSE; no independent component-MSE selection",
               "refit": "Independent all-refit model heads and encoder, commonprefixselected on originaltune",
               "variants": ["candidate", "blend25"], "blend_weight": .25,
               "reference": "Complete original scorecohort;missingNM preservedexactly;frozenfullLightGBMallfinitecontrol compared",
               "raw_hashes": core.read_json(core.ROOT / "private_runs/submission_v2/protocol.json")["raw_hashes"],
               "development": "Authorized exposeddevelopment search; no threshold sweep or fresh holdout claim"}
    if protocol.exists() and core.read_json(protocol) != payload:
        raise ValueError("Residual moment declaration changed")
    core.write_json(protocol, payload)
    snapshot = core.OUT / "source_snapshots"
    snapshot.mkdir(exist_ok=True)
    for path in paths + [Path(core.__file__)]:
        shutil.copyfile(path, snapshot / path.name)
    if args.declare_only:
        print("DECLARED", protocol, "features", len(x.columns), "counts", counts, flush=True)
        return
    records = {}
    controls = {}
    for fold in args.folds:
        records[fold] = core.run(args, fold, x, meta, offset, eligible, info, protocol)
        controls[fold] = compare_control(records[fold], fold, args)
        gc.collect()
    summary = {"created_utc": core.utc_now(), "control_comparison": controls}
    if set(records) == {"F1", "F3"}:
        summary["seasonal_rmse"] = {variant: core.season_score(records["F1"]["reports"][variant]["metrics"]["overall"],
                                                                records["F3"]["reports"][variant]["metrics"]["overall"])
                                     for variant in ("candidate", "blend25")}
    core.write_json(core.OUT / f"summary_s{args.seed}.json", summary)
    print("SUMMARY", summary, flush=True)


if __name__ == "__main__":
    main()
