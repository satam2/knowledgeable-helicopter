"""Feature-augmented full-data GPU comparisons; frozen core remains unmodified."""
import lightgbm
import argparse
import gc
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import run_full as core
import feature_screen

BATCH_PATTERNS = {
    "surface_T": r"^batch_surface_T_",
    "surface_N": r"^batch_surface_N_",
    "source_past": r"^batch_source_(airport|stand)_past_",
    "source_twosided": r"^batch_source_(airport|stand)_twosided_",
}


def batch_cache(root):
    manifest = core.read_json(root / "manifest.json")
    verification = core.read_json(root / "verification.json")
    if manifest.get("status") != "complete" or verification.get("status") != "passed":
        raise ValueError("Batch context cache is not complete and verified")
    if not manifest.get("training_id_order_verified"):
        raise ValueError("Batch context producer has not verified training ID order")
    path = root / "training_features.parquet"
    if core.sha256(path) != manifest["outputs"][path.name]:
        raise ValueError("Batch context hash mismatch")
    return manifest, pd.read_parquet(path).set_index(core.ID)


def augment(x, blocks, batch_root=None):
    receipts = []
    if "trajectory" in blocks:
        x, receipts = feature_screen.augmented(x, "trajectory")
    selected = [block for block in blocks if block in BATCH_PATTERNS]
    if selected:
        root = batch_root or core.ROOT / "private_runs/breakthrough_20260916/batch_context"
        manifest, ext = batch_cache(root)
        if not ext.index.is_unique or not x.index.is_unique or not np.array_equal(ext.index, x.index):
            raise ValueError("Batch context movement IDs do not match model input order exactly")
        columns = []
        for block in selected:
            cols = [column for column in ext if re.match(BATCH_PATTERNS[block], column)]
            if not cols:
                raise ValueError(f"Empty feature block: {block}")
            columns.extend(cols)
        if len(columns) != len(set(columns)) or set(columns).intersection(x.columns):
            raise ValueError("Duplicate augmented feature columns")
        x = pd.concat([x, ext[columns].replace([np.inf, -np.inf], np.nan).fillna(-999999).astype("float32")], axis=1)
        receipts.append({"path": str(root / "manifest.json"), "sha256": core.sha256(root / "manifest.json"),
                         "verification_sha256": core.sha256(root / "verification.json"),
                         "features_sha256": manifest["outputs"]["training_features.parquet"],
                         "columns": columns, "availability": "Explicit retrospective final supplied movement batch"})
    return x, receipts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", required=True, choices=["xgb", "tabm"])
    parser.add_argument("--formulation", required=True, choices=["direct", "aobt_allfinite", "multi_anchor"])
    parser.add_argument("--features", nargs="+", required=True, choices=["trajectory", *BATCH_PATTERNS])
    parser.add_argument("--folds", nargs="+", default=["F1", "F3"])
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--declare-only", action="store_true")
    args = parser.parse_args()
    if len(args.features) != len(set(args.features)):
        raise ValueError("Duplicate feature blocks")
    args.features = sorted(args.features)
    core.OUT = core.external_path(core.OUT / "augmented" / "__".join(args.features))
    original_hashes = core.source_hashes
    core.source_hashes = lambda: {**original_hashes(), "run_augmented.py": core.sha256(__file__),
                                 "feature_screen.py": core.sha256(feature_screen.__file__)}
    x, meta = core.common.load_data()
    x, receipts = augment(x, args.features)
    x, offset, eligible, info = core.anchors(x, meta, args.formulation)
    info = {**info, "feature_blocks": args.features, "feature_receipts": receipts}
    protocol = core.declare(args)
    feature_protocol = protocol.with_name(protocol.stem + "_features.json")
    payload = {"blocks": args.features, "receipts": receipts, "columns": list(x), "source_hashes": core.source_hashes()}
    if feature_protocol.exists() and core.read_json(feature_protocol) != payload:
        raise ValueError("Feature protocol changed")
    core.write_json(feature_protocol, payload)
    snapshot = core.OUT / "source_snapshots" / f"{args.family}_{args.formulation}_s{args.seed}"
    for path in [Path(__file__), Path(feature_screen.__file__)]:
        shutil.copyfile(path, snapshot / path.name)
    if args.declare_only:
        print("DECLARED", protocol, feature_protocol, "shape", x.shape, flush=True)
        return
    records = {}
    for fold in args.folds:
        records[fold] = core.run(args, fold, x, meta, offset, eligible, info, protocol)
        gc.collect()
    if set(records) == {"F1", "F3"}:
        report = {variant: core.season_score(records["F1"]["reports"][variant]["metrics"]["overall"],
                                             records["F3"]["reports"][variant]["metrics"]["overall"])
                  for variant in ("candidate", "blend25")}
        core.write_json(core.OUT / f"{args.family}_{args.formulation}_s{args.seed}_summary.json",
                        {"created_utc": core.utc_now(), "seasonal_rmse": report, "features": payload})
        print("SEASONAL", report, flush=True)


if __name__ == "__main__":
    main()
