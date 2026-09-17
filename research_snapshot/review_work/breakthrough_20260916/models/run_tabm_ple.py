"""Separate TabM piecewise-embedding runs without changing frozen adapters."""
import torch
import argparse
import gc
import importlib
import shutil
from pathlib import Path
from types import SimpleNamespace
import run_full as core
import tabm_ple_gpu as adapter
import run_augmented


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--formulation", default="aobt_allfinite", choices=["direct", "aobt_allfinite", "multi_anchor"])
    parser.add_argument("--members", type=int, default=8, choices=[8, 32])
    parser.add_argument("--features", nargs="*", default=[], choices=["trajectory", *run_augmented.BATCH_PATTERNS])
    parser.add_argument("--folds", nargs="+", default=["F1", "F3"])
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--declare-only", action="store_true")
    args = parser.parse_args()
    if len(args.features) != len(set(args.features)):
        raise ValueError("Duplicate feature blocks")
    args.features = sorted(args.features)
    args.family = f"tabm_ple{args.members}"
    adapter.MEMBERS = args.members
    core.OUT = core.external_path(core.OUT / "tabm_ple" / ("__".join(args.features) or "base"))
    original_hashes = core.source_hashes
    paths = [Path(__file__), Path(adapter.__file__), Path(run_augmented.__file__), Path(run_augmented.feature_screen.__file__)]
    core.source_hashes = lambda: {**original_hashes(), **{path.name: core.sha256(path) for path in paths}}
    core.importlib = SimpleNamespace(import_module=lambda name: adapter if name == "tabm_gpu" else importlib.import_module(name))
    x, meta = core.common.load_data()
    x, receipts = run_augmented.augment(x, args.features)
    x, offset, eligible, info = core.anchors(x, meta, args.formulation)
    info.update(feature_receipts=receipts)
    protocol = core.declare(args)
    extension = protocol.with_name(protocol.stem + "_embedding.json")
    payload = {"members": args.members, "quantile_bins": adapter.N_BINS, "embedding_dim": adapter.D_EMBEDDING,
               "embedding_version": "B", "fit_scope": "All fit rows only; refit learns bins afresh on all refit rows",
               "categorical_handling": "Frozen learned embeddings", "constant_numeric": "Pass through with missing indicators",
               "features": args.features, "receipts": receipts, "columns": list(x), "source_hashes": core.source_hashes()}
    if extension.exists() and core.read_json(extension) != payload:
        raise ValueError("Embedding protocol changed")
    core.write_json(extension, payload)
    snapshot = core.OUT / "source_snapshots" / f"{args.family}_{args.formulation}_s{args.seed}"
    for path in paths:
        shutil.copyfile(path, snapshot / path.name)
    if args.declare_only:
        print("DECLARED", protocol, extension, flush=True)
        return
    records = {}
    for fold in args.folds:
        records[fold] = core.run(args, fold, x, meta, offset, eligible, info, protocol)
        gc.collect()
    if set(records) == {"F1", "F3"}:
        scores = {variant: core.season_score(records["F1"]["reports"][variant]["metrics"]["overall"],
                                            records["F3"]["reports"][variant]["metrics"]["overall"])
                  for variant in ("candidate", "blend25")}
        core.write_json(core.OUT / f"{args.family}_{args.formulation}_s{args.seed}_summary.json",
                        {"created_utc": core.utc_now(), "seasonal_rmse": scores, "embedding": payload})
        print("SEASONAL", scores, flush=True)


if __name__ == "__main__":
    main()
