"""Scheduled full-cohort matched constant versus linear-leaf LightGBM pilot."""
import lightgbm
import argparse
import gc
import importlib
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import run_full as core
import adapter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--leaf", choices=["constant", "linear"], required=True)
    parser.add_argument("--folds", nargs="+", default=["F1", "F3"])
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--declare-only", action="store_true")
    args = parser.parse_args()
    args.family = f"lightgbm_leaf_{args.leaf}"
    args.formulation = "aobt_allfinite"
    adapter.LINEAR_TREE = args.leaf == "linear"
    core.OUT = core.external_path(core.OUT / "linear_tree")
    core.OUT.mkdir(parents=True, exist_ok=True)
    original_hashes = core.source_hashes
    paths = [Path(__file__), Path(adapter.__file__), core.ROOT / "review_work/campaign_20260916/lgbm_adapter.py"]
    core.source_hashes = lambda: {**original_hashes(), **{str(path.relative_to(core.ROOT)): core.sha256(path) for path in paths}}
    core.importlib = SimpleNamespace(import_module=lambda name: adapter if name == "tabm_gpu" else importlib.import_module(name))
    x, meta = core.common.load_data()
    x, offset, eligible, info = core.anchors(x, meta, args.formulation)
    numeric = sum(pd.api.types.is_numeric_dtype(x[column]) for column in x)
    estimates = {}
    for fold in args.folds:
        idx, _, _ = core.common.fold_data(meta, fold, full=True)
        count = max(int(eligible[idx["fit"]].sum() + eligible[idx["tune"]].sum()), int(eligible[idx["refit"]].sum()))
        estimates[fold] = adapter.structural_memory(count, numeric, len(x.columns), args.threads)
    if max(value["conservative_process_gib"] for value in estimates.values()) > adapter.MAX_MEMORY_GIB:
        raise MemoryError("Full-data linear-tree feasibility estimate exceeds16GiB")
    protocol = core.OUT / f"protocol_{args.leaf}_s{args.seed}.json"
    payload = {"leaf": args.leaf, "folds": args.folds, "seed": args.seed, "threads": args.threads,
               "source_hashes": core.source_hashes(), "parameters": adapter.parameters(args.seed, args.threads, adapter.MAX_TREES),
               "features": list(x), "memory_estimates": estimates,
               "target": "Fulloriginalallfinite rawY-NMproxy, includingnegative/longproxy; no clipping",
               "encoding": "Same fitonly standardization/nativecategory encoding for BOTH leafarms;NaNnative;categoryslopes forbidden",
               "selection": "Predeclared200trees15leavesdepth6, originaltune RMSE stopping40; no hypergrid orscoreweights",
               "comparison": "Constantandlinear pilots differ only linear_tree boolean; separatelyselected originaltune treecounts",
               "variants": ["candidate", "blend25"], "blend_weight": .25,
               "resources": "CPUserial learner,2threads,conservativeplanningcap16GiB+4GiBhostreserve",
               "raw_hashes": core.read_json(core.ROOT / "private_runs/submission_v2/protocol.json")["raw_hashes"]}
    if protocol.exists() and core.read_json(protocol) != payload:
        raise ValueError("Frozen linear-tree pilot protocol changed")
    core.write_json(protocol, payload)
    snapshot = core.OUT / "source_snapshots" / args.leaf
    snapshot.mkdir(parents=True, exist_ok=True)
    for path in paths + [Path(core.__file__)]:
        shutil.copyfile(path, snapshot / path.name)
    if args.declare_only:
        print("DECLARED", protocol, "memoryGiB", {fold: value["conservative_process_gib"] for fold, value in estimates.items()}, flush=True)
        return
    records = {}
    for fold in args.folds:
        records[fold] = core.run(args, fold, x, meta, offset, eligible, info, protocol)
        gc.collect()
    if set(records) == {"F1", "F3"}:
        scores = {variant: core.season_score(records["F1"]["reports"][variant]["metrics"]["overall"],
                                            records["F3"]["reports"][variant]["metrics"]["overall"])
                  for variant in ("candidate", "blend25")}
        core.write_json(core.OUT / f"summary_{args.leaf}_s{args.seed}.json", {"created_utc": core.utc_now(), "seasonal_rmse": scores})
        print("SEASONAL", scores, flush=True)


if __name__ == "__main__":
    main()
