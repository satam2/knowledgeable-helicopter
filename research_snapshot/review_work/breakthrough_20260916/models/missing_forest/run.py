"""Coordinator-launched CPU forest comparisons on all missing-clock records."""
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
import run_missing_models as missing
import run_id_context as context


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=["extratrees", "randomforest"], required=True)
    parser.add_argument("--folds", nargs="+", default=["F1", "F3"])
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--declare-only", action="store_true")
    args = parser.parse_args()
    args.formulation = "missing_template_idcontext"
    adapter.FAMILY = args.family
    core.OUT = core.external_path(core.OUT / "missing_forest")
    core.OUT.mkdir(parents=True, exist_ok=True)
    original_hashes = core.source_hashes
    paths = [Path(__file__), Path(adapter.__file__), Path(missing.__file__), Path(context.__file__)]
    core.source_hashes = lambda: {**original_hashes(), **{str(path.relative_to(core.ROOT)): core.sha256(path) for path in paths}}
    core.importlib = SimpleNamespace(import_module=lambda name: adapter if name == "tabm_gpu" else importlib.import_module(name))
    x, meta = missing.load_data()
    audit = core.read_json(context.CACHE / "audit.json")
    peerpath = context.CACHE / "features.parquet"
    if core.sha256(peerpath) != audit["feature_sha256"]:
        raise ValueError("Frozen ID-neighbor cache hash changed")
    peer = pd.read_parquet(peerpath).set_index(core.ID)
    if not np.array_equal(peer.index, meta[core.ID]):
        raise ValueError("ID-neighbor cache movement order differs")
    times = meta.set_index(core.ID).loc[x.index, missing.MOVEMENT]
    extension = context.id_context_features(x, peer.loc[x.index], times)
    x = pd.concat([x, extension], axis=1)
    x[adapter.TIME_COLUMN] = times
    x = x.reindex(pd.Index(meta[core.ID], name=core.ID))
    eligible = ~np.isfinite(meta.proxy_sec.to_numpy(float))
    info = {"rule": "All missing-NM records trained; finite-NM score rows retain reference; original labels unchanged",
            "context": "Retrospective supplied departure record-order neighbors, exact cached features and observed calendar interactions",
            "context_sha256": audit["feature_sha256"], "categorical_encoding": "Fit-only one-hot, unknown categories allzero"}
    protocol = core.OUT / f"protocol_{args.family}_s{args.seed}.json"
    payload = {"family": args.family, "formulation": args.formulation, "seed": args.seed, "threads": args.threads,
               "folds": args.folds, "source_hashes": core.source_hashes(), "features": list(x),
               "id_context_sha256": audit["feature_sha256"], "target": "Direct rawsecond conditional mean; squared_error forests",
               "selection": "Original tune MSE selects minleaf from1,5,20 with300trees,maxfeatures.7; deterministic firstwinner tie rule",
               "templates": "Chronological earlier-month crossfit only; full refit reconstructs prior and encoders",
               "variants": ["candidate", "blend25"], "blend_weight": .25,
               "parameters": {"n_estimators": 300, "minleaf_grid": list(adapter.LEAVES), "max_features": .7, "threads": args.threads},
               "raw_hashes": core.read_json(core.ROOT / "private_runs/submission_v2/protocol.json")["raw_hashes"],
               "availability": "Exposed development folds; retrospective observed ID-order information; ranking transport unproven"}
    if protocol.exists() and core.read_json(protocol) != payload:
        raise ValueError("Frozen forest protocol changed")
    core.write_json(protocol, payload)
    snapshot = core.OUT / "source_snapshots" / f"{args.family}_s{args.seed}"
    snapshot.mkdir(parents=True, exist_ok=True)
    for path in paths + [Path(core.__file__)]:
        shutil.copyfile(path, snapshot / path.name)
    if args.declare_only:
        print("DECLARED", protocol, "missing_rows", int(eligible.sum()), "features", len(x.columns), flush=True)
        return
    records = {}
    for fold in args.folds:
        records[fold] = core.run(args, fold, x, meta, np.zeros(len(meta)), eligible, info, protocol)
        gc.collect()
    if set(records) == {"F1", "F3"}:
        summary = {variant: core.season_score(records["F1"]["reports"][variant]["metrics"]["overall"],
                                              records["F3"]["reports"][variant]["metrics"]["overall"])
                    for variant in ("candidate", "blend25")}
        core.write_json(core.OUT / f"{args.family}_s{args.seed}_summary.json",
                        {"created_utc": core.utc_now(), "seasonal_rmse": summary})
        print("SEASONAL", summary, flush=True)


if __name__ == "__main__":
    main()
