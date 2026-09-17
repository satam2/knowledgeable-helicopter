"""Centrally scheduled TabICLv2 missing-clock evaluation using frozen scoring."""
import argparse
import gc
import importlib
import os
from pathlib import Path
import shutil
from types import SimpleNamespace

for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY", "DO_NOT_TRACK"):
    os.environ[key] = "1"

import torch
import run_full as core
import tabicl_gpu
import numpy as np
from importlib.metadata import version

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", nargs="+", default=["F1", "F3"])
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--declare-only", action="store_true")
    args = parser.parse_args()
    args.family = "tabicl"
    args.formulation = "missing_direct"
    core.OUT = core.external_path(core.OUT / "tabicl_missing")
    core.OUT.mkdir(parents=True, exist_ok=True)
    original_hashes = core.source_hashes
    core.source_hashes = lambda: {**original_hashes(), "run_tabicl_missing.py": core.sha256(__file__),
                                 "tabicl_gpu.py": core.sha256(tabicl_gpu.__file__)}
    # The frozen evaluator chooses its second adapter for every non-XGB family.
    core.importlib = SimpleNamespace(import_module=lambda name: tabicl_gpu if name == "tabm_gpu" else importlib.import_module(name))
    x, meta = core.common.load_data()
    eligible = ~np.isfinite(meta.proxy_sec.to_numpy(float))
    info = {"rule": "All missing-NM original labels as context; missing-NM predictions only; finite-NM rows preserve reference",
            "output": "Public pretrained distribution quantile-grid mean on original target scale"}
    if core.sha256(tabicl_gpu.CHECKPOINT) != tabicl_gpu.EXPECTED_SHA256:
        raise ValueError("Pinned public checkpoint differs")
    counts = {}
    for fold in args.folds:
        idx, _, _ = core.common.fold_data(meta, fold, full=True)
        counts[fold] = {stage: int(eligible[positions].sum()) for stage, positions in idx.items()}
    protocol = core.OUT / f"protocol_s{args.seed}.json"
    payload = {"family": args.family, "formulation": args.formulation, "seed": args.seed, "threads": args.threads,
               "folds": args.folds, "counts": counts, "source_hashes": core.source_hashes(),
               "checkpoint_sha256": tabicl_gpu.EXPECTED_SHA256, "tabicl_version": version("tabicl"),
               "parameters": {"n_estimators": tabicl_gpu.ESTIMATORS, "batch_size": 1, "kv_cache": "repr",
                              "use_amp": True, "use_fa3": False, "offload_mode": "cpu", "query_chunk": tabicl_gpu.CHUNK_SIZE},
               "objective": "In-context pretrained distribution quantile-grid mean, approximate conditional mean; no supervised gradient fitting or tail integration",
               "labels": "All original eligible labels retained including negatives and extreme values; affine standardization only",
               "splits": "Frozen original fit/tune/score and flight-ID purges; independent refit context is fit plus tune",
               "variants": ["candidate", "blend25"], "blend_weight": .25, "no_clipping": True,
               "offline": {key: os.environ[key] for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY", "DO_NOT_TRACK")},
               "allow_auto_download": False, "selection": "Authorized exposed-development search; no new holdout or official score",
               "raw_hashes": core.read_json(core.ROOT / "private_runs/submission_v2/protocol.json")["raw_hashes"]}
    if protocol.exists() and core.read_json(protocol) != payload:
        raise ValueError("Existing TabICL protocol differs")
    core.write_json(protocol, payload)
    snapshot = core.OUT / "source_snapshots"
    snapshot.mkdir(exist_ok=True)
    for path in [Path(__file__), Path(tabicl_gpu.__file__), Path(core.__file__)]:
        shutil.copyfile(path, snapshot / path.name)
    if args.declare_only:
        print("DECLARED", protocol, counts, flush=True)
        return
    records = {}
    for fold in args.folds:
        records[fold] = core.run(args, fold, x, meta, np.zeros(len(x)), eligible, info, protocol)
        gc.collect()
    if set(records) == {"F1", "F3"}:
        scores = {variant: core.season_score(records["F1"]["reports"][variant]["metrics"]["overall"],
                                            records["F3"]["reports"][variant]["metrics"]["overall"])
                  for variant in ("candidate", "blend25")}
        core.write_json(core.OUT / f"summary_s{args.seed}.json", {"created_utc": core.utc_now(), "seasonal_rmse": scores})
        print("SEASONAL", scores, flush=True)


if __name__ == "__main__":
    main()
