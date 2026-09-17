"""Centrally scheduled full finite-NM Student-t source-mixture research pilot."""
import torch
import argparse
import gc
import importlib
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[1]))
import run_full as core
import run_information
import adapter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conventions", action="store_true")
    parser.add_argument("--folds", nargs="+", default=["F1", "F3"])
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--declare-only", action="store_true")
    args = parser.parse_args()
    args.family = "tabm_source_mdn"
    args.formulation = "finite_rawmean"
    core.OUT = core.external_path(core.OUT / "source_mdn" / ("conventions" if args.conventions else "base"))
    core.OUT.mkdir(parents=True, exist_ok=True)
    original_hashes = core.source_hashes
    paths = [Path(__file__), Path(adapter.__file__), Path(run_information.__file__)]
    core.source_hashes = lambda: {**original_hashes(), **{str(path.relative_to(core.ROOT)): core.sha256(path) for path in paths}}
    core.importlib = SimpleNamespace(import_module=lambda name: adapter if name == "tabm_gpu" else importlib.import_module(name))
    x, meta = core.common.load_data()
    receipts = []
    if args.conventions:
        x, receipts = run_information.additional(x, ["conventions"])
    eligible = np.isfinite(meta.proxy_sec.to_numpy(float))
    counts = {}
    for fold in args.folds:
        idx, _, _ = core.common.fold_data(meta, fold, full=True)
        counts[fold] = {stage: {"n": int(eligible[positions].sum()),
                               "id_hash": core.object_hash(meta.iloc[positions[eligible[positions]]][core.ID].tolist())}
                        for stage, positions in idx.items()}
    protocol = core.OUT / f"protocol_s{args.seed}.json"
    payload = {"family": args.family, "formulation": args.formulation, "folds": args.folds, "seed": args.seed, "threads": args.threads,
               "source_hashes": core.source_hashes(), "columns": list(x), "feature_receipts": receipts, "counts": counts,
               "components": adapter.CLOCKS + ["learned_direct_mean"], "fixed_student_t_df": adapter.DEGREES_OF_FREEDOM,
               "location": "Five observedclock proxies plus separatelylearned additivebiases; sixthcomponent learnedrawmean",
               "weights": "Learnedsoftmax sourceweights; unavailableclocks exactlymasked; direct alwaysallowed; initialbias[2,0,0,0,-2,0]",
               "scales": "Softplus learned standardizedscale plus1/targetstd; rawminimum1second",
               "loss": "Average negative loglikelihood over rows AND separateTabMmembers; eachmember has sixcomponentStudentTmixture",
               "prediction": "Analytical sum(weights*componentmeans), averageTabMmembers, affineinverse to rawseconds",
               "epoch_selection": "Originaltune rawMSE of analyticalmeans;64epochsmax/patience8;NLLnotselectionmetric",
               "refit": "Independentfullrefit with freshfit-only encoder/targetscaler; tunedepochcount only",
               "parameters": {"members": 8, "blocks": 3, "width": 256, "dropout": .1, "learning_rate": .001,
                              "weight_decay": .0001, "batch_size": 4096, "epochs": 64, "patience": 8, "gradclip": 10.},
               "eligibility": "EveryoriginalfiniteNMclock, includingnegativeandlongproxies;missingV2exactlypreserved",
               "target": "Alloriginalrawlabels;no clipping,deletion,resamplingorlogtransform;affinestandardizationonly",
               "variants": ["candidate", "blend25"], "blend_weight": .25,
               "research_limit": "Hypothesisaboutmultimodalsourceerrors; no theoreticalsuperiorityclaim; likelihoodfit can sacrificeRMSE",
               "raw_hashes": core.read_json(core.ROOT / "private_runs/submission_v2/protocol.json")["raw_hashes"]}
    if protocol.exists() and core.read_json(protocol) != payload:
        raise ValueError("Frozen source-mixture protocol changed")
    core.write_json(protocol, payload)
    snapshot = core.OUT / "source_snapshots"
    snapshot.mkdir(exist_ok=True)
    for path in paths + [Path(core.__file__), HERE.parent / "encoders.py"]:
        shutil.copyfile(path, snapshot / path.name)
    if args.declare_only:
        print("DECLARED", protocol, "features", len(x.columns), "counts", counts, flush=True)
        return
    info = {"rule": payload["eligibility"], "target": payload["target"], "feature_receipts": receipts, "prediction": payload["prediction"]}
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
