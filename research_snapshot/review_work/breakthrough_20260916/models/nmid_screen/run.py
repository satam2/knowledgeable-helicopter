"""Matched frozen LightGBM controls plus verified seven-feature NM-ID context."""
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
import run_augmented
import run_information
import lgbm_adapter

CACHE = core.ROOT / "private_runs/breakthrough_20260916/missing/nm_id_context/cache_v2"
FEATURES = ["nmid_peer_nm_minus_own_nm", "nmid_peer_eobt_minus_own_nm", "nmid_peer_nm_minus_takeoff",
            "nmid_peer_eobt_minus_takeoff", "nmid_peer_nm_count", "nmid_peer_eobt_count", "nmid_source_id_present"]


def append_nmid(x, root=CACHE):
    marker = core.read_json(root / "manifest.json")
    verification = core.read_json(root / "verification.json")
    if marker.get("status") != "complete" or verification.get("status") != "passed":
        raise ValueError("NM-ID cache must be complete and independently verified")
    if verification.get("manifest_sha256") != core.sha256(root / "manifest.json"):
        raise ValueError("NM-ID verification refers to a stale manifest")
    if not marker.get("training_id_order_verified") or not verification.get("training_id_order_verified"):
        raise ValueError("NM-ID training ID verification missing")
    if marker.get("features") != FEATURES:
        raise ValueError("NM-ID declared feature schema changed")
    if core.sha256(root / "protocol.json") != marker["protocol_sha256"]:
        raise ValueError("NM-ID cache protocol changed")
    path = root / "training_features.parquet"
    if core.sha256(path) != marker["outputs"][path.name]:
        raise ValueError("NM-ID cache feature hash changed")
    extra = pd.read_parquet(path).set_index(core.ID)
    if not x.index.is_unique or not extra.index.is_unique or not np.array_equal(extra.index, x.index):
        raise ValueError("NM-ID movement IDs/order differ from original model inputs")
    if list(extra) != FEATURES or set(extra).intersection(x.columns):
        raise ValueError("NM-ID feature columns missing, extra, or duplicated")
    extra = extra.replace([np.inf, -np.inf], np.nan).fillna(-999999).astype("float32")
    receipt = {"manifest": str(root / "manifest.json"), "manifest_sha256": core.sha256(root / "manifest.json"),
               "verification_sha256": core.sha256(root / "verification.json"), "protocol_sha256": marker["protocol_sha256"],
               "features_sha256": marker["outputs"][path.name], "features": FEATURES,
               "availability": "Retrospective final supplied movement batch; all-airport ARR/DEP NM-ID neighbors; actual UTCmonth isolated; same NMflight excluded"}
    return pd.concat([x, extra], axis=1), receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--combined", action="store_true")
    parser.add_argument("--folds", nargs="+", default=["F1", "F3"])
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--declare-only", action="store_true")
    args = parser.parse_args()
    args.family = "lightgbm_nmid"
    args.formulation = "aobt_allfinite"
    core.OUT = core.external_path(core.OUT / "nmid_screen" / ("combined" if args.combined else "base"))
    paths = [Path(__file__), Path(run_information.__file__), Path(run_augmented.__file__),
             Path(run_augmented.feature_screen.__file__), Path(lgbm_adapter.__file__)]
    original_hashes = core.source_hashes
    core.source_hashes = lambda: {**original_hashes(), **{str(path.relative_to(core.ROOT)): core.sha256(path) for path in paths}}
    core.importlib = SimpleNamespace(import_module=lambda name: lgbm_adapter if name == "tabm_gpu" else importlib.import_module(name))
    x, meta = core.common.load_data()
    receipts = []
    if args.combined:
        x, receipts = run_augmented.augment(x, ["source_past", "source_twosided", "surface_T", "trajectory"])
        x, additional = run_information.additional(x, ["conventions", "geometry", "weather_T"])
        receipts.extend(additional)
    existing_columns = list(x)
    x, nmid_receipt = append_nmid(x)
    receipts.append(nmid_receipt)
    x, offset, eligible, info = core.anchors(x, meta, args.formulation)
    info.update(feature_receipts=receipts)
    protocol = core.declare(args)
    detail = protocol.with_name(protocol.stem + "_nmid.json")
    payload = {"combined": args.combined, "control_columns": existing_columns, "columns": list(x), "receipts": receipts,
               "source_hashes": core.source_hashes(), "comparison": "Frozen LightGBM600 allfinite control plus exactly7NMID features",
               "target": "Allfinite rawY-NMproxy; all original labels and rows retained; missingNMreference unchanged",
               "limitation": "NM-ID ordering is an empirical source signature, not proven clock allocation or airportblock chronology"}
    if detail.exists() and core.read_json(detail) != payload:
        raise ValueError("Frozen NM-ID screen protocol differs")
    core.write_json(detail, payload)
    snapshot = core.OUT / "source_snapshots" / f"{args.family}_{args.formulation}_s{args.seed}"
    for path in paths:
        shutil.copyfile(path, snapshot / path.name)
    if args.declare_only:
        print("DECLARED", protocol, "shape", x.shape, flush=True)
        return
    records = {}
    for fold in args.folds:
        records[fold] = core.run(args, fold, x, meta, offset, eligible, info, protocol)
        gc.collect()
    if set(records) == {"F1", "F3"}:
        summary = {variant: core.season_score(records["F1"]["reports"][variant]["metrics"]["overall"],
                                              records["F3"]["reports"][variant]["metrics"]["overall"])
                    for variant in ("candidate", "blend25")}
        core.write_json(core.OUT / f"summary_s{args.seed}.json", {"created_utc": core.utc_now(), "seasonal_rmse": summary, "features": payload})
        print("SEASONAL", summary, flush=True)


if __name__ == "__main__":
    main()
