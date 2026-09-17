"""Offline-only public TabICLv2 adapter for the missing-clock subgroup."""
import os
from pathlib import Path
import time

for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY", "DO_NOT_TRACK"):
    os.environ[key] = "1"

import numpy as np
import torch
from tabicl import TabICLRegressor

ROOT = Path(__file__).resolve().parents[3]
CACHE = ROOT / "private_runs/breakthrough_20260916/models/public_models/tabicl_v2"
CHECKPOINT = CACHE / "tabicl-regressor-v2-20260212.ckpt"
EXPECTED_SHA256 = "0db9cb538f114e79026bf08f45f41ad8dd7ad2de2aaca9a5ca8cd3bd9748ae7a"
ESTIMATORS = 4
CHUNK_SIZE = 512


def features(x):
    result = x.copy()
    numeric = result.select_dtypes(include=["number"]).columns
    result[numeric] = result[numeric].replace([-999999, np.inf, -np.inf], np.nan)
    return result


def fit(x, y, tuning=None, *, steps=None, seed=20260916, threads=4):
    if not torch.cuda.is_available():
        raise RuntimeError("TabICL subgroup inference requires a centrally scheduled CUDA slot")
    if steps is not None and (steps != 1 or tuning is not None):
        raise ValueError("TabICL uses one fixed context pass without learned tuning")
    import hashlib
    if hashlib.file_digest(CHECKPOINT.open("rb"), "sha256").hexdigest() != EXPECTED_SHA256:
        raise ValueError("Public TabICL checkpoint hash mismatch")
    y = np.asarray(y, dtype=float)
    if not len(y) or len(x) != len(y) or not np.isfinite(y).all():
        raise ValueError("Context labels must be aligned, nonempty and finite")
    torch.set_num_threads(threads)
    started = time.monotonic()
    model = TabICLRegressor(n_estimators=ESTIMATORS, batch_size=1, kv_cache="repr",
                           model_path=str(CHECKPOINT), allow_auto_download=False,
                           checkpoint_version=CHECKPOINT.name, device="cuda", use_amp=True,
                           use_fa3=False, offload_mode="cpu", n_jobs=threads, random_state=seed)
    model.fit(features(x), y)
    evidence = {"steps": 1, "n_context": len(y), "context_target_min": float(y.min()),
                "context_target_max": float(y.max()), "runtime_sec": time.monotonic() - started,
                "configuration": model.get_params(deep=False), "checkpoint_sha256": EXPECTED_SHA256,
                "target_handling": "Unmodified labels; affine StandardScaler within TabICL",
                "prediction": "Arithmetic mean of predicted quantile grid; no explicit tail integration",
                "tuning": "Original tune predictions reported; no parameter or blend selection"}
    return model, evidence


def predict(model, x):
    output = []
    for start in range(0, len(x), CHUNK_SIZE):
        output.append(model.predict(features(x.iloc[start:start + CHUNK_SIZE]), output_type="mean"))
    return np.concatenate(output).astype(float)
