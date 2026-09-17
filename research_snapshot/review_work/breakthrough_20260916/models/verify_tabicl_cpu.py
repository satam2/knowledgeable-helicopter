"""Synthetic CPU-only checkpoint and offline inference smoke test."""
import json
import socket
import time
from pathlib import Path
import tabicl_gpu as adapter
import numpy as np
import torch
from tabicl import TabICLRegressor


def denied(*args, **kwargs):
    raise AssertionError("Offline synthetic inference attempted a network connection")


if __name__ == "__main__":
    socket.socket.connect = denied
    socket.create_connection = denied
    torch.set_num_threads(2)
    rng = np.random.default_rng(20260916)
    x = rng.normal(size=(24, 3))
    y = x[:, 0] * 20 + 100
    y[-1] = 100000
    y[0] = -12
    model = TabICLRegressor(n_estimators=1, batch_size=1, norm_methods=["none"], kv_cache=False,
                            model_path=str(adapter.CHECKPOINT), allow_auto_download=False, device="cpu",
                            use_amp=False, use_fa3=False, offload_mode="cpu", n_jobs=2)
    started = time.monotonic()
    model.fit(x, y)
    prediction = model.predict(x[:3], output_type="mean")
    assert np.isfinite(prediction).all()
    assert np.isclose(model.y_scaler_.mean_[0], y.mean())
    assert model.model_config_["max_classes"] == 0
    record = {"status": "passed", "device": "cpu", "synthetic_only": True,
              "network_connections": "blocked by socket guards", "shape": list(prediction.shape),
              "seconds": time.monotonic() - started, "quantiles": model.model_config_.get("num_quantiles"),
              "target_min": float(y.min()), "target_max": float(y.max()), "target_mean_preserved": True}
    out = adapter.ROOT / "private_runs/breakthrough_20260916/models/validation"
    out.mkdir(parents=True, exist_ok=True)
    (out / "tabicl_cpu_smoke.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2), flush=True)
