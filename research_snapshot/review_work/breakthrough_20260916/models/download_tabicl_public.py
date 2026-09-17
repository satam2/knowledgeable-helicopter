"""Download public TabICL files only; this process never opens competition data."""
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "private_runs/breakthrough_20260916/models/public_models/tabicl_v2"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["DO_NOT_TRACK"] = "1"
os.environ["HF_HOME"] = str(OUT / "hf_cache")
os.environ["HF_HUB_DISABLE_XET"] = "1"
from huggingface_hub import hf_hub_download

REPO = "jingang/TabICL"
REVISION = "4dcd344ece2c00be9e831fdd35bed57b5ad83e19"
FILENAME = "tabicl-regressor-v2-20260212.ckpt"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    card = Path(hf_hub_download(REPO, "README.md", revision=REVISION, local_dir=OUT, token=False))
    if "license: bsd-3-clause" not in card.read_text(encoding="utf-8"):
        raise ValueError("Expected public checkpoint license was not found")
    path = Path(hf_hub_download(REPO, FILENAME, revision=REVISION, local_dir=OUT, token=False))
    record = {"created_utc": datetime.now(timezone.utc).isoformat(), "repo": REPO,
              "revision": REVISION, "filename": FILENAME, "sha256": sha(path),
              "bytes": path.stat().st_size, "model_card_sha256": sha(card),
              "model_card_license": "bsd-3-clause", "source_sha256": sha(Path(__file__)),
              "network_scope": "Unauthenticated public checkpoint and model card download only; no competition files opened"}
    (OUT / "download_manifest.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2), flush=True)
