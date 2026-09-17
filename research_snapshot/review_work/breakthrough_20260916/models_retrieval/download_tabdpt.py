"""Download approved public Apache-2.0 weights, isolated from challenge records."""
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/"output/breakthrough_20260916/models_retrieval/checkpoint"


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1048576),b""):h.update(b)
    return h.hexdigest()


def main():
    licensepath=ROOT/"output/breakthrough_20260916/models_retrieval/estimator/sources/tabdpt_weights_license.txt"
    licensebody=licensepath.read_text()
    assert "Apache License" in licensebody and "Version 2.0" in licensebody
    metadata=json.loads((ROOT/"output/breakthrough_20260916/models_retrieval/code/sources/tabdpt_model_files.txt").read_text())
    assert metadata["gated"] is False and metadata["private"] is False
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/"receipt.json").exists():raise ValueError("Preserve existing checkpoint receipt")
    os.environ["HF_HUB_DISABLE_TELEMETRY"]="1"
    from huggingface_hub import hf_hub_download
    path=Path(hf_hub_download(repo_id="Layer6/TabDPT",filename="tabdpt1_3.safetensors",revision=metadata["sha"],local_dir=OUT,token=False))
    from safetensors import safe_open
    with safe_open(path,framework="np") as file:
        config=json.loads(file.metadata()["cfg"])
        tensor_count=len(file.keys())
    receipt={"created_utc":datetime.now(timezone.utc).isoformat(),"repo":"Layer6/TabDPT","revision":metadata["sha"],"file":str(path),"sha256":sha(path),"bytes":path.stat().st_size,"license":"Apache-2.0","license_sha256":sha(licensepath),"tensor_count":tensor_count,"checkpoint_config":config,"private_data_read":False,"model_constructed":False,"authentication_used":False}
    (OUT/"receipt.json").write_text(json.dumps(receipt,indent=2),encoding="utf-8")
    print(json.dumps(receipt,indent=2),flush=True)


if __name__=="__main__":main()
