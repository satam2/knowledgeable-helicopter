"""Public-only official retrieval-model feasibility sources."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
from urllib.request import Request,urlopen

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/"output/breakthrough_20260916/models_retrieval/sources"
SOURCES={
 "tabr_readme":"https://raw.githubusercontent.com/yandex-research/tabr/main/README.md",
 "tabr_model":"https://raw.githubusercontent.com/yandex-research/tabr/main/bin/tabr.py",
 "tabr_requirements":"https://raw.githubusercontent.com/yandex-research/tabr/main/requirements.txt",
 "tabr_environment":"https://raw.githubusercontent.com/yandex-research/tabr/main/environment.yaml",
 "tabr_license":"https://raw.githubusercontent.com/yandex-research/tabr/main/LICENSE",
 "tabr_repository":"https://api.github.com/repos/yandex-research/tabr",
 "faiss_install":"https://raw.githubusercontent.com/facebookresearch/faiss/main/INSTALL.md",
 "faiss_gpu":"https://raw.githubusercontent.com/facebookresearch/faiss/main/tutorial/python/4-GPU.py",
 "realmlp_readme":"https://raw.githubusercontent.com/dholzmueller/realmlp/main/README.md",
 "pytabkit_readme":"https://raw.githubusercontent.com/dholzmueller/pytabkit/main/README.md",
 "pytabkit_pyproject":"https://raw.githubusercontent.com/dholzmueller/pytabkit/main/pyproject.toml",
 "pytabkit_license":"https://raw.githubusercontent.com/dholzmueller/pytabkit/main/LICENSE",
 "pytabkit_package":"https://pypi.org/pypi/pytabkit/json",
 "tabdpt_readme":"https://raw.githubusercontent.com/layer6ai-labs/TabDPT/main/README.md",
 "tabdpt_repository":"https://api.github.com/repos/layer6ai-labs/TabDPT",
 "tabr_tree":"https://api.github.com/repos/yandex-research/tabr/git/trees/main?recursive=1",
}


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    records={}
    for name,url in SOURCES.items():
        record={"url":url,"retrieved_utc":datetime.now(timezone.utc).isoformat(),"private_files_read":False}
        try:
            with urlopen(Request(url,headers={"User-Agent":"PRC2026-public-model-feasibility/1.0"}),timeout=35) as response:body=response.read()
            (OUT/(name+".txt")).write_bytes(body)
            record.update(status=200,bytes=len(body),sha256=hashlib.sha256(body).hexdigest())
        except Exception as exc:record["error"]=repr(exc)
        records[name]=record
        (OUT.parent/"receipts.json").write_text(json.dumps(records,indent=2),encoding="utf-8")
        print(name,record.get("status"),record.get("error"),flush=True)


if __name__=="__main__":main()
