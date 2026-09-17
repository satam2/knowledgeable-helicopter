"""Install explicitly approved packages without changing any existing package version."""
from datetime import datetime,timezone
import importlib.metadata as metadata
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/"private_runs/breakthrough_20260916/models_retrieval/setup"
PACKAGES=["pytabkit==1.7.3","faiss-cpu==1.12.0","tabdpt==1.3.0"]


def installed():return {d.metadata["Name"].lower().replace("_","-"):d.version for d in metadata.distributions()}


def main():
    assert Path(sys.executable).parent.parent.name==".breakthrough-venv"
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/"install_receipt.json").exists():raise ValueError("Existing installation receipt preserved")
    before=installed()
    (OUT/"constraints.txt").write_text("\n".join(f"{n}=={v}" for n,v in sorted(before.items()))+"\n",encoding="utf-8")
    (OUT/"before.json").write_text(json.dumps(before,indent=2),encoding="utf-8")
    command=[sys.executable,"-m","pip","install","--disable-pip-version-check","--constraint",str(OUT/"constraints.txt"),"--report",str(OUT/"pip_report.json"),*PACKAGES]
    result=subprocess.run(command,check=False)
    after=installed()
    changed={n:{"before":v,"after":after.get(n)} for n,v in before.items() if after.get(n)!=v}
    receipt={"created_utc":datetime.now(timezone.utc).isoformat(),"packages":PACKAGES,"command":command,"exit_code":result.returncode,"changed_existing":changed,"added":{n:v for n,v in after.items() if n not in before},"private_data_read":False}
    (OUT/"install_receipt.json").write_text(json.dumps(receipt,indent=2),encoding="utf-8")
    assert result.returncode==0,receipt
    assert not changed,changed
    print(json.dumps(receipt,indent=2),flush=True)


if __name__=="__main__":main()
