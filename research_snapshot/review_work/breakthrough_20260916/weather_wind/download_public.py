"""Public-only OurAirports runway/elevation source download; no private reads."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
from urllib.request import Request,urlopen

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/"output/breakthrough_20260916/weather_wind/public"
SOURCES={
 "data_page.html":"https://ourairports.com/data/",
 "repository_readme.md":"https://raw.githubusercontent.com/davidmegginson/ourairports-data/main/README.md",
 "repository_commit.json":"https://api.github.com/repos/davidmegginson/ourairports-data/commits/main",
 "data_dictionary.html":"https://ourairports.com/help/data-dictionary.html",
 "iem_fields.html":"https://mesonet.agron.iastate.edu/request/download.phtml",
}


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/"receipt.json").exists():raise ValueError("Preserve prior download receipt")
    receipts={}
    for name,url in SOURCES.items():
        record={"url":url,"retrieved_utc":datetime.now(timezone.utc).isoformat(),"private_data_read":False,"private_request_values":False}
        try:
            with urlopen(Request(url,headers={"User-Agent":"PRC2026-public-runway-wind-research/1.0"}),timeout=60) as response:body=response.read()
            (OUT/name).write_bytes(body)
            record.update(status=200,bytes=len(body),sha256=hashlib.sha256(body).hexdigest())
        except Exception as exc:record["error"]=repr(exc)
        receipts[name]=record
        (OUT/"receipt.json").write_text(json.dumps(receipts,indent=2),encoding="utf-8")
        print(name,record.get("status"),record.get("error"),flush=True)
    metadata=json.loads((OUT/"repository_commit.json").read_text())
    revision=metadata["sha"]
    for name in ("airports.csv","runways.csv"):
        url=f"https://raw.githubusercontent.com/davidmegginson/ourairports-data/{revision}/{name}"
        record={"url":url,"revision":revision,"commit_date":metadata["commit"]["committer"]["date"],"retrieved_utc":datetime.now(timezone.utc).isoformat(),"private_data_read":False,"private_request_values":False}
        with urlopen(Request(url,headers={"User-Agent":"PRC2026-public-runway-wind-research/1.0"}),timeout=90) as response:body=response.read()
        (OUT/name).write_bytes(body)
        record.update(status=200,bytes=len(body),sha256=hashlib.sha256(body).hexdigest())
        receipts[name]=record
        (OUT/"receipt.json").write_text(json.dumps(receipts,indent=2),encoding="utf-8")
        print(name,record["bytes"],revision,flush=True)


if __name__=="__main__":main()
