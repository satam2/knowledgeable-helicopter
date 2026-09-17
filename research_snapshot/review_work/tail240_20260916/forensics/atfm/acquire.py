"""Acquire declared official public ATFM workbooks and documentation only."""
import os
for key in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"]:
    os.environ[key]="1"
import argparse
import hashlib
import json
import time
import urllib.request
import zipfile
from datetime import datetime,timezone
from pathlib import Path
import psutil

ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT / "private_runs/tail240_20260916/forensics/atfm/acquisition_v1"
SOURCES={
    "ATFM_Slot_Adherence.xlsx":"https://www.eurocontrol.int/performance/data/download/xls/ATFM_Slot_Adherence.xlsx",
    "Airport_Arrival_ATFM_Delay.xlsx":"https://www.eurocontrol.int/performance/data/download/xls/Airport_Arrival_ATFM_Delay.xlsx",
    "disclaimer.html":"https://ansperformance.eu/about/disclaimer/",
    "atfm_definition.html":"https://ansperformance.eu/definition/atfm-delay/",
    "indicator_catalog.html":"https://ansperformance.eu/data/",
}


def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""):h.update(chunk)
    return h.hexdigest()


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+"\n",encoding="utf-8")


def now():return datetime.now(timezone.utc).isoformat()


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--declare-only",action="store_true");args=parser.parse_args()
    record={"sources":SOURCES,"source_sha256":digest(__file__),
        "scope":"Public official airport-day slot adherence and arrival ATFM delay only. Preserve original bytes and headers. No taxi-out indicator, no private data in requests, no submission.",
        "coverage_question":"Independently inspect workbook fields,definitions,units and all11 challengeairports in2025 plusJanuary/July2026; do not rely on thirdpartyclaims.",
        "reuse":"Local noncommercial attributed research under previouslyinspected EUROCONTROL terms; fresh terms snapshot retained; prizeeligibility not assumed.",
        "resource":"2CPU6GiBmaximum process,8GiBavailable reserve;1MiB downloadchunks; normalTLSverification."}
    protocol=OUT / "protocol.json"
    if protocol.exists():assert json.loads(protocol.read_text())["declaration"]==record
    else:
        OUT.mkdir(parents=True,exist_ok=False);write(protocol,{"created_utc":now(),"declaration":record})
    if args.declare_only:
        print("Declared public ATFM acquisition; no private inputs",flush=True);return
    receipts=[]
    started=time.monotonic()
    for name,url in SOURCES.items():
        path=OUT / name
        assert not path.exists(),"Preserve prior source bytes"
        temporary=path.with_suffix(path.suffix+".partial")
        assert not temporary.exists(),"Preserve failed download"
        request=urllib.request.Request(url,headers={"User-Agent":"Mozilla/5.0 PRC-local-noncommercial-research"})
        began=time.monotonic()
        with urllib.request.urlopen(request,timeout=180) as response,temporary.open("xb") as stream:
            headers=dict(response.headers.items())
            status=response.status
            final_url=response.url
            count=0
            while True:
                chunk=response.read(1024*1024)
                if not chunk:break
                stream.write(chunk);count+=len(chunk)
                if psutil.virtual_memory().available<8*1024**3:raise MemoryError("Host reserve below8GiB")
            expected=headers.get("Content-Length")
            if expected is not None:assert count==int(expected)
        temporary.rename(path)
        receipt={"filename":name,"url":url,"final_url":final_url,"status":status,"bytes":count,
            "sha256":digest(path),"headers":headers,"elapsed_sec":time.monotonic()-began,"retrieved_utc":now()}
        if name.endswith(".xlsx"):
            with zipfile.ZipFile(path) as archive:
                assert archive.testzip() is None
                receipt["zip_entries"]=len(archive.infolist())
                receipt["uncompressed_bytes"]=sum(info.file_size for info in archive.infolist())
        receipts.append(receipt)
        write(OUT / (name+".receipt.json"),receipt)
        print("ACQUIRED",name,count,receipt["sha256"],flush=True)
    peak=getattr(psutil.Process().memory_info(),"peak_wset",psutil.Process().memory_info().rss)
    assert peak<6*1024**3
    write(OUT / "manifest.json",{"status":"complete","created_utc":now(),"source_sha256":digest(__file__),
        "protocol_sha256":digest(protocol),"elapsed_sec":time.monotonic()-started,"peak_bytes":peak,
        "private_inputs_read":False,"private_data_in_requests":False,"sources":receipts})


if __name__=="__main__":main()
