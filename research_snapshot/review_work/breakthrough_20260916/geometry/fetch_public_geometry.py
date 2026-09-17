"""Download historical public airport geometry; never reads challenge records."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import time

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/"output/breakthrough_20260916/geometry/public"
SNAPSHOT="2024-12-31T23:59:59Z"
BBOX={"LIRF":"41.77,12.20,41.83,12.30","LFPG":"48.98,2.50,49.04,2.62"}
SOURCES={
 "overpass_historical":"https://wiki.openstreetmap.org/wiki/Overpass_API/Overpass_QL#Date",
 "parking_positions":"https://wiki.openstreetmap.org/wiki/Tag:aeroway%3Dparking_position",
 "taxiway":"https://wiki.openstreetmap.org/wiki/Tag:aeroway%3Dtaxiway",
 "runway":"https://wiki.openstreetmap.org/wiki/Tag:aeroway%3Drunway",
 "copyright":"https://www.openstreetmap.org/copyright",
}


def now():return datetime.now(timezone.utc).isoformat()


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/"receipt.json").exists():raise ValueError("Preserve previous receipts; use a new output version")
    records={}
    for airport,bbox in BBOX.items():
        query=f'[out:json][timeout:90][date:"{SNAPSHOT}"];nwr["aeroway"~"^(taxiway|taxilane|parking_position|runway|holding_position)$"]({bbox});(._;>;);out body;'
        SOURCES[airport]="https://overpass-api.de/api/interpreter?"+urlencode({"data":query})
    for name,url in SOURCES.items():
        rec={"url":url,"retrieved_utc":now(),"private_records_read":False,"private_request_values":False,"requested_snapshot":SNAPSHOT if name in BBOX else None}
        start=time.monotonic()
        try:
            request=Request(url,headers={"User-Agent":"PRC2026-local-public-geometry-research/1.0"})
            with urlopen(request,timeout=110) as response:
                body=response.read()
                rec.update(status=response.status,content_type=response.headers.get("Content-Type"),bytes=len(body),sha256=hashlib.sha256(body).hexdigest())
            path=OUT/(name+(".json" if name in BBOX else ".html"))
            path.write_bytes(body)
            if name in BBOX:
                parsed=json.loads(body)
                rec.update(elements=len(parsed.get("elements",[])),osm3s=parsed.get("osm3s"),remark=parsed.get("remark"))
        except Exception as exc:rec["error"]=repr(exc)
        rec["runtime_sec"]=time.monotonic()-start
        records[name]=rec
        (OUT/"receipt.json").write_text(json.dumps(records,indent=2),encoding="utf-8")
        print(name,rec.get("status"),rec.get("elements"),rec.get("error"),flush=True)


if __name__=="__main__":main()
