"""Public metadata geometry with expanded complete airport bounding boxes."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request,urlopen

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/"output/breakthrough_20260916/geometry/public_v2"
DATE="2024-12-31T23:59:59Z"
BBOX={"LIRF":"41.76,12.19,41.86,12.31","LFPG":"48.98,2.49,49.05,2.64"}


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/"receipt.json").exists():raise ValueError("Preserve previous receipt")
    records={"created_utc":datetime.now(timezone.utc).isoformat(),"source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"private_records_read":False,"requested_snapshot":DATE,"airports":{}}
    for airport,bbox in BBOX.items():
        query=f'[out:json][timeout:90][date:"{DATE}"];nwr["aeroway"~"^(taxiway|taxilane|parking_position|runway|holding_position|threshold)$"]({bbox});(._;>;);out meta;'
        url="https://overpass-api.de/api/interpreter?"+urlencode({"data":query})
        rec={"url":url,"bbox":bbox}
        try:
            with urlopen(Request(url,headers={"User-Agent":"PRC2026-local-public-geometry-research/1.0"}),timeout=110) as response:body=response.read()
            (OUT/(airport+".json")).write_bytes(body)
            parsed=json.loads(body)
            assert not parsed.get("remark"),parsed.get("remark")
            timestamps=[e["timestamp"] for e in parsed["elements"]]
            assert max(timestamps)<=DATE
            rec.update(status="passed",sha256=hashlib.sha256(body).hexdigest(),elements=len(timestamps),latest_element_timestamp=max(timestamps),all_element_timestamps_before_snapshot=True)
        except Exception as exc:rec.update(status="failed",error=repr(exc))
        records["airports"][airport]=rec
        records["status"]="complete" if len(records["airports"])==2 and all(x["status"]=="passed" for x in records["airports"].values()) else "incomplete"
        (OUT/"receipt.json").write_text(json.dumps(records,indent=2),encoding="utf-8")
        print(airport,rec.get("status"),rec.get("elements"),rec.get("error"),flush=True)


if __name__=="__main__":main()
