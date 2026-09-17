"""Public-only metadata retrieval to substantiate the requested historical snapshot."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request,urlopen

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/"output/breakthrough_20260916/geometry/public"
DATE="2024-12-31T23:59:59Z"


def main():
    if (OUT/"snapshot_verification.json").exists():raise ValueError("Preserve existing verification")
    report={"requested_snapshot":DATE,"created_utc":datetime.now(timezone.utc).isoformat(),"private_records_read":False,"airports":{}}
    for airport,bbox in {"LIRF":"41.77,12.20,41.83,12.30","LFPG":"48.98,2.50,49.04,2.62"}.items():
        query=f'[out:json][timeout:90][date:"{DATE}"];nwr["aeroway"~"^(taxiway|taxilane|parking_position|runway|holding_position)$"]({bbox});(._;>;);out meta;'
        url="https://overpass-api.de/api/interpreter?"+urlencode({"data":query})
        rec={"url":url}
        try:
            with urlopen(Request(url,headers={"User-Agent":"PRC2026-local-public-geometry-research/1.0"}),timeout=110) as response:body=response.read()
            (OUT/(airport+"_meta.json")).write_bytes(body)
            meta=json.loads(body)
            assert not meta.get("remark"),meta.get("remark")
            original=json.loads((OUT/(airport+".json")).read_text())
            compare=lambda e:{k:v for k,v in e.items() if k not in ("timestamp","version","changeset","user","uid")}
            assert {(e["type"],e["id"]):compare(e) for e in meta["elements"]}=={(e["type"],e["id"]):e for e in original["elements"]}
            timestamps=[e["timestamp"] for e in meta["elements"]]
            assert max(timestamps)<=DATE
            rec.update(status="passed",elements=len(timestamps),oldest_element_timestamp=min(timestamps),latest_element_timestamp=max(timestamps),all_element_timestamps_before_snapshot=True,geometry_equal_to_initial_download=True,sha256=hashlib.sha256(body).hexdigest())
        except Exception as exc:rec.update(status="failed",error=repr(exc))
        report["airports"][airport]=rec
        report["status"]="passed" if len(report["airports"])==2 and all(r["status"]=="passed" for r in report["airports"].values()) else "incomplete"
        (OUT/"snapshot_verification.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
        print(airport,rec.get("status"),rec.get("latest_element_timestamp"),rec.get("error"),flush=True)


if __name__=="__main__":main()
