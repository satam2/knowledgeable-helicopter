"""Primary atmospheric formula documentation; no private input reads."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
from urllib.request import Request,urlopen

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/"output/breakthrough_20260916/weather_wind/public"
SOURCES={
 "metpy_basic.py":"https://raw.githubusercontent.com/Unidata/MetPy/main/src/metpy/calc/basic.py",
 "metpy_thermo.py":"https://raw.githubusercontent.com/Unidata/MetPy/main/src/metpy/calc/thermo.py",
 "metpy_altimeter.html":"https://unidata.github.io/MetPy/latest/api/generated/metpy.calc.altimeter_to_station_pressure.html",
 "metpy_density.html":"https://unidata.github.io/MetPy/latest/api/generated/metpy.calc.density.html",
 "metpy_license":"https://raw.githubusercontent.com/Unidata/MetPy/main/LICENSE",
}


def main():
    if (OUT/"density_receipt.json").exists():raise ValueError("Preserve previous retrieval")
    records={}
    for name,url in SOURCES.items():
        record={"url":url,"retrieved_utc":datetime.now(timezone.utc).isoformat(),"private_data_read":False}
        try:
            with urlopen(Request(url,headers={"User-Agent":"PRC2026-public-physical-features/1.0"}),timeout=45) as response:body=response.read()
            (OUT/name).write_bytes(body)
            record.update(status=200,bytes=len(body),sha256=hashlib.sha256(body).hexdigest())
        except Exception as exc:record["error"]=repr(exc)
        records[name]=record
        (OUT/"density_receipt.json").write_text(json.dumps(records,indent=2),encoding="utf-8")
        print(name,record.get("status"),record.get("error"),flush=True)


if __name__=="__main__":main()
