"""Acquire only fixed public monthly OPDI files, sequentially with TLS verification."""
import os
os.environ["OMP_NUM_THREADS"] = "1"
import hashlib
import json
from pathlib import Path
import sys
import time
import urllib.request
import psutil
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "knowledgeable-helicopter-screening/src"))
from taxiout.paths import external_path
OUT = external_path(ROOT / "private_runs/breakthrough_20260916/missing/opdi_rotation")
PUBLIC = OUT / "public"
PUBLIC.mkdir(parents=True, exist_ok=True)
MONTHS = ["202507"] + [f"2025{x:02d}" for x in range(1, 13) if x != 7] + ["202601", "202607"]
REQUIRED = {"icao24", "flt_id", "adep", "ades", "first_seen", "last_seen"}
records = []
started = time.time()
for month in MONTHS:
    assert psutil.virtual_memory().available >= 8 * 1024**3, "Host reserve below8GiB"
    assert psutil.Process().memory_info().rss < 2 * 1024**3
    url = f"https://www.eurocontrol.int/performance/data/download/OPDI/v002/flight_list/flight_list_{month}.parquet"
    dest = PUBLIC / f"flight_list_{month}.parquet"
    if dest.exists():
        raise RuntimeError(f"Refusing overwrite {dest}")
    partial = dest.with_suffix(".parquet.partial")
    if partial.exists():
        raise RuntimeError(f"Preserve previous partial {partial}")
    sha = hashlib.sha256()
    size = 0
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 PRC-local-noncommercial-research"})
    with urllib.request.urlopen(request, timeout=90) as response, partial.open("xb") as stream:
        headers = dict(response.headers)
        assert response.status == 200
        expected = int(response.headers.get("Content-Length", "0"))
        assert 0 < expected < 100_000_000
        while True:
            data = response.read(1024 * 1024)
            if not data:
                break
            stream.write(data)
            sha.update(data)
            size += len(data)
            assert size <= expected
            assert psutil.virtual_memory().available >= 8 * 1024**3
    assert size == expected
    partial.rename(dest)
    metadata = pq.ParquetFile(dest)
    assert REQUIRED.issubset(metadata.schema_arrow.names), str(metadata.schema_arrow)
    records.append({"month": month, "url": url, "sha256": sha.hexdigest(), "bytes": size,
                    "rows": metadata.metadata.num_rows, "schema": str(metadata.schema_arrow),
                    "headers": headers, "local_path": str(dest)})
    (OUT / "acquisition.json").write_text(json.dumps({
        "status": "complete" if len(records) == len(MONTHS) else "running",
        "attribution": "EUROCONTROL PRC / OpenSky Network, OPDI v0.0.2",
        "scope": "Local noncommercial research; prize eligibility unresolved. Original bytes preserved.",
        "tls_verified": True, "private_data_in_requests": False,
        "elapsed_sec": time.time() - started, "sources": records}, indent=2), encoding="utf-8")
    print("ACQUIRED", month, size, metadata.metadata.num_rows, flush=True)
    if len(records) == 1:
        print("CANARY_SCHEMA", metadata.schema_arrow, flush=True)
