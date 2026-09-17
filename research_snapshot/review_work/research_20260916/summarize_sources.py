"""Summarize public-source receipts, public weather samples, and map tags."""
from collections import Counter
import csv
from datetime import datetime, timezone
import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output" / "research_20260916"
SRC = OUT / "sources"
summary = {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "private_data_read": False}
releases = {}
for path in SRC.glob("*_release.txt"):
    data = json.loads(path.read_text(encoding="utf-8"))
    releases[path.stem] = {key: data.get(key) for key in ("tag_name", "published_at", "html_url", "target_commitish")}
for path in SRC.glob("*_package.txt"):
    data = json.loads(path.read_text(encoding="utf-8"))
    version = data["info"]["version"]
    releases[path.stem] = {"version": version, "upload_time": data["releases"][version][0]["upload_time_iso_8601"], "requires_python": data["info"].get("requires_python")}
summary["releases"] = releases
weather = {}
for path in sorted(SRC.glob("weather_*.txt")):
    rows = list(csv.DictReader(io.StringIO(path.read_text(encoding="utf-8"))))
    times = sorted(datetime.fromisoformat(row["valid"]) for row in rows)
    intervals = [(b-a).total_seconds()/60 for a,b in zip(times, times[1:])]
    weather[path.stem] = {
        "rows": len(rows), "station": sorted({row["station"] for row in rows}),
        "start": times[0].isoformat() if times else None,
        "end": times[-1].isoformat() if times else None,
        "max_internal_gap_minutes": max(intervals) if intervals else None,
        "nonmissing": {col: sum(row.get(col) not in {None, "", "M"} for row in rows) for col in ("tmpf", "drct", "sknt", "vsby", "wxcodes", "p01i", "metar")},
    }
summary["weather_daily_samples"] = weather
geometry = {}
for path in sorted(SRC.glob("geometry_*.txt")):
    data = json.loads(path.read_text(encoding="utf-8"))
    elements = data["elements"]
    counts = Counter(e.get("tags", {}).get("aeroway") for e in elements)
    with_ref = Counter(e.get("tags", {}).get("aeroway") for e in elements if e.get("tags", {}).get("ref"))
    geometry[path.stem] = {"map_timestamp": data.get("osm3s", {}).get("timestamp_osm_base"), "counts_by_aeroway": dict(counts), "with_ref_by_aeroway": dict(with_ref), "geometry_downloaded": False, "private_stand_match_tested": False, "historical_validity_verified": False}
summary["geometry_public_tag_samples"] = geometry
(OUT / "public_pilot_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
