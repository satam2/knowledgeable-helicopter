"""Preserve first builder; also require prior completion before observed takeoff."""
import json
from pathlib import Path
import numpy as np
import build as base

original = base.build_month

def build_month(query, public, month):
    public = public.copy()
    first = base.timestamps(public["first_seen"])
    last = base.timestamps(public["last_seen"])
    public = public.loc[first.notna() & last.notna() & last.gt(first)]
    result = original(query, public, month)
    invalid = result["opdi_previous_leg_age_at_takeoff_sec"].le(0)
    result.loc[invalid, ["opdi_previous_leg_available", "opdi_previous_same_airport"]] = 0
    result.loc[invalid, base.FEATURES[5:]] = np.nan
    return result

if __name__ == "__main__":
    base.build_month = build_month
    base.main()
    path = base.OUT / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["builder_base_sha256"] = manifest["source_sha256"]
    manifest["source_sha256"] = base.sha(__file__)
    manifest["builder_base_path"] = str(Path(base.__file__).resolve())
    manifest["builder_source_path"] = str(Path(__file__).resolve())
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
