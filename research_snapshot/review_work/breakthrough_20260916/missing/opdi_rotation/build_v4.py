"""Reject blank aircraft identity consistently before both linkage passes."""
import json
import build as base
import build_v3 as v3

original = v3.build_month

def build_month(query, public, month):
    public = public.loc[base.normalize(public["icao24"]).ne("")].copy()
    return original(query, public, month)

if __name__ == "__main__":
    v3.build_month = build_month
    v3.main()
    path = base.OUT / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["builder_v3_sha256"] = manifest["source_sha256"]
    manifest["source_sha256"] = base.sha(__file__)
    manifest["builder_v3_path"] = str(base.Path(v3.__file__).resolve())
    manifest["builder_source_path"] = str(base.Path(__file__).resolve())
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
