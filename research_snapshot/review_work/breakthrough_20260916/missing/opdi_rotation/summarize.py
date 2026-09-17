import json
from pathlib import Path
import pandas as pd
from build import OUT, FEATURES, sha

manifest = json.loads((OUT / "manifest.json").read_text())
assert manifest["status"] == "complete"
for filename, digest in manifest["outputs"].items():
    assert sha(OUT / filename) == digest
summary = {"manifest_sha256": sha(OUT / "manifest.json"), "splits": {}, "months": manifest["coverage"]}
for split in ["training", "ranking"]:
    frame = pd.read_parquet(OUT / f"{split}_features.parquet")
    summary["splits"][split] = {"rows": len(frame), "matched": int(frame.opdi_match.sum()),
        "ambiguous": int(frame.opdi_match_ambiguous.sum()),
        "previous_same_airport": int(frame.opdi_previous_same_airport.sum()),
        "match_share": float(frame.opdi_match.mean()),
        "rotation_share": float(frame.opdi_previous_same_airport.mean())}
text = ["# OPDI aircraft-linkage feasibility", "",
    "This is a coverage result, not a prediction result or independently verified aircraft identity. "
    "Original public files are preserved, with EUROCONTROL PRC / OpenSky Network attribution. "
    "Local noncommercial research is authorized; future prize eligibility remains unresolved.", "",
    "All original missing-NM rows are retained. The cache contains eight numeric fields and no raw aircraft identifier. "
    "Matching uses normalized callsign, origin, destination and exactly one public observation start within 600 seconds. "
    "Previous legs are restricted to the same month and must finish before both the current observation start and takeoff; "
    "duration features require the previous destination to equal the departure origin.", "",
    "| Split | Rows | Matched | Match share | Previous same-airport leg | Rotation share |",
    "|---|---:|---:|---:|---:|---:|"]
for split, record in summary["splits"].items():
    text.append(f"| {split} | {record['rows']:,} | {record['matched']:,} | {record['match_share']:.2%} | {record['previous_same_airport']:,} | {record['rotation_share']:.2%} |")
text += ["", "| Month | Missing-clock queries | Matched | Previous same-airport leg |", "|---|---:|---:|---:|"]
for record in manifest["coverage"]:
    text.append(f"| {record['month']} | {record['queries']:,} | {record['matched']:,} | {record['previous_same_airport']:,} |")
text += ["", "January 2026 match coverage is 9.9%, versus 30.0% in January 2025; its rotation coverage is 8.4%, versus 22.6%. "
    "July 2026 match coverage is 37.4% and rotation coverage 31.4%. This is a material seasonal transfer limitation. "
    "The public January 2026 file has fewer observed legs than January 2025, but this does not establish why coverage differs.", "",
    "There were no ambiguous matches in these actual cohorts under the strict rule. Synthetic ambiguity tests still exercise rejection. "
    "A zero ambiguity count is not a measured zero false-match rate. Same ICAO24 and consistent route/time strengthen an association, "
    "but surveillance gaps, inferred airports, identity errors and the difference between observation boundaries and ground/block times remain.", "",
    f"Builder elapsed seconds: {manifest['elapsed_sec']:.2f}; peak process bytes: {manifest['peak_bytes']:,}. "
    "V2 and V3 build attempts were stopped after independent review and preserved. V4 passed nine final-wrapper synthetic tests. "
    "Independent cache verification is separately owned and must pass before training consumes this artifact.", "",
    "Receipts: `acquisition.json`, `manifest.json`, `coverage_summary.json`, `attempt_history.json`, `attempt_v3.json`; "
    "independent validation will write `independent_oracle.json` and `verification.json`."]
for filename, value in [("coverage_summary.json", json.dumps(summary, indent=2)), ("analysis.md", "\n".join(text) + "\n")]:
    path = OUT / filename
    assert not path.exists()
    path.write_text(value, encoding="utf-8")
print(json.dumps(summary["splits"], indent=2))
