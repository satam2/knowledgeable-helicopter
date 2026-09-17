"""Final evidence checks for raw privacy, score integrity and PDF layout."""

import hashlib
import json
import subprocess
from pathlib import Path

import pdfplumber

WORKSPACE = Path(__file__).resolve().parents[1]
OUT = WORKSPACE / "private_runs/screening_230"
SOURCE = WORKSPACE / "knowledgeable-helicopter-screening"
ORIGINAL = WORKSPACE / "knowledgeable-helicopter"
PDF = WORKSPACE / "output/PRC_2026_Screening_Results.pdf"


def digest(file):
    h = hashlib.sha256()
    with file.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


protocol = json.loads((OUT / "protocol.json").read_text())
assert {p.name: digest(p) for p in Path(protocol["raw_root"]).glob("*.parquet")} == protocol["raw_hashes"]
for name, value in protocol["source_hashes"].items():
    assert digest(SOURCE / name) == value, name
assert not [p for r in [SOURCE, ORIGINAL] for glob in ["*.parquet", "*.cbm"] for p in r.rglob(glob)]
original_status = subprocess.check_output(["git", "status", "--porcelain"], cwd=ORIGINAL, text=True).strip()
assert original_status == "?? docs/superpowers/", original_status
for repo in [ORIGINAL, SOURCE]:
    assert not subprocess.check_output(["git", "ls-files", "--", "*.parquet", "*.cbm"], cwd=repo, text=True).strip()
summary = json.loads((OUT / "screening_summary.json").read_text())
assert summary["status"] == "complete" and summary["candidate_folds_complete"] == 8
full = json.loads((OUT / "followup_summary.json").read_text())
historical = json.loads((ORIGINAL / "reports/comparison.json").read_text())["candidates"]["residual_long_proxy_specialist"]
for fold, m in full["candidates"]["baseline"]["folds"].items():
    assert abs(m["overall"]["rmse_sec"] - historical["folds"][fold]["rmse_sec"]) < 1e-9
doc = pdfplumber.open(PDF)
assert len(doc.pages) == 4
bounds, collisions = [], []
for i, page in enumerate(doc.pages):
    chars = [c for c in page.chars if c["text"].strip()]
    for j, char in enumerate(chars):
        rect = (char["x0"], char["top"], char["x1"], char["bottom"])
        if rect[0] < 0 or rect[1] < 0 or rect[2] > page.width or rect[3] > page.height:
            bounds.append([i + 1, char["text"], rect])
        for other in chars[j + 1:]:
            r = (other["x0"], other["top"], other["x1"], other["bottom"])
            if r[1] >= rect[3] or r[3] <= rect[1] or r[0] >= rect[2] or r[2] <= rect[0]:
                continue
            area = (min(rect[2], r[2]) - max(rect[0], r[0])) * (min(rect[3], r[3]) - max(rect[1], r[1]))
            if area > .35 * min((rect[2] - rect[0]) * (rect[3] - rect[1]), (r[2] - r[0]) * (r[3] - r[1])):
                collisions.append([i + 1, char["text"], other["text"]])
assert not bounds, bounds[:10]
assert not collisions, collisions[:10]
text = "\n".join(p.extract_text() for p in doc.pages)
for value in ["331.18", "330.89", "319.00", "331.30", "319.41", "230"]:
    assert value in text, value
report = {"raw_hashes_unchanged": 14, "frozen_source_files_unchanged": len(protocol["source_hashes"]),
          "original_checkout_status": original_status, "private_artifacts_in_checkouts": [],
          "baseline_reproduced_folds": list(full["candidates"]["baseline"]["folds"]),
          "initial_candidate_evaluations": 8, "all_completed_fold_evaluations": full["completed_fold_evaluations"],
          "pdf_pages": len(doc.pages), "pdf_sha256": digest(PDF), "out_of_bounds_characters": bounds,
          "character_collisions": collisions, "pdf": str(PDF), "scope": "Local development; no leaderboard score or release claim"}
(OUT / "final_verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
(OUT / "report_text.txt").write_text(text, encoding="utf-8")
print(json.dumps(report, indent=2))
