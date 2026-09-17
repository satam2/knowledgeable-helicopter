"""Verify the delivered brief, aggregate consistency and private-data boundary."""
import hashlib
import json
import re
import subprocess
from pathlib import Path

import pdfplumber
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
REPO = (ROOT / "knowledgeable-helicopter").resolve()
PRIVATE = (ROOT / "private_analysis").resolve()
RAW = (ROOT / "data/09-15-2026-18-55-03_files_list").resolve()
PDF = ROOT / "output/PRC_2026_Data_Informed_Plan.pdf"
profile = json.loads((PRIVATE / "profile.json").read_text(encoding="utf-8"))
focused = json.loads((PRIVATE / "focused_checks.json").read_text(encoding="utf-8"))
layout = json.loads((PRIVATE / "visual_plan_layout.json").read_text(encoding="utf-8"))


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def normalized(value):
    return re.sub(r"\s+","",value)


def main():
    manifest = json.loads((REPO / "reports/input_manifest.json").read_text(encoding="utf-8"))
    unchanged = all(sha(RAW / f["file"]) == f["sha256"] for f in manifest["files"])
    assert unchanged
    generated_in_repo = [p.relative_to(REPO).as_posix() for p in REPO.rglob("*") if p.is_file() and p.suffix.lower() in {".parquet",".cbm"}]
    assert not generated_in_repo
    status=subprocess.check_output(["git","status","--short"],cwd=REPO,text=True).strip()
    assert status == "?? docs/superpowers/", status
    freeze=json.loads((REPO / "reports/freeze.json").read_text(encoding="utf-8"))
    changed_source=[name for name,digest in freeze["source_hashes"].items() if sha(REPO/name)!=digest]
    assert not changed_source
    assert sum(v["n"] for v in profile["airports"].values())==2085047
    assert sum(v["missing"] for v in profile["airports"].values())==22470
    assert sum(v["missing_tail"] for v in profile["airports"].values())==481
    assert sum(v["n"] for v in profile["ranking_airports"].values())==344841
    assert all(r["missing_nm"]==r["all_eight_missing"] for r in profile["joint_missingness"])
    assert focused["schedule_regimes"]["lirf"]["tail_n"]+focused["schedule_regimes"]["other_airports"]["tail_n"]==481
    reader=PdfReader(PDF)
    assert len(reader.pages)==6
    page_text=[page.extract_text() for page in reader.pages]
    all_text="\n".join(page_text)
    (PRIVATE/"plan_extracted_text.txt").write_text(all_text,encoding="utf-8")
    missing=[]
    for item in layout["paragraphs"]:
        plain=re.sub(r"<[^>]+>","",item["text"])
        if normalized(plain) not in normalized(page_text[item["page"]-1]):
            missing.append({"page":item["page"],"text":plain})
    assert not missing, missing
    outside=[]
    collisions=[]
    with pdfplumber.open(PDF) as pdf:
        for i,page in enumerate(pdf.pages,1):
            for char in page.chars:
                if char["x0"]<40 or char["x1"]>573 or char["top"]<20 or char["bottom"]>787:
                    outside.append({"page":i,"text":char["text"],"box":[char["x0"],char["top"],char["x1"],char["bottom"]]})
            chars=sorted([ch for ch in page.chars if ch["text"].strip()],key=lambda ch:ch["top"])
            for j,a in enumerate(chars):
                for b in chars[j+1:]:
                    if b["top"]>=a["bottom"]:
                        break
                    overlap_x=min(a["x1"],b["x1"])-max(a["x0"],b["x0"])
                    overlap_y=min(a["bottom"],b["bottom"])-max(a["top"],b["top"])
                    if overlap_x>1.4 and overlap_y>min(a["height"],b["height"])*.6:
                        collisions.append({"page":i,"a":a["text"],"b":b["text"],"x":a["x0"],"top":a["top"]})
    assert not outside, outside
    assert not collisions, collisions[:20]
    output={"pages":6,"figures":len(layout["figures"]),"paragraphs_preserved":len(layout["paragraphs"]),
            "pdf_sha256":sha(PDF),"all_14_raw_hashes_unchanged":unchanged,
            "raw_location_outside_repo":not RAW.is_relative_to(REPO),"private_outputs_outside_repo":not PRIVATE.is_relative_to(REPO),
            "parquet_or_model_files_in_repo":generated_in_repo,"changed_frozen_sources":changed_source,
            "git_status_unchanged_from_start":status,"out_of_bounds_characters":outside,"detected_character_collisions":collisions,
            "note":"Private pack was read in place; no copies, links, trained models or flight-level output files were created."}
    (PRIVATE/"plan_verification.json").write_text(json.dumps(output,indent=2),encoding="utf-8")
    print(json.dumps(output,indent=2))


if __name__=="__main__":
    main()
