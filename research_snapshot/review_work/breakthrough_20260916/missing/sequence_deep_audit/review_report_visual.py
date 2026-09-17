"""Bind an already completed human-style visual inspection to artifact hashes."""
import argparse
import hashlib
import json
from pathlib import Path
from pypdf import PdfReader
from PIL import Image

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

parser = argparse.ArgumentParser()
parser.add_argument("--report", type=Path, required=True)
parser.add_argument("--decision", type=Path, required=True)
parser.add_argument("--receipt-name", required=True)
parser.add_argument("--stage", choices=["draft", "final"], required=True)
args = parser.parse_args()
receipt = json.loads((args.report / "report_receipt.json").read_text())
decision = json.loads(args.decision.read_text())
pdf = args.report / "PRC_2026_Breakthrough_Search.pdf"
assert sha(pdf) == receipt["pdf_sha256"]
assert sha(args.decision) == receipt["decision_sha256"]
inventory = ROOT / "private_runs/breakthrough_20260916/inventory_final_1402/inventory.json"
assert sha(inventory) == receipt["inventory_sha256"]
for filename, digest in receipt["supplemental_evidence"].items():
    assert sha(filename) == digest
pages = PdfReader(str(pdf)).pages
assert len(pages) == 5 and receipt["pages"] == 5
assert bool(receipt["final"]) == (args.stage == "final")
text = "\n".join(page.extract_text() for page in pages)
for token in ["272.26", "292.81", "294.626", "342.03", "model_shortlist_final_1402.md", "sources_final_1402.json"]:
    assert token in text, token
pngs = []
for index in range(1, 6):
    path = args.report / f"page_{index}.png"
    with Image.open(path) as image:
        assert image.width >= 800 and image.height >= 1100
        dimensions = [image.width, image.height]
    pngs.append({"path": str(path), "sha256": sha(path), "dimensions": dimensions,
                 "visually_inspected": True, "clipping_overlap_or_unreadable_text": False})
result = {"status": "passed", "stage": args.stage,
    "scope": "All five supplied PNGs visually inspected before this script. PDF, decision, inventory and supplemental hashes checked; five-page count and key score/source text checked.",
    "report_receipt_sha256": sha(args.report / "report_receipt.json"), "pdf_sha256": sha(pdf),
    "decision_sha256": sha(args.decision), "inventory_sha256": sha(inventory),
    "source_sha256": sha(__file__), "pages": pngs,
    "content_limits": "Visual/evidence review does not add model inference, independent holdout, identity truth or official score evidence.",
    "nonblocking_draft_clarity_note": "Draft GRU table blend289.61 is below V2; standalone308.80 loses, and strict replay fails. Make that distinction explicit in the final note." if args.stage == "draft" else None}
path = OUT / args.receipt_name
assert not path.exists()
path.write_text(json.dumps(result, indent=2), encoding="utf-8")
print(path, sha(pdf), "FIVE_PAGES_VISUALLY_REVIEWED_AND_HASH_BOUND")
