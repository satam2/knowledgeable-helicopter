"""Check final PDF structure, text preservation and page geometry."""
import hashlib
import json
import re
from pathlib import Path

import pdfplumber
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "review_work"
PDF = ROOT / "output/PRC_2026_Deep_Review.pdf"


def inline_text(tokens):
    return "".join(inline_text(t["tokens"]) if "tokens" in t else t.get("text", "") for t in tokens)


def normalized(text):
    return re.sub(r"\s+", "", text)


def main():
    tokens = json.loads((WORK / "report_tokens.json").read_text(encoding="utf-8"))
    expected = []
    for token in tokens:
        if token["type"] in {"heading", "paragraph"}:
            expected.append(inline_text(token["tokens"]))
        elif token["type"] == "table":
            expected.extend(inline_text(cell["tokens"]) for row in [token["header"], *token["rows"]] for cell in row)
        elif token["type"] == "list":
            for item in token["items"]:
                expected.extend(inline_text(child["tokens"]) for child in item["tokens"] if "tokens" in child)
    reader = PdfReader(PDF)
    text = "\n".join(page.extract_text() for page in reader.pages)
    (WORK / "pdf_text.txt").write_text(text, encoding="utf-8")
    complete = normalized(text)
    missing = [value[:160] for value in expected if normalized(value) not in complete]
    # A split paragraph has a running header/footer between its two fragments.
    cleaned = re.sub(r"PRC 2026\s*\|\s*Technical review", "", text)
    cleaned = re.sub(r"15 September 2026\s*\|\s*Revision 9e5b9e6\s*\d+", "", cleaned)
    missing = [value[:160] for value in expected if normalized(value) not in normalized(cleaned)]
    geometry, page_records = [], []
    with pdfplumber.open(PDF) as pdf:
        for index, page in enumerate(pdf.pages, 1):
            bad = [c for c in page.chars if c["x0"] < 50 or c["x1"] > 562 or c["top"] < 12 or c["bottom"] > 780]
            geometry.extend({"page": index, "text": c["text"], "x0": c["x0"], "x1": c["x1"], "top": c["top"]} for c in bad)
            page_records.append({"page": index, "characters": len(page.chars), "words": len(page.extract_words())})
    result = {"pages": len(reader.pages), "checked_text_blocks_and_cells": len(expected), "missing_text": missing,
              "out_of_bounds_characters": geometry, "page_records": page_records,
              "sha256": hashlib.sha256(PDF.read_bytes()).hexdigest()}
    (WORK / "document_verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k != "page_records"}, indent=2))
    assert not missing, "PDF lost Markdown text"
    assert not geometry, "PDF text extends outside page margins"
    assert all(record["characters"] > 100 for record in page_records), "Unexpected blank page"


if __name__ == "__main__":
    main()
