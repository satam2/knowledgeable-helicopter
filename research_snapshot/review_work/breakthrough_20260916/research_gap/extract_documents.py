"""Extract primary-document text with page markers; no private inputs."""
import hashlib
import json
from pathlib import Path
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parents[3]
FOLDER=ROOT/"output/breakthrough_20260916/research_gap/documents/sources"


def main():
    report={}
    for path in FOLDER.glob("*.pdf.raw"):
        reader=PdfReader(path)
        pages=[f"\n===== PDF PAGE {i+1} =====\n"+(p.extract_text() or "") for i,p in enumerate(reader.pages)]
        text="\n".join(pages)
        dest=path.with_name(path.name.replace(".pdf.raw",".extracted.txt"))
        dest.write_text(text,encoding="utf-8")
        report[path.name]={"pages":len(reader.pages),"characters":len(text),"pdf_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"text_sha256":hashlib.sha256(text.encode()).hexdigest()}
        print(path.name,len(reader.pages),len(text),flush=True)
    (FOLDER.parent/"extraction.json").write_text(json.dumps(report,indent=2),encoding="utf-8")


if __name__=="__main__":main()
