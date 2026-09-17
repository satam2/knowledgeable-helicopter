"""Fetch only explicitly listed public sources; never reads challenge data."""
from __future__ import annotations

import argparse
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import time
from urllib.request import Request, urlopen
from urllib.parse import urlencode
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output" / "research_20260916" / "sources"
SOURCES = {
    "competition_data": "https://prc-data-challenge-2026.netlify.app/data.html",
    "competition_eligibility": "https://prc-data-challenge-2026.netlify.app/eligibility.html",
    "lightgbm_features": "https://lightgbm.readthedocs.io/en/latest/Advanced-Topics.html",
    "lightgbm_release": "https://api.github.com/repos/microsoft/LightGBM/releases/latest",
    "lightgbm_license": "https://raw.githubusercontent.com/microsoft/LightGBM/master/LICENSE",
    "xgboost_categorical": "https://xgboost.readthedocs.io/en/stable/tutorials/categorical.html",
    "xgboost_gpu": "https://xgboost.readthedocs.io/en/stable/gpu/index.html",
    "xgboost_release": "https://api.github.com/repos/dmlc/xgboost/releases/latest",
    "xgboost_license": "https://raw.githubusercontent.com/dmlc/xgboost/master/LICENSE",
    "catboost_missing": "https://catboost.ai/docs/en/concepts/algorithm-missing-values-processing",
    "catboost_categorical": "https://catboost.ai/docs/en/features/categorical-features",
    "catboost_release": "https://api.github.com/repos/catboost/catboost/releases/latest",
    "catboost_license": "https://raw.githubusercontent.com/catboost/catboost/master/LICENSE",
    "tabm_readme": "https://raw.githubusercontent.com/yandex-research/tabm/main/README.md",
    "tabm_paper": "https://arxiv.org/abs/2410.24258",
    "tabm_package": "https://pypi.org/pypi/tabm/json",
    "tabm_license": "https://raw.githubusercontent.com/yandex-research/tabm/main/LICENSE",
    "ft_transformer_readme": "https://raw.githubusercontent.com/yandex-research/rtdl-revisiting-models/main/README.md",
    "ft_transformer_paper": "https://arxiv.org/abs/2106.11959",
    "ft_transformer_license": "https://raw.githubusercontent.com/yandex-research/rtdl-revisiting-models/main/LICENSE",
    "tabpfn_readme": "https://raw.githubusercontent.com/PriorLabs/TabPFN/main/README.md",
    "tabpfn_license": "https://raw.githubusercontent.com/PriorLabs/TabPFN/main/LICENSE",
    "tabpfn_release": "https://api.github.com/repos/PriorLabs/TabPFN/releases/latest",
    "tabpfn_nature": "https://www.nature.com/articles/s41586-024-08328-6",
    "tabpfn_package": "https://pypi.org/pypi/tabpfn/json",
    "iem_asos": "https://mesonet.agron.iastate.edu/request/download.phtml",
    "iem_disclaimer": "https://mesonet.agron.iastate.edu/disclaimer.php",
    "osm_copyright": "https://www.openstreetmap.org/copyright",
    "osm_airport_mapping": "https://wiki.openstreetmap.org/wiki/Aeroways",
    "era5_license": "https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=overview",
    "tabm_paper_correct": "https://arxiv.org/abs/2410.24210",
    "tabm_paper_code": "https://raw.githubusercontent.com/yandex-research/tabm/main/paper/README.md",
    "tabpfn_models": "https://docs.priorlabs.ai/models",
    "tabpfn_35_weights_license": "https://huggingface.co/Prior-Labs/tabpfn_3_5/raw/main/LICENSE",
    "tabpfn_3_paper": "https://arxiv.org/abs/2605.13986",
    "tabpfn_25_paper": "https://arxiv.org/abs/2511.08667",
    "ft_transformer_package": "https://pypi.org/pypi/rtdl-revisiting-models/json",
    "lightgbm_paper": "https://papers.nips.cc/paper_files/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html",
    "xgboost_paper": "https://arxiv.org/abs/1603.02754",
    "catboost_paper": "https://arxiv.org/abs/1706.09516",
}

for station in ("LIRF", "LFPG"):
    for year, month in ((2025, 1), (2025, 7), (2026, 1), (2026, 7)):
        name = f"weather_{station}_{year}_{month:02d}_01"
        query = dict(station=station, data="all", year1=year, month1=month, day1=1,
                     year2=year, month2=month, day2=2, tz="Etc/UTC", format="onlycomma",
                     latlon="no", missing="M", trace="T", direct="no", report_type=3)
        SOURCES[name] = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?" + urlencode(query)
for airport, bbox in (("LIRF", "41.77,12.20,41.83,12.30"), ("LFPG", "48.98,2.50,49.04,2.62")):
    query = '[out:json][timeout:25];nwr["aeroway"~"^(taxiway|parking_position|runway)$"](' + bbox + ');out tags;'
    SOURCES["geometry_" + airport] = "https://overpass-api.de/api/interpreter?" + urlencode({"data": query})


class PlainText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.skip = 0
        self.parts = []
    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style"}:
            self.skip += 1
    def handle_endtag(self, tag):
        if tag in {"script", "style"}:
            self.skip = max(0, self.skip - 1)
        if tag in {"p", "li", "h1", "h2", "h3", "tr", "section"}:
            self.parts.append("\n")
    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    receipt_path = OUT.parent / "sources.json"
    receipts = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
    for name, url in SOURCES.items():
        if args.only and name not in args.only:
            continue
        start = time.monotonic()
        rec = {"url": url, "fetched_at_utc": datetime.now(timezone.utc).isoformat(), "request_contains_private_data": False}
        try:
            req = Request(url, headers={"User-Agent": "PRC2026-local-public-research/1.0"})
            with urlopen(req, timeout=30) as response:
                body = response.read()
                rec.update(status=response.status, final_url=response.url, content_type=response.headers.get("content-type"), sha256=hashlib.sha256(body).hexdigest(), bytes=len(body))
            text = body.decode("utf-8", errors="replace")
            (OUT / (name + ".raw")).write_bytes(body)
            if "html" in (rec["content_type"] or ""):
                p = PlainText()
                p.feed(text)
                text = "\n".join(line.strip() for line in "".join(p.parts).splitlines() if line.strip())
            (OUT / (name + ".txt")).write_text(text, encoding="utf-8")
            rec["artifact"] = str((OUT / (name + ".txt")).relative_to(ROOT))
        except Exception as exc:
            rec["error"] = repr(exc)
        rec["seconds"] = round(time.monotonic() - start, 3)
        receipts[name] = rec
        receipt_path.write_text(json.dumps(receipts, indent=2), encoding="utf-8")
        print(name, rec.get("status", rec.get("error")), flush=True)


if __name__ == "__main__":
    main()
