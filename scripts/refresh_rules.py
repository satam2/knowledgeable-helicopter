"""Read-only official rule snapshot. No leaderboard scores or external writes."""
import hashlib
import json
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.request import Request, urlopen


class TextParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style"}:
            self.skip += 1

    def handle_endtag(self, tag):
        if tag in {"script", "style"}:
            self.skip = max(0, self.skip - 1)

    def handle_data(self, data):
        if not self.skip and data.strip():
            self.parts.append(data.strip())


def main():
    root = Path(__file__).resolve().parents[1] / "reports/rules"
    root.mkdir(parents=True, exist_ok=True)
    records = []
    for page in ["index", "data", "ranking", "eligibility", "rationale"]:
        url = "https://prc-data-challenge-2026.netlify.app/" + ("" if page == "index" else page + ".html")
        entry = {"url": url, "checked_utc": datetime.now(timezone.utc).isoformat()}
        try:
            request = Request(url, headers={"User-Agent": "PRC-local-reproduction/0.1"})
            with urlopen(request, timeout=30) as response:
                raw = response.read()
                entry["http_status"] = response.status
            (root / (page + ".html")).write_bytes(raw)
            parser = TextParser()
            parser.feed(raw.decode("utf-8"))
            entry["sha256"] = hashlib.sha256(raw).hexdigest()
            print(page, entry["http_status"], entry["sha256"], flush=True)
        except Exception as error:
            entry["error"] = str(error)
            print(page, entry["error"], flush=True)
        records.append(entry)
    (root / "snapshot.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
