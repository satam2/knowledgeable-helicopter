"""Bounded public-only retrieval of four explicitly supplied organizer team pages."""
from pathlib import Path
from html.parser import HTMLParser
from urllib.request import Request, urlopen
from urllib.parse import urljoin
from datetime import datetime, timezone
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'knowledgeable-helicopter-screening/src'))
from taxiout.paths import external_path
OUT = external_path(ROOT / 'output/tail240_20260916/state/public_leader_audit/v1')
TEAMS = ['vigorous-whistle', 'enthusiastic-daisy', 'youthful-giraffe', 'jovial-uniform']
BASE = 'https://ansperformance.eu/study/data-challenge/dc2026/teams/'


class Extract(HTMLParser):
    def __init__(self):
        super().__init__()
        self.text = []
        self.links = []
        self.skip = 0
    def handle_starttag(self, tag, attrs):
        if tag in ['script', 'style']:
            self.skip += 1
        if tag == 'a':
            values = dict(attrs)
            if values.get('href'):
                self.links.append(values['href'])
    def handle_endtag(self, tag):
        if tag in ['script', 'style']:
            self.skip = max(0, self.skip-1)
    def handle_data(self, value):
        if not self.skip and value.strip():
            self.text.append(value.strip())


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    receipts = []
    for team in TEAMS:
        url = BASE + team + '.html'
        req = Request(url, headers={'User-Agent':'PRC-public-method-review/1.0'})
        try:
            with urlopen(req, timeout=30) as response:
                body = response.read()
                status = response.status
                final = response.url
                content_type = response.headers.get('Content-Type')
            (OUT / (team + '.html')).write_bytes(body)
            parsed = Extract()
            parsed.feed(body.decode('utf-8', errors='replace'))
            (OUT / (team + '.txt')).write_text('\n'.join(parsed.text), encoding='utf-8')
            links = sorted(set(urljoin(final, link) for link in parsed.links))
            record = dict(team=team, requested_url=url, final_url=final, http_status=status,
                fetched_utc=datetime.now(timezone.utc).isoformat(), bytes=len(body),
                sha256=hashlib.sha256(body).hexdigest(), content_type=content_type, links=links)
        except Exception as error:
            record = dict(team=team, requested_url=url, fetched_utc=datetime.now(timezone.utc).isoformat(), error=str(error))
        receipts.append(record)
        print(json.dumps(record), flush=True)
    (OUT / 'receipts.json').write_text(json.dumps(dict(source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Only four user-specified public organizer team pages; no private data, authentication, score probing or contacts', pages=receipts), indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
