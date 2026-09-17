"""Fetch public primary-source pages only; never reads confidential inputs."""

import hashlib
import json
import urllib.request
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / 'private_runs/aviation_spike/sources'
URLS = {
    'challenge_data': 'https://prc-data-challenge-2026.netlify.app/data.html',
    'eligibility': 'https://prc-data-challenge-2026.netlify.app/eligibility.html',
    'taxiout': 'https://ansperformance.eu/efficiency/taxiout/',
    'definition': 'https://ansperformance.eu/definition/additional-taxi-out-time/',
    'methodology': 'https://ansperformance.eu/methodology/additional-taxi-out-time/',
}


class MainText(HTMLParser):
    def __init__(self):
        super().__init__()
        self.main = False
        self.skip = 0
        self.chunks = []
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag == 'main':
            self.main = True
        if tag in ['script', 'style']:
            self.skip += 1
        if self.main and tag == 'a':
            values = dict(attrs)
            if 'href' in values:
                self.links.append(values['href'])

    def handle_endtag(self, tag):
        if tag == 'main':
            self.main = False
        if tag in ['script', 'style']:
            self.skip = max(0, self.skip-1)

    def handle_data(self, data):
        if self.main and not self.skip and data.strip():
            self.chunks.append(data.strip())


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    receipts = {}
    for name, url in URLS.items():
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Research/1.0'})
            with urllib.request.urlopen(req, timeout=25) as response:
                data = response.read()
                status = response.status
            parser = MainText()
            parser.feed(data.decode('utf-8'))
            (OUT/f'{name}.html').write_bytes(data)
            (OUT/f'{name}.txt').write_text('\n'.join(parser.chunks), encoding='utf-8')
            receipts[name] = {'url': url, 'status': status, 'sha256': hashlib.sha256(data).hexdigest(),
                              'utc': datetime.now(timezone.utc).isoformat(), 'links': parser.links}
            print(name, status, len(parser.chunks), 'text fragments', flush=True)
        except Exception as exc:
            receipts[name] = {'url': url, 'error': str(exc)}
            print(name, str(exc), flush=True)
    (OUT/'receipts.json').write_text(json.dumps(receipts, indent=2), encoding='utf-8')
