"""Bounded public-only solution search; never reads campaign/private records."""
import argparse
from datetime import datetime, timezone
from html.parser import HTMLParser
import hashlib
import json
from pathlib import Path
import urllib.error
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / 'output/breakthrough_20260916/research_gap/public_solutions_1310'


class Page(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
        self.text = []
        self.hidden = 0

    def handle_starttag(self, tag, attrs):
        if tag in ('script', 'style'):
            self.hidden += 1
        values = dict(attrs)
        if tag == 'a' and 'href' in values:
            self.links.append(values['href'])

    def handle_endtag(self, tag):
        if tag in ('script', 'style'):
            self.hidden = max(0, self.hidden - 1)

    def handle_data(self, data):
        if not self.hidden and data.strip():
            self.text.append(data.strip())


def fetch(key, url):
    path = OUT / f'{key}.raw'
    if path.exists():
        raise FileExistsError(path)
    receipt = {'url': url, 'retrieved_utc': datetime.now(timezone.utc).isoformat()}
    request = urllib.request.Request(url, headers={'User-Agent': 'PRC-public-research/1.0', 'Accept': 'application/json,text/html;q=0.9,*/*;q=0.8'})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            body = response.read(4 * 1024 * 1024)
            receipt.update(status=response.status, final_url=response.url, content_type=response.headers.get('Content-Type'))
    except urllib.error.HTTPError as error:
        body = error.read()
        receipt.update(status=error.code, error=str(error))
    except Exception as error:
        receipt.update(error=repr(error))
        body = b''
    path.write_bytes(body)
    receipt.update(bytes=len(body), sha256=hashlib.sha256(body).hexdigest())
    decoded = body.decode('utf-8', errors='replace')
    try:
        data = json.loads(decoded)
    except (ValueError, TypeError):
        parser = Page()
        parser.feed(decoded)
        (OUT / f'{key}.txt').write_text('\n'.join(parser.text), encoding='utf-8')
        receipt['links'] = sorted(set(urllib.parse.urljoin(url, link) for link in parser.links))
        receipt['text_excerpt'] = '\n'.join(parser.text)[:18000]
    else:
        if isinstance(data, dict) and 'items' in data:
            receipt['total_count'] = data.get('total_count')
            receipt['items'] = [{k: item.get(k) for k in ['full_name', 'html_url', 'description', 'updated_at', 'created_at', 'pushed_at', 'name', 'path']} for item in data['items']]
        else:
            receipt['json'] = data
    (OUT / f'{key}.receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(json.dumps({'key': key, **receipt}, ensure_ascii=True), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', nargs=2, action='append', metavar=('KEY', 'URL'), required=True)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for key, url in args.source:
        fetch(key, url)
