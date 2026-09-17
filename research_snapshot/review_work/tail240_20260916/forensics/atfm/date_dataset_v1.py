"""Snapshot the official dataset dictionary links discovered in the catalog."""
import re
import urllib.request
from date_definition_v1 import Text
from acquire import ROOT, digest, now, write

OUT = ROOT / 'private_runs/tail240_20260916/forensics/atfm/date_dataset_v1'
URLS = {
    'arrival_dataset': 'https://ansperformance.eu/reference/dataset/airport-arrival-atfm-delay/',
    'slot_dataset': 'https://ansperformance.eu/reference/dataset/atfm-slot-adherence/',
    'methodology_mixedcase': 'https://ansperformance.eu/methodology/ATFM-delay-calculation',
}


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    receipts = []
    for name, url in URLS.items():
        receipt = dict(url=url, retrieved_utc=now())
        try:
            request = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 PRC-local-noncommercial-research'})
            with urllib.request.urlopen(request, timeout=45) as response:
                data = response.read(4 * 1024 * 1024)
                assert len(data) < 4 * 1024 * 1024
                path = OUT / (name + '.html')
                path.write_bytes(data)
                parsed = Text()
                parsed.feed(data.decode('utf-8'))
                text = parsed.text()
                (OUT / (name + '.txt')).write_text(text + '\n', encoding='utf-8')
                receipt.update(status=response.status, final_url=response.url, sha256=digest(path), headers=dict(response.headers.items()),
                    matches=[{'line': i + 1, 'text': line} for i, line in enumerate(text.splitlines()) if re.search(r'\bUTC\b|\bGMT\b|time.?zone|local time|date of (the )?flight|midnight|flight date|FLT_DATE|off.block|take.off', line, re.I)],
                    links=parsed.links)
        except Exception as error:
            receipt['error'] = repr(error)
        receipts.append(receipt)
        print(name, receipt.get('matches', receipt.get('error')), flush=True)
    write(OUT / 'evidence.json', dict(source_sha256=digest(__file__), parser_source_sha256=digest(ROOT / 'review_work/tail240_20260916/forensics/atfm/date_definition_v1.py'),
        private_inputs_read=False, sources=receipts))


if __name__ == '__main__':
    main()
