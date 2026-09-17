"""Inspect public ATFM date semantics without reading private flight inputs."""
import argparse
import re
import urllib.request
import zipfile
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from pathlib import Path
from acquire import ROOT, OUT as ORIGINAL, digest, now, write

OUT = ROOT / 'private_runs/tail240_20260916/forensics/atfm/date_definition_v1'
URLS = {
    'linked_methodology.html': 'https://ansperformance.eu/references/methodology/ATFM_delay_calculation.html',
    'current_delay_definition.html': 'https://ansperformance.eu/definition/atfm-delay/',
    'current_slot_definition.html': 'https://ansperformance.eu/definition/atfm-adherence/',
}


class Text(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts, self.links, self.excluded = [], [], 0

    def handle_starttag(self, tag, attrs):
        if tag in ('script', 'style'):
            self.excluded += 1
        if tag == 'a':
            self.links.append(dict(attrs).get('href'))
        if tag in ('p', 'li', 'div', 'h1', 'h2', 'h3', 'tr', 'br'):
            self.parts.append('\n')

    def handle_endtag(self, tag):
        if tag in ('script', 'style'):
            self.excluded -= 1

    def handle_data(self, data):
        if not self.excluded:
            self.parts.append(data)

    def text(self):
        return '\n'.join(line for part in ''.join(self.parts).splitlines() if (line := ' '.join(part.split())))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fetch', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    evidence = {'source_sha256': digest(__file__), 'private_inputs_read': False, 'workbooks': [], 'documents': []}
    for name in ['ATFM_Slot_Adherence.xlsx', 'Airport_Arrival_ATFM_Delay.xlsx']:
        with zipfile.ZipFile(ORIGINAL / name) as archive:
            rel = ET.fromstring(archive.read('xl/worksheets/_rels/sheet2.xml.rels'))
            links = [dict(node.attrib) for node in rel]
        evidence['workbooks'].append({'filename': name, 'sha256': digest(ORIGINAL / name), 'metadata_links': links})
    pages = [(ORIGINAL / name, None) for name in ['atfm_definition.html', 'indicator_catalog.html']]
    if args.fetch:
        for name, url in URLS.items():
            path = OUT / name
            assert not path.exists(), 'Preserve source snapshots'
            try:
                request = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 PRC-local-noncommercial-research'})
                with urllib.request.urlopen(request, timeout=45) as response:
                    data = response.read(4 * 1024 * 1024)
                    assert len(data) < 4 * 1024 * 1024
                    path.write_bytes(data)
                    receipt = {'url': url, 'final_url': response.url, 'status': response.status,
                        'sha256': digest(path), 'retrieved_utc': now(), 'headers': dict(response.headers.items())}
            except Exception as error:
                receipt = {'url': url, 'retrieved_utc': now(), 'error': repr(error)}
            write(OUT / (name + '.receipt.json'), receipt)
            if path.exists():
                pages.append((path, receipt))
    for path, receipt in pages:
        parsed = Text()
        parsed.feed(path.read_text(encoding='utf-8'))
        rendered = parsed.text()
        lines = rendered.splitlines()
        matches = [{'line': index + 1, 'text': line} for index, line in enumerate(lines)
            if re.search(r'\bUTC\b|\bGMT\b|time.?zone|local time|date of (the )?flight|midnight|flight date|off.block|take.off|arrival', line, re.I)]
        text_path = OUT / (path.stem + '.txt')
        assert not text_path.exists(), 'Preserve extracted document text'
        text_path.write_text(rendered + '\n', encoding='utf-8')
        evidence['documents'].append({'path': str(path), 'sha256': digest(path), 'receipt': receipt,
            'matches': matches, 'relevant_links': [link for link in parsed.links if link and any(word in link.lower() for word in ['slot', 'adher', 'method', 'atfm'])]})
    write(OUT / 'evidence.json', evidence)
    print(evidence, flush=True)


if __name__ == '__main__':
    main()
