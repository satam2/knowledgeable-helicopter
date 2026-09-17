"""One bounded official OPDI event file; no private input values or requests."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
from pathlib import Path
import sys
import urllib.request
from html.parser import HTMLParser
import psutil
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'review_work/campaign_20260916'))
import common
OUT = ROOT / 'private_runs/tail240_20260916/forensics/surface_event_pilot/acquisition_v1'
NAME = 'flight_events_20250105_20250115.parquet'
URL = 'https://www.eurocontrol.int/performance/data/download/OPDI/v002/flight_events/' + NAME
CAP = 400 * 1024**2
pa.set_cpu_count(1)
pa.set_io_thread_count(1)


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag == 'a':
            self.links.append(dict(attrs).get('href'))


def guard():
    memory = psutil.Process().memory_info()
    peak = getattr(memory, 'peak_wset', memory.rss)
    assert peak < 2 * 1024**3 and psutil.virtual_memory().available >= 8 * 1024**3
    return peak


def declare():
    listing = ROOT / 'output/breakthrough_20260916/research_gap/public_methods_aviation/primary_probe/opdi_events.html'
    parser = Links()
    parser.feed(listing.read_text(encoding='utf-8'))
    assert URL in parser.links
    terms = ROOT / 'private_runs/tail240_20260916/forensics/atfm/acquisition_v1/disclaimer.html'
    about = ROOT / 'output/breakthrough_20260916/research_gap/public_methods_aviation/primary_docs_v2/opdi_about.html'
    record = dict(source_sha256=common.sha256(__file__), url=URL, file=NAME, max_transfer_bytes=CAP,
        selection='First official ten-day event file wholly within January2025; dates chosen before data inspection or labels.',
        official_listing_sha256=common.sha256(listing), terms_sha256=common.sha256(terms), opdi_about_sha256=common.sha256(about),
        attribution='EUROCONTROL PRC / OpenSky Network OPDI v0.0.2; local attributed noncommercial research; prize eligibility unresolved.',
        download='Exactly one public file, sequential HTTPS with normal TLS validation; require positive Content-Length <=400MiB before body read; preserve original bytes and receipts.',
        resources='1 CPU, <2GiB current/OSpeak, start>=10GiBavailable, runtime>=8GiBavailable.',
        followup='First schema/ID/version/eventbounds/airportcounts. Then separately frozen label-free exactcallsign+bothairports+/-30min join, no learnedaliases and ambiguousabstention. No fit/expansion. Original Y/clock diagnostics only after construction/coverage freeze.',
        private_fields_read=False, private_data_in_requests=False)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'protocol.json'
    if path.exists():
        assert common.read_json(path) == record
    else:
        common.write_json(path, record)
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--declare-only', action='store_true')
    args = parser.parse_args()
    declare()
    if args.declare_only:
        print('DECLARED', URL, 'cap', CAP, flush=True)
        return
    assert psutil.virtual_memory().available >= 10 * 1024**3
    path = OUT / NAME
    partial = OUT / (NAME + '.partial')
    assert not path.exists() and not partial.exists()
    guard()
    request = urllib.request.Request(URL, headers={'User-Agent':'Mozilla/5.0 PRC-local-noncommercial-research'})
    with urllib.request.urlopen(request, timeout=120) as response:
        length = int(response.headers['Content-Length'])
        assert 0 < length <= CAP, 'Declared transfer cap exceeded or unknown size'
        receipt = dict(url=URL, final_url=response.url, status=response.status, headers=dict(response.headers.items()), expected_bytes=length,
            retrieved_utc=common.utc_now(), tls_verified=True, private_data_in_requests=False)
        common.write_json(OUT / 'response_headers.json', receipt)
        count = 0
        with partial.open('xb') as stream:
            while count < length:
                chunk = response.read(min(1024**2, length-count))
                assert chunk, 'Premature EOF; preserve partial file'
                stream.write(chunk)
                count += len(chunk)
                guard()
        assert count == length
    partial.rename(path)
    source = pq.ParquetFile(path)
    receipt.update(status='complete', bytes=count, sha256=common.sha256(path), rows=source.metadata.num_rows,
        row_groups=source.metadata.num_row_groups, schema=str(source.schema_arrow), peak_bytes=guard(),
        protocol_sha256=common.sha256(OUT / 'protocol.json'), source_sha256=common.sha256(__file__))
    common.write_json(OUT / 'manifest.json', receipt)
    print('ACQUIRED', receipt, flush=True)


if __name__ == '__main__':
    main()
