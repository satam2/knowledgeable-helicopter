"""Fetch public station weather only. This process never reads private inputs."""
import csv
import hashlib
import io
import json
import time
from urllib.error import HTTPError
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'output/breakthrough_20260916/public_weather'
# Public stations from the competition's published airport table.
STATIONS = ['EDDF','EDDM','EGLL','EHAM','LEBL','LEMD','LFPG','LIRF','LTFM','LSZH']
PERIODS = [(f'2025_{month:02d}', (2025,month,1),
            (2025,month+1,1) if month<12 else (2026,1,1)) for month in range(1,13)]
PERIODS += [('2024_12_31', (2024,12,31), (2025,1,1)),
            ('2026_01', (2026,1,1), (2026,2,1)),
            ('2026_06_30', (2026,6,30), (2026,7,1)),
            ('2026_07', (2026,7,1), (2026,8,1))]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    receipts = []
    for station in STATIONS:
        for period, start, end in PERIODS:
            dest = OUT / f'{station}_{period}.csv'
            receipt_path = dest.with_suffix('.json')
            if receipt_path.exists() and dest.exists():
                receipt = json.loads(receipt_path.read_text())
                assert hashlib.sha256(dest.read_bytes()).hexdigest() == receipt['sha256']
                receipts.append(receipt)
                continue
            query = dict(station=station, data='all', year1=start[0], month1=start[1], day1=start[2],
                         year2=end[0], month2=end[1], day2=end[2], tz='Etc/UTC', format='onlycomma',
                         latlon='no', missing='M', trace='T', direct='no', report_type=3)
            url = 'https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?' + urlencode(query)
            request = Request(url, headers={'User-Agent':'PRC2026-public-weather-research/1.0'})
            for attempt in range(3):
                try:
                    with urlopen(request, timeout=60) as response:
                        body = response.read()
                    break
                except Exception as error:
                    print('RETRY',station,period,attempt,repr(error),flush=True)
                    if attempt == 2:
                        raise
                    delay = 15*(attempt+1)
                    if isinstance(error, HTTPError) and error.code == 429:
                        delay = max(delay, float(error.headers.get('Retry-After', 30)))
                    time.sleep(min(delay, 60))
            records = list(csv.DictReader(io.StringIO(body.decode('utf-8-sig'))))
            if not records or 'valid' not in records[0] or any(r['station'] != station for r in records):
                raise ValueError(f'Unexpected public weather response for {station}')
            dest.write_bytes(body)
            receipt = {'url':url,'station':station,'period':period,'rows':len(records),
                       'sha256':hashlib.sha256(body).hexdigest(),'bytes':len(body),
                       'fetched_utc':datetime.now(timezone.utc).isoformat(),
                       'public_requests_only':True,'private_files_read':False,
                       'license_source':'https://mesonet.agron.iastate.edu/disclaimer.php',
                       'license':'IEM materials are public domain per preserved research disclaimer'}
            receipt_path.write_text(json.dumps(receipt,indent=2),encoding='utf-8')
            receipts.append(receipt)
            print(station,period,len(records),flush=True)
            time.sleep(5)
    (OUT/'manifest.json').write_text(json.dumps({'status':'complete','receipts':receipts},indent=2),encoding='utf-8')


if __name__ == '__main__':
    main()
