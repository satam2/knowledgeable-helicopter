"""Public-source acquisition only; no private data reads or model imports."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / 'output/breakthrough_20260916/research_gap/faiss_batch_1306'
URL = 'https://raw.githubusercontent.com/facebookresearch/faiss/v1.12.0/faiss/utils/distances.cpp'


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=False)
    with urllib.request.urlopen(URL, timeout=30) as response:
        data = response.read()
        status = response.status
    (OUT / 'distances_v1_12_0.cpp').write_bytes(data)
    receipt = {'url': URL, 'retrieved_utc': datetime.now(timezone.utc).isoformat(),
               'http_status': status, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest(),
               'private_data_read': False}
    (OUT / 'receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(json.dumps(receipt))
