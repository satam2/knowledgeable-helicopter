"""Snapshot JSON run receipts only; no private row reads or model imports."""
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'knowledgeable-helicopter-screening/src'))
from taxiout.paths import external_path


def main():
    destination = external_path(ROOT / 'output/breakthrough_20260916/research_gap/untested_and_limits_1240.json')
    if destination.exists():
        raise FileExistsError('Preserve prior snapshot; use a separately versioned export')
    base = ROOT / 'private_runs/breakthrough_20260916'
    records = []
    for path in sorted(base.rglob('manifest.json')):
        content = path.read_bytes()
        data = json.loads(content)
        records.append({
            'path': str(path.relative_to(ROOT)).replace('\\', '/'),
            'sha256': hashlib.sha256(content).hexdigest(),
            'status': data.get('status'), 'fold': data.get('fold'),
            'family': data.get('family'), 'error': data.get('error'),
            'created_utc': data.get('created_utc'), 'completed_utc': data.get('completed_utc'),
            'runtime_sec': data.get('runtime_sec'),
            'feature_count': len(data['feature_columns']) if 'feature_columns' in data else None,
            'group': path.relative_to(base).parts[0],
        })
    other_paths = [
        'review_work/breakthrough_20260916/gpu_plan_v3.json',
        'review_work/breakthrough_20260916/gpu_queue.py',
        'private_runs/breakthrough_20260916/gpu_queue_1212/results.json',
        'private_runs/breakthrough_20260916/gpu_queue_1131/results.json',
        'private_runs/breakthrough_20260916/models_retrieval/tabdpt_batch_v3/protocol_cuda_s20260916.json',
        'private_runs/breakthrough_20260916/models_retrieval/tabdpt_batch_v3_1/protocol_cuda_s20260916.json',
        'private_runs/breakthrough_20260916/models/sequence_clock_innovations/models/matched_protocol.json',
        'review_work/breakthrough_20260916/deeper_lgb/sequence_leaf63_broader.py',
        'review_work/breakthrough_20260916/deeper_lgb/context_union.py',
    ]
    extra = {p: {'sha256': hashlib.sha256((ROOT / p).read_bytes()).hexdigest(),
                 'exists': True} if (ROOT / p).exists() else {'exists': False} for p in other_paths}
    counts = Counter((r['group'], r['status']) for r in records)
    result = {
        'snapshot_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'All manifest.json files under the campaign private_runs root, plus named pending protocols and queue files.',
        'interpretation': 'Manifest files include caches/canaries and fold attempts; counts are not independent experiments or accepted models. Running statuses are receipt states, not process liveness checks.',
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'counts': [{'group': g, 'status': s, 'count': n} for (g, s), n in sorted(counts.items())],
        'manifests': records,
        'additional_receipts': extra,
        'pending_output_roots': {p: (base / p).exists() for p in ['deeper_sequence_broader', 'deeper_context_union']},
        'private_rows_read': False, 'network_used': False, 'model_launched': False,
    }
    destination.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({'snapshot_utc': result['snapshot_utc'], 'manifest_count': len(records),
                      'counts': result['counts'], 'output': str(destination)}, indent=2))


if __name__ == '__main__':
    main()
