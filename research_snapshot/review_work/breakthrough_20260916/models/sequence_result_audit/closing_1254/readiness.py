"""Read-only final9 artifact readiness, with unresolved acceptance failures explicit."""
from pathlib import Path
import importlib.util
import sys
import json
import hashlib
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[5]
FINAL = ROOT / 'review_work/breakthrough_20260916/models/context_gate/final_simplex9/run.py'
sys.path.insert(0, str(FINAL.parent))
spec = importlib.util.spec_from_file_location('frozen_final9_readiness', FINAL)
frozen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(frozen)
OUT = ROOT / 'private_runs/breakthrough_20260916/models/sequence_result_audit/closing_1254'


def main():
    original = frozen.read_json(frozen.OUT / 'protocol.json')
    states, dependencies = {}, []
    for fold in ('F1', 'F3'):
        states[fold] = {}
        for expert, folder in frozen.registry(fold).items():
            path = folder / 'manifest.json'
            if not path.exists():
                states[fold][expert] = {'status': 'missing', 'manifest': str(path)}
                dependencies.append(f'{expert}/{fold}: missing manifest')
                continue
            record = frozen.read_json(path)
            states[fold][expert] = {'status': record['status'], 'manifest': str(path), 'manifest_sha256': frozen.sha256(path)}
            if record['status'] != 'complete':
                dependencies.append(f'{expert}/{fold}: {record["status"]}')
    failure = ROOT / 'private_runs/breakthrough_20260916/missing/combined_family_audit/xgb/STRICT_REPLAY_NOTE.md'
    assert failure.exists(), 'Preserve strict XGBoost failure receipt'
    gru = ROOT / 'private_runs/breakthrough_20260916/models/sequence_result_audit/gru_cross_device_probe_cuda.json'
    now = datetime.now(timezone.utc)
    report = {'created_utc': now.isoformat(), 'campaign_deadline_utc': '2026-09-16T14:13:00+00:00',
        'source_sha256': frozen.sha256(__file__), 'frozen_final9_protocol_sha256': frozen.sha256(frozen.OUT / 'protocol.json'),
        'frozen_final9_source_matches': original['source_hashes'][str(FINAL.relative_to(ROOT))] == frozen.sha256(FINAL),
        'states': states, 'exact_artifact_dependencies': dependencies, 'all18_model_artifacts_complete': not dependencies,
        'all9_required_no_selective_drop': True,
        'unresolved_acceptance': [{'expert': 'xgb_combined', 'scope': 'IndependentF1CPUversussavedGPUreplay',
            'observed_max_delta_seconds': 0.0005950927734375, 'declared_tolerance_seconds': 0.0001,
            'status': 'strict_failure_preserved_not_waived', 'note_sha256': frozen.sha256(failure), 'note': str(failure),
            'interpretation': 'Complete producer modelmanifest andzero producersavedreload delta do not erase independentstrictCPUfailure. Diagnosticacceptance orpromotionrequires explicitreview.'}],
        'GRU_cuda_probe': {'status': 'available' if gru.exists() else 'not_yet_available', 'path': str(gru)},
        'gate_or_promotion_performed': False, 'gpu_launched': False}
    if gru.exists():
        report['GRU_cuda_probe']['sha256'] = frozen.sha256(gru)
    destination = OUT / ('readiness_' + now.strftime('%H%M%S') + '.json')
    frozen.write_json(destination, report)
    print('FINAL9_DEPENDENCIES', dependencies, 'XGB_strict_acceptance_unresolved', 'GRUprobe', report['GRU_cuda_probe']['status'], flush=True)


if __name__ == '__main__':
    main()
