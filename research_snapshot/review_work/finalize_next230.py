"""Seal external artifacts and verify frozen source, raw inputs and Git privacy."""

import subprocess
import zipfile
from pathlib import Path

from next230_common import OUT, OLD, WORKSPACE, verify_protocol
from taxiout.artifacts import read_json, sha256, source_hashes, write_json


def finalize():
    verify_protocol()
    summary = read_json(OUT / 'summary.json')
    replay = read_json(OUT / 'inference_verification.json')
    split_audit = read_json(OUT / 'split_verification.json')
    complete = sorted(p.parent.name for p in (OUT / 'models').glob('*/manifest.json')
                      if read_json(p)['status'] == 'complete')
    if summary['incomplete'] or complete != sorted(r['run'] for r in replay['verified']):
        raise ValueError('Incomplete training or inference verification')
    if split_audit['status']!='verified' or complete!=sorted(r['run'] for r in split_audit['runs']):
        raise ValueError('Split audit does not cover every completed model')
    for entry in [*replay['verified'],*split_audit['runs']]:
        if sha256(OUT/'models'/entry['run']/'manifest.json')!=entry['manifest_sha256']:
            raise ValueError('Model manifest changed after verification')
    statuses = {}
    for repo in ['knowledgeable-helicopter', 'knowledgeable-helicopter-screening']:
        root = WORKSPACE / repo
        # Only these known repositories are scanned; no raw directory traversal.
        forbidden = [str(p) for p in root.rglob('*') if p.is_file()
                     and p.suffix.lower() in {'.parquet', '.cbm', '.pkl', '.npy', '.npz', '.feather'}]
        if forbidden:
            raise ValueError(f'Private-looking artifacts inside checkout: {forbidden}')
        statuses[repo] = subprocess.run(['git', 'status', '--short'], cwd=root, text=True,
                                        capture_output=True, check=True).stdout
    if source_hashes() != read_json(OLD / 'protocol.json')['source_hashes']:
        raise ValueError('Frozen reference changed')
    files = [Path(__file__).with_name(name) for name in [
        'next230_common.py', 'next230_features.py', 'next230_prepare.py', 'next230_train.py',
        'test_next230.py', 'summarize_next230.py', 'verify_next230_inference.py',
        'compose_next230.py', 'decide_next230.py', 'build_next230_report.py', 'finalize_next230.py',
        'next230_clock_cpu.py', 'next230_schedule_mixture.py', 'next230_calibrate_mixture.py',
        'test_next230_calibration.py', 'next230_gpu_queue.py', 'next230_mixture_seeds.py',
        'next230_validation_queue.py', 'verify_next230_splits.py', 'next230_clock_gate.py',
        'test_next230_clock_gate.py', 'next230_clock_queue.py', 'next230_residual_seeds.py',
        'next230_seed_queue.py', 'next230_seed_sensitivity.py']]
    source_digests = {sha256(p) for p in files}
    for manifest in (OUT/'models').glob('*/manifest.json'):
        run = read_json(manifest)
        if 'runner_sha256' in run and run['runner_sha256'] not in source_digests:
            raise ValueError('Supplemental training runner changed after fit')
    for expert in (OUT/'clock_cpu_experts').glob('*/expert.json'):
        record = read_json(expert)
        if record['runner_sha256'] not in source_digests:
            raise ValueError('Direct expert source changed')
        for file, digest in record['outputs'].items():
            if sha256(expert.parent/file) != digest:
                raise ValueError('Direct expert artifact changed')
    snapshot = OUT / 'next_batch_sources.zip'
    with zipfile.ZipFile(snapshot, 'w', zipfile.ZIP_DEFLATED) as bundle:
        for path in files:
            bundle.write(path, path.name)
        bundle.write(OUT / 'PLAN.md', 'PLAN.md')
    pdf = WORKSPACE / 'output/PRC_2026_Next_Batch_Results.pdf'
    result = {'status': 'verified', 'complete_model_containers': len(complete),
              'replayed_pipelines': replay['count'], 'raw_files_unchanged': 14,
              'split_audited_pipelines':split_audit['count'],
              'frozen_source_files_unchanged': len(source_hashes()), 'private_artifacts_in_checkouts': [],
              'git_status': statuses, 'source_snapshot_sha256': sha256(snapshot),
              'sources': {p.name: sha256(p) for p in files}, 'pdf_sha256': sha256(pdf),
              'evidence_sha256':{name:sha256(OUT/name) for name in [
                  'summary.json','decision.json','scores.csv','report_notes.json',
                  'split_verification.json','inference_verification.json']},
              'no_upload_or_submission': True, 'baseline_source_snapshot': str(OLD / 'screening_source_snapshot.zip')}
    write_json(OUT / 'final_verification.json', result)
    print(result)


if __name__ == '__main__':
    finalize()
