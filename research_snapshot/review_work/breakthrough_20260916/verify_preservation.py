"""Read-only preservation check with a new external receipt per invocation."""
import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'campaign_20260916'))
import common


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    destination = common.external_path(args.output)
    assert not destination.exists()
    original = common.read_json(common.WORKSPACE / 'private_runs/submission_v2/protocol.json')
    raw = {path.name: common.sha256(path) for path in common.RAW.glob('*.parquet')}
    assert raw == original['raw_hashes'] and len(raw) == 14
    assert common.source_hashes() == original['source_hashes']
    prior_path = common.WORKSPACE / 'private_runs/campaign_20260916/validation/existing_evidence_audit.json'
    prior = common.read_json(prior_path)
    for name, digest in prior['frozen_external_runner_hashes'].items():
        assert common.sha256(common.WORKSPACE / 'review_work' / name) == digest
    for name, digest in prior['preserved_evidence_hashes'].items():
        assert common.sha256(common.WORKSPACE / 'private_runs/next_230' / name) == digest
    for record in prior['selected_reference'].values():
        assert common.sha256(Path(record['path']) / 'manifest.json') == record['manifest_sha256']
        assert common.sha256(Path(record['path']) / 'score_predictions.parquet') == record['prediction_sha256']
    checkouts = {}
    for name in ['knowledgeable-helicopter', 'knowledgeable-helicopter-screening']:
        path = common.WORKSPACE / name
        head = subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
        assert head == '9e5b9e6e71e3b548d9bf2d0aad17821c20b0510f'
        checkouts[name] = {'head': head, 'status': subprocess.check_output(
            ['git', '-C', str(path), 'status', '--porcelain=v1', '--untracked-files=all'], text=True)}
    roots = [common.WORKSPACE / prefix / 'breakthrough_20260916' for prefix in ['review_work', 'private_runs', 'output']]
    for root in roots:
        common.external_path(root)
    destination.mkdir(parents=True)
    receipt = {'status': 'passed', 'created_utc': common.utc_now(), 'verifier_sha256': common.sha256(__file__),
               'raw_hashes': raw, 'frozen_screening_source_count': len(original['source_hashes']),
               'frozen_screening_sources_unchanged': True,
               'existing_evidence_audit_sha256': common.sha256(prior_path),
               'prior_external_runners_unchanged': list(prior['frozen_external_runner_hashes']),
               'prior_evidence_unchanged': list(prior['preserved_evidence_hashes']),
               'selected_reference_folds_unchanged': list(prior['selected_reference']),
               'checkouts': checkouts, 'campaign_roots_external_to_git': [str(root) for root in roots],
               'scope': 'Input/source/reference preservation; model quality and individual new artifact integrity require separate run and integration receipts.'}
    common.write_json(destination / 'receipt.json', receipt)
    print('PRESERVATION_PASSED', len(raw), 'raw files;', len(original['source_hashes']), 'screening sources', flush=True)


if __name__ == '__main__':
    main()
