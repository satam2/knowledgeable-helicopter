"""Export one predeclared missing-route variant for independent evaluation."""
import argparse
from pathlib import Path
import sys
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'review_work/campaign_20260916'))
import common
from taxiout.schema import ID


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--source-protocol', type=Path, required=True)
    parser.add_argument('--control-root', type=Path, required=True)
    parser.add_argument('--variant', choices=['candidate', 'blend25'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = common.external_path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    binding_path = ROOT / 'private_runs/tail240_20260916/validation/baseline_binding.json'
    binding = common.read_json(binding_path)
    source_protocol = common.read_json(args.source_protocol)
    protocol = {
        'created_utc': common.utc_now(), 'name': args.source_root.as_posix() + '/' + args.variant,
        'baseline_binding_sha256': common.sha256(binding_path),
        'score_labels_used_for_selection': False, 'routing_uses_score_targets': False,
        'selection_periods': ['fit', 'tune'], 'prediction_transform': 'raw_unclipped',
        'tail_definition': 'AllmissingNM originalroute, no score-tail selection',
        'routing_definition': 'OnlymissingNM predictions can change; other routes frozen completeglobal9',
        'selection_rule': 'One predeclared source variant, exported without selecting or fitting on score labels. Source protocol is authoritative.',
        'inference_columns': source_protocol.get('inference_columns', ['observed_airport_fields', 'declared_context_fields', 'earlier_label_priors']),
        'source_protocol_path': str(args.source_protocol.resolve()),
        'source_protocol_sha256': common.sha256(args.source_protocol),
    }
    common.write_json(out / 'protocol.json', protocol)
    folds, hashes = {}, {str(Path(__file__).relative_to(ROOT)): common.sha256(__file__)}
    for fold in ('F1', 'F3'):
        folder = args.source_root / fold
        marker_path = folder / 'manifest.json'
        marker = common.read_json(marker_path)
        assert marker['status'] == 'complete'
        assert marker['protocol_sha256'] == protocol['source_protocol_sha256']
        actual_ids = marker['fit_ids']
        expected_ids = binding['folds'][fold]['cohorts']['missing_nm']
        assert actual_ids == expected_ids
        assert marker['split']['split_hash'] == binding['folds'][fold]['split_hash']
        filename = args.variant + '.parquet'
        assert common.sha256(folder / filename) == marker['outputs'][filename]
        frame = pd.read_parquet(folder / filename, columns=[ID, 'prediction_sec'])
        assert np.isfinite(frame.prediction_sec).all()
        prediction_name = fold + '.parquet'
        frame.to_parquet(out / prediction_name, index=False)
        control_marker = common.read_json(args.control_root / fold / 'manifest.json')
        control_path = args.control_root / fold / filename
        assert common.sha256(control_path) == control_marker['outputs'][filename]
        control = pd.read_parquet(control_path, columns=[ID, 'prediction_sec'])
        np.testing.assert_array_equal(frame[ID], control[ID])
        control_name = fold + '_control.parquet'
        control.to_parquet(out / control_name, index=False)
        folds[fold] = {'prediction': {'path': prediction_name, 'sha256': common.sha256(out / prediction_name)},
            'split_hash': binding['folds'][fold]['split_hash'], 'training_scope': 'missing_nm',
            'cohorts': actual_ids, 'prediction_scope': 'missing_nm',
            'matched_control': {'path': control_name, 'sha256': common.sha256(out / control_name)},
            'original_manifest_path': str(marker_path.resolve()), 'original_manifest_sha256': common.sha256(marker_path)}
        hashes.update(marker['source_hashes'])
    common.write_json(out / 'manifest.json', {'status': 'complete', 'created_utc': common.utc_now(),
        'protocol_sha256': common.sha256(out / 'protocol.json'), 'folds': folds, 'source_hashes': hashes,
        'scope': 'Exchange conversion only; independent source/model/cache replay and leakage review required separately.'})
    print('EXPORTED', out, flush=True)


if __name__ == '__main__':
    main()
