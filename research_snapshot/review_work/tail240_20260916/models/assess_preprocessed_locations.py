"""Describe frozen preprocessing results without fitting or selecting a variant."""
import preprocessed_locations_tune as trial
import argparse
import numpy as np
import pandas as pd


def run(fold):
    common, root, ident = trial.common, trial.ROOT, trial.ID
    protocol = trial.declare()
    source = trial.OUT / fold
    marker = common.read_json(source / 'manifest.json')
    assert marker['status'] == 'complete'
    assert marker['protocol_sha256'] == common.sha256(trial.OUT / 'protocol.json')
    for name, digest in marker['outputs'].items():
        assert common.sha256(source / name) == digest
    dest = common.external_path(trial.OUT / f'{fold}_diagnostics')
    dest.mkdir(exist_ok=False)
    own = pd.read_parquet(source / 'tune.parquet')
    ids = pd.Index(own[ident])
    assert ids.is_unique
    meta_path = root / 'private_runs/screening_230/data/interim/audit/departures.parquet'
    assert common.sha256(meta_path) == common.read_json(root / 'private_runs/screening_230/reports/data_audit.json')['artifacts'][meta_path.name]
    meta = pd.read_parquet(meta_path, columns=[ident, trial.TARGET, common.MOVEMENT, 'proxy_sec', 'ADEP_mvt']).set_index(ident).loc[ids].reset_index()
    tuples = [(item['path'], item['sha256'], False, item['columns']) for item in marker['feature_receipts']]
    for path, digest, _, names in tuples:
        if 'STAND_mvt' in names:
            assert common.sha256(path) == digest
    tokens = trial.raw_location_tokens(ids, tuples)
    affected = ~trial.semantic.known_location(tokens['STAND_mvt'], encoded=True)
    affected |= ~trial.semantic.known_location(tokens['RUNWAY_mvt'], encoded=True)
    ordinary, current = trial.comparison.baseline(fold, meta, protocol)
    prior = trial.risk.union_folder(fold)
    old_marker = common.read_json(prior / 'manifest.json')
    assert common.sha256(prior / 'manifest.json') == protocol['controls'][fold]
    assert common.sha256(prior / 'tune_predictions.parquet') == old_marker['outputs']['tune_predictions.parquet']
    old = pd.read_parquet(prior / 'tune_predictions.parquet')
    np.testing.assert_array_equal(old[ident], ids)
    candidate = current + marker['coefficient'] * (own.prediction_sec.to_numpy()[ordinary]-old.prediction_sec.to_numpy()[ordinary])
    y = meta[trial.TARGET].to_numpy(float)[ordinary]
    impact = affected[ordinary]
    delta = (y-current)**2-(y-candidate)**2
    slices = {}
    for name, selected in [('all', np.ones(len(y), bool)), ('masked_query', impact), ('unmasked_query', ~impact)]:
        count = int(selected.sum())
        slices[name] = dict(rows=count, reference_rmse=float(np.sqrt(np.mean((y[selected]-current[selected])**2))) if count else None,
            candidate_rmse=float(np.sqrt(np.mean((y[selected]-candidate[selected])**2))) if count else None,
            removed_sse=float(delta[selected].sum()), prediction_changed_rows=int(np.count_nonzero(candidate[selected] != current[selected])))
    order = np.argsort(-delta, kind='stable')
    removals = {}
    for count in [1, 5, 10]:
        keep = np.ones(len(y), bool)
        keep[order[:count]] = False
        removals[str(count)] = float(np.sqrt(np.mean((y[keep]-current[keep])**2))-np.sqrt(np.mean((y[keep]-candidate[keep])**2)))
    record = dict(status='complete', source_sha256=common.sha256(__file__), fold=fold,
        producer_manifest_sha256=common.sha256(source / 'manifest.json'), slices=slices,
        remove_best_rows_gain_sec=removals, primary=marker['results']['primary'],
        limitation='Diagnostic decomposition only. No fitting, alternative gate, score/ranking data or change to declared qualification. Retraining can change predictions on unmasked queries too.')
    common.write_json(dest / 'receipt.json', record)
    print(record, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', choices=['F1', 'F3'], required=True)
    run(parser.parse_args().fold)
