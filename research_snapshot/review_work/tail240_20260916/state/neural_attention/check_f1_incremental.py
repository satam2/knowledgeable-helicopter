"""Evaluate predeclared incremental formula on completed F1; no missing-fold inference."""
import torch
import lightgbm
import numpy as np
import pandas as pd
import assess_incremental as predeclared

subject, common = predeclared.subject, predeclared.common
OUT = common.external_path(subject.OUT.parent / 'f1_incremental_check_v1')


def main():
    protocol = predeclared.declare()
    OUT.mkdir(exist_ok=False)
    folder = subject.OUT / 'F1'
    manifest = common.read_json(folder / 'manifest.json')
    assert manifest['status'] == 'complete'
    assert manifest['protocol_sha256'] == protocol['producer_protocol_sha256']
    for name, digest in manifest['outputs'].items():
        assert common.sha256(folder / name) == digest
    binding = protocol['controls']['F1']
    for name, field in [('F1_weights.json', 'global9_weights_sha256'), ('F1_aligned_tune.parquet', 'global9_tune_sha256')]:
        assert common.sha256(subject.ENSEMBLE / name) == binding[field]
    oldpath = subject.CONTROLS['F1'] / 'tune_predictions.parquet'
    assert common.sha256(oldpath) == binding['tune_predictions_sha256']
    aligned = pd.read_parquet(subject.ENSEMBLE / 'F1_aligned_tune.parquet').set_index(subject.ID)
    own = pd.read_parquet(folder / 'tune_predictions.parquet').set_index(subject.ID)
    assert set(own.loc[own.proxy_sec.between(0, 7200)].index) == set(aligned.index)
    own = own.loc[aligned.index]
    old = pd.read_parquet(oldpath).set_index(subject.ID).loc[aligned.index]
    np.testing.assert_array_equal(old.prediction_sec, own.control387_prediction_sec)
    np.testing.assert_array_equal(old[subject.TARGET], aligned[subject.TARGET])
    np.testing.assert_array_equal(own[subject.TARGET], aligned[subject.TARGET])
    weights = common.read_json(subject.ENSEMBLE / 'F1_weights.json')
    global9 = aligned[weights['experts']].to_numpy() @ np.asarray(weights['global'])
    weight = weights['global'][weights['experts'].index('tabm_ple8')]
    current387 = global9 + weight * (old.prediction_sec.to_numpy() - aligned.tabm_ple8.to_numpy())
    candidate = global9 + weight * (own.prediction_sec.to_numpy() - aligned.tabm_ple8.to_numpy())
    metrics = subject.context.paired(aligned[subject.TARGET].to_numpy(), candidate, current387,
        aligned[subject.TIME].dt.floor('D').to_numpy())
    report = dict(status='complete', source_sha256=common.sha256(__file__),
        predeclared_incremental_protocol_sha256=common.sha256(predeclared.OUT / 'protocol.json'),
        producer_manifest_sha256=common.sha256(folder / 'manifest.json'), fold='F1', ordinary_rows=len(aligned),
        ordinary_id_hash=common.object_hash(aligned.index.tolist()), raw_label_hash=common.object_hash(aligned[subject.TARGET].tolist()),
        frozen_ple_weight=weight, metrics=metrics,
        decision='Incremental current387 qualification fails F1; F3 held by parent, no seasonal estimate or score stage. Original old225 twofoldgate remains incomplete, not falsely reported as evaluated.')
    common.write_json(OUT / 'receipt.json', report)
    print({k:v for k,v in metrics.items() if k not in ['day_removals']}, flush=True)


if __name__ == '__main__':
    main()
