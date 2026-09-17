"""Check current raw ranking observations and submitted predictions, never truth."""
import audit_v1 as a
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def main():
    out = a.common.external_path(a.ROOT / 'private_runs/tail240_20260916/score_gap/drift/observed_verification_v1')
    out.mkdir(parents=True, exist_ok=False)
    columns = [a.ID, 'PHASE_mvt', a.common.MOVEMENT, 'ADEP_mvt', 'AOBT_3_flt']
    raw = pq.read_table(a.ROOT / 'data/09-15-2026-18-55-03_files_list/ranking.parquet',
                        columns=columns, filters=[('PHASE_mvt', '=', 'DEP')], use_threads=False).to_pandas()
    saved = pd.read_parquet(a.OLD / 'ranking_predictions.parquet')
    assert raw[a.ID].is_unique and saved[a.ID].is_unique
    raw = raw.set_index(a.ID).loc[saved[a.ID]].reset_index()
    np.testing.assert_array_equal(raw[a.ID], saved[a.ID])
    np.testing.assert_array_equal(raw.ADEP_mvt.astype(str), saved.ADEP_mvt.astype(str))
    np.testing.assert_array_equal(raw[a.common.MOVEMENT], saved[a.common.MOVEMENT])
    proxy = (raw[a.common.MOVEMENT] - raw.AOBT_3_flt).dt.total_seconds()
    np.testing.assert_array_equal(proxy, saved.proxy_sec)
    prior = a.common.read_json(a.PRIOR)
    raw['month'] = raw[a.common.MOVEMENT].dt.strftime('%Y-%m')
    raw['route'] = a.route(proxy)
    checked = 0
    for month, group in raw.groupby('month'):
        p = prior['records'][month]
        assert len(group) == p['rows']
        for airport, part in group.groupby('ADEP_mvt', observed=True):
            expected = p['per_airport'][airport]
            counts = part.route.value_counts().to_dict()
            assert len(part) == expected['rows']
            assert counts.get('missing', 0) == expected['status'].get('missing', 0)
            assert counts.get('ordinary', 0) == expected['status'].get('ordinary', 0)
            assert counts.get('finite_nonordinary', 0) == expected['status'].get('negative', 0) + expected['status'].get('gt7200', 0)
            checked += 1
    submitted = pd.read_parquet(a.NEW / 'knowledgeable-helicopter_v3.parquet')
    pred = pd.read_parquet(a.NEW / 'ranking_predictions.parquet')
    pred = pred.set_index(a.ID).loc[submitted[a.ID]].reset_index()
    # TAXITIME in this artifact is our uploaded prediction, not ranking ground truth.
    np.testing.assert_array_equal(np.rint(pred.prediction_sec).astype(np.int32), submitted[a.TARGET])
    receipt = dict(status='passed', source_sha256=a.common.sha256(__file__),
        shared_helper_sha256=a.common.sha256(a.__file__), raw_columns_read=columns,
        rows=len(raw), airport_month_groups=checked, raw_observation_to_saved_metadata_exact=True,
        prior_route_count_replay_exact=True, rounded_submission_prediction_identity_exact=True,
        submitted_prediction_sha256=a.common.sha256(a.NEW / 'knowledgeable-helicopter_v3.parquet'),
        ranking_truth_read=False)
    a.common.write_json(out / 'receipt.json', receipt)
    print(receipt, flush=True)


if __name__ == '__main__':
    main()
