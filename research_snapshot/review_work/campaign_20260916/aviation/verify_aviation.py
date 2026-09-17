"""Read-only independent brute-force checks against generated feature cache."""

import os
for variable in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[variable] = '1'

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'knowledgeable-helicopter-screening/src'))

from taxiout.artifacts import read_json, sha256, utc_now, write_json
from taxiout.paths import external_path
from taxiout.schema import BLOCK, FLIGHT_ID, ID, MOVEMENT, PHASE, TARGET, utc

pa.set_cpu_count(1)
pa.set_io_thread_count(1)


def main():
    out = external_path(ROOT / 'private_runs/campaign_20260916/aviation')
    raw_root = external_path(ROOT / 'data/09-15-2026-18-55-03_files_list')
    manifest = read_json(out / 'manifest.json')
    for name, digest in manifest['sources'].items():
        assert sha256(HERE / name) == digest, name
    assert sha256(out / 'features.parquet') == manifest['feature_sha256']
    feature = pd.read_parquet(out / 'features.parquet').set_index(ID)
    assert len(feature) == 2085047 and feature.index.is_unique
    assert np.isfinite(feature.to_numpy()).all()
    all_count = [column for column in feature if 'count_' in column or 'prior_' in column or column == 'surface_arrivals_open_120m']
    assert (feature[all_count].to_numpy() >= 0).all()
    checks = []
    # July and November contain complete screen cohorts. This verifier does not
    # call the feature implementation or its event-query helpers.
    for month in ['07', '11']:
        start = f'2025-{month}-01'
        filename = next(raw_root.glob(f'training_{start}_*.parquet'))
        raw = pq.read_table(filename, columns=[ID, FLIGHT_ID, PHASE, MOVEMENT, BLOCK, 'ADEP_mvt', 'ADES_mvt']).to_pandas()
        raw[MOVEMENT] = utc(raw[MOVEMENT], required=True)
        raw[BLOCK] = utc(raw[BLOCK])
        arrivals = raw.loc[raw[PHASE].eq('ARR')].sort_values(ID).copy()
        arrivals['flight'] = arrivals[FLIGHT_ID].astype('string').fillna('missing-id:' + arrivals[ID].astype(str))
        arrivals = arrivals.drop_duplicates(['flight', 'ADES_mvt', MOVEMENT])
        departures = raw.loc[raw[PHASE].eq('DEP')].sample(n=30, random_state=20260916)
        for _, row in departures.iterrows():
            query = row[MOVEMENT]
            local = arrivals.loc[arrivals.ADES_mvt.eq(row.ADEP_mvt) &
                                 arrivals[MOVEMENT].dt.strftime('%Y-%m').eq(query.strftime('%Y-%m')) &
                                 ~arrivals[FLIGHT_ID].eq(row[FLIGHT_ID])].copy()
            duration = (local[BLOCK] - local[MOVEMENT]).dt.total_seconds()
            completed = local[BLOCK].lt(query) & local[BLOCK].ge(query - pd.Timedelta(minutes=30)) & duration.ge(0)
            expected_count = int(completed.sum())
            expected_mean = duration[completed].mean() if expected_count else -999999.
            cached = feature.loc[row[ID]]
            assert cached.arr_airport_count_30m == expected_count
            assert np.isclose(cached.arr_airport_mean_sec_30m, expected_mean, atol=0.001)
            landed = local[MOVEMENT].lt(query) & local[MOVEMENT].ge(query - pd.Timedelta(minutes=120))
            already_finished = local[BLOCK].lt(query) & duration.ge(0)
            expected_open = int((landed & ~already_finished).sum())
            assert cached.surface_arrivals_open_120m == expected_open
        checks.append({'month': f'2025-{month}', 'brute_force_query_rows': len(departures),
                       'checks': ['completed_30m_count', 'completed_30m_mean', 'recent_arrival_inventory']})
    write_json(out / 'verification.json', {'status': 'passed', 'created_utc': utc_now(),
        'rows': len(feature), 'features': len(feature.columns), 'all_finite': True,
        'all_counts_nonnegative': True, 'brute_force': checks,
        'feature_sha256': manifest['feature_sha256'], 'verifier_sha256': sha256(__file__),
        'limitation': 'Manual real-data checks cover 60 deterministic July/November queries; synthetic tests cover adversarial temporal boundaries.'})
    print('PASS: source/cache hashes, complete IDs, finite/count checks, 60 independent real-data brute-force queries')


if __name__ == '__main__':
    main()
