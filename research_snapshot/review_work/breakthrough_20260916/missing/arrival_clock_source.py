"""Strict-prior ARR airport-vs-NM landing-clock source quality, separate from taxi-in."""

import os
for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[name] = '1'
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import psutil

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'review_work/campaign_20260916'))
import common
from aviation.arrival_features import _window_stats, _query, _token, _month, _ns
from taxiout.artifacts import object_hash, read_json, sha256, utc_now, write_json
from taxiout.availability import make_observations
from taxiout.config import load_config
from taxiout.io import concat_frames
from taxiout.paths import external_path
from taxiout.schema import ID, FLIGHT_ID, MOVEMENT, PHASE, TARGET, utc
from taxiout.splits import make_fold

pa.set_cpu_count(1)
pa.set_io_thread_count(1)
OUT = external_path(ROOT / 'private_runs/breakthrough_20260916/missing/arrival_clock_source')
RAW = external_path(ROOT / 'data/09-15-2026-18-55-03_files_list')


def correlations(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    valid = np.isfinite(a) & np.isfinite(b)
    a, b = a[valid], b[valid]
    return {'n': len(a), 'pearson': float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 and np.std(a) and np.std(b) else None}


def clock_features(dep, events):
    query = _query(dep)
    known = events.loc[np.isfinite(events.value)].copy()
    result = {}
    for label, keys, width in [('airport15', ['airport', 'month'], 15), ('airport60', ['airport', 'month'], 60),
                               ('runway15', ['airport', 'runway', 'month'], 15), ('stand60', ['airport', 'stand', 'month'], 60)]:
        stats = _window_stats(known, query, keys, width)
        n = stats[:, 0]
        mean = np.divide(stats[:, 1], n, out=np.full(len(dep), np.nan), where=n > 0)
        square = np.divide(stats[:, 2], n, out=np.full(len(dep), np.nan), where=n > 0)
        result[f'arrclk_{label}_known_n'] = n
        result[f'arrclk_{label}_mean_sec'] = mean
        result[f'arrclk_{label}_std_sec'] = np.sqrt(np.maximum(0, square - mean ** 2))
        all_events = events.copy(deep=False).assign(value=0.)
        total = _window_stats(all_events, query, keys, width)[:, 0]
        result[f'arrclk_{label}_missing_share'] = np.divide(total - n, total, out=np.full(len(dep), np.nan), where=total > 0)
        outliers = known.copy(deep=False).assign(value=known.value.abs().gt(60).astype(float))
        count = _window_stats(outliers, query, keys, width)[:, 1]
        result[f'arrclk_{label}_abs_over60_share'] = np.divide(count, n, out=np.full(len(dep), np.nan), where=n > 0)
    return pd.DataFrame({key: np.asarray(value, np.float32) for key, value in result.items()}, index=pd.Index(dep[ID], name=ID))


def synthetic_test():
    query = pd.DataFrame({ID: [1], FLIGHT_ID: [10], PHASE: ['DEP'], MOVEMENT: pd.to_datetime(['2025-01-02 12:00'], utc=True),
                          'ADEP_mvt': ['AAA'], 'STAND_mvt': ['A'], 'RUNWAY_mvt': ['01']})
    events = pd.DataFrame({'airport': ['AAA'] * 4, 'month': [202501] * 4, 'stand': ['A'] * 4, 'runway': ['01'] * 4,
        'flight': ['2', '3', '4', '10'], 'time': pd.to_datetime(['2025-01-02 11:55', '2025-01-02 12:00', '2025-01-02 12:05', '2025-01-02 11:56'], utc=True).as_unit('ns').astype('int64'),
        'value': [30., 999., 888., 777.]})
    first = clock_features(query, events)
    assert first.arrclk_airport15_mean_sec.iloc[0] == 30
    assert first.arrclk_airport15_known_n.iloc[0] == 1
    events.loc[2, 'value'] = -9999999.
    pd.testing.assert_frame_equal(first, clock_features(query, events))


def prior_label_smoothness(frame, keys, order):
    ordered = frame.loc[frame.gap.notna()].sort_values(order, kind='stable').copy()
    grouped = ordered.groupby(keys, observed=True, dropna=False)
    lag = grouped.gap.shift(1)
    age = (ordered[MOVEMENT] - grouped[MOVEMENT].shift(1)).dt.total_seconds()
    prior = (age > 0) & (age <= 900)
    center = grouped.gap.transform('mean')
    return {'all_neighbor': correlations(ordered.gap, lag),
            'strict_prior_15m': correlations(ordered.loc[prior, 'gap'], lag[prior]),
            'strict_prior_15m_group_demeaned': correlations((ordered.gap - center)[prior], (lag - center)[prior]),
            'caveat': 'Training-label diagnostic only, not a score feature; group demeaning is retrospective diagnosis.'}


def main():
    synthetic_test()
    OUT.mkdir(parents=True, exist_ok=True)
    began = time.monotonic()
    expected = read_json(ROOT / 'private_runs/submission_v2/protocol.json')['raw_hashes']
    paths = sorted(RAW.glob('training_*.parquet'))
    sources = {str(Path(__file__).relative_to(ROOT)): sha256(__file__),
        'review_work/campaign_20260916/aviation/arrival_features.py': sha256(ROOT / 'review_work/campaign_20260916/aviation/arrival_features.py')}
    write_json(OUT / 'protocol.json', {'created_utc': utc_now(), 'sources': sources,
        'quantity': 'Arrival MVT landing minus NM ARVT3, never NM AOBT and never arrival taxi-in duration.',
        'hypothesis': 'Shared airport-clock offset predicts NEGATIVE departure Y-proxy residual; compare demeaned within-airport correlation.',
        'availability': 'Arrival landing strictly before query, same UTC movement month, ties and sameflight excluded. Final ARVT3 publication time unverified.',
        'windows': ['airport15m', 'airport60m', 'runway15m', 'stand60m'], 'gpu': False, 'cpu_threads': 1})
    arrivals, departures = [], []
    for path in paths:
        assert sha256(path) == expected[path.name]
        fields = [ID, FLIGHT_ID, PHASE, MOVEMENT, TARGET, 'AOBT_3_flt', 'ARVT_3_flt',
                  'ADEP_mvt', 'ADES_mvt', 'STAND_mvt', 'RUNWAY_mvt', 'AIRCRAFT_OPERATOR_flt']
        raw = pq.read_table(path, columns=fields, use_threads=False).to_pandas(strings_to_categorical=True)
        arr = raw.loc[raw[PHASE].eq('ARR')]
        arrivals.append(pd.DataFrame({ID: arr[ID].to_numpy(), 'airport': _token(arr.ADES_mvt), 'stand': _token(arr.STAND_mvt),
            'runway': _token(arr.RUNWAY_mvt), 'month': _month(arr[MOVEMENT]), 'flight': _token(arr[FLIGHT_ID]),
            'time': _ns(arr[MOVEMENT]), 'value': (utc(arr[MOVEMENT]) - utc(arr.ARVT_3_flt)).dt.total_seconds().to_numpy()}))
        departures.append(raw.loc[raw[PHASE].eq('DEP')].drop(columns='ARVT_3_flt'))
    events = pd.concat(arrivals, ignore_index=True)
    events.loc[events.flight.eq('<missing>'), 'flight'] = 'missing-id:' + events.loc[events.flight.eq('<missing>'), ID].astype(str)
    events = events.sort_values(ID).drop_duplicates(['airport', 'flight', 'time'])
    data = concat_frames(departures)
    del arrivals, departures
    write_json(OUT / 'arrival_profile.json', {'arrivals': len(events), 'known_clock_n': int(events.value.notna().sum()),
        'exact_equal_n': int(events.value.eq(0).sum()), 'abs_delta_over60_n': int(events.value.abs().gt(60).sum()),
        'delta_quantiles': events.value.quantile([0, .01, .5, .99, 1]).to_dict()})
    feature_chunks = []
    month = _month(data[MOVEMENT])
    for value in sorted(set(month)):
        raw = data.loc[month == value]
        obs, _, _ = make_observations(raw)
        feature_chunks.append(clock_features(obs, events.loc[events.month.eq(value)]))
        print('Source-clock features', value, len(raw), flush=True)
    features = pd.concat(feature_chunks).loc[data[ID]]
    features.reset_index().to_parquet(OUT / 'features.parquet', index=False)
    data['gap'] = data[TARGET] - (utc(data[MOVEMENT]) - utc(data.AOBT_3_flt)).dt.total_seconds()
    data['day'] = pd.Categorical(utc(data[MOVEMENT]).dt.strftime('%Y-%m-%d'))
    idxmeta = data[[ID, FLIGHT_ID, MOVEMENT]]
    report = {'created_utc': utc_now(), 'sources': sources, 'folds': {}, 'synthetic_tests': 'passed'}
    for fold in ['F1', 'F3']:
        idx, split = make_fold(idxmeta, load_config('configs/folds.yaml')[fold])
        reference, reference_record = common.reference(fold)
        assert object_hash(split) == object_hash(reference_record['split'])
        assert np.array_equal(data.iloc[idx['score']][ID], reference[ID])
        f = {'fit_label_smoothness': {}, 'score_correlation': {}, 'fit_correlation': {}}
        fit = data.iloc[idx['fit']]
        for name, keys, sort in [('airport_day_time', ['ADEP_mvt', 'day'], [MOVEMENT]),
            ('airport_day_operator_time', ['ADEP_mvt', 'day', 'AIRCRAFT_OPERATOR_flt'], [MOVEMENT]),
            ('airport_day_stand_time', ['ADEP_mvt', 'day', 'STAND_mvt'], [MOVEMENT]),
            ('airport_day_runway_time', ['ADEP_mvt', 'day', 'RUNWAY_mvt'], [MOVEMENT]),
            ('airport_day_id_order', ['ADEP_mvt', 'day'], [ID])]:
            f['fit_label_smoothness'][name] = prior_label_smoothness(fit, keys, sort)
        for stage in ['fit', 'score']:
            selected = data.iloc[idx[stage]].copy()
            selected.index = selected[ID]
            gap = selected.gap
            centered_gap = gap - selected.groupby('ADEP_mvt', observed=True).gap.transform('mean')
            for name in ['airport15', 'airport60', 'runway15', 'stand60']:
                clock = features.loc[selected.index, f'arrclk_{name}_mean_sec']
                centered_clock = clock - clock.groupby(selected.ADEP_mvt, observed=True).transform('mean')
                values = {'raw': correlations(clock, gap), 'within_airport': correlations(centered_clock, centered_gap)}
                if stage == 'score':
                    error = reference.prediction_sec.to_numpy() - reference[TARGET].to_numpy()
                    values['reference_error'] = correlations(clock, error)
                    values['tail_enrichment'] = []
                    for threshold in [30, 60, 120, 300]:
                        use = clock.abs().gt(threshold).to_numpy() & features.loc[selected.index, f'arrclk_{name}_known_n'].ge(3).to_numpy()
                        values['tail_enrichment'].append({'mean_abs_threshold': threshold, 'n': int(use.sum()),
                            'large_departure_gap_n': int((np.abs(gap.to_numpy()[use]) > 1800).sum()),
                            'reference_sse_share': float(reference.loc[use, 'squared_error'].sum() / reference.squared_error.sum())})
                f[stage + '_correlation'][name] = values
        report['folds'][fold] = f
    report.update(runtime_sec=time.monotonic() - began, peak_rss_bytes=getattr(psutil.Process().memory_info(), 'peak_wset', psutil.Process().memory_info().rss),
        feature_sha256=sha256(OUT / 'features.parquet'), feature_rows=len(features), feature_columns=list(features),
        scope='Label-free causal-arrival clock cache plus retrospective supervised diagnostics. No fitted predictor or accuracy gain claimed.')
    write_json(OUT / 'analysis.json', report)
    print('DONE arrival clock source', report['runtime_sec'], report['peak_rss_bytes'], flush=True)


if __name__ == '__main__':
    main()
