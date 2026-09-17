"""Fixed-cutoff airport/hour label states; no query-month teacher forcing."""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '2'
from pathlib import Path
import sys
import hashlib
import json
from datetime import datetime, timezone
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'knowledgeable-helicopter-screening/src'))
from taxiout.paths import external_path

ID = 'MVT_ID_mvt'
FID = 'FLIGHT_ID_mvt'
TIME = 'MVT_TIME_UTC_mvt'
AIRPORT = 'ADEP_mvt'
TARGET = 'TAXITIME_SEC_mvt'
SHRINKAGE = 100.0
RECENT_DAYS = 28
HALFLIFE_DAYS = 28.0
SUFFIXES = ['seasonal_mean_sec', 'seasonal_count', 'recent_shift_sec', 'dynamic_mean_sec', 'recent_count', 'history_age_days']
COLUMNS = [f'state_{block}_{suffix}' for block in ('taxi', 'source') for suffix in SUFFIXES]


def guard():
    import psutil
    if psutil.Process().memory_info().rss > 6 * 1024**3 or psutil.virtual_memory().available < 8 * 1024**3:
        raise MemoryError('State features require <=6GiB RSS and >=8GiB available host memory')


def metadata(frame):
    times = pd.to_datetime(frame[TIME], utc=True)
    if times.isna().any() or frame[ID].isna().any() or not frame[ID].is_unique:
        raise ValueError('Unique nonnull movement IDs and times required')
    return pd.DataFrame({'airport': frame[AIRPORT].astype('string').fillna('__MISSING__').to_numpy(),
                         'time': times.to_numpy(), 'how': (times.dt.dayofweek * 24 + times.dt.hour).to_numpy()}, index=frame.index)


class AirportState:
    def __init__(self, block):
        if block not in ('taxi', 'source'):
            raise ValueError(block)
        self.block = block
        self.fallback = 900.0 if block == 'taxi' else 0.0

    def fit(self, history, values):
        data = metadata(history)
        values = np.asarray(values, dtype=float)
        if len(values) != len(data) or not np.isfinite(values).all():
            raise ValueError('History labels must be aligned, raw and finite')
        data['value'] = values
        self.last = data.time.max() if len(data) else None
        self.mean = float(values.mean()) if len(values) else self.fallback
        self.airport = data.groupby('airport', observed=True).agg(mean=('value', 'mean'), last=('time', 'max'))
        self.seasonal = data.groupby(['airport', 'how'], observed=True).agg(mean=('value', 'mean'), n=('value', 'size'))
        # A fixed recent window ends at the permitted history cutoff, not each query.
        recent = data if self.last is None else data.loc[data.time > self.last - pd.Timedelta(days=RECENT_DAYS)]
        self.recent = recent.groupby('airport', observed=True).agg(mean=('value', 'mean'), n=('value', 'size'))
        return self

    def transform(self, queries):
        data = metadata(queries)
        if self.last is not None and len(data) and data.time.min() <= self.last:
            raise ValueError('Every query must strictly follow every permitted history label')
        airport = self.airport.reindex(data.airport.to_numpy())
        seasonal = self.seasonal.reindex(pd.MultiIndex.from_arrays([data.airport, data.how]))
        recent = self.recent.reindex(data.airport.to_numpy())
        prior = airport['mean'].fillna(self.mean).to_numpy(float)
        n = seasonal.n.fillna(0).to_numpy(float)
        means = seasonal['mean'].fillna(pd.Series(prior, index=seasonal.index)).to_numpy(float)
        shrunk = (n * means + SHRINKAGE * prior) / (n + SHRINKAGE)
        ages = (pd.DatetimeIndex(data.time) - pd.DatetimeIndex(pd.to_datetime(airport['last'], utc=True))).total_seconds().to_numpy() / 86400
        shift = recent['mean'].to_numpy(float) - prior
        shift = np.where(np.isfinite(shift), shift, 0.0)
        decay = np.where(np.isfinite(ages), np.exp2(-ages / HALFLIFE_DAYS), 0.0)
        shift *= decay
        values = np.column_stack([shrunk, n, shift, shrunk + shift, recent.n.fillna(0).to_numpy(float), ages])
        return pd.DataFrame(values.astype('float32'), columns=[f'state_{self.block}_{s}' for s in SUFFIXES], index=queries.index)


def stage_features(meta, indices, stage, history_blackout_days=0, query_positions=None):
    """indices must come from original make_fold BEFORE route filtering.

    Returns full-stage features (or an explicit query subset), preserving original
    positional indices. History remains all permitted flights, even for missing-only queries.
    Call separately for fit/tune/refit/score. Blackout is an additional age restriction.
    """
    if stage not in ('fit', 'tune', 'refit', 'score') or history_blackout_days < 0:
        raise ValueError('Invalid stage or history blackout')
    guard()
    positions = np.asarray(indices[stage], dtype=int)
    if query_positions is not None:
        requested = np.asarray(query_positions, dtype=int)
        if not np.isin(requested, positions).all():
            raise ValueError('Query subset outside original stage')
        positions = requested
    if len(np.unique(positions)) != len(positions):
        raise ValueError('Duplicate query positions')
    query = meta.iloc[positions][[ID, FID, TIME, AIRPORT]].copy()
    query.index = positions
    times = pd.to_datetime(meta[TIME], utc=True)
    query_months = pd.to_datetime(query[TIME], utc=True).dt.strftime('%Y-%m')
    pool = np.asarray(indices[stage if stage in ('fit', 'refit') else {'tune': 'fit', 'score': 'refit'}[stage]], dtype=int)
    result, receipts = [], []
    for month in sorted(query_months.unique()):
        current = query.loc[query_months.eq(month)]
        cutoff = pd.Timestamp(month + '-01', tz='UTC') - pd.Timedelta(days=history_blackout_days)
        history_positions = pool[times.iloc[pool].lt(cutoff).to_numpy()]
        # Reapply each whole query-month flight purge before any label aggregation.
        stage_month = times.iloc[np.asarray(indices[stage], dtype=int)].dt.strftime('%Y-%m').eq(month).to_numpy()
        month_positions = np.asarray(indices[stage], dtype=int)[stage_month]
        query_flights = meta.iloc[month_positions][FID].dropna().unique()
        permitted = ~meta.iloc[history_positions][FID].isin(query_flights).to_numpy()
        history_positions = history_positions[permitted]
        history = meta.iloc[history_positions][[ID, FID, TIME, AIRPORT, TARGET, 'proxy_sec']].copy()
        y = history[TARGET].to_numpy(float)
        proxy = history.proxy_sec.to_numpy(float)
        if not np.isfinite(y).all():
            raise ValueError('Nonfinite historical raw target')
        blocks = [AirportState('taxi').fit(history, y).transform(current)]
        finite = np.isfinite(proxy)
        blocks.append(AirportState('source').fit(history.loc[finite], y[finite] - proxy[finite]).transform(current))
        result.append(pd.concat(blocks, axis=1))
        receipts.append({'query_month': month, 'cutoff_utc': cutoff.isoformat(), 'queries': len(current),
                         'taxi_history_rows': len(history), 'source_history_rows': int(finite.sum()),
                         'history_ID_sha256': hashlib.sha256(json.dumps(history[ID].tolist()).encode()).hexdigest(),
                         'history_latest_utc': str(pd.to_datetime(history[TIME], utc=True).max()) if len(history) else None})
        guard()
    output = pd.concat(result).loc[positions] if result else pd.DataFrame(index=positions, columns=COLUMNS, dtype='float32')
    if list(output.columns) != COLUMNS:
        raise AssertionError('Feature schema mismatch')
    return output, {'stage': stage, 'history_blackout_days': history_blackout_days, 'months': receipts,
                    'columns': COLUMNS, 'scope': 'Original stage indices supplied before route filtering; no query target or proxy read'}


def prepare():
    out = external_path(ROOT / 'private_runs/tail240_20260916/state')
    out.mkdir(parents=True, exist_ok=True)
    protocol = {'created_utc': datetime.now(timezone.utc).isoformat(), 'status': 'prepared_no_private_run',
                'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'blocks': {'taxi': 'All permitted historical raw Y; empty-history default900',
                           'source': 'All permitted finite-NM rawY-minus-proxy=N-B; empty-history default0'},
                'parameters': {'seasonal_key': 'airport by UTC hour-of-week', 'shrinkage_rows': SHRINKAGE,
                               'recent_days': RECENT_DAYS, 'decay_half_life_days': HALFLIFE_DAYS},
                'purge': 'Original make_fold all-departure indices before aggregation and route filtering; extra query-month nonnullflight purge before aggregation',
                'crossfit': 'Fit/refit wholeUTCmonth queries use only earliermonth permitted labels; tune fromfit; score fromrefit',
                'ranking': 'January2026 historythroughDec2025; July2026 samehistorythroughDec2025 withactualage/decay. Never readrankingDEPtargets or teacherforce.',
                'blackout_stress': 'Explicit history_blackout_days=182 arm; preserve true dates; not prior-year timestamp shifting',
                'novelty': 'All-DEP airport/hour seasonality plus decaying recentlevel; HistoricalTemplate usesmissing-only airport/schedule/flight/stand hierarchy andlastlabels',
                'comparison': 'Identical observedcovariates/downstreammodel: baseline vs taxi6 vs source6 vs both12. Parent owns scoring; no newdata/model promotion implied.',
                'resources': '2CPU threads, sampled6GiBRSS/8GiBhostreserve; noGPU/network/model fitting', 'columns': COLUMNS}
    with (out / 'protocol.json').open('x', encoding='utf-8') as stream:
        json.dump(protocol, stream, indent=2)
    print(json.dumps(protocol, indent=2))


if __name__ == '__main__':
    prepare()
