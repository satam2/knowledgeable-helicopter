"""Label-free exact flight-ID arrival endpoint discrepancy with fixed ambiguity rules."""
import numpy as np
import pandas as pd

ID = 'MVT_ID_mvt'
FLIGHT = 'FLIGHT_ID_mvt'
TIME = 'MVT_TIME_UTC_mvt'
PHASE = 'PHASE_mvt'
ORIGIN = 'ADEP_mvt'
DEST = 'ADES_mvt'
ARVT = 'ARVT_3_flt'
OPERATOR = 'AIRCRAFT_OPERATOR_flt'
ARR_COLUMNS = [ID, FLIGHT, TIME, PHASE, ORIGIN, DEST]
DEP_COLUMNS = ARR_COLUMNS + [ARVT, OPERATOR]


def deduplicate(frame, phase):
    columns = DEP_COLUMNS if phase == 'DEP' else ARR_COLUMNS
    result = frame[columns].copy()
    result[FLIGHT] = result[FLIGHT].astype('Int64')
    if not result[PHASE].eq(phase).all():
        raise ValueError('Phase filter violated')
    if result[ID].isna().any():
        raise ValueError('Missing movement identity')
    result[TIME] = pd.to_datetime(result[TIME], utc=True)
    if phase == 'DEP':
        result[ARVT] = pd.to_datetime(result[ARVT], utc=True)
    result = result.drop_duplicates()
    if result[ID].duplicated().any():
        raise ValueError('Conflicting duplicate movement identity')
    if phase == 'ARR':
        known = result[FLIGHT].notna()
        good = result.loc[known].sort_values(ID, kind='stable').drop_duplicates([FLIGHT, ORIGIN, DEST, TIME], keep='first')
        result = pd.concat([good, result.loc[~known]], ignore_index=True)
    return result


def link(departures, arrivals):
    dep = deduplicate(departures, 'DEP')
    arr = deduplicate(arrivals, 'ARR')
    dep['month'] = dep[TIME].dt.strftime('%Y-%m')
    arr['month'] = arr[TIME].dt.strftime('%Y-%m')
    known = arr[FLIGHT].notna() & arr['month'].notna()
    candidates = arr.loc[known]
    counts = candidates.groupby(['month', FLIGHT], observed=True).size()
    unique = candidates.loc[~candidates.duplicated(['month', FLIGHT], keep=False)]
    renamed = unique.rename(columns={name:'arrival_' + name for name in ARR_COLUMNS if name != FLIGHT})
    merged = dep.merge(renamed, on=['month', FLIGHT], how='left', validate='many_to_one', sort=False)
    pairs = pd.MultiIndex.from_arrays([merged['month'], merged[FLIGHT]])
    support = counts.reindex(pairs).fillna(0).to_numpy(int)
    route_fields = [ORIGIN, DEST, 'arrival_' + ORIGIN, 'arrival_' + DEST]
    routes_known = np.ones(len(merged), bool)
    for name in route_fields:
        routes_known &= merged[name].astype('string').str.strip().ne('').fillna(False).to_numpy(bool)
    route = routes_known & merged[ORIGIN].eq(merged['arrival_' + ORIGIN]) & merged[DEST].eq(merged['arrival_' + DEST])
    later = merged['arrival_' + TIME].gt(merged[TIME])
    self_match = merged['arrival_' + ID].eq(merged[ID])
    available = (support == 1) & route & later & ~self_match & merged[ARVT].notna()
    status = np.select([merged[FLIGHT].isna(), support == 0, support > 1, ~route, ~later,
                        self_match, merged[ARVT].isna()],
                       ['missing_flight', 'no_arrival_in_month', 'ambiguous_arrival', 'route_mismatch',
                        'not_strictly_later', 'same_movement', 'missing_own_arvt'], default='linked')
    result = dep[[ID, FLIGHT, TIME, ORIGIN, DEST, OPERATOR, ARVT, 'month']].copy()
    result['arrival_id'] = merged['arrival_' + ID].to_numpy()
    result['landing_time'] = merged['arrival_' + TIME].to_numpy()
    result['arrival_count'] = support
    result['linked'] = available.to_numpy(bool)
    result['status'] = status
    delta = (merged[ARVT] - merged['arrival_' + TIME]).dt.total_seconds().to_numpy()
    result['delta_sec'] = np.where(available, delta, np.nan)
    assert np.array_equal(result[ID], dep[ID])
    assert np.array_equal(result.linked, result.status.eq('linked'))
    return result


def oracle(departure, arrivals):
    """Small independent enumeration oracle after raw movement deduplication."""
    fid = departure[FLIGHT]
    if pd.isna(fid):
        return 'missing_flight', np.nan
    month = pd.Timestamp(departure[TIME]).strftime('%Y-%m')
    rows = []
    seen = set()
    for _, row in arrivals.loc[arrivals[FLIGHT].eq(fid)].sort_values(ID, kind='stable').iterrows():
        if pd.isna(row[FLIGHT]) or row[FLIGHT] != fid or pd.isna(row[TIME]):
            continue
        if pd.Timestamp(row[TIME]).strftime('%Y-%m') != month:
            continue
        key = (row[FLIGHT], str(row[ORIGIN]), str(row[DEST]), pd.Timestamp(row[TIME]))
        if key not in seen:
            rows.append(row)
            seen.add(key)
    if len(rows) != 1:
        return ('no_arrival_in_month' if not rows else 'ambiguous_arrival'), np.nan
    row = rows[0]
    if any(pd.isna(row[c]) or pd.isna(departure[c]) or not str(row[c]).strip()
           or not str(departure[c]).strip() or row[c] != departure[c] for c in [ORIGIN, DEST]):
        return 'route_mismatch', np.nan
    if not row[TIME] > departure[TIME]:
        return 'not_strictly_later', np.nan
    if row[ID] == departure[ID]:
        return 'same_movement', np.nan
    if pd.isna(departure[ARVT]):
        return 'missing_own_arvt', np.nan
    return 'linked', (departure[ARVT] - row[TIME]).total_seconds()
