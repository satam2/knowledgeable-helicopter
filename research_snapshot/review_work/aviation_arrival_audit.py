"""Read-only availability audit of arrival outcomes; these are not model features yet."""

import numpy as np
import pandas as pd
from aviation_spike import OUT, PRIOR
from next230_common import config_for
from run_screening import RAW
from taxiout.artifacts import write_json, sha256
from taxiout.io import read_raw, training_paths, concat_frames
from taxiout.schema import ID, FLIGHT_ID, TARGET, BLOCK, MOVEMENT, PHASE


COLS = [ID, FLIGHT_ID, PHASE, TARGET, BLOCK, MOVEMENT, 'AOBT_3_flt',
        'ADEP_mvt', 'ADES_mvt', 'STAND_mvt', 'RUNWAY_mvt']


def profile(frame):
    arr = frame.loc[frame[PHASE].eq('ARR')]
    dep = frame.loc[frame[PHASE].eq('DEP')]
    values = pd.to_numeric(arr[TARGET], errors='coerce')
    aobt = pd.to_datetime(arr.AOBT_3_flt, utc=True, errors='coerce')
    block = pd.to_datetime(arr[BLOCK], utc=True, errors='coerce')
    move = pd.to_datetime(arr[MOVEMENT], utc=True, errors='coerce')
    duration = (block-move).dt.total_seconds()
    comparable = values.notna() & duration.notna()
    known = frame.loc[frame[FLIGHT_ID].notna() & frame.AOBT_3_flt.notna(), FLIGHT_ID]
    missing_dep = dep.AOBT_3_flt.isna()
    recovered = missing_dep & dep[FLIGHT_ID].notna() & dep[FLIGHT_ID].isin(known)
    return {'arrivals': len(arr), 'arrival_taxi_label_present_pct': float(values.notna().mean()*100),
        'arrival_block_present_pct': float(block.notna().mean()*100),
        'arrival_identity_exact_when_observed': bool(np.array_equal(values[comparable], duration[comparable])),
        'arrival_proxy_present_pct': float(aobt.notna().mean()*100),
        'arrival_taxi_negative_count': int((values<0).sum()),
        'arrival_taxi_over_7200_count': int((values>7200).sum()),
        'departures': len(dep), 'departure_nm_missing_count': int(missing_dep.sum()),
        'missing_nm_recoverable_from_other_same_flight_rows': int(recovered.sum()),
        'departure_hidden_label_present_count': int(dep[TARGET].notna().sum())}


def main():
    train = concat_frames([read_raw(p,COLS) for p in training_paths(config_for('baseline'))])
    ranking = read_raw(RAW/'ranking.parquet',COLS)
    result = {'training': profile(train), 'ranking': profile(ranking), 'folds': {},
        'scope': 'Availability audit only; no new feature used in trained models; arrival data must be complete before a causal query.'}
    # At takeoff time, an arrival taxi-in duration is observable only after in-block.
    for fold in ['F1', 'F3']:
        sr = pd.read_parquet(OUT/fold/'score_reference.parquet')
        dep = train.set_index(ID).loc[sr[ID]]
        qtimes = pd.to_datetime(dep[MOVEMENT],utc=True)
        month = qtimes.dt.strftime('%Y-%m').iloc[0]
        ready = pd.to_datetime(train[BLOCK],utc=True,errors='coerce')
        move = pd.to_datetime(train[MOVEMENT],utc=True,errors='coerce')
        arr = train.loc[train[PHASE].eq('ARR') & move.dt.strftime('%Y-%m').eq(month) & ready.notna()].copy()
        arr['ready'] = pd.to_datetime(arr[BLOCK],utc=True)
        arr = arr.loc[arr.ready>=pd.to_datetime(arr[MOVEMENT],utc=True)]
        groups = {str(k): np.sort(a.ready.dt.as_unit('ns').astype('int64').to_numpy()) for k,a in arr.groupby('ADES_mvt', observed=True)}
        counts = np.zeros(len(dep))
        qarray = qtimes.dt.as_unit('ns').astype('int64').to_numpy()
        for airport, pos in dep.groupby('ADEP_mvt',observed=True).indices.items():
            times = groups.get(str(airport), np.empty(0,dtype=np.int64))
            counts[pos] = np.searchsorted(times,qarray[pos],side='left')-np.searchsorted(times,qarray[pos]-1800*10**9,side='left')
        fit_end = pd.Timestamp('2025-06-01' if fold=='F1' else '2025-10-01',tz='UTC')
        fit_arr = train.loc[train[PHASE].eq('ARR') & (ready<fit_end)]
        support = set(zip(fit_arr.ADES_mvt.astype(str), fit_arr.STAND_mvt.astype(str)))
        stand_known = [key in support for key in zip(dep.ADEP_mvt.astype(str),dep.STAND_mvt.astype(str))]
        result['folds'][fold] = {'query_n': len(dep), 'recent_completed_arrival_30m_available_pct': float((counts>0).mean()*100),
            'median_completed_arrivals_30m': float(np.median(counts)),
            'departure_stand_seen_in_fit_arrivals_pct': float(np.mean(stand_known)*100)}
    result['runner_sha256'] = sha256(__file__)
    write_json(OUT/'arrival_availability.json', result)
    print(result)


if __name__ == '__main__':
    main()
