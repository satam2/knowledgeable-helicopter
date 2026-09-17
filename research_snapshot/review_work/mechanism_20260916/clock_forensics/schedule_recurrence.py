"""Exact flight/airport/schedule recurrence among departure records only."""

import os
for variable in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[variable] = '1'
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'review_work/campaign_20260916'))
import common
from taxiout.artifacts import read_json, sha256, utc_now, write_json
from taxiout.io import concat_frames
from taxiout.paths import external_path
from taxiout.schema import BLOCK, ID, MOVEMENT, PHASE, TARGET, utc
from analyze_clocks import json_finite

pa.set_cpu_count(1)
pa.set_io_thread_count(1)
OUT = external_path(ROOT / 'private_runs/mechanism_20260916/clock_forensics')
RAW = external_path(ROOT / 'data/09-15-2026-18-55-03_files_list')
KEY = ['FLIGHT_mvt', 'ADEP_mvt', 'ADES_mvt', 'SCHED_TIME_UTC_mvt']


def summarize(frame):
    def counts(data):
        return {'n': len(data), 'any_exact_schedule_peer': int(data.peer_any.sum()),
            'any_earlier_peer': int(data.peer_earlier.sum()),
            'earlier_nm_known_peer': int(data.peer_nm_earlier.sum()),
            'earlier_nm_proxy_0_7200': int(data.peer_nm_plausible.sum()),
            'earlier_nm_hidden_block_within60': int(data.peer_nm_matches_block.sum()),
            'different_takeoff_utc_day_peer': int(data.peer_otherday.sum()),
            'exact_same_takeoff_peer': int(data.peer_sametime.sum())}
    missing = frame.AOBT_3_flt.isna()
    report = {'all': counts(frame), 'missing': counts(frame.loc[missing]),
        'missing_gt12h': counts(frame.loc[missing & frame[TARGET].gt(43200)]),
        'missing_24h_plus_0_2h': counts(frame.loc[missing & frame[TARGET].between(86400, 93600)]),
        'known': counts(frame.loc[~missing])}
    for fold in ['F1', 'F3']:
        ref, _ = common.reference(fold)
        score = frame.loc[frame[ID].isin(ref[ID])]
        assert len(score) == len(ref)
        report[fold] = {'all': counts(score), 'missing': counts(score.loc[score.AOBT_3_flt.isna()]),
            'missing_gt12h': counts(score.loc[score.AOBT_3_flt.isna() & score[TARGET].gt(43200)])}
    return report


def main():
    expected = read_json(ROOT / 'private_runs/submission_v2/protocol.json')['raw_hashes']
    frames = []
    for path in sorted(RAW.glob('training_*.parquet')):
        assert sha256(path) == expected[path.name]
        raw = pq.read_table(path, columns=[ID, PHASE, MOVEMENT, BLOCK, TARGET, 'AOBT_3_flt', *KEY], use_threads=False).to_pandas(strings_to_categorical=True)
        frames.append(raw.loc[raw[PHASE].eq('DEP')].drop(columns=PHASE))
    frame = concat_frames(frames)
    for column in ['peer_any', 'peer_earlier', 'peer_nm_earlier', 'peer_nm_plausible', 'peer_nm_matches_block', 'peer_otherday', 'peer_sametime']:
        frame[column] = False
    eligible = frame.loc[frame[KEY].notna().all(axis=1)]
    group_size = eligible.groupby(KEY, observed=True).size()
    repeated = group_size[group_size > 1].index
    print('Repeated exact flight/route/schedule keys', len(repeated), flush=True)
    grouped = eligible.groupby(KEY, observed=True)
    for key in repeated:
        sub = grouped.get_group(key)
        for index, row in sub.iterrows():
            other = sub.loc[sub[ID].ne(row[ID])]
            earlier = other.loc[other[MOVEMENT].lt(row[MOVEMENT])]
            known = earlier.loc[earlier.AOBT_3_flt.notna()]
            frame.loc[index, 'peer_any'] = len(other) > 0
            frame.loc[index, 'peer_earlier'] = len(earlier) > 0
            frame.loc[index, 'peer_nm_earlier'] = len(known) > 0
            frame.loc[index, 'peer_otherday'] = utc(other[MOVEMENT]).dt.normalize().ne(row[MOVEMENT].normalize()).any()
            frame.loc[index, 'peer_sametime'] = other[MOVEMENT].eq(row[MOVEMENT]).any()
            if len(known):
                proxy = (row[MOVEMENT] - utc(known.AOBT_3_flt)).dt.total_seconds()
                frame.loc[index, 'peer_nm_plausible'] = proxy.between(0, 7200).any()
                frame.loc[index, 'peer_nm_matches_block'] = (utc(known.AOBT_3_flt) - row[BLOCK]).dt.total_seconds().abs().le(60).any()
    report = {'created_utc': utc_now(), 'source_sha256': sha256(__file__),
        'scope': 'Exact flight-number text + departure airport + destination airport + complete scheduled UTC timestamp; DEP records only. No normalization, no aircraft rotation inference.',
        'availability': 'Earlier-peer counts require prior takeoff event; all-peer recurrence is separately retrospective diagnostic only. Hidden block closeness is diagnostic, not selection.',
        'duplicate_schedule_groups': len(repeated), 'maximum_group_n': int(group_size.max()), 'results': summarize(frame)}
    write_json(OUT / 'schedule_recurrence.json', json_finite(report))
    print('DONE exact schedule recurrence', flush=True)


if __name__ == '__main__':
    main()
