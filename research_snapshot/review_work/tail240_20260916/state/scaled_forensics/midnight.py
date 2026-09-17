"""Bounded timestamp interpretation of one exposed-tune error; no correction rule."""
import lightgbm
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT/'review_work/tail240_20260916/models'))
import normalized_missing_tune as base
common, ID, TARGET, TIME = base.common, base.ID, base.TARGET, base.TIME
OUT = common.external_path(ROOT/'private_runs/tail240_20260916/state/scaled_forensics/midnight_v1')


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    x, meta = base.shared.load_missing()
    qid = 190201378.
    raw_path = ROOT/'data/09-15-2026-18-55-03_files_list/training_2025-06-01_2025-07-01.parquet'
    raw_hash = common.read_json(ROOT/'private_runs/submission_v2/protocol.json')['raw_hashes'][raw_path.name]
    assert common.sha256(raw_path) == raw_hash
    fields = [ID, TIME, 'SCHED_TIME_UTC_mvt', 'ADEP_mvt', 'ADES_mvt', 'FLIGHT_mvt', 'STAND_mvt', 'RUNWAY_mvt']
    raw = pq.read_table(raw_path, columns=fields, filters=[(ID,'==',qid)], use_threads=False).to_pandas()
    assert len(raw) == 1
    row = raw.iloc[0]
    observed = pd.to_datetime(row[TIME],utc=True)
    schedule = pd.to_datetime(row.SCHED_TIME_UTC_mvt,utc=True)
    label = float(meta.set_index(ID).loc[qid,TARGET])
    implied = observed-pd.Timedelta(seconds=label)
    query = dict(id=qid, airport=str(row.ADEP_mvt), destination=str(row.ADES_mvt), flight=str(row.FLIGHT_mvt),
        stand=str(row.STAND_mvt), runway=str(row.RUNWAY_mvt), raw_Y=label,
        observed_schedule_proxy=float((observed-schedule).total_seconds()), label_modulo_day=label%86400,
        UTC=dict(takeoff=str(observed), schedule=str(schedule), implied_airport_block=str(implied)),
        Europe_Rome=dict(takeoff=str(observed.tz_convert('Europe/Rome')), schedule=str(schedule.tz_convert('Europe/Rome')),
            implied_airport_block=str(implied.tz_convert('Europe/Rome'))),
        explanation='Adding86400seconds to impliedblock would yield768seconds taxi; purely arithmetical alternative, not an authorized labelcorrection or establishedrecordingmechanism.')
    reports = {}
    witnesses = []
    for fold in ['F1','F3']:
        idx, _, _ = common.fold_data(meta,fold,full=True)
        for stage in ['fit','tune']:
            p = idx[stage]
            p = p[~np.isfinite(meta.iloc[p].proxy_sec)]
            rows = meta.iloc[p].copy().set_index(ID)
            rows = rows.loc[rows.ADEP_mvt.astype(str).eq('LIRF')]
            f = x.loc[rows.index]
            t = pd.to_datetime(rows[TIME],utc=True)
            gap = f.schedule_proxy_sec.to_numpy(float)
            s = t-pd.to_timedelta(np.where(gap==-999999,np.nan,gap),unit='s')
            local = t.dt.tz_convert('Europe/Rome')
            slocal = s.dt.tz_convert('Europe/Rome')
            nearutc = t.dt.hour.lt(4)
            nearlocal = local.dt.hour.lt(4)
            sentinel = slocal.dt.hour.eq(23)&slocal.dt.minute.eq(59)
            y = rows[TARGET]
            masks = {'all_missing_Rome':np.ones(len(rows),bool), 'UTC_takeoff_00_04':nearutc,
                'Rome_takeoff_00_04':nearlocal, 'Rome_schedule_23_59':sentinel,
                'Rome_schedule23_59_takeoff00_04':sentinel&nearlocal}
            summary = {}
            for name, mask in masks.items():
                v = y.loc[mask]
                summary[name] = dict(rows=len(v), raw_Y_ge86400=int(v.ge(86400).sum()),
                    raw_Y_86400_to93600=int(v.between(86400,93600).sum()), raw_Y_0_to7200=int(v.between(0,7200).sum()),
                    median_raw_Y=float(v.median()) if len(v) else None)
            reports[fold+'_'+stage] = summary
            use = rows.loc[nearlocal|sentinel].copy()
            use['schedule'] = s.loc[use.index]
            use['takeoff_Rome'] = local.loc[use.index]
            use['schedule_Rome'] = slocal.loc[use.index]
            use['flight'] = f.loc[use.index,'flight']
            use['stand'] = f.loc[use.index,'stand']
            use['runway'] = f.loc[use.index,'runway']
            use['raw_Y_mod_day'] = use[TARGET]%86400
            use['implied_airport_block_UTC'] = use[TIME]-pd.to_timedelta(use[TARGET],unit='s')
            use['fold_stage'] = fold+'_'+stage
            witnesses.append(use.reset_index())
    pd.concat(witnesses,ignore_index=True).to_parquet(OUT/'near_midnight_witnesses.parquet',index=False)
    common.write_json(OUT/'report.json',dict(status='complete',source_sha256=common.sha256(__file__),query=query,counts=reports,
        raw_input_sha256=raw_hash,witnesses_sha256=common.sha256(OUT/'near_midnight_witnesses.parquet'),
        limits='Exposedfit/tune rawlabel diagnostic only. Fixedclock windows, no scoredata, no fitting, no labelcorrection. Same clocktime modulo with differentdates iscompatiblewithrollovererror but cannot distinguishtrue24hinterval; schedule23:59 isobserved, notprovenplaceholder.'))
    print(query, reports,flush=True)


if __name__ == '__main__':
    main()
