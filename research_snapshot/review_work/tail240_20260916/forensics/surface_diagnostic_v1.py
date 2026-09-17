"""Post-freeze fit-period surface surrogate diagnostics, not a model."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import psutil

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'review_work/campaign_20260916'))
import common
from surface_acquire_v1 import guard
JOIN=ROOT/'private_runs/tail240_20260916/forensics/surface_event_pilot/join_v2'
OUT=ROOT/'private_runs/tail240_20260916/forensics/surface_event_pilot/diagnostic_v1'
ID,TARGET,TIME=common.ID,common.TARGET,common.MOVEMENT


def summary(values):
    values=np.asarray(values,dtype=float)
    values=values[np.isfinite(values)]
    if not len(values):return dict(n=0)
    return dict(n=len(values),mean=float(values.mean()),rmse=float(np.sqrt(np.mean(values**2))),mae=float(np.mean(np.abs(values))),
        quantiles={str(q):float(v) for q,v in zip([0,.01,.1,.5,.9,.99,1],np.quantile(values,[0,.01,.1,.5,.9,.99,1]))})


def summarize(rows):
    finite=np.isfinite(rows.proxy_sec)
    unique=rows.unique_exit_offset_sec.notna()
    paired=rows.loc[finite&unique]
    nm=paired[TARGET]-paired.proxy_sec
    event=paired[TARGET]-paired.unique_exit_offset_sec
    multiple=rows.loc[rows.eligible_parking_exit_count.gt(1)]
    return dict(queries=len(rows),finite_nm=int(finite.sum()),unique_exit=int(unique.sum()),multiple_exits=len(multiple),
        raw_y=summary(rows[TARGET]),nm_residual_allfinite=summary(rows.loc[finite,TARGET]-rows.loc[finite,'proxy_sec']),
        unique_event_offset=summary(rows.loc[unique,'unique_exit_offset_sec']),
        unique_event_residual=summary(rows.loc[unique,TARGET]-rows.loc[unique,'unique_exit_offset_sec']),
        paired_count=len(paired),paired_nm_residual=summary(nm),paired_event_residual=summary(event),
        paired_event_closer=int((event.abs()<nm.abs()).sum()),paired_event_within60=int(event.abs().le(60).sum()),
        paired_event_within300=int(event.abs().le(300).sum()),paired_nm_within60=int(nm.abs().le(60).sum()),
        paired_event_minus_nm=summary(paired.unique_exit_offset_sec-paired.proxy_sec),
        paired_unique_firstseen_offset=summary(paired.opdi_first_seen_offset_sec),
        raw_y_within_multiple_exit_range=int((multiple[TARGET].ge(multiple.exit_offset_min_sec)&multiple[TARGET].le(multiple.exit_offset_max_sec)).sum()),
        multiple_exit_range_width=summary(multiple.exit_offset_max_sec-multiple.exit_offset_min_sec),
        no_multiple_exit_choice=True)


def declare():
    marker=common.read_json(JOIN/'manifest.json')
    assert marker['status']=='complete' and marker['private_targets_read'] is False
    protocol=dict(source_sha256=common.sha256(__file__),frozen_join_manifest_sha256=common.sha256(JOIN/'manifest.json'),
        scope='Original F1 January05-15fit-period rows retained in frozen join, everyrowincluded. Metadata labels/proxy read only for frozenqueryIDs. No score/ranking/model.',
        fixed_comparison='Y minus unique eligible parking-exit offset versus Y minus NM proxy onidenticalfinitepairedrows. All-query/airport/matchmode/NMstatus coverage explicit. No calibration/weight/offset tuning.',
        multiple='Only descriptive rawY inclusion within observed min/max offset range and range width; no choice of exit or nearesttruth selection.',
        limitations='Eventexit is geometric surveillance surrogate, not guaranteedactualoffblock. In-sample earlierfitperiod diagnostic cannot establish tune/scorebenefit; matching/censoring can selecteasiercases.',
        resources='1CPU,<2GiBOSpeak,8GiBhostreserve,start10GiB. No newpublicrequestoracquisition.')
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/'protocol.json'
    if path.exists():assert common.read_json(path)==protocol
    else:common.write_json(path,protocol)
    return protocol


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--declare-only',action='store_true');args=parser.parse_args()
    protocol=declare()
    if args.declare_only:
        print('DECLARED_POST_FREEZE_FIT_DIAGNOSTIC',flush=True);return
    assert psutil.virtual_memory().available>=10*1024**3
    assert not (OUT/'summary.json').exists()
    marker=common.read_json(JOIN/'manifest.json')
    for name,expected in marker['outputs'].items():assert common.sha256(JOIN/name)==expected
    features=pd.read_parquet(JOIN/'features.parquet')
    queries=pd.read_parquet(JOIN/'queries.parquet',columns=[ID,TIME])
    np.testing.assert_array_equal(features[ID],queries[ID])
    assert queries[TIME].ge(pd.Timestamp('2025-01-05',tz='UTC')).all() and queries[TIME].lt(pd.Timestamp('2025-01-15',tz='UTC')).all()
    meta_path=ROOT/'private_runs/screening_230/data/interim/audit/departures.parquet'
    assert common.sha256(meta_path)==common.read_json(ROOT/'private_runs/screening_230/reports/data_audit.json')['artifacts'][meta_path.name]
    # Only metadata for split eligibility is loaded globally; targets stay filtered to frozen IDs.
    meta=pd.read_parquet(meta_path,columns=[ID,'FLIGHT_ID_mvt',TIME,'ADEP_mvt'])
    meta[TARGET]=np.nan
    idx,split,_=common.fold_data(meta,'F1',full=True)
    assert features[ID].isin(meta.iloc[idx['fit']][ID]).all()
    truth=pd.read_parquet(meta_path,columns=[ID,TARGET,'proxy_sec'],filters=[(ID,'in',features[ID].tolist())])
    rows=features.merge(truth,on=ID,how='left',validate='one_to_one')
    assert len(rows)==len(features) and np.isfinite(rows[TARGET]).all()
    results={'all':summarize(rows),'airports':{},'match_modes':{},'nm_status':{}}
    for airport,group in rows.groupby('ADEP_mvt'):results['airports'][airport]=summarize(group)
    for mode,group in rows.groupby('match_mode'):results['match_modes'][mode]=summarize(group)
    for status,mask in [('finite',np.isfinite(rows.proxy_sec)),('missing',~np.isfinite(rows.proxy_sec))]:results['nm_status'][status]=summarize(rows.loc[mask])
    rows.to_parquet(OUT/'diagnostic_rows.parquet',index=False)
    common.write_json(OUT/'summary.json',dict(status='complete',source_sha256=common.sha256(__file__),protocol_sha256=common.sha256(OUT/'protocol.json'),
        frozen_join_manifest_sha256=protocol['frozen_join_manifest_sha256'],original_f1_split=split,results=results,
        diagnostic_rows_sha256=common.sha256(OUT/'diagnostic_rows.parquet'),peak_bytes=guard(),no_model=True,no_score=True))
    print('SURFACE_DIAGNOSTIC',results,flush=True)


if __name__=='__main__':main()
