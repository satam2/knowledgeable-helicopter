"""Fixed post-freeze comparison of first-seen and takeoff-event linkage arms."""
import surface_diagnostic_v1 as metrics
import argparse
import numpy as np
import pandas as pd

ROOT,common,ID,TARGET,TIME=metrics.ROOT,metrics.common,metrics.ID,metrics.TARGET,metrics.TIME
BASE=ROOT/'private_runs/tail240_20260916/forensics/surface_event_pilot'
ARMS={'first_seen':BASE/'join_v2','takeoff_event':BASE/'takeoff_join_v1'}
OUT=BASE/'dual_diagnostic_v1'


def declare():
    records={arm:common.read_json(folder/'manifest.json') for arm,folder in ARMS.items()}
    assert all(record['status']=='complete' and record['private_targets_read'] is False for record in records.values())
    assert len({record['query_id_hash'] for record in records.values()})==1
    record=dict(source_sha256=common.sha256(__file__),metric_source_sha256=common.sha256(metrics.__file__),
        arm_manifests={arm:common.sha256(folder/'manifest.json') for arm,folder in ARMS.items()},
        scope='Same49,522retainedoriginalF1fitJan05-Jan15queryIDs. No targetreaduntilbothlabel-freearmsfrozen. No score/ranking/model/newdownloads.',
        comparisons='Forbothfixedarms, rawY minusuniqueeligibleparkingoffset versusrawY minusNMproxy onidenticalfinitepairedrows. Reportcoverage byairport/mode/NMstatus, fullresidualdistributions. Nocalibration/offset/blend/choiceofarmfromlabels.',
        multiple='Rangecontainment andwidth only; multipleeligibleexit rows remainunresolved, nonearesttruthchoice.',
        matched='Also report pairedNM andeventerrors onintersection and newlycoveredrows fromeachfixedarm; no resultingrouteorselector.',
        resources='1CPU,<2GiBOSpeak,start10GiBavailable/runtime8reserve.',
        limitations='Earlierfitperiod descriptivecheck, censored/selectedobservations; parkinggeometricexitnotguaranteedAOBT, no predictorbenefitclaim.')
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/'protocol.json'
    if path.exists():assert common.read_json(path)==record
    else:common.write_json(path,record)
    return record


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--declare-only',action='store_true');args=parser.parse_args()
    protocol=declare()
    if args.declare_only:print('DECLARED_DUAL_FIT_DIAGNOSTIC',flush=True);return
    assert metrics.psutil.virtual_memory().available>=10*1024**3 and not (OUT/'summary.json').exists()
    features={}
    for arm,folder in ARMS.items():
        marker=common.read_json(folder/'manifest.json')
        assert common.sha256(folder/'manifest.json')==protocol['arm_manifests'][arm]
        for name,expected in marker['outputs'].items():assert common.sha256(folder/name)==expected
        features[arm]=pd.read_parquet(folder/'features.parquet')
    ids=features['first_seen'][ID]
    np.testing.assert_array_equal(ids,features['takeoff_event'][ID])
    queries=pd.read_parquet(ARMS['first_seen']/'queries.parquet',columns=[ID,TIME])
    np.testing.assert_array_equal(ids,queries[ID])
    assert queries[TIME].ge(pd.Timestamp('2025-01-05',tz='UTC')).all() and queries[TIME].lt(pd.Timestamp('2025-01-15',tz='UTC')).all()
    meta_path=ROOT/'private_runs/screening_230/data/interim/audit/departures.parquet'
    assert common.sha256(meta_path)==common.read_json(ROOT/'private_runs/screening_230/reports/data_audit.json')['artifacts'][meta_path.name]
    meta=pd.read_parquet(meta_path,columns=[ID,'FLIGHT_ID_mvt',TIME,'ADEP_mvt'])
    meta[TARGET]=np.nan
    idx,split,_=common.fold_data(meta,'F1',full=True)
    assert ids.isin(meta.iloc[idx['fit']][ID]).all()
    truth=pd.read_parquet(meta_path,columns=[ID,TARGET,'proxy_sec'],filters=[(ID,'in',ids.tolist())])
    results,frames={},{}
    for arm,frame in features.items():
        rows=frame.merge(truth,on=ID,how='left',validate='one_to_one')
        assert len(rows)==len(ids) and np.isfinite(rows[TARGET]).all()
        result={'all':metrics.summarize(rows),'airports':{},'match_modes':{},'nm_status':{}}
        for airport,group in rows.groupby('ADEP_mvt'):result['airports'][airport]=metrics.summarize(group)
        for mode,group in rows.groupby('match_mode'):result['match_modes'][mode]=metrics.summarize(group)
        for status,mask in [('finite',np.isfinite(rows.proxy_sec)),('missing',~np.isfinite(rows.proxy_sec))]:result['nm_status'][status]=metrics.summarize(rows.loc[mask])
        rows.to_parquet(OUT/(arm+'_rows.parquet'),index=False)
        frames[arm]=rows
        results[arm]=result
    unique={arm:set(frame.loc[frame.unique_exit_offset_sec.notna(),ID]) for arm,frame in frames.items()}
    intersection=unique['first_seen']&unique['takeoff_event']
    comparisons={}
    for arm,frame in frames.items():
        other='takeoff_event' if arm=='first_seen' else 'first_seen'
        comparisons[arm]={'shared_unique_exits':metrics.summarize(frame.loc[frame[ID].isin(intersection)]),
            'new_unique_exits':metrics.summarize(frame.loc[frame[ID].isin(unique[arm]-unique[other])])}
    common.write_json(OUT/'summary.json',dict(status='complete',source_sha256=common.sha256(__file__),protocol_sha256=common.sha256(OUT/'protocol.json'),
        results=results,comparisons=comparisons,original_f1_split=split,peak_bytes=metrics.guard(),no_model=True,no_score=True,
        outputs={arm+'_rows.parquet':common.sha256(OUT/(arm+'_rows.parquet')) for arm in ARMS}))
    print('DUAL_SURFACE', {arm:result['all'] for arm,result in results.items()},flush=True)


if __name__=='__main__':main()
