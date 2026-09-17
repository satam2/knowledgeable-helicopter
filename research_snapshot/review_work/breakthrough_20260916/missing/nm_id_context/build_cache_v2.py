"""Actual-month isolation after combining nominal monthly source packs."""
import build_cache as core
import audit_ids as base
import numpy as np
import pandas as pd

CACHE=base.external_path(base.OUT/'cache_v2')


def main():
    if (CACHE/'manifest.json').exists():
        raise ValueError('Completed cache retained')
    CACHE.mkdir(parents=True,exist_ok=True)
    frozen=base.read_json(base.ROOT/'private_runs/submission_v2/protocol.json')
    protocol=dict(created_utc=base.utc_now(),source_hashes={name:base.sha256(base.HERE/name) for name in ['audit_ids.py','build_cache.py','build_cache_v2.py']},
        raw_columns=base.COLS,raw_hashes=frozen['raw_hashes'],features=core.FEATURES,width_each_side=8,
        context='Combine all suppliedtrainingpacks; then isolate actualUTC movementmonth. Ranking similarlymonthisolated. Allairports ARR/DEP. SameNMflight entirelyexcluded. Perflight clockmedians.',
        availability='RETROSPECTIVE suppliedbatch; numericIDneighbors canbelater movements. No hiddenlabel or departureBLOCK input.',
        failed_attempt='cache_v1 preserved: nominal monthlypacks have seven outofmonth rows includingtwo departures.',
        interpretation='Empirical nearIOBTordering, not provenNMIDallocation or physicalB ordering.',score_label_rules=False)
    base.write_json(CACHE/'protocol.json',protocol)
    frames=[]
    boundaries=[]
    for path in sorted(base.common.RAW.glob('training*.parquet')):
        assert base.sha256(path)==frozen['raw_hashes'][path.name]
        frame=base.pq.read_table(path,columns=base.COLS,use_threads=False).to_pandas()
        months=pd.to_datetime(frame[base.MOVEMENT],utc=True).dt.strftime('%Y-%m')
        expected=path.name[len('training_'):len('training_')+7]
        boundaries.append(dict(file=path.name,outside_nominal_month=int(months.ne(expected).sum())))
        frames.append(frame)
    raw=pd.concat(frames,ignore_index=True)
    del frames
    training=core.transform(raw)
    del raw
    metadata=base.ROOT/'private_runs/screening_230/data/interim/audit/departures.parquet'
    audit=base.read_json(base.ROOT/'private_runs/screening_230/reports/data_audit.json')
    assert base.sha256(metadata)==audit['artifacts'][metadata.name]
    ids=pd.read_parquet(metadata,columns=[base.ID])[base.ID]
    training=training.set_index(base.ID).loc[ids].reset_index()
    np.testing.assert_array_equal(training[base.ID],ids)
    training.to_parquet(CACHE/'training_features.parquet',index=False)
    rank=base.common.RAW/'ranking.parquet'
    assert base.sha256(rank)==frozen['raw_hashes'][rank.name]
    raw=base.pq.read_table(rank,columns=base.COLS,use_threads=False).to_pandas()
    ranking=core.transform(raw)
    np.testing.assert_array_equal(ranking[base.ID],raw.loc[raw[base.PHASE].eq('DEP'),base.ID])
    ranking.to_parquet(CACHE/'ranking_features.parquet',index=False)
    summary={}
    for name,frame in [('training',training),('ranking',ranking)]:
        summary[name]=dict(rows=len(frame),peer_nm_minus_own_nm=base.stats(frame.nmid_peer_nm_minus_own_nm),
            available_id=int(frame.nmid_source_id_present.sum()),
            finite_counts={col:int(np.isfinite(frame[col]).sum()) for col in core.FEATURES})
    base.write_json(CACHE/'manifest.json',dict(status='complete',created_utc=base.utc_now(),
        protocol_sha256=base.sha256(CACHE/'protocol.json'),source_hashes=protocol['source_hashes'],
        features=core.FEATURES,training_id_order_verified=True,ranking_id_order_verified=True,
        summary=summary,nominal_pack_boundary_rows=boundaries,
        outputs={name:base.sha256(CACHE/name) for name in ['training_features.parquet','ranking_features.parquet']}))
    print('CACHE_V2',summary,flush=True)


if __name__=='__main__':
    main()
