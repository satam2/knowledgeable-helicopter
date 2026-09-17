"""Seven label-free NMID neighbor-clock fields for controlled downstream tests."""
import audit_ids as base
import numpy as np
import pandas as pd

CACHE=base.external_path(base.OUT/'cache_v1')
FEATURES=['nmid_peer_nm_minus_own_nm','nmid_peer_eobt_minus_own_nm',
    'nmid_peer_nm_minus_takeoff','nmid_peer_eobt_minus_takeoff',
    'nmid_peer_nm_count','nmid_peer_eobt_count','nmid_source_id_present']


def transform(context):
    if any(column in context for column in [base.TARGET,'BLOCK_TIME_UTC_mvt']):
        raise ValueError('Hidden columns forbidden in NMID feature producer')
    dep=context.loc[context[base.PHASE].eq('DEP')].copy()
    if not dep[base.ID].is_unique:
        raise ValueError('Duplicate departure movement IDs')
    months=pd.to_datetime(context[base.MOVEMENT],utc=True).dt.strftime('%Y-%m')
    pieces=[]
    for month in sorted(months.unique()):
        events=context.loc[months.eq(month)]
        query=events.loc[events[base.PHASE].eq('DEP')]
        if query.empty:
            continue
        peers=base.neighbors(query,events,width=8)
        own=base.seconds(query.AOBT_3_flt)
        takeoff=base.seconds(query[base.MOVEMENT])
        piece=pd.DataFrame({base.ID:query[base.ID].to_numpy(),
            FEATURES[0]:peers.nmid_peer_nm_seconds.to_numpy()-own,
            FEATURES[1]:peers.nmid_peer_initial_seconds.to_numpy()-own,
            FEATURES[2]:peers.nmid_peer_nm_seconds.to_numpy()-takeoff,
            FEATURES[3]:peers.nmid_peer_initial_seconds.to_numpy()-takeoff,
            FEATURES[4]:peers.nmid_peer_nm_count.to_numpy(),
            FEATURES[5]:peers.nmid_peer_initial_count.to_numpy(),
            FEATURES[6]:query[base.FLIGHT_ID].notna().to_numpy(float)})
        pieces.append(piece)
    output=pd.concat(pieces,ignore_index=True).set_index(base.ID).loc[dep[base.ID]].reset_index()
    output[FEATURES]=output[FEATURES].astype('float32')
    return output


def main():
    if (CACHE/'manifest.json').exists():
        raise ValueError('Completed cache preserved')
    CACHE.mkdir(parents=True,exist_ok=True)
    frozen=base.read_json(base.ROOT/'private_runs/submission_v2/protocol.json')
    protocol=dict(created_utc=base.utc_now(),source_hashes={name:base.sha256(base.HERE/name) for name in ['audit_ids.py','build_cache.py']},
        fields=FEATURES,raw_columns=base.COLS,raw_hashes=frozen['raw_hashes'],
        width_each_side=8,context='Same UTCmovementmonth, all suppliedairport ARR/DEP, sameNMflight allrows excluded; one clock median perdistinctNMflight.',
        availability='RETROSPECTIVE finalsuppliedbatch. IDneighbors may be futuremovementrows. Nohiddenlabels or departureBLOCK used.',
        missing='MissingNMID => no peers. Missing ownNM => ownNM differences missing; takeoff differences may remain for rareIDonly records.',
        interpretation='Empirical filing/initialoffblock order, not documented NMallocation semantics, aircraftidentity or trueoffblockorder.',
        model_evidence='Fit/tune13binmeans retain~2.4-2.6s after ownIOBT/EOBT/schedule controls; no V2 scoreimprovement or standalone model established.',
        score_label_rules=False)
    base.write_json(CACHE/'protocol.json',protocol)
    parts=[]
    for path in sorted(base.common.RAW.glob('training*.parquet')):
        assert base.sha256(path)==frozen['raw_hashes'][path.name]
        raw=base.pq.read_table(path,columns=base.COLS,use_threads=False).to_pandas()
        assert pd.to_datetime(raw[base.MOVEMENT],utc=True).dt.strftime('%Y-%m').nunique()==1
        parts.append(transform(raw))
    training=pd.concat(parts,ignore_index=True)
    audit=base.read_json(base.ROOT/'private_runs/screening_230/reports/data_audit.json')
    meta_path=base.ROOT/'private_runs/screening_230/data/interim/audit/departures.parquet'
    assert base.sha256(meta_path)==audit['artifacts'][meta_path.name]
    ids=pd.read_parquet(meta_path,columns=[base.ID])[base.ID]
    training=training.set_index(base.ID).loc[ids].reset_index()
    np.testing.assert_array_equal(training[base.ID],ids)
    training.to_parquet(CACHE/'training_features.parquet',index=False)
    rank_path=base.common.RAW/'ranking.parquet'
    assert base.sha256(rank_path)==frozen['raw_hashes'][rank_path.name]
    raw=base.pq.read_table(rank_path,columns=base.COLS,use_threads=False).to_pandas()
    ranking=transform(raw)
    np.testing.assert_array_equal(ranking[base.ID],raw.loc[raw[base.PHASE].eq('DEP'),base.ID])
    ranking.to_parquet(CACHE/'ranking_features.parquet',index=False)
    summaries={}
    for name,frame in [('training',training),('ranking',ranking)]:
        summaries[name]=dict(rows=len(frame),own_nm_peer=base.stats(frame[FEATURES[0]]),
            available_id=int(frame.nmid_source_id_present.sum()),
            finite_counts={c:int(np.isfinite(frame[c]).sum()) for c in FEATURES})
    base.write_json(CACHE/'manifest.json',dict(status='complete',created_utc=base.utc_now(),
        protocol_sha256=base.sha256(CACHE/'protocol.json'),source_hashes=protocol['source_hashes'],
        features=FEATURES,training_id_order_verified=True,ranking_id_order_verified=True,summary=summaries,
        outputs={name:base.sha256(CACHE/name) for name in ['training_features.parquet','ranking_features.parquet']}))
    print('CACHE',summaries,flush=True)


if __name__=='__main__':
    main()
