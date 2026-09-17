"""Observed prior-hour proxy-minute match support; no provenance claim."""
import audit as core
import numpy as np
import pandas as pd
from pathlib import Path

OUT=core.external_path(core.OUT/'context_cache')
COLS=['prov_prior60_count','prov_prior60_ownminute_fraction','prov_own_proxy_available']


def transform(meta):
    if core.TARGET in meta or 'BLOCK_TIME_UTC_mvt' in meta:
        raise ValueError('Hiddenlabels forbidden in contextbuilder')
    temp=core.prior_matching_fraction(meta)
    p=meta.proxy_sec.to_numpy(float)
    values=pd.DataFrame({core.ID:meta[core.ID].to_numpy(),COLS[0]:temp.prior_count,
        COLS[1]:temp.ownminute_match_fraction,COLS[2]:(np.isfinite(p)&(p>=0)&(p<=7200)).astype(float)})
    values[COLS]=values[COLS].astype('float32')
    return values


def main():
    if (OUT/'manifest.json').exists():
        raise ValueError('Completedcache retained')
    OUT.mkdir(parents=True,exist_ok=True)
    frozen=core.read_json(core.ROOT/'private_runs/submission_v2/protocol.json')
    audit=core.read_json(core.ROOT/'private_runs/screening_230/reports/data_audit.json')
    path=core.ROOT/'private_runs/screening_230/data/interim/audit/departures.parquet'
    assert core.sha256(path)==audit['artifacts'][path.name]
    meta=pd.read_parquet(path,columns=[core.ID,core.MOVEMENT,'ADEP_mvt','proxy_sec'])
    protocol=dict(created_utc=core.utc_now(),source_hashes={str(p.relative_to(core.ROOT)):core.sha256(p) for p in [Path(__file__),Path(core.__file__)]},
        features=COLS,eligibility='Contextotherdepartures finiteproxy0..7200 only; no predictive label clipping.',
        context='Airport/actualUTCmovementmonth,strict [T-3600,T), excludesALLmovementties andhenceownrow; observedproxy roundedminute floor((proxy+30)/60).',
        availability='Prior takeoffevent times withfinalNMactualoffblock; real-time publicationnotproven. No futuretakeoff context.',
        meaning='Localobservedproxyhistogrammatchsupport; not fallbacksource identification.',labels_loaded=False)
    core.write_json(OUT/'protocol.json',protocol)
    training=transform(meta)
    np.testing.assert_array_equal(training[core.ID],meta[core.ID])
    training.to_parquet(OUT/'training_features.parquet',index=False)
    rank_path=core.common.RAW/'ranking.parquet'
    assert core.sha256(rank_path)==frozen['raw_hashes'][rank_path.name]
    raw=core.pq.read_table(rank_path,columns=[core.ID,'PHASE_mvt',core.MOVEMENT,'ADEP_mvt','AOBT_3_flt'],use_threads=False).to_pandas()
    rank=raw.loc[raw.PHASE_mvt.eq('DEP')].copy()
    rank['proxy_sec']=(pd.to_datetime(rank[core.MOVEMENT],utc=True)-pd.to_datetime(rank.AOBT_3_flt,utc=True)).dt.total_seconds()
    ranking=transform(rank)
    np.testing.assert_array_equal(ranking[core.ID],rank[core.ID])
    ranking.to_parquet(OUT/'ranking_features.parquet',index=False)
    core.write_json(OUT/'manifest.json',dict(status='complete',created_utc=core.utc_now(),protocol_sha256=core.sha256(OUT/'protocol.json'),
        source_hashes=protocol['source_hashes'],features=COLS,training_rows=len(training),ranking_rows=len(ranking),
        training_id_order_verified=True,ranking_id_order_verified=True,
        outputs={name:core.sha256(OUT/name) for name in ['training_features.parquet','ranking_features.parquet']}))
    print('CONTEXT_CACHE',training.shape,ranking.shape,flush=True)


if __name__=='__main__':
    main()
