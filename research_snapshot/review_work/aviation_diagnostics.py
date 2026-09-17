"""Aggregate private error diagnostics. No row-level details printed or published."""

import numpy as np
import pandas as pd
from aviation_spike import OUT, PRIOR, ARMS
from next230_common import config_for, load_data, load_reference
from taxiout.artifacts import write_json, read_json, sha256, source_hashes
from taxiout.availability import make_observations
from taxiout.io import read_raw, training_paths, concat_frames
from taxiout.schema import ID, TARGET, MOVEMENT, PHASE


def summarize():
    x, meta, labels = load_data()
    cols = [ID, PHASE, MOVEMENT, 'ADEP_mvt', 'ADES_mvt', 'ADEP_flt', 'ADES_flt',
            'AIRCRAFT_TYPE_mvt', 'AIRCRAFT_TYPE_flt', 'AOBT_3_flt', 'LOBT_flt',
            'SCHED_TIME_UTC_mvt', 'STAND_mvt', 'RUNWAY_mvt']
    raw = make_observations(concat_frames([read_raw(p, cols) for p in training_paths(config_for('baseline'))]))[0]
    dep = raw.loc[raw[PHASE].eq('DEP')].set_index(ID).loc[x.index]
    aggregates = {'source_mismatch': {}, 'error_budget': {}, 'airports': {}, 'folds': {}, 'pair_baseline': {}}
    oracle = {'weighted_mse': 0., 'folds': {},
              'definition': 'Label-aware closest point in frozen residual/direct expert interval; only residual route changes. Other routes fixed. Diagnostic only.'}
    candidates = {arm: {'folds': {}, 'weighted_mse': 0.} for arm in ARMS}
    for fold, weight in [('F1', 192122/344841), ('F3', 152719/344841)]:
        folder = OUT/fold
        _, _, idx, _ = load_reference(fold, meta)
        sr = pd.read_parquet(folder/'score_reference.parquet')
        obs = dep.loc[sr[ID]]
        y = sr[TARGET].to_numpy(float)
        p = sr.prediction_sec.to_numpy(float)
        se = (p-y)**2
        ideal = np.clip(y, np.minimum(sr.residual_expert, sr.direct_expert), np.maximum(sr.residual_expert, sr.direct_expert))
        oracle_p = p.copy()
        eligible = sr.route.eq('residual').to_numpy()
        oracle_p[eligible] = ideal[eligible]
        oracle_mse = float(np.mean((oracle_p-y)**2))
        oracle['folds'][fold] = float(np.sqrt(oracle_mse))
        oracle['weighted_mse'] += weight*oracle_mse
        proxy = meta.iloc[idx['score']].proxy_sec.to_numpy(float)
        disagree = np.abs(y-proxy)
        source_bad = np.zeros(len(sr), dtype=bool)
        for prefix in ['ADEP', 'ADES', 'AIRCRAFT_TYPE']:
            a, b = obs[prefix+'_mvt'].astype('string'), obs[prefix+'_flt'].astype('string')
            mismatch = (a.notna() & b.notna() & a.ne(b)).fillna(False).to_numpy(bool)
            source_bad |= mismatch
        observed_clocks_disagree = np.abs((pd.to_datetime(obs.AOBT_3_flt, utc=True) - pd.to_datetime(obs.LOBT_flt, utc=True)).dt.total_seconds().to_numpy()) > 300
        conditions = {
            'NM missing': ~np.isfinite(proxy),
            'NM target gap <=60s': np.isfinite(proxy) & (disagree <= 60),
            'NM target gap 60..300s': (disagree > 60) & (disagree <= 300),
            'NM target gap 300..1800s': (disagree > 300) & (disagree <= 1800),
            'NM target gap >1800s': disagree > 1800,
        }
        aggregate_records = {}
        for name, mask in {**conditions, 'observed source mismatch': source_bad,
                           'observed AOBT-LOBT >300s': observed_clocks_disagree,
                           'target >3600s': y > 3600, 'target >7200s': y > 7200,
                           'absolute error top 1pct': np.abs(p-y) >= np.quantile(np.abs(p-y), .99)}.items():
            record = {'rows': int(mask.sum()), 'row_pct': 100*float(mask.mean()),
                      'sse_pct': 100*float(se[mask].sum()/se.sum()),
                      'rmse_sec': float(np.sqrt(se[mask].mean())) if mask.any() else None}
            aggregate_records[name] = record
            key = 'error_budget' if name in conditions else 'source_mismatch'
            acc = aggregates[key].setdefault(name, {'weighted_mse': 0., 'weighted_row_pct': 0.})
            acc['weighted_mse'] += weight*float(se[mask].sum()/len(sr))
            acc['weighted_row_pct'] += weight*record['row_pct']
        aggregates['folds'][fold] = aggregate_records
        for airport in sorted(obs.ADEP_mvt.astype(str).unique()):
            mask = obs.ADEP_mvt.astype(str).eq(airport).to_numpy()
            acc = aggregates['airports'].setdefault(airport, {'weighted_mse': 0., 'folds': {}})
            acc['weighted_mse'] += weight*float(se[mask].sum()/len(sr))
            acc['folds'][fold] = {'n': int(mask.sum()), 'rmse_sec': float(np.sqrt(se[mask].mean()))}
        table = pd.read_parquet(folder/'pair_priors.parquet').reset_index()
        supported = table.loc[table.n>=50]
        spreads = supported.groupby('ADEP_mvt').q20.agg(lambda a: a.quantile(.9)-a.quantile(.1))
        aggregates['pair_baseline'][fold] = {'supported_pairs': len(supported),
            'airport_q20_pair_spread_p90_p10_sec': {str(k): float(v) for k,v in spreads.items()},
            'coverage': read_json(folder/'evidence.json')}
        day = pd.to_datetime(sr[MOVEMENT], utc=True).dt.strftime('%Y-%m-%d')
        for arm in ARMS:
            pred = pd.read_parquet(folder/f'{arm}_predictions.parquet')
            assert np.array_equal(pred[ID], sr[ID]) and np.array_equal(pred[TARGET], sr[TARGET])
            e2 = (pred.prediction_sec.to_numpy()-y)**2
            daily = pd.DataFrame({'day': day, 'delta': e2-se, 'n': 1}).groupby('day').agg({'delta':'sum','n':'sum'})
            drop_delta = ((e2-se).sum() - daily.delta) / (len(sr)-daily.n)
            rng = np.random.default_rng(20260916)
            draws = rng.integers(0, len(daily), (2000, len(daily)))
            values = daily.delta.to_numpy()[draws].sum(axis=1)/daily.n.to_numpy()[draws].sum(axis=1)
            record = {'rmse_sec': float(np.sqrt(e2.mean())), 'delta_sec': float(np.sqrt(e2.mean())-np.sqrt(se.mean())),
                'delta_mse_day_bootstrap_95': np.quantile(values,[.025,.975]).tolist(),
                'every_day_removal_improves': bool((drop_delta<0).all()),
                'days_improved': int((daily.delta<0).sum()), 'days': len(daily)}
            candidates[arm]['folds'][fold] = record
            candidates[arm]['weighted_mse'] += weight*float(e2.mean())
    total = candidates['control']['weighted_mse']
    for group in ['error_budget', 'source_mismatch', 'airports']:
        for record in aggregates[group].values():
            record['sse_pct'] = 100*record['weighted_mse']/total
    for candidate in candidates.values():
        candidate['seasonal_rmse_sec'] = float(np.sqrt(candidate['weighted_mse']))
        candidate['delta_sec'] = candidate['seasonal_rmse_sec']-np.sqrt(total)
    oracle['seasonal_rmse_sec'] = float(np.sqrt(oracle['weighted_mse']))
    assert source_hashes() == read_json(OUT/'protocol.json')['source_hashes']
    outputs = {'diagnostics': aggregates, 'candidates': candidates, 'fixed_expert_oracle_gate': oracle,
        'frozen_source_unchanged': True, 'input_hashes_reverified_by_load_data': True,
        'analysis_source_sha256': sha256(__file__)}
    write_json(OUT/'analysis.json', outputs)
    print('CANDIDATES', {a: v['seasonal_rmse_sec'] for a,v in candidates.items()})
    print('ERROR_BUDGET', aggregates['error_budget'])
    print('OTHER_DIAGNOSTICS', aggregates['source_mismatch'])
    print('AIRPORT_SSE', {a: r['sse_pct'] for a,r in aggregates['airports'].items()})


if __name__ == '__main__':
    summarize()
