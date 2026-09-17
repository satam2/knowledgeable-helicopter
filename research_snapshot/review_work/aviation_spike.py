"""Private, bounded feature spike. Frozen production and prior runners are untouched."""

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from next230_common import load_data, load_reference, OUT as PRIOR, config_for, WORKSPACE
from next230_clock_gate import gate_features, gate_targets, PARAMS, checked
from taxiout.artifacts import read_json, write_json, sha256, object_hash, utc_now, source_hashes
from taxiout.availability import make_observations, assert_observations
from taxiout.io import training_paths, read_raw, concat_frames
from taxiout.paths import external_path
from taxiout.schema import ID, TARGET, MOVEMENT, PHASE

OUT = external_path(WORKSPACE / 'private_runs/aviation_spike')
ARMS = ['control', 'geometry', 'operations']


def location_priors(x, y, fit_rows, queries, store_path):
    """All label-derived tables use fit rows only, and remain frozen through score."""
    keys = ['ADEP_mvt', 'RUNWAY_mvt', 'STAND_mvt']
    train = x.iloc[fit_rows][keys].astype(str).copy()
    train['y'] = np.asarray(y)[fit_rows]
    query = x.iloc[queries][keys].astype(str)
    global_q20 = float(train.y.quantile(.2))
    global_q50 = float(train.y.median())
    ap = train.groupby(keys[:1], observed=True).y.agg(n='size', q20=lambda v: v.quantile(.2), q50='median')
    ap.to_parquet(store_path / 'airport_priors.parquet')
    a20 = query.ADEP_mvt.map(ap.q20).fillna(global_q20).to_numpy()
    a50 = query.ADEP_mvt.map(ap.q50).fillna(global_q50).to_numpy()
    values = {'physical_airport_q20': a20}
    for name, group in [('runway', keys[:2]), ('stand', [keys[0], keys[2]]), ('pair', keys)]:
        table = train.groupby(group, observed=True).y.agg(n='size', q20=lambda v: v.quantile(.2), q50='median')
        table.to_parquet(store_path / f'{name}_priors.parquet')
        matched = query[group].merge(table.reset_index(), on=group, how='left', sort=False, validate='many_to_one')
        n = matched.n.fillna(0).to_numpy()
        strength = n / (n + 50.)
        for q, parent in [('q20', a20), ('q50', a50)]:
            local = matched[q].to_numpy()
            local = np.where(np.isfinite(local), local, parent)
            values[f'physical_{name}_{q}'] = strength * local + (1 - strength) * parent
        values[f'physical_{name}_support'] = np.log1p(n)
    result = pd.DataFrame(values, index=x.index[queries]).astype('float32')
    proxy = x.iloc[queries].takeoff_minus_AOBT_3_flt.to_numpy()
    result['physical_proxy_excess'] = proxy - result.physical_pair_q20.to_numpy()
    return result


def operations(dep, context):
    """Use only strictly earlier movement observations within the query month."""
    assert_observations(dep)
    assert_observations(context)
    q = pd.DataFrame({'airport': dep.ADEP_mvt.astype(str).to_numpy(),
                      'runway': dep.RUNWAY_mvt.astype('string').fillna('MISSING').to_numpy(),
                      'month': dep[MOVEMENT].dt.strftime('%Y-%m').to_numpy()})
    e = pd.DataFrame({'airport': np.where(context[PHASE].eq('DEP'), context.ADEP_mvt, context.ADES_mvt),
                      'runway': context.RUNWAY_mvt.astype('string').fillna('MISSING').to_numpy(),
                      'month': context[MOVEMENT].dt.strftime('%Y-%m').to_numpy(),
                      'phase': context[PHASE].to_numpy()})
    e.airport = e.airport.astype(str)
    qt = dep[MOVEMENT].dt.as_unit('ns').astype('int64').to_numpy()
    et = context[MOVEMENT].dt.as_unit('ns').astype('int64').to_numpy()
    clock = pd.to_datetime(context.AOBT_3_flt, utc=True, errors='coerce')
    proxy = (context[MOVEMENT] - clock).dt.total_seconds().to_numpy()
    heavy = context.WK_TBL_CAT_flt.isin(['H', 'J']).to_numpy(float)
    columns = {}
    for scope, keys in [('ap', ['airport', 'month']), ('rw', ['airport', 'runway', 'month'])]:
        event_groups = e.groupby([*keys, 'phase'], observed=True).indices
        for phase in ['DEP', 'ARR']:
            prefix = f'op_{scope}_{phase.lower()}'
            counts = {n: np.zeros(len(dep), dtype=float) for n in [5, 15, 30, 60]}
            age = np.full(len(dep), -1.)
            spacing = np.full(len(dep), -1.)
            spread = np.full(len(dep), -1.)
            peers = {n: np.full(len(dep), -1.) for n in [15, 60]}
            support = {n: np.zeros(len(dep)) for n in [15, 60]}
            heavy_share = np.zeros(len(dep))
            runway_change = np.zeros(len(dep))
            for key, positions in q.groupby(keys, observed=True).indices.items():
                ev = event_groups.get((*key, phase))
                if ev is None:
                    continue
                order = ev[np.argsort(et[ev], kind='stable')]
                times = et[order]
                query = qt[positions]
                right = np.searchsorted(times, query, side='left')
                has = right > 0
                age[positions[has]] = (query[has] - times[right[has]-1]) / 1e9
                lefts = {}
                for minutes in counts:
                    lefts[minutes] = np.searchsorted(times, query - minutes * 60 * 10**9, side='left')
                    counts[minutes][positions] = right - lefts[minutes]
                # Spacing among up to eight strictly preceding movements, not query-to-last.
                gaps = np.r_[0., np.diff(times) / 1e9]
                cs, cs2 = np.r_[0., np.cumsum(gaps)], np.r_[0., np.cumsum(gaps**2)]
                start = np.maximum(1, right - 7)
                denom = right - start
                valid = denom > 0
                avg = (cs[right[valid]] - cs[start[valid]]) / denom[valid]
                var = (cs2[right[valid]] - cs2[start[valid]]) / denom[valid] - avg**2
                spacing[positions[valid]], spread[positions[valid]] = avg, np.sqrt(np.maximum(0, var))
                if phase == 'DEP':
                    p = proxy[order]
                    usable = np.isfinite(p) & (p >= 0) & (p <= 7200)
                    pc = np.r_[0., np.cumsum(usable)]
                    ps = np.r_[0., np.cumsum(np.where(usable, p, 0.))]
                    for minutes in peers:
                        left = lefts[minutes]
                        n = pc[right] - pc[left]
                        means = np.divide(ps[right] - ps[left], n, out=np.full(len(n), -1.), where=n > 0)
                        peers[minutes][positions], support[minutes][positions] = means, n
                    hc = np.r_[0., np.cumsum(heavy[order])]
                    denom = right - lefts[15]
                    heavy_share[positions] = np.divide(hc[right] - hc[lefts[15]], denom,
                        out=np.zeros(len(right)), where=denom > 0)
                    if scope == 'ap':
                        runways = e.iloc[order].runway.to_numpy()
                        runway_change[positions[has]] = (runways[right[has]-1] != q.iloc[positions[has]].runway.to_numpy())
            columns[prefix + '_age_sec'] = age
            columns[prefix + '_gap_mean_sec'] = spacing
            columns[prefix + '_gap_sd_sec'] = spread
            for a, b in [(0, 5), (5, 15), (15, 30), (30, 60)]:
                columns[f'{prefix}_{a}_{b}m'] = counts[b] - (counts[a] if a else 0)
            columns[prefix + '_rate_change'] = counts[5] / 5 - (counts[30] - counts[5]) / 25
            if phase == 'DEP':
                for minutes in peers:
                    columns[f'{prefix}_peer_proxy_{minutes}m'] = peers[minutes]
                    columns[f'{prefix}_peer_support_{minutes}m'] = support[minutes]
                columns[prefix + '_heavy_share'] = heavy_share
                if scope == 'ap':
                    columns[prefix + '_last_runway_differs'] = runway_change
    result = pd.DataFrame(columns, index=dep[ID]).astype('float32')
    assert np.isfinite(result.to_numpy()).all()
    return result


def invariants():
    times = pd.to_datetime(['2025-06-01T09:58Z', '2025-06-01T09:59Z', '2025-06-01T10:00Z', '2025-06-01T10:01Z'])
    context = pd.DataFrame({ID: [1, 2, 3, 4], MOVEMENT: times, PHASE: ['DEP']*4,
        'ADEP_mvt': ['TEST']*4, 'ADES_mvt': ['ELSE']*4, 'RUNWAY_mvt': ['09']*4,
        'AOBT_3_flt': times - pd.Timedelta(minutes=10), 'WK_TBL_CAT_flt': ['M']*4})
    dep = context.iloc[[2]]
    original = operations(dep, context)
    categorical = context.copy()
    categorical.RUNWAY_mvt = categorical.RUNWAY_mvt.astype('category')
    categorical.loc[3, 'RUNWAY_mvt'] = np.nan
    pd.testing.assert_frame_equal(original, operations(categorical.iloc[[2]], categorical))
    altered = context.copy()
    altered.loc[[2, 3], 'AOBT_3_flt'] = times[[2, 3]] - pd.Timedelta(hours=1)
    altered.loc[3, 'RUNWAY_mvt'] = '27'
    pd.testing.assert_frame_equal(original, operations(dep, altered))
    assert original.op_ap_dep_0_5m.iloc[0] == 2
    assert original.op_ap_dep_peer_proxy_15m.iloc[0] == 600
    assert original.op_ap_dep_gap_mean_sec.iloc[0] == 60
    tied = context.copy()
    tied.loc[1, MOVEMENT] = times[2]
    assert operations(dep, tied).op_ap_dep_0_5m.iloc[0] == 1
    previous = context.copy()
    previous.loc[0, MOVEMENT] = pd.Timestamp('2025-05-31T23:59Z')
    assert operations(dep, previous).op_ap_dep_0_5m.iloc[0] == 1
    hidden = context.assign(**{TARGET: 600})
    try:
        operations(dep, hidden)
    except ValueError:
        pass
    else:
        raise AssertionError('Hidden target accepted')
    return {'strict_prior_and_self_exclusion': True, 'ties_excluded': True,
            'month_isolation': True, 'hidden_columns_rejected': True,
            'known_peer_mean_and_spacing': True, 'categorical_missing_runway': True}


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = {'created_utc': utc_now(), 'runner_sha256': sha256(__file__), 'arms': ARMS,
        'folds': ['F1', 'F3'], 'params': PARAMS,
        'geometry': 'Fit-period label q20/q50 by airport, runway, stand, pair; alpha=50; frozen through score.',
        'operations': 'Strict-prior movement observations; month-isolated; no current-flight peer contribution.',
        'peer_filter': 'Only peer input duration 0..7200 is summarized; no labels or scored rows removed.',
        'design': 'Same saved experts; tune-month gate training; identical 300-tree depth-3 gates. Two additions vs exact control.',
        'limits': 'Gating probe only; not a full physical model, queue reconstruction, or new external validation.',
        'source_hashes': source_hashes(), 'raw_hashes': read_json(PRIOR/'protocol.json')['raw_hashes'],
        'invariants': invariants()}
    if (OUT/'protocol.json').exists():
        raise ValueError('Existing spike must be inspected; do not overwrite')
    write_json(OUT/'protocol.json', protocol)
    x, meta, labels = load_data()
    columns = [ID, MOVEMENT, PHASE, 'ADEP_mvt', 'ADES_mvt', 'RUNWAY_mvt', 'AOBT_3_flt', 'WK_TBL_CAT_flt']
    context = make_observations(concat_frames([read_raw(p, columns) for p in training_paths(config_for('baseline'))]))[0]
    for fold in ['F1', 'F3']:
        folder = OUT/fold
        folder.mkdir()
        _, _, idx, split = load_reference(fold, meta)
        residual_path = PRIOR/'models'/f'capacity_d8_5000_{fold}_s20260910'
        direct_path = PRIOR/'clock_cpu_experts'/fold
        checked(residual_path, 'manifest.json')
        checked(direct_path, 'expert.json')
        tr, td = pd.read_parquet(residual_path/'tune_predictions.parquet'), pd.read_parquet(direct_path/'tune_direct.parquet')
        assert np.array_equal(tr[ID], td[ID])
        loc = pd.Index(meta[ID]).get_indexer(tr[ID])
        assert (loc >= 0).all() and np.isin(loc, idx['tune']).all()
        assert np.array_equal(td[TARGET], labels.iloc[loc][TARGET])
        sr, sd = pd.read_parquet(residual_path/'score_predictions.parquet'), pd.read_parquet(direct_path/'score_direct.parquet')
        cp = PRIOR/'models'/f'clock_and_rome_ensemble_{fold}_s20260910'
        checked(cp, 'manifest.json')
        champion = pd.read_parquet(cp/'score_predictions.parquet')
        assert np.array_equal(sr[ID], sd[ID]) and np.array_equal(sr[ID], champion[ID])
        assert np.array_equal(sr[ID], meta.iloc[idx['score']][ID])
        tune_residual = meta.iloc[loc].proxy_sec.to_numpy() + tr.prediction.to_numpy()
        base_t = gate_features(x.iloc[loc], tune_residual, td.direct.to_numpy())
        base_s = gate_features(x.iloc[idx['score']], sr.prediction_sec, sd.direct)
        base_t.to_parquet(folder/'tune_base.parquet')
        base_s.to_parquet(folder/'score_base.parquet')
        query = np.r_[loc, idx['score']]
        physical = location_priors(x, labels[TARGET].to_numpy(), idx['fit'], query, folder)
        physical.to_parquet(folder/'geometry.parquet')
        dep = context.loc[context[ID].isin(x.index[query])].set_index(ID, drop=False).loc[x.index[query]].reset_index(drop=True)
        months = dep[MOVEMENT].dt.strftime('%Y-%m').unique()
        local = context.loc[context[MOVEMENT].dt.strftime('%Y-%m').isin(months)]
        ops = operations(dep, local)
        assert np.array_equal(ops.index, physical.index)
        ops.to_parquet(folder/'operations.parquet')
        td.to_parquet(folder/'tune_labels.parquet', index=False)
        score = champion.copy()
        score['residual_expert'] = sr.prediction_sec
        score['direct_expert'] = sd.direct
        score.to_parquet(folder/'score_reference.parquet', index=False)
        write_json(folder/'evidence.json', {'split': split, 'tune_n': len(loc), 'score_n': len(sr),
            'prior_fit_id_hash': object_hash(x.index[idx['fit']].tolist()),
            'prior_fit_n': len(idx['fit']), 'prior_fit_before_tune': bool(meta.iloc[idx['fit']][MOVEMENT].max() < meta.iloc[loc][MOVEMENT].min()),
            'tune_id_hash': object_hash(tr[ID].tolist()), 'score_id_hash': object_hash(sr[ID].tolist()),
            'geometry_pair_support_ge50_pct': float((np.expm1(physical.iloc[len(loc):].physical_pair_support)>=49.999).mean()*100),
            'peer15_available_pct': float((ops.iloc[len(loc):].op_rw_dep_peer_support_15m>0).mean()*100),
            'source_manifest_hashes': {'residual': sha256(residual_path/'manifest.json'), 'direct': sha256(direct_path/'expert.json'), 'champion': sha256(cp/'manifest.json')},
            'files': {p.name: sha256(p) for p in folder.iterdir() if p.is_file()}})
        print(f'PREPARED {fold}: tune={len(loc)}, score={len(sr)}, physical={len(physical.columns)}, operations={len(ops.columns)}', flush=True)
        del physical, ops, dep, base_t, base_s
        gc.collect()
    write_json(OUT/'prepared.json', {'runner_sha256': sha256(__file__), 'complete': True})


def run(fold):
    if read_json(OUT/'protocol.json')['runner_sha256'] != sha256(__file__):
        raise ValueError('Spike source changed after freeze')
    folder = OUT/fold
    evidence = read_json(folder/'evidence.json')
    for name, digest in evidence['files'].items():
        assert sha256(folder/name) == digest
    xt, xs = pd.read_parquet(folder/'tune_base.parquet'), pd.read_parquet(folder/'score_base.parquet')
    td, sr = pd.read_parquet(folder/'tune_labels.parquet'), pd.read_parquet(folder/'score_reference.parquet')
    eligible, target, weights, _ = gate_targets(td[TARGET], xt.gate_residual_prediction, xt.gate_direct_prediction)
    # Float32 gate features are not authoritative expert outputs: use original saved predictions.
    tr = pd.read_parquet(PRIOR/'models'/f'capacity_d8_5000_{fold}_s20260910'/'tune_predictions.parquet')
    tune_y = td[TARGET].to_numpy(float)
    residual = tune_y - tr.target.to_numpy() + tr.prediction.to_numpy()
    eligible, target, weights, _ = gate_targets(tune_y, residual, td.direct.to_numpy())
    reference = sr.prediction_sec.to_numpy(float)
    changed = sr.route.eq('residual').to_numpy()
    result = {}
    for arm in ARMS:
        ft, fs = xt, xs
        if arm != 'control':
            ext = pd.read_parquet(folder/f'{arm}.parquet')
            assert np.array_equal(ext.index[:len(xt)], xt.index) and np.array_equal(ext.index[len(xt):], xs.index)
            ft, fs = pd.concat([xt, ext.iloc[:len(xt)]], axis=1), pd.concat([xs, ext.iloc[len(xt):]], axis=1)
        start = time.monotonic()
        model = CatBoostRegressor(**PARAMS)
        model.fit(ft.iloc[np.flatnonzero(eligible)], target, sample_weight=weights,
                  cat_features=list(ft.select_dtypes('category').columns))
        model.save_model(str(folder/f'{arm}.cbm'))
        gate = np.clip(model.predict(fs, thread_count=4), 0, 1)
        prediction = reference.copy()
        blend = sr.residual_expert.to_numpy() + gate * (sr.direct_expert.to_numpy() - sr.residual_expert.to_numpy())
        prediction[changed] = blend[changed]
        assert np.array_equal(prediction[~changed], reference[~changed])
        saved = CatBoostRegressor().load_model(str(folder/f'{arm}.cbm'))
        assert np.array_equal(gate, np.clip(saved.predict(fs, thread_count=4), 0, 1))
        if arm == 'control':
            delta = float(np.max(np.abs(prediction - reference)))
            if delta > 1e-9:
                raise ValueError(f'Control must reproduce champion exactly, got {delta}')
        error = prediction - sr[TARGET].to_numpy(float)
        record = {'rmse_sec': float(np.sqrt(np.mean(error**2))), 'mae_sec': float(np.mean(np.abs(error))),
            'runtime_sec': time.monotonic()-start, 'rows': len(sr), 'changed_rows': int(changed.sum()),
            'reload_exact': True, 'protected_routes_exact': True,
            'importance': sorted([{'feature': n, 'value': float(v)} for n,v in zip(ft.columns, model.feature_importances_)], key=lambda p:p['value'], reverse=True)[:15]}
        pd.DataFrame({ID: sr[ID], 'prediction_sec': prediction, TARGET: sr[TARGET]}).to_parquet(folder/f'{arm}_predictions.parquet', index=False)
        write_json(folder/f'{arm}_result.json', record)
        result[arm] = record
        print(f'RESULT {fold} {arm}: {record["rmse_sec"]:.6f}s, {record["runtime_sec"]:.1f}s runtime', flush=True)
    write_json(folder/'complete.json', {'arms': list(result), 'source': sha256(__file__), 'completed_utc': utc_now()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'run', 'test'])
    parser.add_argument('--fold', choices=['F1', 'F3'])
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare()
    elif args.action == 'test':
        print(json.dumps(invariants()))
    else:
        run(args.fold)
