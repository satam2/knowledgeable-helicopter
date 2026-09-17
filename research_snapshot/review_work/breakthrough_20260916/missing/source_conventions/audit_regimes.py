"""Fit-only clock-convention regimes and their untouched tune-month transfer."""

import os
for variable in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[variable] = '1'
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / 'review_work/campaign_20260916'))
import common
from taxiout.artifacts import sha256, read_json, write_json, utc_now
from taxiout.paths import external_path
from taxiout.schema import ID, TARGET, CLOCKS

OUT = external_path(ROOT / 'private_runs/breakthrough_20260916/missing/source_conventions')
NAMES = ['nm', 'est', 'init', 'last', 'sched']


def stats(y, p):
    error = np.asarray(p, float) - np.asarray(y, float)
    return {'n': len(error), 'rmse_sec': float(np.sqrt(np.mean(error ** 2))),
            'bias_sec': float(np.mean(error)), 'mae_sec': float(np.mean(np.abs(error)))}


def main():
    marker = read_json(OUT / 'manifest.json')
    assert sha256(OUT / 'features.parquet') == marker['feature_sha256']
    x, meta = common.load_data()
    extension = pd.read_parquet(OUT / 'features.parquet', columns=[ID, 'conv_clock_rank_signature',
        'conv_clock_equality_signature', 'conv_nm_est_gap_bucket', 'conv_nm_second']).set_index(ID)
    assert np.array_equal(extension.index, x.index)
    clocks = np.column_stack([x['takeoff_minus_' + clock].to_numpy(float) for clock in CLOCKS])
    valid = np.isfinite(clocks).all(axis=1) & (clocks != -999999).all(axis=1) & (clocks[:, 0] >= 0) & (clocks[:, 0] <= 7200)
    data = extension.copy()
    data['airport'] = meta.ADEP_mvt.astype('string').to_numpy()
    data['y'] = meta[TARGET].to_numpy(float)
    for i, name in enumerate(NAMES):
        data['proxy_' + name] = clocks[:, i]
        data['error_' + name] = data.y - clocks[:, i]
        data['squared_' + name] = data['error_' + name] ** 2
    report = {'created_utc': utc_now(), 'source_sha256': sha256(__file__), 'feature_manifest_sha256': sha256(OUT / 'manifest.json'),
        'scope': 'Training-derived conditional source costs transferred to original untouched tune month; no score labels used here. Descriptive screen, not model promotion.',
        'eligibility': 'All five clock proxies available and actual proxy0..7200. Original labels unbounded. Groups below200 fit rows fall back to airport proxy mean bias.',
        'folds': {}}
    for fold in ['F1', 'F3']:
        idx, _, _ = common.fold_data(meta, fold, full=True)
        fit = data.iloc[idx['fit'][valid[idx['fit']]]]
        tune = data.iloc[idx['tune'][valid[idx['tune']]]]
        airport_bias = fit.groupby('airport', observed=True).error_nm.mean()
        fallback_bias = tune.airport.map(airport_bias).fillna(fit.error_nm.mean()).to_numpy(float)
        baseline = tune.proxy_nm.to_numpy(float) + fallback_bias
        result = {'fit_n': len(fit), 'tune_n': len(tune), 'tune_airport_mean_corrected_actual': stats(tune.y, baseline), 'regimes': {}}
        for mode, keys in [('order', ['airport', 'conv_clock_rank_signature']),
                           ('order_gap', ['airport', 'conv_clock_rank_signature', 'conv_nm_est_gap_bucket'])]:
            grouped = fit.groupby(keys, observed=True)
            table = grouped.y.agg(n='size')
            for name in NAMES:
                table['bias_' + name] = grouped['error_' + name].mean()
                table['raw_mse_' + name] = grouped['squared_' + name].mean()
                table['centered_mse_' + name] = table['raw_mse_' + name] - table['bias_' + name] ** 2
            table = table.loc[table.n >= 200].copy()
            costs = table[['centered_mse_' + name for name in NAMES]].to_numpy()
            selected = np.argmin(costs, axis=1)
            table['chosen_source'] = np.asarray(NAMES)[selected]
            table['source_bias'] = table[['bias_' + name for name in NAMES]].to_numpy()[np.arange(len(table)), selected]
            table['fit_gain_mse_vs_actual'] = table.centered_mse_nm - costs[np.arange(len(table)), selected]
            mapped = tune.reset_index().merge(table.reset_index(), on=keys, how='left', validate='many_to_one')
            selected_prediction = baseline.copy()
            actual_group = baseline.copy()
            covered = mapped.chosen_source.notna().to_numpy()
            actual_group[covered] = mapped.loc[covered, 'proxy_nm'].to_numpy() + mapped.loc[covered, 'bias_nm'].to_numpy()
            for name in NAMES:
                use = mapped.chosen_source.eq(name).to_numpy()
                selected_prediction[use] = mapped.loc[use, 'proxy_' + name].to_numpy() + mapped.loc[use, 'source_bias'].to_numpy()
            record = {'fit_supported_groups': len(table), 'tune_supported_n': int(covered.sum()),
                'tune_source_selected': stats(tune.y, selected_prediction), 'tune_actual_with_same_group_bias': stats(tune.y, actual_group),
                'chosen_fit_groups': table.chosen_source.value_counts().to_dict(),
                'top_fit_potential_groups': table.assign(fit_total_gain=table.n * table.fit_gain_mse_vs_actual).sort_values('fit_total_gain', ascending=False).head(20).reset_index().to_dict('records')}
            result['regimes'][mode] = record
        report['folds'][fold] = result
    write_json(OUT / 'regime_audit.json', report)
    print({fold: {mode: r['tune_source_selected']['rmse_sec'] for mode,r in result['regimes'].items()} for fold,result in report['folds'].items()}, flush=True)


if __name__ == '__main__':
    main()
