"""Label-free ranking drift, prediction changes, and descriptive mix reweighting."""
import os
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '2'
import lightgbm
from datetime import datetime, timezone
import math
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
import pyarrow as pa
import psutil

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'review_work/campaign_20260916'))
import common
ID, TARGET = common.ID, common.TARGET
OUT = common.external_path(ROOT / 'private_runs/tail240_20260916/score_gap/drift/v1')
REPORT = common.external_path(ROOT / 'output/tail240_20260916/score_gap/drift')
OLD = ROOT / 'private_runs/submission_v2'
NEW = ROOT / 'private_runs/tail240_20260916/final_submission_v3_v2'
LOCAL = ROOT / 'private_runs/tail240_20260916/models/neural_missing_integration_v1/replacement'
PRIOR = ROOT / 'private_runs/breakthrough_20260916/retrospective_research/validation_transfer/aggregate.json'
PRIOR_SOURCE = ROOT / 'review_work/breakthrough_20260916/models_retrieval/retrospective_research/transfer_audit.py'
WEIGHTS = {'F1': 192122 / 344841, 'F3': 152719 / 344841}
pa.set_cpu_count(2)
pa.set_io_thread_count(1)
PEAK = 0


def guard():
    global PEAK
    PEAK = max(PEAK, psutil.Process().memory_info().rss)
    if PEAK > 4 * 1024**3 or psutil.virtual_memory().available < 8 * 1024**3:
        raise MemoryError('Drift audit resource budget exceeded')


def route(proxy):
    proxy = np.asarray(proxy, dtype=float)
    return np.select([~np.isfinite(proxy), (proxy >= 0) & (proxy <= 7200)],
                     ['missing', 'ordinary'], default='finite_nonordinary')


def describe(values):
    a = np.asarray(values, float)
    return dict(rows=len(a), mean=float(a.mean()), rms=float(np.sqrt(np.mean(a*a))),
                mean_absolute=float(np.abs(a).mean()), minimum=float(a.min()), maximum=float(a.max()),
                negative=int((a < 0).sum()), quantiles=dict(zip(['p01', 'p50', 'p90', 'p99', 'p999'],
                np.quantile(a, [.01, .5, .9, .99, .999]).tolist())))


def change_groups(frame, columns):
    total = frame.delta.pow(2).sum()
    rows = []
    for keys, group in frame.groupby(columns, observed=True, sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        record = dict(zip(columns, map(str, keys)))
        record.update(describe(group.delta))
        record['squared_change_share'] = float(group.delta.pow(2).sum() / total)
        record['v2_negative'] = int((group.v2 < 0).sum())
        record['v3_negative'] = int((group.v3 < 0).sum())
        rows.append(record)
    return rows


def main():
    start = time.monotonic()
    guard()
    OUT.mkdir(parents=True, exist_ok=False)
    REPORT.mkdir(parents=True, exist_ok=True)
    prior = common.read_json(PRIOR)
    assert common.sha256(PRIOR_SOURCE) == prior['source_sha256']
    raw_protocol = common.read_json(OLD / 'protocol.json')
    raw_receipts = {x['file']: x['sha256'] for x in prior['raw_receipts']}
    raw_bound = {}
    for name in ['ranking.parquet', 'training_2025-07-01_2025-08-01.parquet', 'training_2025-11-01_2025-12-01.parquet']:
        digest = common.sha256(ROOT / 'data/09-15-2026-18-55-03_files_list' / name)
        assert digest == raw_protocol['raw_hashes'][name] == raw_receipts[name]
        raw_bound[name] = digest
    receipts = [common.read_json(folder / 'submission_ready.json') for folder in [OLD, NEW]]
    pred_paths = [folder / 'ranking_predictions.parquet' for folder in [OLD, NEW]]
    pred_hashes = [common.sha256(path) for path in pred_paths]
    assert pred_hashes[0] == receipts[0]['ranking_predictions_sha256']
    assert pred_hashes[1] == receipts[1]['prediction_sha256']
    a = pd.read_parquet(pred_paths[0], columns=[ID, 'ADEP_mvt', common.MOVEMENT, 'proxy_sec', 'schedule_sec', 'prediction_sec', 'route'])
    b = pd.read_parquet(pred_paths[1], columns=[ID, 'proxy_sec', 'prediction_sec', 'route'])
    assert a[ID].is_unique and b[ID].is_unique and len(a) == len(b) == 344841
    b = b.set_index(ID).loc[a[ID]].reset_index()
    np.testing.assert_array_equal(a[ID], b[ID])
    np.testing.assert_array_equal(a.proxy_sec, b.proxy_sec)
    a['v2'], a['v3'] = a.prediction_sec, b.prediction_sec
    a['route'] = route(a.proxy_sec)
    np.testing.assert_array_equal(a.route, b.route)
    a['month'] = a[common.MOVEMENT].dt.strftime('%Y-%m')
    a['airport'] = a.ADEP_mvt.astype(str)
    a['delta'] = a.v3 - a.v2
    assert np.isfinite(a[['v2', 'v3', 'delta']]).all().all()
    changes = dict(overall=describe(a.delta), v2=describe(a.v2), v3=describe(a.v3),
                   rounded_v2=describe(np.rint(a.v2)), rounded_v3=describe(np.rint(a.v3)))
    for cols in [['route'], ['month'], ['airport'], ['month', 'route'], ['airport', 'route'], ['month', 'airport', 'route']]:
        changes['_'.join(cols)] = change_groups(a, cols)
    order = np.argsort(-np.abs(a.delta.to_numpy()), kind='stable')
    n = math.ceil(.01 * len(a))
    top = a.iloc[order[:n]]
    changes['largest_one_percent'] = dict(rows=n, minimum_absolute_change=float(top.delta.abs().min()),
        absolute_change_share=float(top.delta.abs().sum()/a.delta.abs().sum()),
        squared_change_share=float(top.delta.pow(2).sum()/a.delta.pow(2).sum()),
        groups=change_groups(top, ['month', 'airport', 'route']))
    guard()
    manifest = common.read_json(LOCAL / 'manifest.json')
    binding = common.read_json(ROOT / 'private_runs/tail240_20260916/validation/baseline_binding.json')
    reweighted = {}
    all_cells = []
    for fold, rank_month, local_month in [('F1', '2026-07', '2025-07'), ('F3', '2026-01', '2025-11')]:
        ref, refmark = common.reference(fold)
        path = LOCAL / f'{fold}.parquet'
        assert common.sha256(path) == manifest['folds'][fold]['prediction']['sha256']
        pred = pd.read_parquet(path, columns=[ID, 'prediction_sec'])
        np.testing.assert_array_equal(ref[ID], pred[ID])
        assert common.object_hash(ref[ID].tolist()) == binding['folds'][fold]['cohorts']['all']['score']['hash']
        ref['airport'], ref['route'] = ref.ADEP_mvt.astype(str), route(ref.proxy_sec)
        ref['mse'] = (pred.prediction_sec.to_numpy() - ref[TARGET].to_numpy())**2
        rank = a.loc[a.month.eq(rank_month)]
        assert len(ref) == prior['records'][local_month]['rows']
        local_counts = ref.groupby(['airport', 'route'], observed=True).size()
        rank_counts = rank.groupby(['airport', 'route'], observed=True).size()
        conditional = ref.groupby(['airport', 'route'], observed=True).mse.mean()
        cells = []
        for airport, rt in sorted(set(local_counts.index) | set(rank_counts.index)):
            key = (airport, rt)
            nr, nl = int(rank_counts.get(key, 0)), int(local_counts.get(key, 0))
            # No fallback is invented for a ranking group with no local support.
            assert nr == 0 or nl > 0, f'Unsupported ranking stratum: {fold} {key}'
            mse = float(conditional.get(key, 0))
            delta_mse = (nr/len(rank) - nl/len(ref))*mse
            cell = dict(fold=fold, local_month=local_month, ranking_month=rank_month, airport=airport, route=rt,
                        local_rows=nl, ranking_rows=nr, local_share=nl/len(ref), ranking_share=nr/len(rank),
                        local_conditional_mse=mse, mix_delta_mse=delta_mse,
                        seasonal_mix_delta_mse=WEIGHTS[fold]*delta_mse)
            cells.append(cell)
        local_mse = float(ref.mse.mean())
        new_mse = sum(c['ranking_share']*c['local_conditional_mse'] for c in cells)
        assert abs(sum(c['local_share']*c['local_conditional_mse'] for c in cells)-local_mse) < 1e-7
        reweighted[fold] = dict(local_rmse=math.sqrt(local_mse), reweighted_rmse=math.sqrt(new_mse),
                               local_mse=local_mse, reweighted_mse=new_mse,
                               prediction_sha256=common.sha256(path), reference_outputs=refmark['outputs'])
        all_cells.extend(cells)
    oldmse = sum(WEIGHTS[f]*v['local_mse'] for f,v in reweighted.items())
    newmse = sum(WEIGHTS[f]*v['reweighted_mse'] for f,v in reweighted.items())
    assert abs(math.sqrt(oldmse)-268.662991126314) < 1e-8
    reweighted['seasonal'] = dict(local_rmse=math.sqrt(oldmse), reweighted_rmse=math.sqrt(newmse),
                                  mix_only_rmse_delta=math.sqrt(newmse)-math.sqrt(oldmse))
    common.write_json(OUT / 'airport_route_cells.json', all_cells)
    pd.DataFrame(all_cells).to_csv(OUT / 'airport_route_cells.csv', index=False)
    guard()
    result = dict(status='complete', created_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=common.sha256(__file__), raw_hashes_verified=raw_bound,
        prior_aggregate_sha256=common.sha256(PRIOR), prior_source_sha256=common.sha256(PRIOR_SOURCE),
        ranking_prediction_hashes=dict(v2=pred_hashes[0], v3=pred_hashes[1]),
        reused_drift_comparisons=prior['comparisons'], prediction_changes=changes, composition_diagnostic=reweighted,
        airport_route_cells_sha256=common.sha256(OUT / 'airport_route_cells.json'), peak_observed_rss=PEAK,
        duration_sec=time.monotonic()-start, limitations=[
            'Ranking target and departure block columns never read; raw files only hashed in this continuation.',
            'Prediction changes are not prediction errors and do not identify which ranking rows improved.',
            'Composition reweighting assumes unchanged local conditional MSE within airport/route/season; not an official score forecast or causal attribution.',
            'Finite nonordinary combines negative and above7200 proxies; bins in reused audit are left-closed and not exact routing boundaries.',
            'Local folds are exposed development data; January2026 uses November2025 as winter proxy.',
            'Marginal covariate shifts cannot identify hidden conditional error shift.'])
    common.write_json(OUT / 'receipt.json', result)
    print(common.read_json(OUT / 'receipt.json')['composition_diagnostic']['seasonal'], flush=True)
    print(changes['overall'], flush=True)
    print(changes['largest_one_percent']['squared_change_share'], flush=True)


if __name__ == '__main__':
    main()
