"""Fixed missing-clock component ablations from saved score predictions only."""
import os
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[key] = '2'
import lightgbm
import argparse
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
import psutil
import pyarrow as pa

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'review_work/campaign_20260916'))
import common
ID, TARGET = common.ID, common.TARGET
OUT = common.external_path(ROOT / 'private_runs/tail240_20260916/robustness/tail/v1')
V3 = ROOT / 'private_runs/tail240_20260916/models/neural_missing_integration_v1/replacement'
NORMAL = ROOT / 'private_runs/tail240_20260916/models/normalized_missing_refit_v2/observed_schedule_scale_candidate'
FOREST = ROOT / 'private_runs/breakthrough_20260916/models/missing_forest'
BINDING = ROOT / 'private_runs/tail240_20260916/validation/baseline_binding.json'
WEIGHTS = {'F1': 192122/344841, 'F3': 152719/344841}
ARMS = {'current': '.5625 V2 + .1875 ExtraTrees + .25 normalized',
        'remove_normalized': '.75 V2 + .25 ExtraTrees',
        'remove_ExtraTrees': '.75 V2 + .25 normalized',
        'remove_both': 'V2'}
pa.set_cpu_count(2)
pa.set_io_thread_count(1)


def inputs():
    sources = {}
    for fold in WEIGHTS:
        sources[fold] = {}
        for name, folder, filename in [('current', V3, fold+'.parquet'), ('normalized', NORMAL, fold+'.parquet'),
             ('forest_blend25', FOREST/f'extratrees_missing_template_idcontext_{fold}_s20260916', 'blend25.parquet')]:
            marker = common.read_json(folder/'manifest.json')
            expected = marker['outputs'][filename] if name == 'forest_blend25' else marker['folds'][fold]['prediction']['sha256']
            assert common.sha256(folder/filename) == expected
            sources[fold][name] = dict(path=str(folder/filename), sha256=expected,
                                       manifest_sha256=common.sha256(folder/'manifest.json'))
    return sources


def declare():
    OUT.mkdir(parents=True, exist_ok=True)
    declaration = dict(source_sha256=common.sha256(__file__), arms=ARMS, sources=inputs(),
        baseline_binding_sha256=common.sha256(BINDING), weights=WEIGHTS,
        scope='Only missing NM; finite V3 predictions bit-identical. Original complete F1/F3 score rows.',
        selection='Fixed ablations only, no coefficient search, new fit, clipping, ranking outcomes, or promotion.',
        influence='Paired MSE and RMSE versus current; remove1/2/5/10 largest beneficial rows per fold and each UTC day; both helpful and harmful top10 rows retained.',
        exposure='Post-selection descriptive ablations on exposed development folds, not fresh confirmation.')
    path = OUT/'protocol.json'
    if path.exists():
        assert common.read_json(path)['declaration'] == declaration
    else:
        common.write_json(path, dict(created_utc=common.utc_now(), declaration=declaration,
                                   declared_before_this_run_score_reads=True))
    return declaration


def metrics(y, p):
    e = p-y
    return dict(rows=len(y), mse=float(np.mean(e*e)), rmse=float(np.sqrt(np.mean(e*e))),
                mae=float(np.abs(e).mean()), bias=float(e.mean()))


def run():
    started = time.monotonic()
    protocol = declare()
    assert not (OUT/'summary.json').exists()
    binding = common.read_json(BINDING)
    results, replays, peaks = {}, {}, []
    for fold in WEIGHTS:
        rss = psutil.Process().memory_info().rss
        peaks.append(rss)
        assert rss < 4*1024**3 and psutil.virtual_memory().available > 8*1024**3
        ref, refmarker = common.reference(fold)
        assert common.object_hash(ref[ID].tolist()) == binding['folds'][fold]['cohorts']['all']['score']['hash']
        arr = {}
        for name, record in protocol['sources'][fold].items():
            frame = pd.read_parquet(record['path'], columns=[ID, 'prediction_sec'])
            np.testing.assert_array_equal(frame[ID], ref[ID])
            arr[name] = frame.prediction_sec.to_numpy(float)
        missing = ~np.isfinite(ref.proxy_sec.to_numpy(float))
        y, base = ref[TARGET].to_numpy(float), ref.prediction_sec.to_numpy(float)
        expected = arr['forest_blend25'][missing] + .25*(arr['normalized'][missing]-arr['forest_blend25'][missing])
        np.testing.assert_allclose(expected, arr['current'][missing], atol=1e-9, rtol=0)
        pred = {name:arr['current'].copy() for name in ARMS}
        pred['remove_normalized'][missing] = arr['forest_blend25'][missing]
        pred['remove_ExtraTrees'][missing] = base[missing] + .25*(arr['normalized'][missing]-base[missing])
        pred['remove_both'][missing] = base[missing]
        days = ref[common.MOVEMENT].dt.strftime('%Y-%m-%d').to_numpy()
        replay = pd.DataFrame({ID:ref[ID], TARGET:y, 'airport':ref.ADEP_mvt.astype(str),
                               'day':days, 'missing':missing, 'v2':base, **pred})
        replay.to_parquet(OUT/f'{fold}_replay.parquet', index=False)
        current_e2 = (pred['current']-y)**2
        result = {}
        for name,p in pred.items():
            assert np.isfinite(p).all()
            np.testing.assert_array_equal(p[~missing], pred['current'][~missing])
            e2 = (p-y)**2
            gain = current_e2-e2
            order = np.flatnonzero(missing)[np.argsort(-gain[missing], kind='stable')]
            bad = np.flatnonzero(missing)[np.argsort(gain[missing], kind='stable')]
            influence = {}
            for k in [1,2,5,10]:
                keep = np.ones(len(y), bool)
                keep[order[:k]] = False
                influence[str(k)] = dict(current=metrics(y[keep], pred['current'][keep]),
                    ablation=metrics(y[keep], p[keep]), removed_sse_gain=float(gain[order[:k]].sum()),
                    removed_ids=ref.iloc[order[:k]][ID].tolist())
            daily = []
            for day in sorted(set(days)):
                keep = days != day
                daily.append(dict(day=day, rows_removed=int((~keep).sum()),
                    current_mse=float(current_e2[keep].mean()), ablation_mse=float(e2[keep].mean()),
                    removed_sse_gain=float(gain[~keep].sum()),
                    paired_mse_gain=float(gain[keep].mean()),
                    paired_rmse_gain=float(np.sqrt(current_e2[keep].mean())-np.sqrt(e2[keep].mean()))))
            def rows(idx):
                return [dict(id=float(ref.iloc[i][ID]), airport=str(ref.iloc[i].ADEP_mvt), day=str(days[i]),
                             target=float(y[i]), current=float(pred['current'][i]), ablation=float(p[i]),
                             sse_gain=float(gain[i])) for i in idx]
            result[name] = dict(all=metrics(y,p), missing=metrics(y[missing],p[missing]),
                rome_missing=metrics(y[missing & ref.ADEP_mvt.eq('LIRF').to_numpy()],p[missing & ref.ADEP_mvt.eq('LIRF').to_numpy()]),
                paired_mse_gain=float(gain.mean()), paired_rmse_gain=float(np.sqrt(current_e2.mean())-np.sqrt(e2.mean())),
                prediction_range=[float(p.min()),float(p.max())], negative_predictions=int((p<0).sum()),
                top_beneficial=rows(order[:10]), top_harmful=rows(bad[:10]), remove_top_beneficial=influence,
                day_removals=daily, every_day_removal_improves=all(d['paired_mse_gain']>0 for d in daily))
        results[fold] = result
        replays[fold] = dict(sha256=common.sha256(OUT/f'{fold}_replay.parquet'), reference_outputs=refmarker['outputs'],
                            current_reconstruction_max_abs=float(np.max(np.abs(expected-arr['current'][missing]))),
                            missing_target_negative=int((y[missing]<0).sum()), score_id_hash=common.object_hash(ref[ID].tolist()))
        peaks.append(psutil.Process().memory_info().rss)
    seasonal = {}
    current_mse = sum(WEIGHTS[f]*results[f]['current']['all']['mse'] for f in WEIGHTS)
    for name in ARMS:
        mse = sum(WEIGHTS[f]*results[f][name]['all']['mse'] for f in WEIGHTS)
        removal = {}
        for k in ['1','2','5','10']:
            cur = sum(WEIGHTS[f]*results[f][name]['remove_top_beneficial'][k]['current']['mse'] for f in WEIGHTS)
            alt = sum(WEIGHTS[f]*results[f][name]['remove_top_beneficial'][k]['ablation']['mse'] for f in WEIGHTS)
            removal[k] = dict(current_rmse=float(np.sqrt(cur)), ablation_rmse=float(np.sqrt(alt)), gain=float(np.sqrt(cur)-np.sqrt(alt)))
        seasonal[name] = dict(rmse=float(np.sqrt(mse)), mse=mse,
            mse_gain_vs_current=current_mse-mse, rmse_gain_vs_current=float(np.sqrt(current_mse)-np.sqrt(mse)),
            remove_k_most_beneficial_per_fold=removal)
    assert abs(seasonal['current']['rmse']-268.662991126314)<1e-8
    common.write_json(OUT/'summary.json', dict(status='complete', protocol_sha256=common.sha256(OUT/'protocol.json'),
        source_sha256=common.sha256(__file__), folds=results, seasonal=seasonal, replay_bindings=replays,
        max_sampled_rss=max(peaks), duration_sec=time.monotonic()-started,
        positive_gain_definition='Current error minus ablation error; positive favors removing the component.',
        limitation='Exposed score cohorts, descriptive component dependence, no independent generalization or ranking guarantee.'))
    print(seasonal, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--declare-only', action='store_true')
    args = parser.parse_args()
    if args.declare_only:
        declare()
        print('FIXED_FOUR_ARM_PROTOCOL_DECLARED', flush=True)
    else:
        run()
