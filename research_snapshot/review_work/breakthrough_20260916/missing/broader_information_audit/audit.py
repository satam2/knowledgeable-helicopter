"""Low-memory independent broader-fold and matched information/capacity audit."""
import os
for k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[k] = '1'
import gc
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT/'review_work/campaign_20260916'))
import common
from taxiout.artifacts import sha256, object_hash, read_json, write_json
ID, TARGET, TIME = common.ID, common.TARGET, common.MOVEMENT
OUT = ROOT/'private_runs/breakthrough_20260916/missing/broader_information_audit'
BASE = ROOT/'private_runs/breakthrough_20260916'
HASHES = {}
pa.set_cpu_count(1)
pa.set_io_thread_count(1)


def sha(path):
    p = str(Path(path).resolve())
    if p not in HASHES:
        HASHES[p] = sha256(p)
    return HASHES[p]


def stats(a, b, mask):
    n = int(mask.sum())
    s1, s2 = float(a[mask].sum()), float(b[mask].sum())
    return dict(n=n, reference_rmse=float(np.sqrt(s1/n)) if n else None,
        candidate_rmse=float(np.sqrt(s2/n)) if n else None,
        reference_sse=s1, candidate_sse=s2, sse_gain=s1-s2)


def days(a, b, dates):
    result = []
    for day in sorted(np.unique(dates)):
        keep = dates != day
        row = stats(a, b, keep)
        row.update(removed_day=str(day), delta_rmse=row['candidate_rmse']-row['reference_rmse'])
        result.append(row)
    return dict(all_improve=all(r['delta_rmse'] < 0 for r in result),
        min_delta=min(r['delta_rmse'] for r in result), max_delta=max(r['delta_rmse'] for r in result), rows=result)


def independent_purges(meta, spec):
    t = pd.to_datetime(meta[TIME], utc=True)
    masks = {s: ((t >= pd.Timestamp(v[0], tz='UTC')) & (t < pd.Timestamp(v[1], tz='UTC'))).to_numpy()
             for s, v in spec.items() if s in ['fit', 'tune', 'refit', 'score']}
    counts = {}
    for stage, against in [('fit','score'),('tune','score'),('refit','score'),('fit','tune')]:
        flights = set(meta.loc[masks[against], 'FLIGHT_ID_mvt'].dropna().tolist())
        remove = masks[stage] & meta.FLIGHT_ID_mvt.isin(flights).to_numpy()
        counts[stage+'_against_'+against] = int(remove.sum())
        masks[stage] &= ~remove
    return {s: np.flatnonzero(m) for s,m in masks.items()}, counts


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    meta_path = ROOT/'private_runs/screening_230/data/interim/audit/departures.parquet'
    assert sha(meta_path) == read_json(ROOT/'private_runs/screening_230/reports/data_audit.json')['artifacts'][meta_path.name]
    meta = pq.read_table(meta_path, columns=[ID,TARGET,TIME,'FLIGHT_ID_mvt','ADEP_mvt','proxy_sec'], use_threads=False).to_pandas()
    finite = np.isfinite(meta.proxy_sec.to_numpy())
    references, indices, splits = {}, {}, {}
    for fold in ['F1','F2','F3','G1']:
        idx, split, _ = common.fold_data(meta, fold, full=True)
        independent, purges = independent_purges(meta, split['spec'])
        assert purges == split['purged_related_departures']
        for stage in idx:
            np.testing.assert_array_equal(idx[stage], independent[stage])
        reference, refrec = common.reference(fold)
        assert object_hash(split) == object_hash(refrec['split'])
        np.testing.assert_array_equal(reference[ID], meta.iloc[idx['score']][ID])
        np.testing.assert_array_equal(reference[TARGET], meta.iloc[idx['score']][TARGET])
        references[fold], indices[fold], splits[fold] = reference, idx, split
    raw_checks = {}
    frozen = read_json(ROOT/'private_runs/submission_v2/protocol.json')
    for month, fold in [('07','F1'),('10','F2'),('11','F3')]:
        path = next((ROOT/'data/09-15-2026-18-55-03_files_list').glob('training_2025-'+month+'-01_*.parquet'))
        assert sha(path) == frozen['raw_hashes'][path.name]
        raw = pq.read_table(path, columns=[ID,TARGET,'PHASE_mvt'], filters=[('PHASE_mvt','=','DEP')], use_threads=False).to_pandas().set_index(ID)
        expected = meta.iloc[indices[fold]['score']]
        np.testing.assert_array_equal(raw.loc[expected[ID], TARGET], expected[TARGET])
        raw_checks[month] = len(expected)
        del raw
    registry = {
        ('600','base'): 'cpu/base/lightgbm_aobt_allfinite_{fold}_s20260916',
        ('600','combined'): 'information_models/conventions__geometry__source_past__source_twosided__surface_T__trajectory__weather_T/lightgbm_aobt_allfinite_{fold}_s20260916',
        ('leaf63','base'): 'deeper_lgb/base/lightgbm_leaf63_aobt_allfinite_{fold}_s20260916',
        ('leaf63','combined'): 'deeper_lgb/combined/lightgbm_leaf63_aobt_allfinite_{fold}_s20260916'}
    rows, manifests, predictions = {}, {}, {}
    for (capacity, info), pattern in registry.items():
        for fold in (['F1','F3','F2','G1'] if (capacity,info)==('leaf63','combined') else ['F1','F3']):
            key = capacity+'_'+info+'_'+fold
            directory = BASE/pattern.format(fold=fold)
            m = read_json(directory/'manifest.json')
            assert m['status'] == 'complete'
            for name,digest in m['outputs'].items():
                assert sha(directory/name) == digest, (key,name)
            protocol_name = m['family']+'_aobt_allfinite_s20260916'
            snapshot = directory.parent/'source_snapshots'/protocol_name
            for name,digest in m['source_hashes'].items():
                assert sha(snapshot/Path(name).name) == digest, (key,name)
            protocol_path = directory.parent/'protocols'/(protocol_name+'.json')
            protocol = read_json(protocol_path)
            assert sha(protocol_path) == m['protocol_sha256']
            for receipt in m['anchor'].get('feature_receipts', []):
                manifest_path = Path(receipt.get('path', receipt.get('manifest')))
                assert sha(manifest_path) == receipt.get('sha256',receipt.get('manifest_sha256'))
                fm = read_json(manifest_path)
                filename = 'features.parquet' if 'feature_sha256' in fm else 'training_features.parquet'
                expected = fm.get('feature_sha256', fm.get('outputs',{}).get(filename))
                assert expected and sha(manifest_path.parent/filename) == expected
                if 'features_sha256' in receipt:
                    assert expected == receipt['features_sha256']
                if 'verification_sha256' in receipt:
                    assert sha(manifest_path.parent/'verification.json') == receipt['verification_sha256']
                feature_ids = pq.read_table(manifest_path.parent/filename, columns=[ID],use_threads=False).column(0).to_numpy()
                np.testing.assert_array_equal(feature_ids,meta[ID])
            idx = indices[fold]
            assert object_hash(splits[fold]) == object_hash(m['split'])
            for stage, position in idx.items():
                eligible = position[finite[position]]
                assert m['fit_ids'][stage] == dict(n=len(eligible),hash=object_hash(meta.iloc[eligible][ID].tolist()))
            assert m['fit']['rows'] == m['fit_ids']['fit']['n']
            assert m['refit']['rows'] == m['fit_ids']['refit']['n']
            assert m['fit']['steps'] == m['refit']['steps']
            tune = pd.read_parquet(directory/'tune_predictions.parquet')
            np.testing.assert_array_equal(tune[ID],meta.iloc[idx['tune'][finite[idx['tune']]]][ID])
            candidate = pd.read_parquet(directory/'candidate.parquet')
            ref = references[fold]
            np.testing.assert_array_equal(candidate[ID],ref[ID])
            np.testing.assert_array_equal(candidate[TARGET],ref[TARGET])
            pred, y = candidate.prediction_sec.to_numpy(), candidate[TARGET].to_numpy()
            assert np.isfinite(pred).all()
            a, b = (ref.prediction_sec.to_numpy()-y)**2, (pred-y)**2
            np.testing.assert_array_equal(b,candidate.squared_error)
            proxy = meta.iloc[idx['score']].proxy_sec.to_numpy()
            missing = ~np.isfinite(proxy)
            np.testing.assert_array_equal(pred[missing],ref.prediction_sec.to_numpy()[missing])
            dates = pd.to_datetime(meta.iloc[idx['score']][TIME],utc=True).dt.strftime('%Y-%m-%d').to_numpy()
            masks = {'missing':missing,'negative':np.isfinite(proxy)&(proxy<0),
                'ordinary':np.isfinite(proxy)&(proxy>=0)&(proxy<=7200),'long':np.isfinite(proxy)&(proxy>7200)}
            gap = np.abs(y-proxy)
            gap_masks = {'missing':missing,'0_to_60':gap<=60,'60_to_300':(gap>60)&(gap<=300),
                '300_to_1800':(gap>300)&(gap<=1800),'1800_to_7200':(gap>1800)&(gap<=7200),'over_7200':gap>7200}
            order = np.argsort(a-b)[::-1]
            removal = {}
            for n in [1,2,5,10]:
                keep = np.ones(len(y),bool)
                keep[order[:n]] = False
                removal[str(n)] = stats(a,b,keep)
            overall = stats(a,b,np.ones(len(y),bool))
            assert np.isclose(overall['candidate_rmse'],m['reports']['candidate']['metrics']['overall']['rmse_sec'],rtol=0,atol=1e-10)
            rows[key] = dict(manifest_sha256=sha(directory/'manifest.json'), overall=overall,
                features=len(m['feature_columns']),fit_ids=m['fit_ids'],steps=m['fit']['steps'],
                purges=splits[fold]['purged_related_departures'],all_output_hashes_pass=True,
                saved_replay_delta=m['reload_max_abs_delta'],protocol_lists_fold=fold in protocol['folds'],
                day_removals=days(a,b,dates),support={k:stats(a,b,v) for k,v in masks.items()},
                absolute_source_gap={k:stats(a,b,v) for k,v in gap_masks.items()},
                largest_gain_row_removals=removal,missing_sse_share=float(b[missing].sum()/b.sum()))
            predictions[key] = b
            manifests[key] = m
            print('AUDITED',key,overall['candidate_rmse'],rows[key]['day_removals']['all_improve'],flush=True)
            gc.collect()
    table = pd.read_csv(BASE/'information_capacity_1100/two_by_two.csv')
    matched = {}
    for capacity in ['600','leaf63']:
        fold_results = {}
        seasonal = {}
        for info in ['base','combined']:
            mse = {f:rows[capacity+'_'+info+'_'+f]['overall']['candidate_sse']/rows[capacity+'_'+info+'_'+f]['overall']['n'] for f in ['F1','F3']}
            seasonal[info] = float(np.sqrt((192122*mse['F1']+152719*mse['F3'])/344841))
            selected = table[(table.capacity==('600trees31leaves' if capacity=='600' else '2500trees63leaves'))&(table.information==('base30' if info=='base' else 'combined225'))]
            assert len(selected)==1 and np.isclose(seasonal[info],selected.seasonal_rmse.iloc[0],rtol=0,atol=1e-10)
        for fold in ['F1','F3']:
            k1,k2=capacity+'_base_'+fold,capacity+'_combined_'+fold
            m1,m2=manifests[k1],manifests[k2]
            assert m1['fit_ids']==m2['fit_ids']
            assert m1['fit']['params']==m2['fit']['params']
            assert set(m1['feature_columns']).issubset(m2['feature_columns'])
            a,b=predictions[k1],predictions[k2]
            dates=pd.to_datetime(meta.iloc[indices[fold]['score']][TIME],utc=True).dt.strftime('%Y-%m-%d').to_numpy()
            fold_results[fold]=dict(overall=stats(a,b,np.ones(len(a),bool)),day_removals=days(a,b,dates))
        matched[capacity]=dict(seasonal_rmse=seasonal,information_gain=seasonal['base']-seasonal['combined'],folds=fold_results)
    report=dict(status='passed_with_provenance_limits',source_sha256=sha(__file__),rows=rows,matched=matched,
        raw_score_labels_checked=raw_checks,score_fold_F2_G1_same_ids=True,
        current_encoder_sha256=sha(ROOT/'review_work/campaign_20260916/lgbm_adapter.py'),
        current_encoder_path=str(ROOT/'review_work/campaign_20260916/lgbm_adapter.py'),
        limits=['deeper runner omits imported lgbm_adapter.py from original frozen source hashes; current hash is not historical proof',
            'F2/G1 runtime manifests use shared original protocol that lists only F1/F3; actual fold splits/IDs audited independently',
            'all folds development-exposed; F2/G1 share October score rows; no fresh holdout',
            'combined source_twosided/surface/trajectory fields use retrospective supplied final batch; no strict live availability claim',
            'saved replay receipts verified, no new model inference or fitting',
            'source gap uses absolute Y-minus-proxy (NM offblock minus airport offblock); diagnostic only, not a feature or exclusion'])
    write_json(OUT/'audit.json',report)
    print('COMPLETE',json.dumps({k:v['seasonal_rmse'] for k,v in matched.items()}),flush=True)


if __name__ == '__main__':
    main()
