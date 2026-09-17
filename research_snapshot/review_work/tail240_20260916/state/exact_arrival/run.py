"""Staged label-free exact-arrival join and bounded source-offset association screen."""
import os
for variable in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[variable] = '1'
import argparse
import gc
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import psutil
from threadpoolctl import threadpool_limits
import join
import association_stats as stats

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'review_work/campaign_20260916'))
import common
ID, TARGET, TIME = common.ID, common.TARGET, common.MOVEMENT
OUT = common.external_path(ROOT / 'private_runs/tail240_20260916/state/exact_arrival/v1')
RAW = common.RAW
META = ROOT / 'private_runs/screening_230/data/interim/audit/departures.parquet'
FOLDS = {'F1':('2025-06-01', '2025-07-01'), 'F3':('2025-10-01', '2025-11-01')}


def union_folder(fold):
    return ROOT / 'private_runs/breakthrough_20260916/deeper_context_union' / f'lightgbm_leaf63_sequence_aobt_allfinite_{fold}_s20260916'


def guard():
    info = psutil.Process().memory_info()
    peak = getattr(info, 'peak_wset', info.rss)
    available = psutil.virtual_memory().available
    assert max(info.rss, peak) < 2*1024**3, f'2GiB process budget {info}'
    assert available >= 8*1024**3, '8GiB host reserve'
    return dict(rss_bytes=info.rss, peak_wset_bytes=peak, available_bytes=available)


def declare():
    paths = [Path(__file__), Path(join.__file__), Path(stats.__file__), Path(common.__file__),
             Path(__file__).with_name('test_join.py'), Path(__file__).with_name('test_statistics.py')]
    frozen = common.read_json(ROOT / 'private_runs/submission_v2/protocol.json')
    controls = {}
    for fold in FOLDS:
        folder = union_folder(fold)
        marker = common.read_json(folder / 'manifest.json')
        assert marker['status'] == 'complete' and len(marker['feature_columns']) == 387
        controls[fold] = dict(manifest_sha256=common.sha256(folder / 'manifest.json'),
            predictions_sha256=marker['outputs']['tune_predictions.parquet'],
            fit_model_sha256=marker['outputs']['fit_model.joblib'], fit_ids=marker['fit_ids'], split=marker['split'])
    value = dict(source_hashes={str(p.relative_to(ROOT)):common.sha256(p) for p in paths},
        metadata_sha256='503df9cb08f3ed7260fd7d969a3505d7b58434e20bccc80f1526d517a292e976',
        raw_hashes={k:v for k,v in frozen['raw_hashes'].items() if k.startswith('training_')},
        controls=controls, folds=FOLDS,
        source='All12 training packs scanned perlogicalJune/Octmonth with Arrow phase/month predicates; no ranking or external data',
        projection=dict(DEP=join.DEP_COLUMNS, ARR=join.ARR_COLUMNS),
        primitive='delta_sec=ownARVT3-linkedairportlanding; noARRblock/ARRlabel/DEPblock read',
        join='ExactnonnullNMFLIGHT_ID andsameUTCmovementmonth; matchingknownorigin+destination; uniqueARRcounterpart afterdedup; landingstrictlyafterqueryT; sameMVT_IDrejected; ownARVTfinite',
        dedup='Whitelist-identical duplicate movement IDs collapse; conflicting IDs fail. Sameflight/origin/destination/landing duplicateARRrecords collapse tosmallestMVT_ID beforeambiguity. AnyothermultipleARRcounterpart withinmonth invalidatesflight, evenifoneroutematches.',
        cohort='Everyoriginalunion387fittuneID first; ordinary0<=proxy<=7200 retainedcompletely. No target-basedeligibility or clipping. Month/logicalrawquerycoverage asserted.',
        label_order='Separatebuild-joins action freezes bothlabel-free joinParquets/coverage/sourceoracles/joinmanifest beforeassociate canload ONLYJune/Octoberquerylabels. Label metadata projection isID/TARGET with queryIDpredicate.',
        baseline='Savedunion387fitmodeltune predictions, neverglobal9same-tuneweights. F1fitJanMay/F3fitJanSep; theseareearlierfitmodelsbuttune-developmentexposed.',
        calibration='June1-15linkedrows only; originmeansofe=Y-union387 anddelta; globalmeanfallback; populationdeltaSTD fromcalibration (zero=>1); withinorigincenteredstandarddelta tocenterede, ONEpooled Ridgealpha100/nointercept/cholesky',
        arms=['unchangedunion387', 'linkedonlyoriginbias', 'identicaloriginbias+ONEdeltaSlope'],
        evaluation='June16-30+allOctober separately, fullordinaryandlinkedonly; Octoberchangesnuisancefitsoassociation/transferdiagnostic, notfullfoldqualification',
        statistics='RawSSE/RMSE/MAE/bias,paired300daybootstrapseed20260916,everydayremoval,top1/5/10beneficialremoval,airport/daylinksupport',
        permutation='25fixedseed20260916 delta permutations withinorigin/destination/operator/day linkedrows, missingoperatorretainedasnullgroup, fixedcoefficient, report singletonandunchangedshares',
        screen='Informativeonlyifdelta vs identicalbias improvesbothperiods/everydayremoval and95%pairedMSElowerbound>0. No scores/refit/ranking/promotion or variant grid authorized.',
        resources='1CPU,2GiB currentRSS+historicalpeak,8GiB hostreserve,noGPU/network; periodicnotOSenforced',
        oracle='Atleast64fixedseedrawdeparturequeriespermonth plusup to4perstatus; independently enumerate rawARRcounterparts usingonlywhitelist; exactstatus/delta mustmatch')
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'protocol.json'
    if path.exists():
        assert common.object_hash(common.read_json(path)) == common.object_hash(value), 'Frozen source/protocol changed'
    else:
        common.write_json(path, value)
    return value


def month_rows(path, phase, start, end):
    columns = join.DEP_COLUMNS if phase == 'DEP' else join.ARR_COLUMNS
    source = ds.dataset(path, format='parquet')
    dtype = source.schema.field(TIME).type
    lower = pa.scalar(pd.Timestamp(start, tz='UTC').to_pydatetime(), type=dtype)
    upper = pa.scalar(pd.Timestamp(end, tz='UTC').to_pydatetime(), type=dtype)
    predicate = (ds.field(join.PHASE) == phase) & (ds.field(TIME) >= lower) & (ds.field(TIME) < upper)
    scanner = source.scanner(columns=columns, filter=predicate, batch_size=16384,
        batch_readahead=1, fragment_readahead=1, use_threads=False)
    frames = []
    for batch in scanner.to_batches():
        frames.append(batch.to_pandas())
        guard()
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)


def build_joins():
    protocol = declare()
    assert not (OUT / 'join_manifest.json').exists(), 'Preserve frozen joins'
    started = time.monotonic()
    assert common.sha256(META) == protocol['metadata_sha256']
    meta = pd.read_parquet(META, columns=[ID, TIME, 'proxy_sec']).set_index(ID)
    assert meta.index.is_unique
    rawfiles = sorted(RAW.glob('training_*.parquet'))
    assert len(rawfiles) == 12 and set(p.name for p in rawfiles) == set(protocol['raw_hashes'])
    for path in rawfiles:
        assert common.sha256(path) == protocol['raw_hashes'][path.name]
    records = {}
    outputs = {}
    for fold, (start, end) in FOLDS.items():
        folder = union_folder(fold)
        assert common.sha256(folder / 'manifest.json') == protocol['controls'][fold]['manifest_sha256']
        assert common.sha256(folder / 'tune_predictions.parquet') == protocol['controls'][fold]['predictions_sha256']
        ids = pd.read_parquet(folder / 'tune_predictions.parquet', columns=[ID])[ID]
        assert ids.is_unique
        assert common.object_hash(ids.tolist()) == protocol['controls'][fold]['fit_ids']['tune']['hash']
        query = meta.loc[ids].copy()
        assert np.isfinite(query.proxy_sec).all()
        ordinary_ids = pd.Index(query.loc[query.proxy_sec.between(0, 7200)].index)
        dep_parts, arr_parts, reads = [], [], []
        for path in rawfiles:
            dep = month_rows(path, 'DEP', start, end)
            arr = month_rows(path, 'ARR', start, end)
            reads.append(dict(file=path.name, logical_dep_rows=len(dep), logical_arr_rows=len(arr)))
            if len(dep):
                dep_parts.append(dep)
            if len(arr):
                arr_parts.append(arr)
            guard()
        dep = join.deduplicate(pd.concat(dep_parts, ignore_index=True), 'DEP')
        arr_raw = pd.concat(arr_parts, ignore_index=True)
        del dep_parts, arr_parts
        dep = dep.set_index(ID).loc[ordinary_ids].reset_index()
        assert np.array_equal(dep[ID], ordinary_ids)
        pd.testing.assert_series_equal(dep[TIME].reset_index(drop=True), meta.loc[ordinary_ids, TIME].reset_index(drop=True), check_dtype=False, check_names=False)
        linked = join.link(dep, arr_raw)
        rng = np.random.default_rng(stats.SEED)
        selected = list(rng.choice(len(dep), min(64, len(dep)), replace=False))
        for status in sorted(linked.status.unique()):
            positions = np.flatnonzero(linked.status.eq(status))
            selected.extend(positions[:4])
        oracles = []
        for index in sorted(set(selected)):
            status, delta = join.oracle(dep.iloc[index], arr_raw)
            actual = linked.iloc[index]
            assert actual.status == status
            np.testing.assert_allclose(actual.delta_sec, delta, rtol=0, atol=0, equal_nan=True)
            oracles.append(dict(query_id=float(actual[ID]), status=status, delta_sec=None if not np.isfinite(delta) else float(delta)))
        path = OUT / f'{fold}_join.parquet'
        assert not path.exists(), 'Preserve partial prior join'
        linked.to_parquet(path, index=False)
        outputs[path.name] = common.sha256(path)
        count = linked.groupby('status', dropna=False).size().to_dict()
        support = linked.assign(day=linked[TIME].dt.floor('D')).groupby([join.ORIGIN, join.DEST, 'day'], observed=True, dropna=False).agg(rows=(ID, 'size'), linked=('linked', 'sum')).reset_index()
        support_path = OUT / f'{fold}_coverage.parquet'
        support.to_parquet(support_path, index=False)
        outputs[support_path.name] = common.sha256(support_path)
        records[fold] = dict(rows=len(linked), id_hash=common.object_hash(linked[ID].tolist()),
            original_finite_tune_rows=len(ids), linked_rows=int(linked.linked.sum()), status_counts=count,
            reads=reads, raw_oracles=oracles, raw_oracle_count=len(oracles), resources=guard())
        common.write_json(OUT / f'{fold}_join_receipt.json', records[fold])
        outputs[f'{fold}_join_receipt.json'] = common.sha256(OUT / f'{fold}_join_receipt.json')
        print('JOIN_COMPLETE', fold, len(linked), count, flush=True)
        del dep, arr, arr_raw, linked, query, support
        gc.collect()
    record = dict(status='complete', protocol_sha256=common.sha256(OUT / 'protocol.json'),
        labels_loaded=False, forbidden_columns_projected=False, raw_hashes=protocol['raw_hashes'],
        folds=records, outputs=outputs, runtime_sec=time.monotonic()-started, resources=guard())
    common.write_json(OUT / 'join_manifest.json', record)
    print('JOINS_FROZEN', common.sha256(OUT / 'join_manifest.json'), flush=True)


def associate():
    protocol = declare()
    assert not (OUT / 'association.json').exists(), 'Preserve completed association'
    manifest = common.read_json(OUT / 'join_manifest.json')
    assert manifest['status'] == 'complete' and manifest['labels_loaded'] is False
    assert manifest['protocol_sha256'] == common.sha256(OUT / 'protocol.json')
    for name, digest in manifest['outputs'].items():
        assert common.sha256(OUT / name) == digest
    assert common.sha256(META) == protocol['metadata_sha256']
    frames = {}
    label_receipts = {}
    for fold in FOLDS:
        frame = pd.read_parquet(OUT / f'{fold}_join.parquet')
        assert common.object_hash(frame[ID].tolist()) == manifest['folds'][fold]['id_hash']
        source = ds.dataset(META, format='parquet')
        labels = source.to_table(columns=[ID, TARGET], filter=ds.field(ID).isin(frame[ID].to_numpy()), use_threads=False).to_pandas().set_index(ID)
        assert labels.index.is_unique and set(labels.index) == set(frame[ID])
        y = labels.loc[frame[ID], TARGET].to_numpy(float)
        assert np.isfinite(y).all()
        path = union_folder(fold) / 'tune_predictions.parquet'
        assert common.sha256(path) == protocol['controls'][fold]['predictions_sha256']
        prediction = pd.read_parquet(path).set_index(ID).loc[frame[ID], 'prediction_sec'].to_numpy(float)
        assert np.isfinite(prediction).all()
        frame['raw_target_sec'] = y
        frame['baseline_prediction_sec'] = prediction
        frames[fold] = frame
        label_receipts[fold] = dict(rows=len(y), id_hash=common.object_hash(frame[ID].tolist()),
            raw_label_hash=common.object_hash(y.tolist()), baseline_prediction_hash=common.object_hash(prediction.tolist()))
        guard()
    calibration_mask = frames['F1'][TIME] < pd.Timestamp('2025-06-16', tz='UTC')
    calibration = stats.fit_calibration(frames['F1'].loc[calibration_mask])
    calibration['calibration_id_hash'] = common.object_hash(frames['F1'].loc[calibration_mask, ID].tolist())
    common.write_json(OUT / 'calibration.json', calibration)
    evaluations = {'June16_30':frames['F1'].loc[~calibration_mask].copy(), 'October':frames['F3'].copy()}
    results = {}
    for name, frame in evaluations.items():
        bias, candidate = stats.predictions(frame, calibration)
        frame['airport_bias_prediction_sec'] = bias
        frame['delta_prediction_sec'] = candidate
        frame.to_parquet(OUT / f'{name}_diagnostic.parquet', index=False)
        y, baseline, days = frame.raw_target_sec.to_numpy(float), frame.baseline_prediction_sec.to_numpy(float), frame[TIME].dt.floor('D')
        report = {}
        for slice_name, selected in [('full_ordinary', np.ones(len(frame), bool)), ('linked', frame.linked.to_numpy(bool))]:
            report[slice_name] = dict(baseline=stats.scores(y[selected], baseline[selected]),
                airport_bias=stats.scores(y[selected], bias[selected]), delta=stats.scores(y[selected], candidate[selected]),
                delta_vs_bias=stats.paired(y[selected], candidate[selected], bias[selected], days.iloc[np.flatnonzero(selected)]),
                delta_vs_baseline=stats.paired(y[selected], candidate[selected], baseline[selected], days.iloc[np.flatnonzero(selected)]))
        report['permutation_null'] = stats.permutation_null(frame, calibration)
        results[name] = report
        guard()
    passes = all(r['full_ordinary']['delta_vs_bias']['gain'] > 0 and
        r['full_ordinary']['delta_vs_bias']['all_day_removals_improve'] and
        r['full_ordinary']['delta_vs_bias']['mse_gain_ci95'][0] > 0 for r in results.values())
    report = dict(status='complete', protocol_sha256=common.sha256(OUT / 'protocol.json'),
        frozen_join_manifest_sha256=common.sha256(OUT / 'join_manifest.json'),
        label_receipts=label_receipts, calibration=calibration, results=results,
        association_screen_passed=passes, no_model_promotion=True, no_score_stage=True, resources=guard(),
        outputs={p.name:common.sha256(p) for p in OUT.glob('*_diagnostic.parquet')})
    common.write_json(OUT / 'association.json', report)
    print('ASSOCIATION_COMPLETE', dict(screen_passed=passes, coefficient=calibration['coefficient'],
        gains={k:v['full_ordinary']['delta_vs_bias']['gain'] for k,v in results.items()}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--declare-only', action='store_true')
    modes.add_argument('--build-joins', action='store_true')
    modes.add_argument('--associate', action='store_true')
    args = parser.parse_args()
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    with threadpool_limits(1):
        if args.declare_only:
            declare()
            print(common.sha256(OUT / 'protocol.json'), flush=True)
        elif args.build_joins:
            build_joins()
        else:
            associate()
