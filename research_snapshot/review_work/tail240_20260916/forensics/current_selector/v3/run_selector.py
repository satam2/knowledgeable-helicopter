"""Declared exposed-development selector pilot; no score/refit path."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'v1'))
import selector
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'v2'))
import watchdog
import argparse
import gc
import importlib.util
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import psutil
import pyarrow as pa
import pyarrow.parquet as pq
from threadpoolctl import threadpool_limits

ROOT = selector.ROOT


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


old = load_module('selector_frozen_additive', ROOT / 'review_work/tail240_20260916/forensics/current_calibration/run_v1.py')
context = load_module('selector_frozen_context', ROOT / 'review_work/tail240_20260916/state/neural_context/run.py')
common, risk, ID, TARGET = old.common, old.risk, old.ID, old.TARGET
OUT = common.external_path(ROOT / 'private_runs/tail240_20260916/forensics/current_selector/v3')
CAP = 4*1024**3
FIT_SECONDS = 900


def guard():
    memory = psutil.Process().memory_info()
    peak = max(memory.rss, getattr(memory, 'peak_wset', memory.rss))
    assert peak < CAP and psutil.virtual_memory().available >= 8*1024**3
    return peak


def declare():
    previous = common.read_json(old.OUT / 'protocol.json')
    assert common.sha256(old.OUT / 'protocol.json') == '995efd4bbee01885108c6673ec74bb1bdc473a7fa589268536400e3cb271e364'
    sources = [Path(__file__), Path(selector.__file__), Path(selector.__file__).with_name('test_selector.py'),
               Path(watchdog.__file__), Path(watchdog.__file__).with_name('test_watchdog.py'),
               Path(__file__).with_name('test_cpu_isolation.py'),
               Path(old.__file__), Path(context.__file__), Path(risk.__file__),
               Path(old.schema_discovery.__file__), Path(selector.gate.__file__),
               ROOT / 'review_work/breakthrough_20260916/models/encoders.py']
    folds = {}
    for fold in ['F1', 'F3']:
        folder = old.OUT / fold
        marker = common.read_json(folder / 'manifest.json')
        assert marker['status'] == 'complete'
        folds[fold] = {name: common.sha256(folder / name) for name in
                       ['inputs.parquet', 'preflight.json', 'weights.json', 'manifest.json']}
    record = dict(status='declared', preserved_v1_protocol_sha256=common.sha256(OUT.parent/'v1/protocol.json'),
        preserved_v2_protocol_sha256=common.sha256(OUT.parent/'v2/protocol.json'),
        operational_change='V2recursiveprocess-treewatchdogretained;CUDA_VISIBLE_DEVICES=-1beforeTorchimportpreventsCPUAdamWCUDAcontextinitialization;scienceunchanged', source_hashes={str(p.relative_to(ROOT)): common.sha256(p) for p in sources},
        prior_protocol_sha256=common.sha256(old.OUT / 'protocol.json'), folds=folds,
        columns=previous['columns'], experts=old.EXPERTS, contrasts=previous['contrast_columns'],
        architecture='FrozenearlyFrameEncoder;15x4categoryembeddings+762numeric/missing=822;Linear822to32/ReLU/Linear32to9/softmax;unknownembeddingzero;rawconvexmixture',
        training=dict(seed=20260916, epochs=20, batch_size=4096, optimizer='AdamW', lr=.001,
            weight_decay=.0001, betas=[.9,.999], eps=1e-8, initialization='zerooutputweight/logsmoothedearlysimplex',
            loss='rawMSE/earlyconstantMSE', no_eval_set=True, fixed_final_epoch=True),
        comparator='Freshgate.constant_weights rawSSE earlyonly;exactexistingdefaults; nativeweightsmatchprioradditiveearlysimplex',
        cohorts='Exactverifiedadditiveearly60/late40ordinarytune;earlynonnullFIDpurgedagainstlate;PLE387onlyreplacement',
        resources=dict(cpu_threads=2, interop_threads=1, process_cap_bytes=CAP,
            startup_available_bytes=12*1024**3, host_reserve_bytes=8*1024**3,
            parent_sample_sec=.1, fit_timeout_sec=FIT_SECONDS, device='cpu'),
        gate='F1positiveandeverydaypositiveelseholdF3;bothpositive/everydaypositive/weightedlateordinarygain>=2sec',
        weights=old.WEIGHTS, exposure='Fulltunealreadyselectedupstreamexpertstopping;lateisnotfreshOOF;nonewbasemodels/score/refit/ranking',
        inference='Float32softmaxweightsrenormalizedfloat64;mean+sumweights*rawcenteredpredictions;noclipping',
        fit_label_boundary='Training API receives early labels only; saved inputs read without target; early targets filtered by earlyIDs; late targets opened only after saved predictions')
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'protocol.json'
    if path.exists():
        assert common.read_json(path) == record, 'Frozen declaration changed; preserve and version'
    else:
        common.write_json(path, record)
    return record


def inputs(fold, protocol):
    folder = old.OUT / fold
    for name, digest in protocol['folds'][fold].items():
        assert common.sha256(folder/name) == digest
    names = [name for name in pq.read_schema(folder/'inputs.parquet').names if name != TARGET]
    frame = pd.read_parquet(folder/'inputs.parquet', columns=names)
    saved = common.read_json(folder/'preflight.json')
    fit, early, late, purged, cutoff = old.partitions(frame)
    for name, values in [('fit',fit), ('early',early), ('late',late), ('purged',purged)]:
        np.testing.assert_array_equal(frame[name], values)
    assert str(cutoff) == saved['cutoff']
    assert common.object_hash(frame.loc[fit, ID].tolist()) == saved['fit_ids_hash']
    assert common.object_hash(frame.loc[late, ID].tolist()) == saved['late_ids_hash']
    return frame, saved


def preflight(fold):
    protocol = declare()
    frame, saved = inputs(fold, protocol)
    folder = OUT/fold
    folder.mkdir(exist_ok=False)
    common.write_json(folder/'preflight.json', dict(status='passed', protocol_sha256=common.sha256(OUT/'protocol.json'),
        old_preflight_sha256=protocol['folds'][fold]['preflight.json'], fit_ids_hash=saved['fit_ids_hash'],
        late_ids_hash=saved['late_ids_hash'], fit_rows=int(frame.fit.sum()), late_rows=int(frame.late.sum()),
        read_target=False, model_fit=False, peak_bytes=guard()))
    print('PREFLIGHT', fold, len(frame), guard(), flush=True)


def labels_for(fold, ids):
    target = pd.read_parquet(old.OUT/fold/'inputs.parquet', columns=[ID,TARGET], filters=[(ID,'in',ids.tolist())])
    assert target[ID].is_unique and len(target) == len(ids)
    values = target.set_index(ID).loc[ids,TARGET].to_numpy(float)
    assert np.isfinite(values).all()
    return values


def child_fit(fold):
    protocol = declare()
    folder = OUT/fold
    pre = common.read_json(folder/'preflight.json')
    assert pre['protocol_sha256'] == common.sha256(OUT/'protocol.json') and pre['status'] == 'passed'
    assert not (folder/'state.pt').exists() and not (folder/'manifest.json').exists()
    frame, saved = inputs(fold, protocol)
    fit_mask, late = frame.fit.to_numpy(bool), frame.late.to_numpy(bool)
    order = np.r_[np.flatnonzero(fit_mask), np.flatnonzero(~fit_mask)]
    inverse, nfit = np.argsort(order), int(fit_mask.sum())
    sources, discovery = old.schema_discovery.cached_discovery(old.ORIGINAL_SOURCES, protocol['columns'])
    assert [dict(path=str(p),sha256=h,fill=f,columns=c) for p,h,f,c in sources] == saved['source_receipts']
    risk.feature_sources = lambda columns: sources if columns == protocol['columns'] else (_ for _ in ()).throw(ValueError('Schema changed'))
    risk.guard = guard
    matrix, vocab, receipts = risk.load_matrix(pd.Index(frame.iloc[order][ID]), nfit, protocol['columns'], folder)
    fill = {name:filled for _,_,filled,names in sources for name in names}
    for j,name in enumerate(protocol['columns']):
        if name not in vocab:
            matrix[:,j] = old.restore_numeric(matrix[:,j], fill[name])
    features = context.decode_frame(matrix, vocab, protocol['columns'])
    p = frame[protocol['experts']].to_numpy(float)
    for j,name in enumerate(protocol['contrasts']):
        features[name] = old.contrasts(p)[order,j]
    del matrix
    gc.collect()
    early_ids = frame.loc[fit_mask,ID]
    y_early = labels_for(fold, early_ids)
    common.write_json(folder/'fit_started.json', dict(utc=common.utc_now(), fit_timeout_sec=FIT_SECONDS))
    start = time.monotonic()
    def fit_guard():
        guard()
        if time.monotonic()-start > FIT_SECONDS:
            raise TimeoutError('Fixed 15-minute fit watchdog exceeded')
    with threadpool_limits(2):
        model = selector.fit(features.iloc[:nfit], p[fit_mask], y_early, guard=fit_guard)
    assert not selector.torch.cuda.is_initialized(), 'CPU selector unexpectedly initialized CUDA'
    assert len(model['encoder'].numeric) == 381 and len(model['encoder'].categories) == 15
    assert model['dimensions']['numeric_dimensions'] == 762
    old_weights = np.asarray(common.read_json(old.OUT/fold/'weights.json')['weights'])
    np.testing.assert_allclose(model['weights'], old_weights, rtol=0, atol=1e-10)
    selector.save(model, folder)
    common.write_json(folder/'fit_manifest.json', dict(status='fit_frozen_before_late_labels',
        fit_ids_hash=saved['fit_ids_hash'], dimensions=model['dimensions'], evidence=model['evidence'],
        weights=model['weights'].tolist(), outputs={n:common.sha256(folder/n) for n in ['state.pt','model.joblib']}))
    prediction_ordered, weights_ordered = selector.predict(model, features, p[order], guard=guard)
    native = selector.load(folder)
    replay, replay_weights = selector.predict(native, features, p[order], guard=guard)
    np.testing.assert_array_equal(prediction_ordered, replay)
    np.testing.assert_array_equal(weights_ordered, replay_weights)
    prediction, weights = prediction_ordered[inverse], weights_ordered[inverse]
    baseline = p@model['weights']
    output = frame[[ID,old.FID,old.TIME,'fit','late','frozen_current_sec']].copy()
    output['prediction_sec'], output['early_constant_sec'] = prediction, baseline
    for j,name in enumerate(protocol['experts']):
        output['weight_'+name] = weights[:,j]
    output.to_parquet(folder/'predictions.parquet', index=False)
    common.write_json(folder/'prediction_frozen.json',dict(predictions_sha256=common.sha256(folder/'predictions.parquet'),
        late_labels_opened=False,native_replay_max_abs_delta=0.))
    y_late = labels_for(fold, frame.loc[late,ID])
    result = dict(primary=old.comparison(y_late,prediction[late],baseline[late],frame.loc[late,old.TIME]),
        frozen_current_diagnostic=old.comparison(y_late,prediction[late],frame.loc[late,'frozen_current_sec'],frame.loc[late,old.TIME]),
        early_training=dict(constant=old.metrics(y_early,baseline[fit_mask]),candidate=old.metrics(y_early,prediction[fit_mask])))
    common.write_json(folder/'metrics.json', result)
    assert declare() == protocol
    common.write_json(folder/'manifest.json',dict(status='complete',protocol_sha256=common.sha256(OUT/'protocol.json'),
        fit_rows=nfit,late_rows=int(late.sum()),fit_ids_hash=saved['fit_ids_hash'],late_ids_hash=saved['late_ids_hash'],
        feature_receipts=receipts,discovery=discovery,metrics=result,peak_bytes=guard(),
        native_replay_max_abs_delta=0.,fit_evidence=model['evidence'],
        outputs={n:common.sha256(folder/n) for n in ['state.pt','model.joblib','fit_manifest.json','predictions.parquet','prediction_frozen.json','metrics.json']}))
    print('SELECTOR',fold,result['primary'],guard(),flush=True)


def launch(fold):
    declare()
    if fold == 'F3':
        first = common.read_json(OUT/'F1/manifest.json')['metrics']['primary']
        assert first['gain'] > 0 and first['every_day_positive'], 'F1 failure holds F3'
    assert psutil.virtual_memory().available >= 12*1024**3
    folder = OUT/fold
    assert (folder/'preflight.json').exists() and not (folder/'launch.json').exists()
    common.write_json(folder/'launch.json',dict(utc=common.utc_now(),protocol_sha256=common.sha256(OUT/'protocol.json')))
    with (folder/'stdout.log').open('w') as stream:
        proc = subprocess.Popen([sys.executable,'-B','-u',str(Path(__file__)),'--stage','child','--fold',fold],
                                stdout=stream,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        watched = psutil.Process(proc.pid)
        fit_start, failure, maximum = None, None, 0
        process_peaks = {}
        while proc.poll() is None:
            sample = watchdog.sample_tree(watched, process_peaks)
            maximum = max(maximum, sample['rss_bytes'], sample['conservative_peak_bytes'])
            if (folder/'fit_started.json').exists() and fit_start is None:
                fit_start = time.monotonic()
            if maximum >= CAP or psutil.virtual_memory().available < 8*1024**3:
                failure = 'process cap or host reserve'
            if fit_start is not None and time.monotonic()-fit_start > FIT_SECONDS and not (folder/'fit_manifest.json').exists():
                failure = 'fit time budget'
            if failure:
                watchdog.terminate_tree(watched)
                break
            time.sleep(.1)
        returncode = proc.wait()
    common.write_json(folder/'watchdog.json',dict(returncode=returncode,failure=failure,peak_bytes=maximum,per_process_peaks=process_peaks,
        accounting='ConservativesumofperPIDsampledOShistoricalpeaks;currentRSSsum;launcherandrecursivedescendants'))
    if returncode or failure:
        raise RuntimeError(f'Selector failed: returncode={returncode},failure={failure}; see preserved stdout.log')
    print(common.read_json(folder/'metrics.json')['primary'],flush=True)


def assess():
    protocol = declare()
    records = {f:common.read_json(OUT/f/'manifest.json') for f in ['F1','F3']}
    assert all(r['status']=='complete' and r['protocol_sha256']==common.sha256(OUT/'protocol.json') for r in records.values())
    before = np.sqrt(sum(old.WEIGHTS[f]*r['metrics']['primary']['baseline']['mse'] for f,r in records.items()))
    after = np.sqrt(sum(old.WEIGHTS[f]*r['metrics']['primary']['candidate']['mse'] for f,r in records.items()))
    passed = before-after >= 2 and all(r['metrics']['primary']['gain']>0 and r['metrics']['primary']['every_day_positive'] for r in records.values())
    assert not (OUT/'assessment.json').exists()
    common.write_json(OUT/'assessment.json',dict(primary_gate_passed=bool(passed),constant_rmse=float(before),candidate_rmse=float(after),
        gain=float(before-after),weights=protocol['weights'],scope='Exposedlateordinarytuneonly;noscoreclaim'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage',required=True,choices=['declare','preflight','fit','child','assess'])
    parser.add_argument('--fold',choices=['F1','F3'])
    args = parser.parse_args()
    pa.set_cpu_count(2)
    pa.set_io_thread_count(1)
    if args.stage == 'declare':
        declare()
        print('DECLARED',common.sha256(OUT/'protocol.json'),guard(),flush=True)
    elif args.stage == 'assess':
        assess()
    else:
        assert args.fold
        {'preflight':preflight,'fit':launch,'child':child_fit}[args.stage](args.fold)
