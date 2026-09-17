"""Independent static and identifier-only half-data learning-curve review."""
import ast
import hashlib
import importlib.util
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import psutil

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT/'knowledgeable-helicopter-screening/src'))
from taxiout.artifacts import read_json, write_json, sha256, object_hash
from taxiout.paths import external_path
from taxiout.schema import ID, FLIGHT_ID, MOVEMENT


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    started = time.monotonic()
    source = ROOT/'review_work/tail240_20260916/state/learning_curve_design'
    artifact = ROOT/'private_runs/tail240_20260916/state/learning_curve_design/v3'
    destination = external_path(ROOT/'private_runs/tail240_20260916/forensics/learning_curve_preflight/v3')
    assert not destination.exists()
    assert sha256(artifact/'protocol.json') == '0ae69ee30d4a1cfcfb8f5d66b964434019f25e6833bf0345a1c62565d6618437'
    protocol, selection = read_json(artifact/'protocol.json'), read_json(artifact/'selection.json')
    for name, digest in protocol['sources'].items():
        assert sha256(ROOT/name) == digest
    assert selection['protocol_sha256'] == sha256(artifact/'protocol.json')
    assert selection['outcome_columns_read'] == [] and selection['selected_rows'] == 408030
    control = ROOT/'private_runs/breakthrough_20260916/deeper_context_union/lightgbm_leaf63_sequence_aobt_allfinite_F1_s20260916'
    marker = read_json(control/'manifest.json')
    assert sha256(control/'manifest.json') == protocol['control_manifest_sha256']
    assert protocol['parameters'] == dict(marker['fit']['params'], n_estimators=2001)
    native = ROOT/'private_runs/tail240_20260916/models/following_groups_tune_v1/F1/control387'
    native_marker = read_json(native/'manifest.json')
    assert sha256(native/'manifest.json') == protocol['native_manifest_sha256']
    for name in ['model.txt','encoder.json']:
        assert sha256(native/name) == native_marker['outputs'][name]
    metadata = ROOT/'private_runs/screening_230/data/interim/audit/departures.parquet'
    assert sha256(metadata) == protocol['metadata_sha256']
    columns = [ID,FLIGHT_ID,MOVEMENT,'ADEP_mvt','proxy_sec']
    meta = pd.read_parquet(metadata, columns=columns)
    times = pd.to_datetime(meta[MOVEMENT],utc=True)
    fit = times.ge('2025-01-01') & times.lt('2025-06-01')
    tune = times.ge('2025-06-01') & times.lt('2025-07-01')
    score = times.ge('2025-07-01') & times.lt('2025-08-01')
    related_score = meta.loc[score,FLIGHT_ID].dropna()
    fit &= ~(meta[FLIGHT_ID].notna() & meta[FLIGHT_ID].isin(related_score))
    tune &= ~(meta[FLIGHT_ID].notna() & meta[FLIGHT_ID].isin(related_score))
    fit &= ~(meta[FLIGHT_ID].notna() & meta[FLIGHT_ID].isin(meta.loc[tune,FLIGHT_ID].dropna()))
    finite = np.isfinite(meta.proxy_sec.to_numpy())
    fit_frame, tune_frame = meta.loc[fit & finite], meta.loc[tune & finite]
    for stage, frame in [('fit',fit_frame),('tune',tune_frame)]:
        assert dict(n=len(frame),hash=object_hash(frame[ID].tolist())) == marker['fit_ids'][stage]
    ids = fit_frame[ID].to_numpy()
    digests = np.asarray([hashlib.sha256(('prc2026-uniform-half-v1|20260916|'+str(int(i))).encode('ascii')).hexdigest() for i in ids], dtype='S64')
    order = np.sort(np.lexsort((ids,digests))[:408030])
    selected = ids[order]
    saved_ids = pd.read_parquet(artifact/'selected_ids.parquet')[ID].to_numpy()
    np.testing.assert_array_equal(saved_ids, selected)
    assert sha256(artifact/'selected_ids.parquet') == selection['selected_ids_file_sha256']
    assert object_hash(selected.tolist()) == selection['selected_fit_ids_hash']
    chosen = fit_frame.iloc[order]
    assert chosen[MOVEMENT].dt.strftime('%Y-%m').value_counts().sort_index().to_dict() == selection['by_month']
    assert chosen.ADEP_mvt.value_counts().sort_index().to_dict() == selection['by_airport']
    sampling = import_file('independent_sample_subject', source/'sampling.py')
    vocab, mapping = sampling.selected_vocab(np.array([2,3,4,0,1]), np.array([0,3,4]), ['a','b','c'])
    assert vocab == ['a']
    np.testing.assert_array_equal(mapping, [0,1,2,1,1])
    reference = import_file('independent_reference_subject',source/'reference.py')
    stored = pd.read_parquet(control/'tune_predictions.parquet')
    assert set(stored.columns) == {ID,'prediction_sec'}
    reference.reference_predictions(stored,tune_frame[ID])
    assert sha256(control/'tune_predictions.parquet') == marker['outputs']['tune_predictions.parquet']
    tree = ast.parse((source/'run_half_v3.py').read_text())
    fit_calls = [n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
                 and isinstance(n.func.value,ast.Name) and n.func.value.id == 'model' and n.func.attr == 'fit']
    assert len(fit_calls) == 1
    assert set(k.arg for k in fit_calls[0].keywords) == {'categorical_feature','feature_name','callbacks'}
    assert [ast.unparse(a) for a in fit_calls[0].args] == ['half[:len(selected)]','yfit']
    assert protocol['parameters']['n_estimators'] == 2001 and protocol['parameters']['n_jobs'] == 2
    selection_func = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name == 'selection')
    assert not any(isinstance(n,ast.Name) and n.id == 'TARGET' for n in ast.walk(selection_func))
    run_text = ast.unparse(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name == 'run'))
    assert run_text.index('reference_predictions(stored') < run_text.index('model.fit(')
    assert run_text.index('np.testing.assert_array_equal(replay, stored.prediction_sec.to_numpy())') < run_text.index('model.fit(')
    assert "filters=[(ID, 'in', eligible_label_ids)]" in run_text
    assert "labels.loc[fit[ID]]" in run_text and "labels.loc[parts['tune'][ID]]" in run_text
    memory = psutil.Process().memory_info()
    peak = max(memory.rss,getattr(memory,'peak_wset',memory.rss))
    assert peak < 2*1024**3 and psutil.virtual_memory().available >= 8*1024**3
    destination.mkdir(parents=True,exist_ok=False)
    write_json(destination/'receipt.json',dict(status='passed_static_and_identifier_preflight',
        source_sha256=sha256(__file__),protocol_sha256=sha256(artifact/'protocol.json'),
        selection_sha256=sha256(artifact/'selection.json'),selected_rows=len(selected),
        selected_fit_ids_hash=object_hash(selected.tolist()),fit_rows=len(fit_frame),tune_rows=len(tune_frame),
        target_columns_read=[],input_columns=columns,full_control_native_replay='Statically required before fit; not executed by this preflight',
        checks=['All declared hashes','Independent manual outer flight purges','Independent SHA256 sample reconstruction',
                'Selected-only categorical vocab synthetic check','Exact full reference ID and prediction schema',
                'Fixed2001 and noevalset in model.fit','Selected-fit+tune metadata label predicate',
                'No outcome read by membership function'],no_model_fit=True,no_gpu=True,
        peak_bytes=peak,runtime_sec=time.monotonic()-started))
    print(read_json(destination/'receipt.json'),flush=True)


if __name__ == '__main__':
    main()
