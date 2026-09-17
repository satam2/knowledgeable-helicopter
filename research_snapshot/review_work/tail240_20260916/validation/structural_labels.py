"""Diagnostic raw-label clock consistency; never changes labels or builds features."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import psutil

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'review_work/campaign_20260916'))
import common
from taxiout.artifacts import read_json, write_json, sha256, utc_now
from taxiout.schema import ID, TARGET, MOVEMENT, BLOCK

OUT = common.external_path(ROOT / 'private_runs/tail240_20260916/validation/structural_labels_v1')
RAW = common.RAW
pa.set_cpu_count(2)
pa.set_io_thread_count(1)


def summary(frame):
    if frame.empty:
        return {'rows': 0}
    y = frame[TARGET].to_numpy(float)
    delta = (pd.to_datetime(frame[MOVEMENT], utc=True) - pd.to_datetime(frame[BLOCK], utc=True)).dt.total_seconds().to_numpy(float)
    error = y - delta
    finite = np.isfinite(error)
    return {'rows': len(frame), 'target_nonfinite': int((~np.isfinite(y)).sum()),
        'target_negative': int((y < 0).sum()), 'target_zero': int((y == 0).sum()),
        'target_over7200': int((y > 7200).sum()), 'target_over86400': int((y > 86400).sum()),
        'missing_movement_time': int(frame[MOVEMENT].isna().sum()), 'missing_block_time': int(frame[BLOCK].isna().sum()),
        'comparable_rows': int(finite.sum()), 'label_exact_clock_difference': int((finite & (error == 0)).sum()),
        'label_within_one_second': int((finite & (np.abs(error) <= 1)).sum()),
        'label_disagreement_over_one_second': int((finite & (np.abs(error) > 1)).sum()),
        'max_absolute_label_clock_disagreement': float(np.abs(error[finite]).max()) if finite.any() else None,
        'target_quantiles': {str(q): float(np.quantile(y[np.isfinite(y)], q)) for q in (0, .5, .95, .99, 1)} if np.isfinite(y).any() else {}}


def main():
    assert not OUT.exists()
    OUT.mkdir(parents=True)
    binding = read_json(ROOT / 'private_runs/tail240_20260916/validation/baseline_binding.json')
    frozen = read_json(ROOT / 'private_runs/submission_v2/protocol.json')['raw_hashes']
    selected = {}
    for fold, record in binding['folds'].items():
        frame = pd.read_parquet(record['prediction_path'])
        squared = (frame.prediction_sec.to_numpy() - frame[TARGET].to_numpy())**2
        top = np.argsort(squared)[-int(np.ceil(.01*len(frame))):]
        selected[fold] = {'all': set(frame[ID]), 'baseline_top1pct': set(frame.iloc[top][ID])}
    parts, records, score_parts = [], [], {fold: [] for fold in selected}
    for path in sorted(RAW.glob('training_*.parquet')):
        assert sha256(path) == frozen[path.name]
        frame = pq.read_table(path, columns=[ID, TARGET, MOVEMENT, BLOCK], filters=[('PHASE_mvt', '=', 'DEP')], use_threads=False).to_pandas()
        assert frame[ID].notna().all()
        records.append({'file': path.name, 'sha256': frozen[path.name], 'within_file_duplicate_ID_rows': int(frame[ID].duplicated().sum()), **summary(frame)})
        parts.append(frame[[ID, TARGET]])
        for fold in selected:
            subset = frame.loc[frame[ID].isin(selected[fold]['all'])].copy()
            score_parts[fold].append(subset)
        peak = getattr(psutil.Process().memory_info(), 'peak_wset', psutil.Process().memory_info().rss)
        assert peak < 4*1024**3 and psutil.virtual_memory().available >= 8*1024**3
    labels = pd.concat(parts, ignore_index=True)
    meta = pd.read_parquet(ROOT / 'private_runs/screening_230/data/interim/audit/departures.parquet', columns=[ID, TARGET])
    np.testing.assert_array_equal(labels[ID], meta[ID])
    np.testing.assert_array_equal(labels[TARGET], meta[TARGET])
    assert labels[ID].is_unique and len(labels) == 2085047
    scored = {}
    for fold, pieces in score_parts.items():
        frame = pd.concat(pieces, ignore_index=True)
        assert set(frame[ID]) == selected[fold]['all']
        tail = frame.loc[frame[ID].isin(selected[fold]['baseline_top1pct'])]
        scored[fold] = {'all_score_rows': summary(frame), 'baseline_top1pct_diagnostic': summary(tail)}
        delta = (pd.to_datetime(tail[MOVEMENT], utc=True)-pd.to_datetime(tail[BLOCK], utc=True)).dt.total_seconds()
        details = tail.copy()
        details['raw_clock_duration_sec'] = delta
        details['label_minus_clock_sec'] = details[TARGET]-delta
        details.to_parquet(OUT / f'{fold}_top1pct_diagnostic.parquet', index=False)
    additive = [key for key, value in records[0].items() if isinstance(value, int) and key != 'rows']
    totals = {'rows': len(labels), **{key: sum(row[key] for row in records) for key in additive},
        'all_training_IDs_unique': True, 'exact_original_metadata_ID_order_and_labels': True}
    write_json(OUT / 'audit.json', {'status': 'passed', 'created_utc': utc_now(), 'source_sha256': sha256(__file__),
        'baseline_binding_sha256': sha256(ROOT / 'private_runs/tail240_20260916/validation/baseline_binding.json'),
        'training': totals, 'months': records, 'score_diagnostics': scored,
        'outputs': {p.name: sha256(p) for p in OUT.glob('*.parquet')},
        'scope': 'Raw training DEP target versus private movement-minus-block clock consistency only; no ranking departure hidden data read, no labels altered, no predictive features produced.',
        'limits': 'Arithmetic agreement does not prove operational validity of the airport timestamps or causal availability. Target-defined tail selections are diagnostic and forbidden at inference.'})
    print('STRUCTURAL_LABELS', totals, flush=True)


if __name__ == '__main__':
    main()
