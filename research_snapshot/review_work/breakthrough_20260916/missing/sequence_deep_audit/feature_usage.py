"""Saved-booster feature usage only; no data, predictions, or model fitting."""
import os
for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[name] = '2'
import lightgbm
import gc
import hashlib
import json
from pathlib import Path
import re
import sys
import joblib
import numpy as np
import pandas as pd
import psutil

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'review_work/campaign_20260916'))
BASE = ROOT / 'private_runs/breakthrough_20260916'
OUT = BASE / 'missing/sequence_deep_audit/feature_usage'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def guard():
    info = psutil.Process().memory_info()
    peak = getattr(info, 'peak_wset', info.rss)
    assert peak < 2 * 1024**3, peak
    return peak


def classify(feature):
    matched = re.fullmatch(r'flat_(dep|arr)([1-4])_(.+)', feature)
    if not matched:
        return dict(block='base225', phase='base', rank=0, field=feature, group='base225')
    phase, rank, field = matched.groups()
    if field.startswith('offset_') or field == 'aobt_second':
        group = 'departure_clock_context'
    elif field == 'arrival_duration':
        group = 'completed_arrival_duration'
    elif field.startswith('same_'):
        group = 'neighbor_category_match'
    elif field == 'padding_missing':
        group = 'neighbor_presence'
    else:
        assert field in ['age_sec', 'landing_age_sec', 'movement_second'], field
        group = 'event_timing'
    return dict(block='added112', phase=phase, rank=int(rank), field=field, group=group)


def main():
    assert not OUT.exists()
    OUT.mkdir(parents=True)
    protocol = dict(source_sha256=sha(__file__), sources=['deep337', 'lgb600_337'],
        stages=['fit', 'refit'], folds=['F1', 'F3'], cpu_threads=2, peak_rss_limit=2*1024**3,
        no_training_or_inference=True, no_feature_or_label_data_read=True,
        limitation='In-sample tree split gain/frequency measures model usage, not causal contribution or held-out gain. Correlated substitutes and split opportunities bias importance.')
    (OUT/'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    rows, receipts = [], []
    for family in protocol['sources']:
        for fold in protocol['folds']:
            folder = (BASE/'deeper_sequence_v2'/f'lightgbm_leaf63_sequence_aobt_allfinite_{fold}_s20260916'
                      if family == 'deep337' else BASE/'missing/sequence_flatten/models'/f'lightgbm_aobt_allfinite_{fold}_s20260916')
            record = json.loads((folder/'manifest.json').read_text())
            assert record['status'] == 'complete' and len(record['feature_columns']) == 337
            for stage, filename in [('fit', 'fit_model.joblib'), ('refit', 'model.joblib')]:
                path = folder/filename
                digest = sha(path)
                assert digest == record['outputs'][filename]
                model = joblib.load(path)
                estimator = model['estimator']
                booster = estimator.booster_
                columns = booster.feature_name()
                assert columns == model['encoder'].columns == record['feature_columns']
                steps = model['steps']
                gain = booster.feature_importance(importance_type='gain', iteration=steps)
                split = booster.feature_importance(importance_type='split', iteration=steps)
                assert len(gain) == len(split) == 337 and np.isfinite(gain).all()
                for feature, g, s in zip(columns, gain, split):
                    rows.append(dict(family=family, fold=fold, stage=stage, feature=feature,
                        gain=float(g), splits=int(s), gain_share=float(g/gain.sum()),
                        split_share=float(s/split.sum()), **classify(feature)))
                receipts.append(dict(family=family, fold=fold, stage=stage, path=str(path),
                    model_sha256=digest, manifest_sha256=sha(folder/'manifest.json'), steps=steps,
                    gain_total=float(gain.sum()), splits_total=int(split.sum()), peak_rss_bytes=guard()))
                print('INSPECTED',family,fold,stage,steps,'peak',guard(),flush=True)
                del model, estimator, booster
                gc.collect()
    data = pd.DataFrame(rows)
    data.to_csv(OUT/'feature_usage.csv', index=False)
    summaries = {}
    keys = ['family', 'fold', 'stage']
    for name, dimensions in {'block':['block'], 'group':['block','group'],
                             'phase':['block','phase'], 'phase_rank':['block','phase','rank'],
                             'field':['block','field'], 'phase_field':['block','phase','field']}.items():
        table = data.groupby(keys+dimensions, as_index=False)[['gain','splits','gain_share','split_share']].sum()
        table.to_csv(OUT/(name+'.csv'), index=False)
        summaries[name] = table.to_dict('records')
    added = data[data.block.eq('added112')]
    top = added.sort_values('gain', ascending=False).groupby(keys, sort=False).head(20)
    top.to_csv(OUT/'top20_added.csv', index=False)
    zero = added[added.splits.eq(0)]
    zero.to_csv(OUT/'unused_added.csv', index=False)
    result = dict(status='passed', protocol_sha256=sha(OUT/'protocol.json'), receipts=receipts,
        source_sha256=sha(__file__), peak_rss_bytes=guard(), summaries=summaries,
        outputs={p.name:sha(p) for p in OUT.glob('*.csv')}, limitation=protocol['limitation'])
    (OUT/'summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('COMPLETE',guard(),flush=True)


if __name__ == '__main__':
    main()
