"""Earlier-period distinction test for public observation date hypotheses."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'review_work/campaign_20260916'))
import common
from taxiout.schema import ID, TARGET, MOVEMENT


def main():
    cache = ROOT / 'private_runs/tail240_20260916/source_distinctions/opdi_dates_v1'
    manifest = common.read_json(cache / 'manifest.json')
    for name, digest in manifest['outputs'].items():
        assert common.sha256(cache / name) == digest
    out = common.external_path(ROOT / 'private_runs/tail240_20260916/source_distinctions/opdi_dates_assessment_v1')
    out.mkdir(parents=True, exist_ok=False)
    protocol = {'created_utc': common.utc_now(), 'source_sha256': common.sha256(__file__),
        'cache_manifest_sha256': common.sha256(cache / 'manifest.json'),
        'scope': 'Fit/tune only source-mechanism association; originalfoldpurges. No score labels analyzed or predictor selected.',
        'groups': 'MissingNM+Rome/other, mode raw/learned/lexical, same-daypublicobserved, onlyprevious-dayobserved, bothdaysobserved, nopublictimehypothesis.',
        'outcomes': 'Rawtaximean/median, Y>12h, Y24..26h, |Y-schedule|<=60. Diagnosticassociation only.'}
    common.write_json(out / 'protocol.json', protocol)
    meta_path = ROOT / 'private_runs/screening_230/data/interim/audit/departures.parquet'
    assert common.sha256(meta_path) == common.read_json(ROOT / 'private_runs/screening_230/reports/data_audit.json')['artifacts'][meta_path.name]
    meta = pd.read_parquet(meta_path)
    features = pd.read_parquet(cache / 'features.parquet').set_index(ID)
    output, rows = {}, []
    for fold in ('F1', 'F3'):
        idx, split, _ = common.fold_data(meta, fold, full=True)
        output[fold] = {'split_hash': split['split_hash'], 'stages': {}}
        for stage in ('fit', 'tune'):
            query = meta.iloc[idx[stage]]
            query = query.loc[~np.isfinite(query.proxy_sec)].set_index(ID)
            data = query.join(features, how='left', validate='one_to_one')
            assert data['opdi_raw_day1_count_30min'].notna().all()
            data['fold'], data['stage'] = fold, stage
            for mode in ('raw', 'learned', 'lexical'):
                same = data[f'opdi_{mode}_day1_count_30min'].gt(0)
                before = data[f'opdi_{mode}_day2_count_30min'].gt(0)
                after = data[f'opdi_{mode}_day0_count_30min'].gt(0)
                data[f'{mode}_group'] = np.select([same & before, same & ~before, ~same & before, ~same & ~before & after],
                    ['same_and_before', 'same_only', 'before_without_same', 'after_only'], default='none')
            stats = []
            for mode in ('raw', 'learned', 'lexical'):
                for (rome, category), group in data.groupby([data.ADEP_mvt.eq('LIRF'), f'{mode}_group'], observed=True):
                    y = group[TARGET].to_numpy()
                    stats.append({'mode': mode, 'rome': bool(rome), 'category': category, 'n': len(group),
                        'mean_y': float(y.mean()), 'median_y': float(np.median(y)),
                        'over12h_n': int(np.sum(y > 43200)), 'day_short_n': int(np.sum((y >= 86400) & (y <= 93600))),
                        'near_schedule_n': int(np.sum(np.abs(y - group.schedule_sec.to_numpy()) <= 60))})
            output[fold]['stages'][stage] = stats
            rows.append(data.reset_index())
            print(fold, stage, [s for s in stats if s['rome'] and s['mode'] == 'learned'], flush=True)
    pd.concat(rows, ignore_index=True).to_parquet(out / 'earlier_rows.parquet', index=False)
    common.write_json(out / 'summary.json', {'status': 'complete', 'folds': output,
        'protocol_sha256': common.sha256(out / 'protocol.json'), 'earlier_rows_sha256': common.sha256(out / 'earlier_rows.parquet')})


if __name__ == '__main__':
    main()
