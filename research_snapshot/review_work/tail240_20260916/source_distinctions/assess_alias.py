"""Assess recovered clocks on allowed missing-source fit/tune periods only."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'review_work/campaign_20260916'))
import common
from taxiout.schema import ID, TARGET


def main():
    base = ROOT / 'private_runs/tail240_20260916/source_distinctions/alias_v1'
    summary = common.read_json(base / 'summary.json')
    assert summary['status'] == 'complete'
    assert common.sha256(base / 'selected.parquet') == summary['output_sha256']
    selected = pd.read_parquet(base / 'selected.parquet')
    selected = selected.loc[selected.query_nm_missing].copy()
    path = ROOT / 'private_runs/screening_230/data/interim/audit/departures.parquet'
    assert common.sha256(path) == common.read_json(ROOT / 'private_runs/screening_230/reports/data_audit.json')['artifacts'][path.name]
    meta = pd.read_parquet(path)
    out = common.external_path(ROOT / 'private_runs/tail240_20260916/source_distinctions/alias_fit_tune_v1')
    out.mkdir(parents=True, exist_ok=False)
    protocol = {'created_utc': common.utc_now(), 'source_sha256': common.sha256(__file__),
        'alias_manifest_sha256': common.sha256(base / 'summary.json'),
        'scope': 'Fit/tune missing-source label diagnostics only. No score labels selected or scored. No predictor trained.',
        'groups': ['all', 'LIRF', 'other'], 'modes': ['exact', 'alias', 'alias_only']}
    common.write_json(out / 'protocol.json', protocol)
    results = {}
    for fold in ('F1', 'F3'):
        idx, split, _ = common.fold_data(meta, fold, full=True)
        result = {'split_hash': split['split_hash'], 'stages': {}}
        for stage in ('fit', 'tune'):
            m = meta.iloc[idx[stage]].set_index(ID)
            known = selected.loc[selected[ID + '_query'].isin(m.index)]
            modes = {'exact': known.loc[known.match_mode.eq('exact')],
                     'alias': known.loc[known.match_mode.eq('alias')]}
            modes['alias_only'] = modes['alias'].loc[~modes['alias'][ID + '_query'].isin(modes['exact'][ID + '_query'])]
            report = {}
            for mode, frame in modes.items():
                target = m.loc[frame[ID + '_query'], TARGET].to_numpy()
                values = frame.candidate_proxy_sec.to_numpy()
                airports = m.loc[frame[ID + '_query'], 'ADEP_mvt'].astype(str).to_numpy()
                report[mode] = {}
                for group, mask in [('all', np.ones(len(frame), bool)), ('LIRF', airports == 'LIRF'), ('other', airports != 'LIRF')]:
                    errors = values[mask] - target[mask]
                    report[mode][group] = {'n': int(mask.sum()),
                        'rmse': float(np.sqrt(np.mean(errors ** 2))) if len(errors) else None,
                        'mae': float(np.mean(np.abs(errors))) if len(errors) else None,
                        'within60_fraction': float(np.mean(np.abs(errors) <= 60)) if len(errors) else None,
                        'over1800_count': int(np.sum(np.abs(errors) > 1800)),
                        'bias': float(np.mean(errors)) if len(errors) else None}
            result['stages'][stage] = report
        results[fold] = result
        print(fold, result['stages'], flush=True)
    common.write_json(out / 'summary.json', {'status': 'complete', 'folds': results,
        'protocol_sha256': common.sha256(out / 'protocol.json')})


if __name__ == '__main__':
    main()
