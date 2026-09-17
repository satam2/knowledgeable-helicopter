"""Consolidate public receipts and completed experiment metadata, without raw data."""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / 'output/breakthrough_20260916/research_gap'
BASE = ROOT / 'private_runs/breakthrough_20260916'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def experiment(name, relative, prefix):
    directory = BASE / relative
    records = {}
    for fold in ['F1', 'F3']:
        paths = sorted(directory.glob(f'{prefix}*_{fold}_*/manifest.json'))
        if len(paths) != 1:
            records[fold] = {'status': 'not uniquely completed', 'matches': len(paths)}
            continue
        path = paths[0]
        source = read(path)
        record = {'path': str(path.relative_to(ROOT)), 'sha256': digest(path), 'status': source['status']}
        if source['status'] == 'complete':
            record.update(training_scope=source.get('training_scope'), rows=source.get('fit_ids'),
                feature_count=len(source.get('feature_columns', [])), runtime_sec=source.get('runtime_sec'),
                peak_rss_bytes=source.get('peak_rss_bytes'), reload_max_abs_delta=source.get('reload_max_abs_delta'),
                metrics={variant: source['reports'][variant]['metrics']['overall'] for variant in ['candidate', 'blend25']})
            for key in ['fit', 'refit']:
                fit = source.get(key, {})
                record[key] = {k: fit[k] for k in ['rows', 'steps', 'features', 'device', 'gpu_peak_allocated_bytes', 'gpu_peak_bytes', 'fit_runtime_sec', 'runtime_sec'] if k in fit}
        records[fold] = record
    result = {'name': name, 'folds': records}
    if all(records[f]['status'] == 'complete' for f in ['F1', 'F3']):
        result['seasonal_rmse'] = {variant: ((192122 * records['F1']['metrics'][variant]['rmse_sec'] ** 2 + 152719 * records['F3']['metrics'][variant]['rmse_sec'] ** 2) / 344841) ** .5 for variant in ['candidate', 'blend25']}
    return result


def main():
    target = OUT / 'sources_final.json'
    if target.exists():
        raise ValueError('Final snapshot exists; preserve and create versioned later snapshot')
    prior = OUT / 'sources.json'
    result = deepcopy(read(prior))
    result['snapshot_utc'] = datetime.now(timezone.utc).isoformat()
    result['prior_snapshot'] = {'path': str(prior.relative_to(ROOT)), 'sha256': digest(prior)}
    result['purpose'] = 'Final research handoff snapshot; retained primary receipts and current local manifests; zero new network or private row reads'
    statuses = {'tabm': 'Completed base, source-context augmented, combined225, PLE8, multi-anchor and source-mixture evaluations; inspect named experiments rather than pooling formulations',
        'tabicl_v2': 'Original missingNM subgroup evaluated; augmented wrapper prepared only, no completed folds at cutoff',
        'tabdpt_v13': 'Pinned checkpoint; original CUDA synthetic canary failed missing attention kernel; mathSDPA v2 CPU synthetic canary passed; no complete private fold scores at cutoff'}
    for source in result['sources']:
        if source['id'] in statuses:
            source['status'] = statuses[source['id']]
    definitions = [
        ('TabM base', 'models', 'tabm_aobt_allfinite'),
        ('TabM source context', 'models/augmented/source_past__source_twosided', 'tabm_aobt_allfinite'),
        ('TabM combined225', 'full_neural', 'tabm_combined_standard_aobt_allfinite'),
        ('Source StudentT MDN conventions', 'models/source_mdn/conventions', 'tabm_source_mdn_finite_rawmean'),
        ('Convex clock fusion conventions', 'clock_fusion/conventions', 'clockfusion_convex_finite_rawmean'),
        ('RealMLP bounded200K', 'models_retrieval', 'realmlp_aobt_allfinite'),
        ('TabICL missingNM', 'models/tabicl_missing', 'tabicl_missing_direct'),
        ('Combined LGB600', 'information_models/conventions__geometry__source_past__source_twosided__surface_T__trajectory__weather_T', 'lightgbm_aobt_allfinite'),
        ('Combined LGB63', 'deeper_lgb/combined', 'lightgbm_leaf63_aobt_allfinite'),
        ('Combined LGB63 seed20260917', 'deeper_replication/s20260917', 'lightgbm_leaf63_aobt_allfinite'),
        ('Combined CatBoost', 'combined_catboost', 'catboost_combined_aobt_allfinite'),
    ]
    result['experiments'] = [experiment(*args) for args in definitions]
    result['canaries'] = []
    for relative in ['models_retrieval/tabdpt_v2/canaries/tabdpt_cpu_s20260916/manifest.json',
        'models_retrieval/canaries/tabdpt_cuda_s20260916/manifest.json',
        'sequence_context/models/canaries/cuda_s20260919/manifest.json',
        'models_retrieval/source_mdn_cuda_canary/cuda_s20260916/manifest.json']:
        path = BASE / relative
        source = read(path)
        result['canaries'].append({'path': str(path.relative_to(ROOT)), 'sha256': digest(path), **{k: source[k] for k in ['status', 'error', 'runtime_sec', 'peak_rss_bytes', 'peak_vram_bytes', 'median_step_sec', 'gpu_peak_allocated_bytes', 'gpu_peak_reserved_bytes', 'private_data_loaded'] if k in source}})
    result['retrospective_context'] = {'path': str((BASE / 'retrospective_research/manifest.json').relative_to(ROOT)),
        'sha256': digest(BASE / 'retrospective_research/manifest.json'), 'verification': read(BASE / 'retrospective_research/verification.json'),
        'independent_oracle': 'private_runs/breakthrough_20260916/models/sequence_result_audit/retrospective_oracle.json',
        'features': 50, 'groups': {'past': 18, 'future': 18, 'day': 14}, 'training_rows': 2085047, 'ranking_rows': 344841,
        'interpretation': 'Observed final supplied batch context; future arrival completion and uncensored peer-clock moments; coverage and verification are not evidence of predictive gain'}
    result['sources'].append({'id': 'competition_availability', 'primary_urls': ['https://prc-data-challenge-2026.netlify.app/data.html', 'https://prc-data-challenge-2026.netlify.app/eligibility.html'],
        'receipt': 'output/research_20260916/sources.json', 'keys': ['competition_data', 'competition_eligibility'],
        'claim': 'Ranking removes departure block and taxi labels; retained arrival outcomes and final logs support declared retrospective modeling; no examined rule establishes a real-time timestamp-publication cutoff'})
    result['queue_at_snapshot'] = read(BASE / 'gpu_queue_1131/results.json')
    result['status_warning'] = 'Snapshot only. Active or queued experiments may complete later. Full-cohort evaluation is distinct from full eligible training. Development scores are exposed, not official or new holdouts. No architecture claims superiority from generic benchmarks.'
    target.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({'snapshot_utc': result['snapshot_utc'], 'sources_final_sha256': digest(target),
        'experiments': [{'name': x['name'], 'seasonal_rmse': x.get('seasonal_rmse'), 'status': {f: y['status'] for f, y in x['folds'].items()}} for x in result['experiments']]}, indent=2))


if __name__ == '__main__':
    main()
