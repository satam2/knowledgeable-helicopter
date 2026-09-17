"""Create a versioned final research snapshot from receipts and metadata only."""
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'knowledgeable-helicopter-screening/src'))
from taxiout.paths import external_path

OUT = external_path(ROOT / 'output/breakthrough_20260916/research_gap')
BASE = ROOT / 'private_runs/breakthrough_20260916'
CONTROL = 'information_models/conventions__geometry__source_past__source_twosided__surface_T__trajectory__weather_T'
WEIGHTS = {'F1': 192122 / 344841, 'F3': 152719 / 344841}
DEFINITIONS = [
    ('TabM base', 'models', 'tabm_aobt_allfinite', 20260916, 'Full eligible finite-NM; unchanged missing route'),
    ('TabM source context', 'models/augmented/source_past__source_twosided', 'tabm_aobt_allfinite', 20260916, 'Same family plus observed source context'),
    ('TabM combined225', 'full_neural', 'tabm_combined_standard_aobt_allfinite', 20260916, 'Full eligible finite-NM; 225 fields'),
    ('TabM combined225 second seed', 'full_neural', 'tabm_combined_standard_aobt_allfinite', 20260917, 'Residual-expert seed replication only'),
    ('TabM PLE8 combined225', 'full_neural', 'tabm_combined_ple8_aobt_allfinite', 20260916, 'Numeric embedding alternative'),
    ('LightGBM600 combined225', CONTROL, 'lightgbm_aobt_allfinite', 20260916, 'Matched information baseline; 600-tree cap'),
    ('LightGBM63 combined225', 'deeper_lgb/combined', 'lightgbm_leaf63_aobt_allfinite', 20260916, 'Deeper/longer capacity'),
    ('LightGBM63 combined225 second seed', 'deeper_replication/s20260917', 'lightgbm_leaf63_aobt_allfinite', 20260917, 'Residual-expert seed replication only'),
    ('CatBoost combined225', 'combined_catboost', 'catboost_combined_aobt_allfinite', 20260916, 'Matched input contract; different model capacity'),
    ('XGBoost combined225', 'combined_xgb', 'xgb_aobt_allfinite', 20260916, 'Matched input contract; different model capacity'),
    ('LightGBM600 ordered337', 'missing/sequence_flatten/models', 'lightgbm_aobt_allfinite', 20260916, '225 base plus 112 ordered fields'),
    ('LightGBM600 no peer-DEP clocks313', 'models/sequence_clock_ablation', 'lightgbm_sequence_no_dep_clocks_aobt_allfinite', 20260916, 'Removes 24 peer-DEP clock channels'),
    ('LightGBM63 ordered337', 'deeper_sequence_v2', 'lightgbm_leaf63_sequence_aobt_allfinite', 20260916, 'Full eligible finite-NM ordered context'),
    ('LightGBM63 ordered8', 'deeper_sequence8', 'lightgbm_leaf63_sequence8_aobt_allfinite', 20260916, 'Longer ordered representation'),
    ('LightGBM63 context union387', 'deeper_context_union', 'lightgbm_leaf63_sequence_aobt_allfinite', 20260916, 'Ordered337 plus retrospective50'),
    ('LightGBM600 retrospective past243', 'retrospective_models/past', 'lightgbm_aobt_allfinite', 20260916, 'Past18 plus combined225'),
    ('LightGBM600 retrospective past+future261', 'retrospective_models/future__past', 'lightgbm_aobt_allfinite', 20260916, 'Separately declared future18'),
    ('LightGBM600 retrospective past+day257', 'retrospective_models/day__past', 'lightgbm_aobt_allfinite', 20260916, 'Separately declared whole-day14'),
    ('LightGBM600 retrospective all275', 'retrospective_models/day__future__past', 'lightgbm_aobt_allfinite', 20260916, 'Past18, future18 and day14'),
    ('LightGBM600 monthly ARR240 v3', 'retrospective_models/monthly_arrival_v3', 'lightgbm_monthly_arrival_aobt_allfinite', 20260916, 'Whole-month ARR moments/quantiles; adaptive follow-up'),
    ('LightGBM600 clock innovations', 'models/sequence_clock_innovations/models', 'lightgbm_clock_innovations_aobt_allfinite', 20260916, 'Explicit query-minus-peer clock representation'),
    ('Source Student-t MDN', 'models/source_mdn/conventions', 'tabm_source_mdn_finite_rawmean', 20260916, '115 fields; raw analytical mixture mean'),
    ('Convex clock fusion', 'clock_fusion/conventions', 'clockfusion_convex_finite_rawmean', 20260916, '115 fields; convex own-clock residual fusion'),
    ('RealMLP bounded200K', 'models_retrieval', 'realmlp_aobt_allfinite', 20260917, '200K fit/refit reference; full tune/score'),
    ('TabICL missing-NM', 'models/tabicl_missing', 'tabicl_missing_direct', 20260916, 'Missing-NM specialist; V2 outside subgroup'),
    ('TabDPT v3.1 reference32K', 'models_retrieval/tabdpt_batch_v3_1', 'tabdpt_aobt_allfinite', 20260916, '32K reference, context256, PCA, mathSDPA batch16'),
    ('GRU static200K', 'sequence_context/models', 'static', 20260919, 'Budgeted static control; 200K fit/refit'),
    ('GRU context200K', 'sequence_context/models', 'context', 20260919, 'Budgeted ordered-event GRU; 200K fit/refit'),
    ('Missing historical template control', 'missing/id_context_v1/models', 'historical_template', 20260916, 'Original matched missing-route control; finite-NM predictions retain V2'),
    ('OPDI missing historical template v3', 'models/opdi_missing_v3/models', 'historical_template', 20260916, 'Added eight external identity fields; adaptive missing-route follow-up'),
    ('Monthly ARR missing historical template', 'missing/monthly_arrival_model/models', 'historical_template', 20260916, 'Added fifteen monthly ARR fields; adaptive missing-route follow-up'),
    ('ExtraTrees missing template/ID context', 'models/missing_forest', 'extratrees_missing_template_idcontext', 20260916, 'Non-boosting missing-route comparator; finite-NM retains V2'),
    ('RandomForest missing template/ID context', 'models/missing_forest', 'randomforest_missing_template_idcontext', 20260916, 'Non-boosting missing-route comparator; finite-NM retains V2'),
]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def relative(path):
    return Path(path).relative_to(ROOT).as_posix()


def reference(path):
    path = Path(path)
    return {'path': relative(path), 'exists': path.exists(), **({'sha256': digest(path)} if path.exists() else {})}


def experiment(definition):
    name, directory, prefix, seed, scope = definition
    records = {}
    for fold in WEIGHTS:
        path = BASE / directory / f'{prefix}_{fold}_s{seed}/manifest.json'
        record = reference(path)
        record['status'] = 'not completed at snapshot'
        if path.exists():
            data = read(path)
            record['status'] = data.get('status', 'status not recorded')
            for key in ('created_utc', 'completed_utc', 'training_scope', 'scope', 'complete_score_rows',
                        'runtime_sec', 'peak_rss_bytes', 'peak_vram_bytes', 'reload_max_abs_delta',
                        'replay_delta', 'replay_scope', 'fit_ids', 'sampled_stage_rows', 'eligible_stage_rows'):
                if key in data:
                    record[key] = data[key]
            columns = data.get('feature_columns', data.get('columns'))
            if isinstance(columns, (list, dict)):
                record['feature_count'] = len(columns)
            if data.get('status') == 'complete':
                record['metrics'] = {variant: report['metrics']['overall'] for variant, report in data.get('reports', {}).items()
                                     if 'metrics' in report and 'overall' in report['metrics']}
                for stage in ('fit', 'refit'):
                    record[stage] = {key: value for key, value in data.get(stage, {}).items() if key != 'history'}
        records[fold] = record
    result = {'name': name, 'scope': scope, 'folds': records}
    if all(records[fold]['status'] == 'complete' for fold in WEIGHTS):
        common = set.intersection(*(set(records[fold].get('metrics', {})) for fold in WEIGHTS))
        result['seasonal_rmse'] = {variant: sum(WEIGHTS[fold] * records[fold]['metrics'][variant]['rmse_sec'] ** 2
                                               for fold in WEIGHTS) ** .5 for variant in sorted(common)}
    return result


def public_sources():
    repos = [
        ('javidmardanov/PRC-Data-Challenge-2026', '845d0d15bc4065892ae126efc5a14e083dfd1de3', 'public_solution_findings_1316.md', 'Airport-hour TimesFM forecast covariates for CatBoost, month ARR context, LOBT residual follow-up'),
        ('Phoenix-Ops-LTD/prc2026-taxiout', '4477d9b088abdd16d0008a2857c7ce07ca639ab2', 'public_methods_validation/PUBLIC_METHODS_REVIEW.md', 'Following runway/stand clock context and specialist ensemble; label clipping and different splits'),
        ('sergiuv11/prc-data-challenge-2026', 'e3b653b31b79793bd10e1d7276b4075ae4ed36f2', 'public_methods_validation/PUBLIC_METHODS_REVIEW.md', 'Direct LightGBM plus airport experts; smaller ranking cohort and changed final floor'),
        ('ahmetabdullahgultekin/prc-taxiout-2026', '9d50bf676cef2f74bfebeb88ba650aa76fe3986e', 'public_methods_aviation/review.md', 'Empirical P10 reference, retrospective traffic, weather and daily external ATFM'),
        ('ChandanHegde07/OpenAir', '97f603cb181a779e2807e358493f253ee411efc6', 'public_methods_main/review.md', 'OPDI aircraft address and previous observed leg; surveillance bounds are not block times'),
        ('skylinkapi/prc-data-challenge-2026-kind-mango', 'b8f3321aee0f81fcaade4c91c6e22e3596945d43', 'public_methods_main/review.md', 'Linear trees, neighboring clocks, stand reuse; alleged organizer clock ban unsubstantiated'),
    ]
    return [{'repository': repo, 'revision': revision, 'url': f'https://github.com/{repo}/tree/{revision}',
             'review': reference(OUT / report), 'implemented_idea': idea,
             'score_evidence': 'Author claims only; no independent organizer-score verification; not matched local folds',
             'code_execution': False} for repo, revision, report, idea in repos]


def build():
    prior = OUT / 'sources_final.json'
    result = deepcopy(read(prior))
    result['snapshot_utc'] = datetime.now(timezone.utc).isoformat()
    result['purpose'] = 'Final versioned evidence snapshot; reads source receipts and local JSON metadata only; no network, private row reads, model loading or training'
    result['prior_snapshot'] = reference(prior)
    result['exporter'] = reference(Path(__file__))
    result['template'] = reference(Path(__file__).with_name('shortlist_final_1402.template.md'))
    result['seasonal_weights'] = WEIGHTS
    result['experiments'] = [experiment(definition) for definition in DEFINITIONS]
    statuses = {
        'tabm': 'Completed full-eligible base/context/combined, PLE8 and second residual-expert seed; see individually named experiments',
        'tabdpt_v13': 'Completed v3.1 full-score F1/F3 evaluation with 32K fit/refit references, context256, PCA and CUDA mathSDPA batch16; negative bounded result. Original v1 kernel failure and v2 runtime stop preserved.',
        'tabicl_v2': 'Original missing-NM specialist completed; augmented wrapper remains prepared only unless separately completed manifests appear in inventory',
        'ft_transformer': 'Researched and feasibility assessed; no implemented wrapper or completed run',
        'tabpfn35': 'Researched; gated weights not downloaded and terms not accepted; no completed run',
    }
    for item in result['sources']:
        if item['id'] in statuses:
            item['status'] = statuses[item['id']]
    result['sources'].extend([
        {'id': 'timesfm3', 'primary_urls': ['https://huggingface.co/google/timesfm-3.0-pytorch'],
         'revision': '43046b85ec22d584a13f8098c2ed39c889e129c2', 'license': 'TimesFM Non-Commercial License v1.0',
         'license_sha256': '3e36db7240d23adb6ac6d7d931892dcc5706a14a7d31581f08c35d2f25736dfd',
         'status': 'Official card/license text retrieved; no weights downloaded, terms accepted or model run',
         'limits': 'Noncommercial/nonproduction; distribution of model/derivatives prohibited; prize eligibility not established',
         'receipt': reference(OUT / 'public_solutions_1310/timesfm_official_license.receipt.json')},
        {'id': 'organizer_rules_current', 'primary_urls': ['https://github.com/euctrl-pru/prc_data_challenge_website_2026'],
         'revision': '8a2651f22c66d8f63bd4e933485ab7cc3aaa0dd9',
         'claim': 'Inspected ranking warning bans exploitation of ranking process; no explicit supplied AOBT/LOBT ban found in data/ranking/eligibility/rationale/index. Not organizer endorsement of a model.',
         'review': reference(OUT / 'public_solution_findings_1316.md')},
        {'id': 'opdi_flight_lists', 'primary_urls': ['https://www.opdi.aero/flight-list-data.html', 'https://www.opdi.aero/data.html', 'https://www.opdi.aero/about.html'],
         'version': '0.0.2', 'meaning': 'Aircraft identity and surveillance observation bounds; no actual block times',
         'license': 'Noncommercial language and attribution requirements; specific permissive prize/release rights not established',
         'review': reference(OUT / 'public_methods_aviation/review.md'),
         'current_local_evidence': [reference(BASE / 'missing/opdi_rotation' / name) for name in ('acquisition.json', 'manifest.json', 'verification.json')]},
        {'id': 'eurocontrol_daily_atfm', 'primary_urls': ['https://ansperformance.eu/data/', 'https://ansperformance.eu/definition/atfm-delay/'],
         'status': 'Primary catalog and workbook HEAD verified; 215417012 combined bytes, range through July2026; no workbook download or model run',
         'meaning': 'Published airport-day regulation/slot adherence and arrival ATFM delay, not individual taxi or queue labels',
         'review': reference(OUT / 'public_methods_aviation/review.md')},
    ])
    result['public_solutions'] = public_sources()
    receipts = []
    for directory in ('public_solutions_1310', 'public_methods_main', 'public_methods_validation', 'public_methods_aviation'):
        for path in sorted((OUT / directory).rglob('*.json')):
            if 'receipt' in path.name or 'manifest' in path.name:
                receipts.append(reference(path))
    result['new_public_receipt_index'] = receipts
    result['queue_at_snapshot'] = {relative(path): {'reference': reference(path), 'records': read(path)}
                                   for path in sorted(BASE.glob('gpu_queue_*/results.json'))}
    evidence = [
        'closing_implementation_attempts.json',
        'retrospective_research/monthly_arrival/manifest.json',
        'retrospective_research/monthly_arrival/independent_oracle.json',
        'retrospective_research/factorial_analysis/protocol.json',
        'retrospective_research/factorial_analysis/analysis.json',
        'retrospective_research/state_canary/analysis.json',
        'retrospective_research/faiss_batch_diagnostic/analysis.json',
        'models_retrieval/tabdpt_batch_v3_1/batch_canary_cuda_s20260916/manifest.json',
        'models/sequence_result_audit/clock_innovations_independent.json',
        'models/sequence_result_audit/innovations_native_replay.json',
        'models/sequence_result_audit/innovations_matched_audit.json',
        'models/sequence_clock_innovations/models/summary.json',
        'models/sequence_result_audit/monthly_native_replay.json',
        'models/sequence_result_audit/monthly_matched_audit.json',
        'models/sequence_result_audit/missing_information/audit.json',
        'missing/route_composition_v3/summary.json',
        'missing/route_composition_v4/summary.json',
        'missing/route_composition_v4/verification.json',
        'missing/route_composition_v4/error_budget/summary.json',
        'models/context_gate/final_simplex9_v1/score_summary.json',
        'models/context_gate/final_simplex9_v1/independent_verification.json',
        'models/opdi_missing_v3/summary.json',
        'missing/monthly_arrival_model/summary.json',
    ]
    result['followup_evidence'] = [reference(BASE / path) for path in evidence]
    factorial = BASE / 'retrospective_research/factorial_analysis/analysis.json'
    result['factorial_analysis'] = {'reference': reference(factorial)}
    if factorial.exists():
        fact = read(factorial)
        result['factorial_analysis'].update({key: fact[key] for key in ('status', 'seasonal_rmse_sec', 'seasonal_contrasts', 'scope', 'limits', 'decision') if key in fact})
    result['compositions'] = []
    for generation in (3, 4):
        path = BASE / f'missing/route_composition_v{generation}/summary.json'
        if path.exists():
            result['compositions'].append({'generation': generation, 'reference': reference(path), 'result': read(path),
                                           'independent_verification': reference(path.with_name('verification.json'))})
    result['manifest_inventory'] = []
    for path in sorted(BASE.rglob('manifest.json')):
        data = read(path)
        result['manifest_inventory'].append({**reference(path), 'status': data.get('status', 'status not recorded'),
                                            'completed_utc': data.get('completed_utc')})
    result['manifest_status_counts'] = dict(Counter(item['status'] for item in result['manifest_inventory']))
    result['unlaunched_scope'] = {'broader_ordered337_F2_G1': 'Parent reserved deeper_sequence_broader; no launch as of13:39 due union/new-information jobs and deadline. Inventory is authoritative if status changes.',
                                'FTTransformer': 'No implemented wrapper', 'TabPFN35': 'No gated weight download or acceptance',
                                'TimesFM3': 'No weight download or model run', 'daily_ATFM': 'Catalog and HEAD only',
                                'TabICL_augmented': 'Prepared wrapper only; original missing-NM result is distinct'}
    result['status_warning'] = 'Metadata snapshot, not a new audit or promotion. Completed producer manifest is not independent replay. Full score cohorts are distinct from full training. F1/F3 are exposed development folds, not official or fresh holdouts. Public author scores are unverified and noncomparable. Preserve original failure and intermediate receipts.'
    return result


def render(result, tag):
    rows = []
    for exp in result['experiments']:
        scores = exp.get('seasonal_rmse', {})
        state = ', '.join(f"{fold}: {record['status']}" for fold, record in exp['folds'].items())
        candidate = f"{scores['candidate']:.6f}" if 'candidate' in scores else 'not complete'
        blend = f"{scores['blend25']:.6f}" if 'blend25' in scores else 'not complete'
        rows.append(f"| {exp['name']} | {candidate} | {blend} | {exp['scope']} ({state}) |")
    template = Path(__file__).with_name('shortlist_final_1402.template.md').read_text(encoding='utf-8')
    composition_rows = []
    for item in result['compositions']:
        for name, variant in sorted(item['result'].get('variants', {}).items()):
            metrics = variant.get('seasonal_rmse', {})
            if 'composed' in metrics:
                composition_rows.append(f"| v{item['generation']} {name} | {metrics['composed']:.6f} | {metrics.get('V2', float('nan')):.6f} | {variant.get('every_day_removal_improves_vs_previous', 'not recorded')} |")
    compositions = '\n'.join(composition_rows) or 'No complete composition summary at snapshot.'
    factorial = result['factorial_analysis']
    factorial_text = 'The four-arm independent factorial analysis is not complete at this snapshot. No interaction conclusion is drawn.'
    if factorial.get('status') == 'passed':
        changes = factorial['seasonal_contrasts']
        lines = ['The frozen independent four-arm analysis passed its cohort/label/settings/prediction-hash checks; it checks the producer replay receipt but does not claim another model reload.', '',
                 '| Contrast | Seasonal MSE difference (s^2) | Leave-one-day-out range (s^2) |', '|---|---:|---|']
        for name, contrast in changes.items():
            bounds = contrast['seasonal_leave_one_day_out_range_sec2']
            lines.append(f"| {name.replace('_', ' ')} | {contrast['delta_mse_sec2']:.6f} | {bounds[0]:.6f} to {bounds[1]:.6f} |")
        lines.extend(['', 'Negative MSE differences favor the added block; the interaction is MSE(all)-MSE(past+future)-MSE(past+day)+MSE(past). Its positive value indicates overlapping/subadditive gains, not synergy. It is an error-reduction interaction under this model/training contract, not a causal estimate. Day after future adds only 0.087217 seasonal seconds. Every seasonal increment retains its sign under day removal, but the tiny November future-after-day increment does not: its MSE difference is -1.619982 with within-November day-removal range -25.728818 to +23.847378. The overlap interaction remains positive in each fold under all day removals. These adaptive follow-ups are excluded from final9 and do not justify selecting a new blend on score outcomes.'])
        factorial_text = '\n'.join(lines)
    return (template.replace('{{SNAPSHOT}}', result['snapshot_utc']).replace('{{TAG}}', tag)
            .replace('{{TABLE}}', '\n'.join(rows)).replace('{{COMPOSITIONS}}', compositions)
            .replace('{{FACTORIAL}}', factorial_text))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', default='1402')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    if not args.tag.isdigit():
        raise ValueError('Numeric snapshot tag required')
    result = build()
    rendered = render(result, args.tag)
    serialized = json.dumps(result, indent=2, allow_nan=False)
    assert '{{' not in rendered
    assert len({item['name'] for item in result['experiments']}) == len(result['experiments'])
    if args.check:
        print(json.dumps({'snapshot_utc': result['snapshot_utc'], 'manifest_status_counts': result['manifest_status_counts'],
                          'experiments': [{'name': item['name'], 'scores': item.get('seasonal_rmse'),
                                           'status': {fold: record['status'] for fold, record in item['folds'].items()}}
                                          for item in result['experiments']]}, indent=2))
        return
    targets = [OUT / f'sources_final_{args.tag}.json', OUT / f'model_shortlist_final_{args.tag}.md']
    if any(path.exists() for path in targets):
        raise FileExistsError('Preserve previous snapshot; use a new numeric tag')
    with targets[0].open('x', encoding='utf-8') as stream:
        stream.write(serialized)
    with targets[1].open('x', encoding='utf-8') as stream:
        stream.write(rendered)
    print(json.dumps({'snapshot_utc': result['snapshot_utc'], 'outputs': [reference(path) for path in targets]}, indent=2))


if __name__ == '__main__':
    main()
