"""Classify diagnostic/control rows and enforce cross-fold pairing in a new inventory."""
import contextlib
import io
import json
import re
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import collect_evidence as frozen


def role(recipe, variant):
    if 'support_guard/' in recipe or variant == 'support_projected':
        return 'support_diagnostic'
    if '/simplex' in recipe and variant in {'lgb_allinfo', 'lgb_leaf63', 'tabm_source'}:
        return 'component_control'
    if variant.startswith('control_') or '/matched_models/control/' in recipe:
        return 'matched_control'
    if variant in {'candidate', 'blend25', 'raw', 'global_simplex', 'airport_shrunk_simplex',
                   'airport3', 'global3', 'selected_candidate', 'selected_blend25',
                   'adapted_candidate', 'adapted_blend25'}:
        return 'experimental_predictor'
    return 'unclassified_review_required'


def seed(record, path):
    if record.get('seed') is not None:
        return str(record['seed'])
    match = re.search(r'_s(\d+)(?:_|/|$)', path.as_posix())
    return match.group(1) if match else None


def main():
    if '--output' not in sys.argv:
        raise ValueError('Explicit new --output directory required')
    output = Path(sys.argv[sys.argv.index('--output') + 1])
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        frozen.main()
    (output / 'v1_collection.log').write_text(log.getvalue(), encoding='utf-8')
    table = pd.read_csv(output / 'folds.csv')
    seasonal = pd.read_csv(output / 'seasonal.csv')
    inventory = frozen.common.read_json(output / 'inventory.json')
    if table.duplicated(['recipe', 'variant', 'fold']).any():
        raise ValueError('Duplicate recipe/variant/fold would make seasonal or broader aggregation ambiguous')
    reference_receipts = {}
    for fold in frozen.FOLDS:
        directory = frozen.common.WORKSPACE / 'private_runs/next_230/models' / f'clock_and_rome_ensemble_{fold}_s20260910'
        manifest = frozen.common.read_json(directory / 'manifest.json')
        prediction = directory / 'score_predictions.parquet'
        if frozen.common.sha256(prediction) != manifest['outputs'][prediction.name]:
            raise ValueError('Reference predictions changed')
        reference_receipts[fold] = {'manifest': str(directory / 'manifest.json'),
            'manifest_sha256': frozen.common.sha256(directory / 'manifest.json'),
            'prediction_sha256': manifest['outputs'][prediction.name]}
    cache = {}
    for filename in table.manifest.unique():
        path = Path(filename)
        record = frozen.common.read_json(path)
        if frozen.common.sha256(path) != inventory['manifest_receipts'][filename]:
            raise ValueError('Source manifest changed during collection')
        cache[filename] = record
    table['evidence_role'] = [role(r.recipe, r.variant) for r in table.itertuples()]
    table['seed_recorded_or_path'] = [seed(cache[r.manifest], Path(r.manifest)) for r in table.itertuples()]
    table['fold_role'] = table.fold.map({'F1': 'July exposed development', 'F3': 'November exposed winter proxy',
                                       'F2': 'October exposed development', 'G1': 'Same October score rows, longer training gap'})
    reports = {}
    rows = []
    for entry in seasonal.to_dict('records'):
        group = table[(table.recipe == entry['recipe']) & (table.variant == entry['variant'])]
        pair = group[group.fold.isin(frozen.WEIGHTS)]
        issues = []
        if len(pair) != 2 or set(pair.fold) != set(frozen.WEIGHTS):
            issues.append('Expected exactly oneF1 and oneF3')
        known_seeds = set(pair.seed_recorded_or_path.dropna())
        if len(known_seeds) > 1:
            issues.append('Different explicit seeds')
        records = [cache[path] for path in pair.manifest]
        for key in ('family', 'formulation', 'feature_columns', 'source_hashes', 'protocol_sha256'):
            values = [record.get(key) for record in records]
            if all(value is not None for value in values) and values[0] != values[1]:
                issues.append('Different crossfold contract field:' + key)
        kind = role(entry['recipe'], entry['variant'])
        entry.update(evidence_role=kind, seed_recorded_or_path=','.join(sorted(known_seeds)) or None,
            pairing_status='requires_review' if issues else 'passed_available_contracts',
            pairing_issues='; '.join(issues),
            eligible_for_predictive_comparison=kind in {'experimental_predictor', 'matched_control'} and not issues,
            broader_fold_overlap='F2/G1 share original October score cohort; not two independent test months',
            score_receipts=json.dumps({row.fold: {'manifest': row.manifest,
                'manifest_sha256': row.manifest_sha256, 'prediction_sha256': row.prediction_sha256}
                for row in pair.itertuples()}, sort_keys=True))
        rows.append(entry)
        reports[entry['recipe'] + '/' + entry['variant']] = {'issues': issues,
            'source_hashes_present_both': all(bool(r.get('source_hashes')) for r in records),
            'protocol_sha256_present_both': all(bool(r.get('protocol_sha256')) for r in records),
            'explicit_seed_present_both': all(r.get('seed') is not None for r in records),
            'limitation': 'Missing optional direct manifest fields are not reconstructed; composition source contracts require their independent receipts.'}
    result = pd.DataFrame(rows)
    result = result.rename(columns={'every_day_removal_improves': 'every_day_removal_improves_vs_reference'})
    result.sort_values(['eligible_for_predictive_comparison', 'rmse'], ascending=[False, True]).to_csv(output / 'seasonal.csv', index=False)
    table.to_csv(output / 'folds.csv', index=False)
    inventory.update(wrapper_sha256=frozen.common.sha256(__file__),
        classifications={str(k): int(v) for k, v in result.evidence_role.value_counts().items()},
        pairing_review=reports, reference_receipts=reference_receipts,
        promotion='No automatic promotion. Predictive comparison eligibility is a reporting classification, not independent validation or a deployment decision.',
        ranking_warning='Support diagnostics must never be presented as the best predictive model. Fit-derived floor changes zero existing V2 ranking predictions; its effect on unbuilt new ranking models is unknown.',
        source_files={'v1': frozen.common.sha256(frozen.__file__), 'v2': frozen.common.sha256(__file__)})
    frozen.common.write_json(output / 'inventory.json', inventory)
    frozen.common.write_json(output / 'outputs.json', {'created_utc': frozen.common.utc_now(),
        'hashes': {name: frozen.common.sha256(output / name) for name in ['folds.csv', 'seasonal.csv', 'inventory.json', 'v1_collection.log']}})
    print(result[result.eligible_for_predictive_comparison][['recipe', 'variant', 'rmse']].sort_values('rmse').head(20).to_string(index=False), flush=True)
    print('CLASSIFIED', inventory['classifications'], flush=True)


if __name__ == '__main__':
    main()
