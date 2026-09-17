"""Add independently reviewed route variants to the frozen evidence collector."""
import sys
from pathlib import Path
import collect_evidence_v2 as frozen

MISSING_ROUTE_RECIPES = {
    'feature_screens/models/lightgbm_direct_linkage_full_{fold}_s20260916',
    'feature_screens/models/lightgbm_direct_trajectory_full_{fold}_s20260916',
    'physical_v2/models/basephysicalfull_{fold}_s20260916',
    'physical_v2/models/surfacegeometry_{fold}_s20260916',
    'physical_v2/models/weather_{fold}_s20260916',
}
CONTEXT_SIMPLEX_RECIPE = 'models/context_gate/simplex4_context_v1/{fold}_score'


def main():
    common = frozen.frozen.common
    receipt_path = common.WORKSPACE / 'private_runs/breakthrough_20260916/missing/inventory_roles/verification.json'
    receipt_digest = common.sha256(receipt_path)
    old_role = frozen.role

    def reviewed_role(recipe, variant):
        if recipe in MISSING_ROUTE_RECIPES and variant in {'missing_only', 'missing_blend25'}:
            return 'experimental_predictor'
        if recipe == CONTEXT_SIMPLEX_RECIPE and variant in {'global4', 'airport4'}:
            return 'experimental_predictor'
        return old_role(recipe, variant)

    frozen.role = reviewed_role
    try:
        frozen.main()
    finally:
        frozen.role = old_role
    output = Path(sys.argv[sys.argv.index('--output') + 1])
    inventory = common.read_json(output / 'inventory.json')
    assert common.sha256(receipt_path) == receipt_digest
    inventory.update(classifier_v3_sha256=common.sha256(__file__),
                     missing_route_review={'path': str(receipt_path), 'sha256': receipt_digest},
                     v3_scope={'missing_routes': sorted(MISSING_ROUTE_RECIPES),
                               'context_simplex': CONTEXT_SIMPLEX_RECIPE},
                     classifier_warning='Classification permits comparison only; no automatic model promotion. Unknown variants remain unclassified.')
    common.write_json(output / 'inventory.json', inventory)
    common.write_json(output / 'outputs.json', {'created_utc': common.utc_now(),
        'hashes': {name: common.sha256(output / name) for name in ['folds.csv', 'seasonal.csv', 'inventory.json', 'v1_collection.log']}})
    print('V3_REVIEW_BOUND', receipt_digest, flush=True)


if __name__ == '__main__':
    main()
