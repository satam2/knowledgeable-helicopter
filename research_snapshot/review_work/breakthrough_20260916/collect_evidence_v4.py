"""Classify the predeclared nine-expert blends without changing frozen collectors."""
import sys
from pathlib import Path
import collect_evidence_v3 as prior

RECIPE = 'models/context_gate/final_simplex9_v1/{fold}_score'


def main():
    classifier = prior.frozen
    common = classifier.frozen.common
    protocol = common.WORKSPACE / 'private_runs/breakthrough_20260916/models/context_gate/final_simplex9_v1/protocol.json'
    payload = common.read_json(protocol)
    assert len(payload['experts']) == 9 and payload['folds'] == ['F1', 'F3']
    digest = common.sha256(protocol)
    original = classifier.role

    def role(recipe, variant):
        if recipe == RECIPE and variant in {'global9', 'airport9'}:
            return 'experimental_predictor'
        return original(recipe, variant)

    classifier.role = role
    try:
        prior.main()
    finally:
        classifier.role = original
    output = Path(sys.argv[sys.argv.index('--output') + 1])
    assert common.sha256(protocol) == digest
    inventory = common.read_json(output / 'inventory.json')
    inventory.update(classifier_v4_sha256=common.sha256(__file__),
        final9_classification={'recipe': RECIPE, 'variants': ['global9', 'airport9'],
            'declaration_sha256': digest,
            'scope': 'Reporting role only; does not waive model replay failures or constitute independent validation.'})
    common.write_json(output / 'inventory.json', inventory)
    common.write_json(output / 'outputs.json', {'created_utc': common.utc_now(),
        'hashes': {name: common.sha256(output / name) for name in ['folds.csv', 'seasonal.csv', 'inventory.json', 'v1_collection.log']}})
    print('V4_FINAL9_DECLARATION_BOUND', digest, flush=True)


if __name__ == '__main__':
    main()
