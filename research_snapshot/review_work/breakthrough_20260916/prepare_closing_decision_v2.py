"""Clarify the GRU table rationale while preserving the earlier decision builder."""
import hashlib
import json
from pathlib import Path
import sys
import prepare_closing_decision as prior


def main():
    output_index = sys.argv.index('--output') + 1
    destination = prior.external_path(Path(sys.argv[output_index]))
    assert not destination.exists()
    original_arguments = list(sys.argv)
    intermediate = destination.with_name(destination.stem + '_preclarification.json')
    sys.argv[output_index] = str(intermediate)
    try:
        prior.main()
    finally:
        sys.argv[:] = original_arguments
    decision = prior.read(intermediate)
    rows = [row for row in decision['comparison_rows'] if row['label'] == 'GRU sequence, 25% V2 blend']
    assert len(rows) == 1
    assert rows[0]['note'] == '200K fit sample; standalone 308.80; loses'
    rows[0]['note'] = '200K fit sample; standalone 308.80; replay fails'
    decision['clarification'] = {
        'scope': 'GRU note clarifies failed replay; the fixed blend itself improves V2 locally.',
        'prior_decision': str(intermediate),
        'prior_decision_sha256': hashlib.sha256(intermediate.read_bytes()).hexdigest(),
        'wrapper_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    destination.write_text(json.dumps(decision, indent=2) + '\n', encoding='utf-8')
    print('CLARIFIED_DECISION', destination, flush=True)


if __name__ == '__main__':
    main()
