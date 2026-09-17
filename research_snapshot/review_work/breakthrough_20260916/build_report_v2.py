"""Report provenance and typography supplement; original rendered draft retained."""
import argparse
import json
from pathlib import Path
import build_report as frozen

REPLACEMENTS = {
    'scores282.91': 'scores 282.91', 'about11': 'about 11', 'only2-3': 'only 2-3',
    'Original30': 'Original 30', 'Combined225': 'Combined 225',
    'over30 minutes': 'over 30 minutes', 'scores219.47': 'scores 219.47',
    'to230': 'to 230', 'that230': 'that 230',
    'from238.29': 'from 238.29', 'to232.03': 'to 232.03',
    'from260.79': 'from 260.79', 'to256.51': 'to 256.51',
    '600trees31leaves': '600 trees / 31 leaves', '2500trees63leaves': '2500 trees / 63 leaves',
    '287.16 control': '287.16 control', 'historical leaf63': 'historical leaf63',
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--decision', type=Path, required=True)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    paths = [
        frozen.ROOT / 'private_runs/breakthrough_20260916/information_capacity_1100/audit.json',
        frozen.ROOT / 'private_runs/breakthrough_20260916/oracle_diversity_1050/summary.json',
        frozen.ROOT / 'private_runs/breakthrough_20260916/oracle_diversity_1050/protocol.json',
        frozen.ROOT / 'output/breakthrough_20260916/research_gap/model_shortlist.md',
        frozen.ROOT / 'output/breakthrough_20260916/research_gap/sources.json',
    ]
    supplemental = {str(path): frozen.sha(path) for path in paths}
    oracle = frozen.read(paths[1])['seasonal_rmse']['discrete_oracle']
    if round(oracle, 2) != 219.47:
        raise ValueError('Frozen oracle narrative no longer matches evidence')
    original_p = frozen.layout.p
    def p(text, style='BodyPRC'):
        for before, after in REPLACEMENTS.items():
            text = text.replace(before, after)
        return original_p(text, style)
    frozen.layout.p = p
    try:
        frozen.build(args)
    finally:
        frozen.layout.p = original_p
    for path, digest in supplemental.items():
        if frozen.sha(path) != digest:
            raise ValueError('Supplemental evidence changed during report build')
    receipt_path = args.output / 'report_receipt.json'
    receipt = frozen.read(receipt_path)
    receipt.update(wrapper_sha256=frozen.sha(__file__), supplemental_evidence=supplemental,
                   text_changes='Exact typography-only word/number spacing; numerical results unchanged.')
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print('SUPPLEMENTAL_EVIDENCE_BOUND', len(supplemental), flush=True)


if __name__ == '__main__':
    main()
