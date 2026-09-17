"""Corrected report narrative with decision-declared supplementary evidence."""
import argparse
import json
from pathlib import Path
import build_report as frozen
import build_report_v2 as prior

REPLACEMENTS = {
    **prior.REPLACEMENTS,
    'about2-3': 'about 2-3',
    'matched287.16': 'matched 287.16',
    'up to 2500trees63leaves': 'up to 2,500 trees / 63 leaves',
    'Matched information gains are about 11 seconds; the capacity change adds about 2-3 seconds.':
        'Matched information gains are about 11 seconds; the tested capacity package adds about 2-3 seconds.',
    'A fit-derived lower bound removes two extreme negative V2 local predictions, but changes zero existing V2 ranking predictions.':
        'A fit-derived lower bound clips five July V2 predictions; two extreme rows dominate the gain. It changes zero existing submitted V2 ranking predictions.',
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--decision', type=Path, required=True)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    decision = frozen.read(args.decision)
    supplemental_paths = [
        frozen.ROOT / 'private_runs/breakthrough_20260916/information_capacity_1100/audit.json',
        frozen.ROOT / 'private_runs/breakthrough_20260916/oracle_diversity_1050/summary.json',
        frozen.ROOT / 'private_runs/breakthrough_20260916/oracle_diversity_1050/protocol.json',
        *(frozen.ROOT / path for path in decision['supplemental_evidence']),
    ]
    supplemental = {str(path): frozen.sha(path) for path in supplemental_paths}
    assert round(frozen.read(supplemental_paths[1])['seasonal_rmse']['discrete_oracle'], 2) == 219.47
    original_p = frozen.layout.p

    def corrected(text, style='BodyPRC'):
        for before, after in REPLACEMENTS.items():
            text = text.replace(before, after)
        return original_p(text, style)

    frozen.layout.p = corrected
    try:
        frozen.build(args)
    finally:
        frozen.layout.p = original_p
    for path, digest in supplemental.items():
        if frozen.sha(path) != digest:
            raise ValueError('Supplemental evidence changed during report rendering')
    receipt_path = args.output / 'report_receipt.json'
    receipt = frozen.read(receipt_path)
    receipt.update(wrapper_sha256=frozen.sha(__file__), typography_source_sha256=frozen.sha(prior.__file__),
                   supplemental_evidence=supplemental, evidence_cutoff_utc=decision['evidence_cutoff_utc'],
                   corrections=['Explicit standalone/blend variants in decision table',
                                'Five July support-floor changes, two dominant',
                                'Tested capacity package, not universal model ceiling',
                                'Selected predictor error chart computed from bound inventory'])
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print('FINAL_REPORT_EVIDENCE_BOUND', len(supplemental), flush=True)


if __name__ == '__main__':
    main()
