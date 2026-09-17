"""Finish minor label/source-link corrections without altering prior report versions."""
from pathlib import Path
import json
import sys
import build_report_v3 as prior


def main():
    frozen = prior.frozen
    old_band = frozen.ErrorBand
    decision = frozen.read(Path(sys.argv[sys.argv.index('--decision') + 1]))
    shortlist = decision.get('shortlist_filename', 'model_shortlist_final.md')
    sources = decision.get('sources_filename', 'sources_final.json')

    class ErrorBand(old_band):
        def __init__(self, parts):
            super().__init__([(name.replace('over30', 'over 30'), fraction, color)
                              for name, fraction, color in parts])

    prior.REPLACEMENTS.update({
        'model_shortlist.md and sources.json in output/breakthrough_20260916/research_gap':
            shortlist + ' and ' + sources + ' in output/breakthrough_20260916/research_gap',
    })
    frozen.ErrorBand = ErrorBand
    try:
        prior.main()
    finally:
        frozen.ErrorBand = old_band
    output = Path(sys.argv[sys.argv.index('--output') + 1])
    path = output / 'report_receipt.json'
    receipt = frozen.read(path)
    receipt['final_typography_wrapper_sha256'] = frozen.sha(__file__)
    receipt['corrections'].extend(['Error-band label spacing', 'Source-list filenames match bound snapshot'])
    path.write_text(json.dumps(receipt, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
