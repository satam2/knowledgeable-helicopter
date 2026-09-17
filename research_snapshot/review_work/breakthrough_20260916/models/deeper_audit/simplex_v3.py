"""Tune-only replacement of LGB600 by frozen leaf63, preserving simplex_v2."""
import argparse
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(HERE.parent / 'stacking'))
import audit_tuning
import run_simplex as frozen
from run_simplex_v2 import day_sensitivity

OUT = ROOT / 'private_runs/breakthrough_20260916/models/deeper_audit/simplex_v3'
TUNE = OUT / 'tune_only_audit'
original_registry = audit_tuning.registry


def registry(fold):
    return {'tabm_source': original_registry(fold)['tabm_source'],
            'lgb_leaf63': ROOT / 'private_runs/breakthrough_20260916/deeper_lgb/combined' /
            f'lightgbm_leaf63_aobt_allfinite_{fold}_s20260916'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true', help='Only original tune audit and pre-score declaration')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    audit_tuning.registry = registry
    audit_tuning.OUT = TUNE
    frozen.OUT = OUT
    frozen.EXPERTS = ('tabm_source', 'lgb_leaf63')
    frozen.day_sensitivity = day_sensitivity
    receipt = {'source_sha256': frozen.sha256(__file__), 'replacement': 'LightGBM600 -> combined leaf63',
               'unchanged_objective': 'Original raw tune SSE; global and airport-shrunk simplex; alpha in [0,1], no intercept',
               'shrinkage_rows': frozen.simplex.SHRINKAGE_ROWS,
               'score_exposure': 'Leaf63 chosen after exposed development score; stack weights learned only from original tune labels',
               'prior_v2_retained': str(ROOT / 'private_runs/breakthrough_20260916/models/stacking/simplex_v2'),
               'frozen_dependencies': {str(path.relative_to(ROOT)): frozen.sha256(path) for path in
                   [Path(frozen.__file__), Path(audit_tuning.__file__), Path(frozen.simplex.__file__),
                    HERE.parent / 'stacking/run_simplex_v2.py']}}
    declaration = OUT / 'replacement_declaration.json'
    if declaration.exists():
        assert frozen.read_json(declaration) == receipt
    else:
        frozen.write_json(declaration, receipt)
    if not (TUNE / 'tuning_audit.json').exists():
        audit_tuning.main()
    shutil.copyfile(__file__, OUT / Path(__file__).name)
    sys.argv = [__file__, '--declare-only'] if args.prepare else [__file__]
    frozen.main()


if __name__ == '__main__':
    main()
