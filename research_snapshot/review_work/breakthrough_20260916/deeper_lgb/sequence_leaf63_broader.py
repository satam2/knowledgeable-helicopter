"""Same verified four-event model on the original F2/G1 temporal contracts."""
import lightgbm
import shutil
from pathlib import Path
import sequence_leaf63 as frozen

CORE = frozen.core
ROOT = CORE.ROOT
ORIGINAL_OUT = ROOT / 'private_runs/breakthrough_20260916/deeper_sequence'
OUT = ROOT / 'private_runs/breakthrough_20260916/deeper_sequence_broader'


def main():
    real_external = CORE.external_path
    CORE.external_path = lambda path: real_external(OUT if Path(path) == ORIGINAL_OUT else path)
    original_hashes = CORE.source_hashes
    CORE.source_hashes = lambda: {**original_hashes(), str(Path(__file__).relative_to(ROOT)): CORE.sha256(__file__)}
    real_declare = CORE.declare

    def declare(args):
        if set(args.folds) != {'F2', 'G1'}:
            raise ValueError('Broader verification requires the unchanged original F2 and G1 folds')
        path = real_declare(args)
        snapshot = CORE.OUT / 'source_snapshots' / f'{args.family}_{args.formulation}_s{args.seed}'
        shutil.copyfile(__file__, snapshot / Path(__file__).name)
        receipt = {'wrapper_sha256': CORE.sha256(__file__), 'frozen_runner_sha256': CORE.sha256(frozen.__file__),
                   'execution': 'LightGBM imported first; original337features/leaf63parameters/seed2CPU/fullcohorts.',
                   'cohort_warning': 'F2 and G1 share October scoring rows. Different training gaps, not two independent test months.',
                   'selection': 'Follow-up to F1/F3 ordered-information development result, not a fresh holdout.',
                   'gpu_used': False}
        target = CORE.OUT / 'broader_receipt.json'
        if target.exists() and CORE.read_json(target) != receipt:
            raise ValueError('Frozen broader declaration changed')
        CORE.write_json(target, receipt)
        return path

    CORE.declare = declare
    frozen.main()


if __name__ == '__main__':
    main()
