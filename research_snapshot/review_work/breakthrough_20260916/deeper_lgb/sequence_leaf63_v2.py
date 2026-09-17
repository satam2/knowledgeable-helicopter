"""Import-order repair only; frozen337-feature leaf63 experiment, new artifacts."""
import lightgbm
import shutil
from pathlib import Path
import sequence_leaf63 as frozen

CORE=frozen.core
ROOT=CORE.ROOT
ORIGINAL_OUT=ROOT/'private_runs/breakthrough_20260916/deeper_sequence'
OUT=ROOT/'private_runs/breakthrough_20260916/deeper_sequence_v2'


def main():
    real_external=CORE.external_path
    CORE.external_path=lambda path: real_external(OUT if Path(path)==ORIGINAL_OUT else path)
    original_hashes=CORE.source_hashes
    CORE.source_hashes=lambda:{**original_hashes(),str(Path(__file__).relative_to(ROOT)):CORE.sha256(__file__)}
    real_declare=CORE.declare

    def declare(args):
        path=real_declare(args)
        snapshot=CORE.OUT/'source_snapshots'/f'{args.family}_{args.formulation}_s{args.seed}'
        shutil.copyfile(__file__,snapshot/Path(__file__).name)
        receipt={'wrapper_sha256':CORE.sha256(__file__),'frozen_runner_sha256':CORE.sha256(frozen.__file__),
                 'repair':'Import lightgbm before numpy,pandas,xgboost; samefrozenadapter/features/labels/parameters/cohorts',
                 'prior_failed_attempt':str(ORIGINAL_OUT/'lightgbm_leaf63_sequence_aobt_allfinite_F1_s20260916'),
                 'diagnostic_fixture':'private_runs/breakthrough_20260916/models/sequence_result_audit/native_canary/fixture.json',
                 'gpu_used':False}
        target=CORE.OUT/'repair_receipt.json'
        if target.exists() and CORE.read_json(target)!=receipt:
            raise ValueError('Frozen importorder repair changed')
        CORE.write_json(target,receipt)
        return path

    CORE.declare=declare
    frozen.main()


if __name__=='__main__':
    main()
