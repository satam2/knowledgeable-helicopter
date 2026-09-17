"""F3-only scheduling change; fixed trees and single-thread predictions."""
import argparse
from pathlib import Path
import run as original
import parallel_fit

common = original.common
OUT = common.external_path(original.OUT.parent / 'v2')


def declare():
    prior = original.declare()
    record = dict(prior, wrapper_sha256=common.sha256(__file__),
        helper_sha256=common.sha256(parallel_fit.__file__),
        canary_source_sha256=common.sha256(Path(__file__).with_name('test_parallel_fit.py')),
        prior_protocol_sha256=common.sha256(original.OUT / 'protocol.json'),
        allocation_only='F3only ExtraTrees.fit n_jobs4; prediction n_jobs1 restoredafter everyfit. Same300 per-tree seeds/inputs/targets/weights/.7features/minleaf1 and modelclass algorithm. OriginalF1v1unchanged.',
        canary='Beforedeclaration, fixed601x17synthetic weighted+unweighted300tree fits at1/4workers had bitwiseequal fullper-tree state (children/features/thresholds/impurity/counts/weightedcounts/values) and single-thread predictions.',
        resources='F3:4CPU duringforestfit,1CPU BLAS/prediction;3GiBcurrent+historicalpeak,8GiBhostreserve,start11GiB, noGPU; parentcoordinated.')
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/'protocol.json'
    if path.exists():
        assert common.read_json(path)==record
    else:
        common.write_json(path,record)
    return record


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--declare-only',action='store_true')
    args=parser.parse_args()
    protocol=declare()
    if args.declare_only:
        print(common.sha256(OUT/'protocol.json'),flush=True)
    else:
        original.OUT=OUT
        original.declare=lambda:protocol
        original.ExtraTreesRegressor=parallel_fit.ParallelFitExtraTrees
        original.forest.ExtraTreesRegressor=parallel_fit.ParallelFitExtraTrees
        with original.threadpool_limits(1):
            original.run('F3')
