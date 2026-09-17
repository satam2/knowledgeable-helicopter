"""Isolated v2 canary/residual commands; parent owns all CUDA launches."""
import torch
import argparse
import sys
from pathlib import Path
import tabdpt
import tabdpt.model
import tabdpt.utils
import tabdpt.estimator

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
import run_pilot as core
import run_residual as residual
import adapter

OUT=core.external_path(core.OUT/'tabdpt_v2')


def source_paths():
    return {'v2_run.py':Path(__file__),'v2_adapter.py':Path(adapter.__file__),
            'frozen_run_pilot.py':Path(core.__file__),'frozen_run_residual.py':Path(residual.__file__),
            'frozen_tabdpt_adapter.py':Path(adapter.frozen.__file__),
            'frozen_preprocessing.py':HERE.parent/'preprocessing.py','prior_common.py':Path(core.common.__file__),
            'official_tabdpt_model.py':Path(tabdpt.model.__file__),'official_tabdpt_utils.py':Path(tabdpt.utils.__file__),
            'official_tabdpt_estimator.py':Path(tabdpt.estimator.__file__)}


def hashes():
    return {name:core.sha256(path) for name,path in source_paths().items()}


def configure():
    core.OUT=OUT
    core.module=lambda family:adapter if family=='tabdpt' else (_ for _ in ()).throw(ValueError('V2 only supports TabDPT'))
    core.hashes=hashes
    residual.source_paths=source_paths
    residual.hashes=hashes


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--canary',action='store_true')
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--device',choices=['cpu','cuda'],required=True)
    parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--seed',type=int,default=20260916)
    parser.add_argument('--folds',nargs='+',choices=['F1','F3'],default=['F1','F3'])
    parser.add_argument('--max-score-seconds',type=float,default=600)
    args=parser.parse_args()
    if args.threads!=2:
        raise ValueError('V2 declared two CPU threads')
    args.family='tabdpt'
    configure()
    OUT.mkdir(parents=True,exist_ok=True)
    protocol=OUT/f'protocol_{args.device}_s{args.seed}.json'
    payload={'source_hashes':hashes(),'seed':args.seed,'device':args.device,'threads':args.threads,
             'folds':args.folds,'reference_limit':32000,'context':256,'batch':16,'ensembles':1,
             'attention':'officialuse_flashFalse plusPyTorchSDPBackend.MATH; noinstalledpackageedits',
             'numeric_change':'NoGPUautocast; modeldefaultfloat32','max_score_seconds':args.max_score_seconds,
             'labels':'Unchanged original rawY-P; no clipping/no changedregressionbin resolution',
             'evaluation':'Frozen32Kfit/refitreferences; alltunerows saved; every originalscorerow;missingNMV2exact'}
    if protocol.exists() and core.read_json(protocol)!=payload:
        raise ValueError('Frozenv2protocol changed')
    core.write_json(protocol,payload)
    snapshot=OUT/'source'
    snapshot.mkdir(exist_ok=True)
    for name,path in source_paths().items():
        destination=snapshot/name
        if destination.exists() and core.sha256(destination)!=core.sha256(path):
            raise ValueError('Frozenv2snapshot changed')
        if not destination.exists():
            core.shutil.copyfile(path,destination)
    if args.prepare_only:
        assert not torch.cuda.is_initialized()
        print('PREPARED',protocol,'CUDAuninitialized',flush=True)
        return
    if args.canary:
        core.canary(args)
        return
    forwarded=[str(Path(residual.__file__)),'--family','tabdpt','--device',args.device,'--threads',str(args.threads),
               '--seed',str(args.seed),'--folds',*args.folds,'--max-score-seconds',str(args.max_score_seconds)]
    sys.argv=forwarded
    residual.main()


if __name__=='__main__':
    main()
