"""F3-only grouped exact PLE quantiles to bound temporary CPU sort memory."""
import torch
import argparse
from pathlib import Path
import run as frozen

OUT=frozen.common.external_path(frozen.ROOT/'private_runs/tail240_20260916/state/neural_context/v2')


def grouped_fit_bins(numbers,n_original_numeric):
    varying=(numbers[:,:n_original_numeric]!=numbers[0,:n_original_numeric]).any(dim=0)
    embedded=torch.where(varying)[0]
    chosen=set(embedded.tolist())
    passthrough=torch.tensor([i for i in range(numbers.shape[1]) if i not in chosen],dtype=torch.long)
    bins=[]
    for columns in embedded.split(16):
        bins.extend(frozen.adapter.rtdl_num_embeddings.compute_bins(numbers[:,columns],n_bins=min(frozen.adapter.N_BINS,len(numbers)-1)))
    return bins,embedded,passthrough


def declare():
    old_root=frozen.ROOT/'private_runs/tail240_20260916/state/neural_context/v1'
    old=frozen.common.read_json(old_root/'protocol.json')
    assert frozen.common.sha256(frozen.__file__)==old['source_sha256']
    assert frozen.common.sha256(frozen.adapter.__file__)==old['adapter_sha256']
    assert frozen.common.sha256(frozen.tabm_gpu.__file__)==old['trainer_sha256']
    assert frozen.common.sha256(frozen.encoders.__file__)==old['encoder_sha256']
    record=dict(old,folds=['F3'],wrapper_sha256=frozen.common.sha256(__file__),
        prior_protocol_sha256=frozen.common.sha256(old_root/'protocol.json'),
        allocation_only_change='Sameofficialcompute_bins per16varyingnumericcolumns ratherthanallatonce; torch.quantile dim0 columnwise identical, allfitrows and48binsretained. Same order/architecture/trainer/encoder/seed/batch.',
        resources='ExclusiveCUDA,2CPU;20GiBsampledRSS,8GiBhostreserve; awaitparentCPUjobcompletionand>=25GiBfree beforeload; groupedbinworkspace avoidsfull-columnsortpeak.')
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/'protocol.json'
    if path.exists():
        assert frozen.common.read_json(path)==record
    else:
        frozen.common.write_json(path,record)
    return record


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--declare-only',action='store_true')
    args=parser.parse_args()
    protocol=declare()
    if not args.declare_only:
        assert frozen.psutil.virtual_memory().available>=25*1024**3
        frozen.OUT=OUT
        frozen.freeze=lambda:protocol
        frozen.adapter.fit_bins=grouped_fit_bins
        frozen.pa.set_cpu_count(2);frozen.pa.set_io_thread_count(1)
        with frozen.threadpool_limits(2):
            frozen.run('F3')
