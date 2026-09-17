"""Bounded combined-width PLE8 mechanics; no real fit or CUDA initialization."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
import sys
import time
import io
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import joblib
import psutil

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'review_work/breakthrough_20260916/models'))
sys.path.insert(0,str(ROOT/'review_work/campaign_20260916'))
import common
import tabm_ple_gpu as ple


def main():
    torch.set_num_threads(2)
    torch.manual_seed(20260916)
    rng=np.random.default_rng(20260916)
    x=pd.DataFrame({f'n{i}':rng.normal(size=128).astype('float32') for i in range(214)})
    for i in range(11):
        x[f'c{i}']=pd.Categorical(rng.integers(0,32,size=len(x)).astype(str))
    x.loc[::5,'n0']=np.nan
    x['n1']=1.
    labels=rng.normal(size=len(x))
    original=ple.frozen.Network
    captured=[]
    real_bins=ple.fit_bins

    def bins(numbers,count):
        value=real_bins(numbers,count)
        captured.append([b.clone() for b in value[0]])
        return value

    with patch.object(ple.frozen,'fit',return_value=({}, {'params':{}})),patch.object(ple,'fit_bins',side_effect=bins):
        ple.fit(x,labels,(x.copy(),labels),threads=2)
        altered=x.copy()
        altered['n0']=np.full(len(altered),1e12,dtype=np.float32)
        ple.fit(x,labels,(altered,np.full(len(x),1e12)),threads=2)
    assert ple.frozen.Network is original
    assert len(captured[0])==len(captured[1])
    for a,b in zip(*captured):
        torch.testing.assert_close(a,b,rtol=0.,atol=0.)
    encoder=ple.FrameEncoder().fit(x,neural=True)
    numbers,categories=ple.frozen.tensors(encoder,x)
    bins,embedded,passthrough=real_bins(numbers,len(encoder.numeric))
    network=ple.Network(numbers.shape[1],[len(m)+2 for m in encoder.categories.values()],bins,embedded,passthrough,8)
    began=time.monotonic()
    output=network(numbers[:64],categories[:64])
    assert output.shape==(64,8)
    loss=(output-torch.from_numpy(labels[:64]).float()[:,None]).square().mean()
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in network.parameters())
    network.eval()
    with torch.inference_mode():
        expected=network(numbers[:16],categories[:16])
    saved=io.BytesIO()
    joblib.dump(network,saved)
    saved.seek(0)
    reloaded=joblib.load(saved)
    with torch.inference_mode():
        actual=reloaded(numbers[:16],categories[:16])
    torch.testing.assert_close(expected,actual,rtol=0.,atol=0.)
    assert not torch.cuda.is_initialized()
    embedded_width=len(embedded)*16+len(passthrough)+sum(e.embedding_dim for e in network.embeddings)
    record={'status':'passed','input_width':225,'numeric_columns':214,'categorical_columns':11,'members':8,
            'fit_only_bins_tune_mutation_equal':True,'frozen_constructor_restored':True,'forward_shape':[64,8],
            'finite_backward':True,'optimizer_step':False,'saved_replay_delta':0.,'cuda_initialized':False,
            'optimizer_omission':'Torch2.14CPUAdamWstep attempts acceleratorstream initialization; priorcanarycontexts exited, noGPUtensors/fit wereused',
            'parameters':sum(p.numel() for p in network.parameters()),'expanded_input_width':embedded_width,
            'batch4096_expanded_input_8members_float32_bytes':4096*8*embedded_width*4,
            'runtime_sec':time.monotonic()-began,'peak_rss_bytes':getattr(psutil.Process().memory_info(),'peak_wset',psutil.Process().memory_info().rss),
            'source_hashes':{str(p.relative_to(ROOT)):common.sha256(p) for p in [Path(__file__),Path(ple.__file__),Path(ple.frozen.__file__),ROOT/'review_work/breakthrough_20260916/full_neural/run.py']},
            'limitation':'Synthetic225width/64batch CPU mechanics only; memoryestimateexcludesembeddingintermediates/gradients/optimizer/fullfit tensors'}
    destination=ROOT/'private_runs/breakthrough_20260916/models/sequence_result_audit/ple8_cpu_canary.json'
    assert not destination.exists()
    common.write_json(destination,record)
    print('PLE8_CPU_CANARY',record,flush=True)


if __name__=='__main__':
    with patch.object(torch.cuda,'_lazy_init',side_effect=RuntimeError('CPU-only canary attempted CUDA initialization')):
        main()
