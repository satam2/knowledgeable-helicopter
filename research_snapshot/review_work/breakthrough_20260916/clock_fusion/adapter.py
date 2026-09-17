"""Differentiable supplied-clock fusion with a learned direct expert."""
import math
import time
import torch
from torch import nn
import numpy as np
import tabm
from encoders import FrameEncoder

CLOCKS=['AOBT_3_flt','EOBT_1_flt','IOBT_flt','LOBT_flt','SCHED_TIME_UTC_mvt']
MODE='convex'


def raw_anchors(x):
    values=x[['takeoff_minus_'+clock for clock in CLOCKS]].to_numpy(dtype=np.float32)
    mask=np.isfinite(values)&(values!=-999999)
    return np.where(mask,values,0).astype(np.float32),mask


class Fusion(nn.Module):
    def __init__(self,n_numeric,cardinalities,mode):
        super().__init__()
        self.mode=mode
        dimensions=[min(32,max(4,math.ceil(math.sqrt(count)))) for count in cardinalities]
        self.embeddings=nn.ModuleList([nn.Embedding(count,dimension,padding_idx=0) for count,dimension in zip(cardinalities,dimensions)])
        for embedding in self.embeddings:
            with torch.no_grad():
                embedding.weight[1].zero_()
        self.backbone=tabm.TabM.make(n_num_features=n_numeric+sum(dimensions),d_out=8 if mode=='affine' else 7,
                                    n_blocks=3,d_block=256,k=8,dropout=.1,arch_type='tabm')
        self.register_buffer('gate_bias',torch.tensor([2.,0.,0.,0.,-2.,0.]))

    def forward(self,numbers,categories,anchors,available,return_details=False):
        features=[numbers]+[embedding(categories[:,i]) for i,embedding in enumerate(self.embeddings)]
        output=self.backbone(torch.cat(features,dim=1))
        observed=anchors[:,None,:].expand(-1,output.shape[1],-1)
        experts=torch.cat([observed,output[:,:,6:7]],dim=-1)
        allowed=torch.cat([available,torch.ones((len(available),1),device=available.device,dtype=torch.bool)],dim=1)
        logits=(output[:,:,:6]+self.gate_bias).masked_fill(~allowed[:,None,:],float('-inf'))
        weights=torch.softmax(logits,dim=-1)
        prediction=(weights*experts).sum(dim=-1)
        if self.mode=='affine':
            prediction=prediction+output[:,:,7]
        return (prediction,weights,experts) if return_details else prediction


def tensors(encoder,x,center,scale):
    numbers,categories=encoder.transform(x)
    anchors,available=raw_anchors(x)
    anchors=np.where(available,(anchors-center)/scale,0).astype(np.float32)
    return tuple(torch.from_numpy(a) for a in (numbers,categories,anchors,available))


def infer(network,values,device='cuda'):
    network.eval()
    result=np.empty(len(values[0]),dtype=float)
    with torch.inference_mode():
        for start in range(0,len(result),8192):
            end=min(start+8192,len(result))
            result[start:end]=network(*(v[start:end].to(device) for v in values)).mean(dim=1).cpu().numpy()
    return result


def fit(x,y,tuning=None,*,steps=None,seed=20260916,threads=4):
    assert torch.cuda.is_available()
    assert steps is None or (steps>=1 and tuning is None)
    y=np.asarray(y,float)
    assert len(y)==len(x) and np.isfinite(y).all()
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    started=time.monotonic()
    encoder=FrameEncoder().fit(x,neural=True)
    center=float(y.mean());scale=max(float(y.std()),1.)
    train=tensors(encoder,x,center,scale)
    tune=tensors(encoder,tuning[0],center,scale) if tuning is not None else None
    labels=torch.from_numpy(((y-center)/scale).astype(np.float32))
    network=Fusion(train[0].shape[1],[len(mapping)+2 for mapping in encoder.categories.values()],MODE).cuda()
    optimizer=torch.optim.AdamW(network.parameters(),lr=.001,weight_decay=.0001)
    generator=torch.Generator().manual_seed(seed)
    best=float('inf');best_epoch=1;best_state=None;history=[]
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(1,(80 if steps is None else int(steps))+1):
        network.train()
        total=0.
        for ids in torch.randperm(len(y),generator=generator).split(4096):
            optimizer.zero_grad(set_to_none=True)
            pred=network(*(v[ids].cuda() for v in train))
            loss=(pred-labels[ids].cuda()[:,None]).square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('Clock fusion produced nonfinite raw squared loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(),10.)
            optimizer.step()
            total+=float(loss.detach())*len(ids)
        row={'epoch':epoch,'train_mse_sec2':total/len(y)*scale**2}
        if tune is not None:
            pred=infer(network,tune)*scale+center
            mse=float(np.mean((pred-np.asarray(tuning[1],float))**2))
            row['tune_mse_sec2']=mse
            if mse<best:
                best,best_epoch=mse,epoch
                best_state={name:value.detach().cpu().clone() for name,value in network.state_dict().items()}
        else:
            best_epoch=epoch
        history.append(row)
        print('CLOCK_FUSION',MODE,row,flush=True)
        if tune is not None and epoch-best_epoch>=8:
            break
    if best_state is not None:
        network.load_state_dict(best_state)
    network.cpu().eval()
    peak=torch.cuda.max_memory_allocated()
    torch.cuda.empty_cache()
    return {'network':network,'encoder':encoder,'center':center,'scale':scale,'steps':best_epoch,'threads':threads},{
        'steps':best_epoch,'runtime_sec':time.monotonic()-started,'history':history,'mode':MODE,'rows':len(y),
        'gpu_peak_bytes':peak,'target':'unaltered raw seconds via affine standardization','anchors':CLOCKS,
        'parameters':{'members':8,'width':256,'blocks':3,'dropout':.1,'learning_rate':.001,'max_epochs':80,'patience':8}}


def predict(model,x):
    torch.set_num_threads(model['threads'])
    network=model['network'].cuda()
    result=infer(network,tensors(model['encoder'],x,model['center'],model['scale']))*model['scale']+model['center']
    network.cpu()
    torch.cuda.empty_cache()
    return result
