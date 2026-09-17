"""Minibatch full-data TabM with train-only preprocessing and affine target scale."""
import copy
import math
import time
from importlib.metadata import version
import torch
from torch import nn
import numpy as np
import tabm
from encoders import FrameEncoder


class Network(nn.Module):
    def __init__(self,n_numeric,cardinalities):
        super().__init__()
        dimensions=[min(32,max(4,math.ceil(math.sqrt(count)))) for count in cardinalities]
        self.embeddings=nn.ModuleList([nn.Embedding(count,dimension,padding_idx=0) for count,dimension in zip(cardinalities,dimensions)])
        for embedding in self.embeddings:
            with torch.no_grad():
                embedding.weight[1].zero_()
        self.backbone=tabm.TabM.make(n_num_features=n_numeric+sum(dimensions),d_out=1,
            n_blocks=3,d_block=256,k=8,dropout=.1,arch_type="tabm")

    def forward(self,numbers,categories):
        values=[numbers]+[embedding(categories[:,index]) for index,embedding in enumerate(self.embeddings)]
        return self.backbone(torch.cat(values,dim=1)).squeeze(-1)


def tensors(encoder,x):
    numbers,categories=encoder.transform(x)
    return torch.from_numpy(numbers),torch.from_numpy(categories)


def infer(estimator,values,device,batch_size=8192):
    estimator.eval()
    result=np.empty(len(values[0]),dtype=np.float64)
    with torch.inference_mode():
        for start in range(0,len(result),batch_size):
            end=min(start+batch_size,len(result))
            numbers=values[0][start:end].to(device)
            categories=values[1][start:end].to(device)
            result[start:end]=estimator(numbers,categories).mean(dim=1).cpu().numpy()
    return result


def fit(x,y,tuning=None,*,steps=None,seed=20260916,threads=4):
    if not torch.cuda.is_available():
        raise RuntimeError("GPU TabM requires the separately configured CUDA runtime")
    if steps is not None and (steps<1 or tuning is not None):
        raise ValueError("Refit must use selected epochs without tuning")
    y=np.asarray(y,float)
    if not len(y) or len(y)!=len(x) or not np.isfinite(y).all():
        raise ValueError("Training requires nonempty aligned finite targets")
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False
    started=time.monotonic()
    encoder=FrameEncoder().fit(x,neural=True)
    train=tensors(encoder,x)
    tune=tensors(encoder,tuning[0]) if tuning is not None else None
    ymean=float(y.mean()); yscale=max(float(y.std()),1.)
    labels=torch.from_numpy(((y-ymean)/yscale).astype(np.float32))
    estimator=Network(train[0].shape[1],[len(mapping)+2 for mapping in encoder.categories.values()]).cuda()
    optimizer=torch.optim.AdamW(estimator.parameters(),lr=.001,weight_decay=.0001)
    epochs=40 if steps is None else int(steps)
    generator=torch.Generator().manual_seed(seed)
    history=[]; best_epoch=1; best_loss=float("inf"); best_state=None
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(1,epochs+1):
        epoch_start=time.monotonic()
        estimator.train()
        order=torch.randperm(len(labels),generator=generator)
        total=0.
        for ids in order.split(4096):
            optimizer.zero_grad(set_to_none=True)
            predicted=estimator(train[0][ids].cuda(),train[1][ids].cuda())
            target=labels[ids].cuda()
            loss=(predicted-target[:,None]).square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite TabM squared loss")
            loss.backward()
            gradient_norm=torch.nn.utils.clip_grad_norm_(estimator.parameters(),10.)
            optimizer.step()
            total+=float(loss.detach())*len(ids)
        row={"epoch":epoch,"train_mse_sec2":total/len(labels)*yscale**2,"seconds":time.monotonic()-epoch_start}
        if tune is not None:
            prediction=infer(estimator,tune,"cuda")*yscale+ymean
            mse=float(np.mean((prediction-np.asarray(tuning[1],float))**2))
            row["tune_mse_sec2"]=mse
            if mse<best_loss:
                best_loss,best_epoch=mse,epoch
                best_state={name:value.detach().cpu().clone() for name,value in estimator.state_dict().items()}
        else:
            best_epoch=epoch
        history.append(row)
        print("TABM_GPU_EPOCH",row,flush=True)
        if tune is not None and epoch-best_epoch>=5:
            break
    if best_state is not None:
        estimator.load_state_dict(best_state)
    gpu_peak=torch.cuda.max_memory_allocated()
    estimator.cpu().eval()
    torch.cuda.empty_cache()
    model={"estimator":estimator,"encoder":encoder,"steps":best_epoch,"y_mean":ymean,"y_scale":yscale,"threads":threads}
    return model,{"steps":best_epoch,"fit_runtime_sec":time.monotonic()-started,"rows":len(x),"features":len(x.columns),
        "version":version("tabm"),"torch_version":torch.__version__,"device":"cuda:0","gpu_peak_allocated_bytes":gpu_peak,
        "history":history,"params":{"blocks":3,"width":256,"members":8,"dropout":.1,"batch_size":4096,"max_epochs":epochs,
        "patience":5,"learning_rate":.001,"weight_decay":.0001,"gradient_norm_cap":10.,"embedding_dim":"ceil(sqrt(cardinality)) bounded4..32"},
        "encoding":"Fit-only category embeddings; median imputation/mean standardization/missing indicators; mean raw squared loss via affine target scale; no label clipping or oversampling."}


def predict(model,x):
    torch.set_num_threads(model["threads"])
    estimator=model["estimator"].cuda()
    values=infer(estimator,tensors(model["encoder"],x),"cuda")*model["y_scale"]+model["y_mean"]
    estimator.cpu()
    torch.cuda.empty_cache()
    return values
