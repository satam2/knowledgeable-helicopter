import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'models'))
import torch
import numpy as np
import pandas as pd
import io
from adapter import Fusion, CLOCKS, tensors
from encoders import FrameEncoder


def test_convex_mask_and_mean_gradients():
    torch.manual_seed(8)
    model=Fusion(3,[4],'convex')
    nums=torch.randn(5,3)
    cats=torch.zeros(5,1,dtype=torch.long)
    anchors=torch.randn(5,5)
    available=torch.zeros(5,5,dtype=torch.bool)
    available[:3,:2]=True
    pred,weights,experts=model(nums,cats,anchors,available,return_details=True)
    assert torch.isfinite(pred).all()
    assert torch.allclose(weights.sum(-1),torch.ones_like(pred))
    assert (weights[:3,:,2:5]==0).all()
    assert (weights[3:,:,:5]==0).all()
    assert torch.allclose(pred[3:],experts[3:,:,5])
    assert (pred>=experts.min(-1).values-1e-6).all()
    assert (pred<=experts.max(-1).values+1e-6).all()
    pred.square().mean().backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.parameters())


def test_masked_anchor_values_are_irrelevant():
    torch.manual_seed(9)
    model=Fusion(3,[],'affine').eval()
    nums=torch.randn(3,3)
    cats=torch.empty(3,0,dtype=torch.long)
    anchors=torch.randn(3,5)
    available=torch.zeros(3,5,dtype=torch.bool)
    available[:,0]=True
    changed=anchors.clone();changed[:,1:]=1e8
    with torch.no_grad():
        assert torch.equal(model(nums,cats,anchors,available),model(nums,cats,changed,available))


def test_affine_anchor_reconstruction_and_serialization():
    frame=pd.DataFrame({'takeoff_minus_'+clock:[-12.,100000.,-999999.] for clock in CLOCKS})
    encoder=FrameEncoder().fit(frame,neural=True)
    values=tensors(encoder,frame,1200.,300.)
    anchors,available=values[2].numpy(),values[3].numpy()
    assert np.allclose(anchors[:2]*300+1200,frame.iloc[:2].to_numpy(),atol=.01)
    assert not available[2].any()
    assert (anchors[2]==0).all()
    model=Fusion(values[0].shape[1],[],'convex').eval()
    buffer=io.BytesIO()
    torch.save(model.state_dict(),buffer)
    buffer.seek(0)
    loaded=Fusion(values[0].shape[1],[],'convex').eval()
    loaded.load_state_dict(torch.load(buffer,weights_only=True))
    with torch.no_grad():
        assert torch.equal(model(*values),loaded(*values))
