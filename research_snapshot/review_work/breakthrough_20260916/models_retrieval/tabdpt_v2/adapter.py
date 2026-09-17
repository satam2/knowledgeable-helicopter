"""Versioned official non-flash TabDPT plus explicit supported math SDPA."""
import os
os.environ['HF_HUB_OFFLINE']='1'
os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
import sys
import time
from pathlib import Path
from importlib.metadata import version
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
import tabdpt_adapter as frozen

REFERENCE_LIMIT=frozen.REFERENCE_LIMIT
CONTEXT=frozen.CONTEXT
BATCH=frozen.BATCH
ENSEMBLES=frozen.ENSEMBLES


def fit(x,y,tuning=None,*,steps=None,seed=20260916,threads=2,device='cuda',max_epochs=None):
    if len(x)>REFERENCE_LIMIT:
        raise ValueError('Reference subset exceeds unchanged32K budget')
    labels=np.asarray(y,dtype=float)
    if not len(x) or len(x)!=len(labels) or not np.isfinite(labels).all():
        raise ValueError('Invalid original reference rows/labels')
    if tuning is not None and steps is not None:
        raise ValueError('Ambiguous fit/refit request')
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    import faiss
    faiss.omp_set_num_threads(threads)
    path,receipt=frozen.checked_checkpoint()
    began=time.monotonic()
    processor=frozen.RetrievalPreprocessor().fit(x)
    encoded=processor.transform(x)
    params=dict(device=device,context_reduction='retrieval',compile=False,use_flash=False,
                model_weight_path=str(path),missing_indicators=False,feature_reduction='pca',verbose=False)
    estimator=frozen.estimator_class()(**params)
    if estimator.use_flash or estimator.model.use_flash:
        raise AssertionError('Official flash-only context unexpectedly enabled')
    estimator.fit(encoded,labels)
    model={'estimator':estimator,'processor':processor,'steps':1,'threads':threads,'seed':seed,
           'context_size':min(CONTEXT,len(x)),'attention_backend':'math','use_flash':False}
    evidence={'library':'tabdpt','version':version('tabdpt'),'steps':1,'family':'TabDPT1.3_reference32K_mathSDPA_v2',
              'rows':len(x),'reference_rows':len(x),'fit_seconds':time.monotonic()-began,'seed':seed,'threads':threads,
              'params':params,'prediction_params':{'context_size':min(CONTEXT,len(x)),'batch_size':BATCH,'n_ensembles':ENSEMBLES,'seed':seed},
              'checkpoint_sha256':receipt['sha256'],'checkpoint_revision':receipt['revision'],'encoded_features':encoded.shape[1],
              'checkpoint_max_features':int(estimator.max_features),'pca_used':encoded.shape[1]>estimator.max_features,
              'preprocessing':'Unchanged frozen fit-reference-only RetrievalPreprocessor; original labels passed directly',
              'objective':'Unchanged pretrained regression-bin conditional mean; no label-resolution or checkpoint changes',
              'tune_labels_in_reference':False,'network_disabled':True,'device':device,
              'attention':'Officialuse_flashFalse bypasses Flash-only SDPA context and CUDAautocast; explicit PyTorchMATH SDPA forpredict',
              'numerics':'Default model float32 instead of v1 CUDA bfloat16/float16 autocast; numerical equality to failed CUDA path not claimed'}
    return model,evidence


def predict(model,x):
    if model['estimator'].use_flash or model['estimator'].model.use_flash:
        raise AssertionError('V2 prediction requires official use_flash=False')
    with sdpa_kernel(SDPBackend.MATH):
        return frozen.predict(model,x)
