"""Outcome-free nested membership and selected-fit categorical identities."""
import hashlib
import numpy as np

PREFIX = 'prc2026-uniform-half-v1|20260916|'


def selected_positions(ids, count):
    values=np.asarray(ids,dtype=np.float64)
    if not np.isfinite(values).all() or len(np.unique(values))!=len(values):
        raise ValueError('IDs must be unique finite values')
    if not np.equal(values,np.floor(values)).all() or np.any(np.abs(values)>2**53):
        raise ValueError('IDs must have exact integral float64 representation')
    if not 0<count<=len(values):
        raise ValueError('Invalid nested sample size')
    ranked=sorted((hashlib.sha256((PREFIX+str(int(v))).encode('ascii')).digest(),int(v),i)
                  for i,v in enumerate(values))
    return np.sort(np.fromiter((row[2] for row in ranked[:count]),dtype=np.int64))


def selected_vocab(full_codes, selected, vocabulary):
    codes=np.asarray(full_codes)[selected].astype(np.int64)
    if np.any(codes<0) or np.any(codes>=len(vocabulary)+2):
        raise ValueError('Invalid full-vocabulary code')
    observed=np.unique(codes[codes>=2])
    vocab=[vocabulary[i-2] for i in observed]
    mapping=np.ones(len(vocabulary)+2,dtype=np.float32)
    mapping[0]=0
    mapping[observed]=np.arange(2,len(observed)+2,dtype=np.float32)
    return vocab,mapping
