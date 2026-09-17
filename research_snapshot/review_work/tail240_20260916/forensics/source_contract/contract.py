"""Exact per-feature row coverage and loss-of-meaning checks before float32."""
from pathlib import Path
import hashlib

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ID = 'MVT_ID_mvt'
SENTINEL = -999999.


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024**2),b''):
            digest.update(chunk)
    return digest.hexdigest()


def numeric_type(dtype):
    if pa.types.is_dictionary(dtype):
        dtype = dtype.value_type
    return pa.types.is_integer(dtype) or pa.types.is_floating(dtype) or pa.types.is_boolean(dtype)


def validate(sources, ids, columns, *, batch_size=8192, guard=lambda:None):
    """Validate requested rows only; source tuples are (path, sha256, fill, names)."""
    ids = pd.Index(ids)
    columns = list(columns)
    if not len(ids) or ids.hasnans or not ids.is_unique:
        raise ValueError('Requested IDs must be nonempty, nonnull and unique')
    if not columns or len(set(columns)) != len(columns) or ID in columns:
        raise ValueError('Declared feature names must be nonempty, unique and exclude ID')
    bound = [(Path(path),digest,fill,list(names)) for path,digest,fill,names in sources]
    if not bound:
        raise ValueError('Missing source tuples')
    # One bitset per feature catches overlap that aggregate row counts cannot.
    seen = {name:np.zeros((len(ids)+7)//8,dtype=np.uint8) for name in columns}
    stats = {name:dict(rows=0,preexisting_sentinel=0,nonfinite_input=0) for name in columns}
    kinds, receipts = {}, []
    for path,digest,fill,names in bound:
        if not names or len(set(names)) != len(names) or not set(names) <= set(columns):
            raise ValueError(f'Invalid source feature names: {path}')
        if not isinstance(fill,(bool,np.bool_)) or sha256(path) != digest:
            raise ValueError(f'Source hash or fill declaration mismatch: {path}')
        schema = pq.read_schema(path)
        if ID not in schema.names or not set(names) <= set(schema.names):
            raise ValueError(f'Source schema missing declared fields: {path}')
        for name in names:
            numeric = numeric_type(schema.field(name).type)
            if name in kinds and kinds[name] != numeric:
                raise ValueError(f'Inconsistent numeric/categorical schema: {name}')
            kinds[name] = numeric
        selected, outside = 0, 0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size,columns=[ID,*names],use_threads=False):
            frame = batch.to_pandas()
            positions = ids.get_indexer(frame[ID])
            keep = positions >= 0
            outside += int((~keep).sum())
            positions = positions[keep]
            if len(np.unique(positions)) != len(positions):
                raise ValueError(f'Within-file duplicate requested IDs: {path}')
            selected += len(positions)
            rows = frame.loc[keep]
            byte, bit = positions//8, np.left_shift(np.uint8(1),(positions%8).astype(np.uint8))
            for name in names:
                if np.any(seen[name][byte] & bit):
                    raise ValueError(f'Cross-source or cross-batch duplicate requested ID for feature {name}: {path}')
                np.bitwise_or.at(seen[name],byte,bit)
                stats[name]['rows'] += len(positions)
                if kinds[name]:
                    values = pd.to_numeric(rows[name]).to_numpy(dtype=np.float64,na_value=np.nan)
                    with np.errstate(over='ignore',invalid='ignore'):
                        cast = values.astype(np.float32)
                    finite = np.isfinite(values)
                    if np.any(finite & ~np.isfinite(cast)):
                        raise ValueError(f'Finite-to-nonfinite float32 conversion: {name} in {path}')
                    if np.any(finite & (values != SENTINEL) & (cast == SENTINEL)):
                        raise ValueError(f'Nonsentinel value rounds to sentinel: {name} in {path}')
                    stats[name]['preexisting_sentinel'] += int(np.sum(values == SENTINEL))
                    stats[name]['nonfinite_input'] += int(np.sum(~finite))
            guard()
        receipts.append(dict(path=str(path),sha256=digest,fill=bool(fill),columns=names,
                             requested_rows=selected,outside_requested_rows=outside))
    for name in columns:
        count = int(np.unpackbits(seen[name],bitorder='little')[:len(ids)].sum())
        if count != len(ids) or stats[name]['rows'] != len(ids):
            raise ValueError(f'Feature {name} missing {len(ids)-count} requested IDs')
    guard()
    return dict(status='passed',requested_rows=len(ids),columns=columns,coverage=stats,sources=receipts,
                policy='No mutation, normalization, imputation or filtering; outside-cohort rows ignored; preexisting sentinel and existing nonfinite values preserved for frozen loader')


def load_checked(risk, sources, ids, nfit, columns, folder, *, guard=None):
    """Bind validated sources to a synchronous frozen loader invocation."""
    if not 0 < nfit <= len(ids):
        raise ValueError('Fit prefix must be within requested IDs')
    bound = [(Path(p),h,f,list(c)) for p,h,f,c in sources]
    previous_guard = getattr(risk, "guard", lambda:None)
    effective_guard = guard or previous_guard
    receipt = validate(bound,ids,columns,guard=effective_guard)
    for path,digest,_,_ in bound:
        if sha256(path) != digest:
            raise ValueError(f'Source changed after validation: {path}')
    previous = risk.feature_sources
    def fixed(requested):
        if list(requested) != list(columns):
            raise ValueError('Frozen loader requested another schema')
        return bound
    risk.feature_sources = fixed
    risk.guard = effective_guard
    try:
        matrix,vocab,loaded = risk.load_matrix(pd.Index(ids),nfit,list(columns),folder)
    finally:
        risk.feature_sources = previous
        risk.guard = previous_guard
    for path,digest,_,_ in bound:
        if sha256(path) != digest:
            raise ValueError(f'Source changed during loading: {path}')
    return matrix,vocab,loaded,receipt
