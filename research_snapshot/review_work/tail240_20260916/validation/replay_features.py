"""Read-only reconstruction of saved tune encodings from bound feature receipts."""
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
import validate_candidate as validation


def rebuild(ids, encoder, receipts):
    ids=pd.Index(ids)
    assert ids.is_unique
    columns=encoder['columns']
    positions={column:i for i,column in enumerate(columns)}
    matrix=np.full((len(ids),len(columns)),np.nan,dtype='float32',order='F')
    counts=np.zeros(len(columns),dtype=int)
    for source in receipts:
        path=Path(source['path'])
        assert validation.sha256(path)==source['sha256']
        names=source['columns']
        seen=np.zeros(len(ids),dtype=bool)
        for batch in pq.ParquetFile(path).iter_batches(batch_size=8192,columns=[validation.ID,*names],use_threads=False):
            frame=batch.to_pandas()
            rows=ids.get_indexer(frame[validation.ID])
            keep=rows>=0
            rows=rows[keep]
            assert len(np.unique(rows))==len(rows) and not seen[rows].any()
            seen[rows]=True
            frame=frame.loc[keep]
            for name in names:
                if name in encoder['vocab']:
                    mapping={value:i+2 for i,value in enumerate(encoder['vocab'][name])}
                    values=frame[name].astype('string').map(mapping).fillna(1).where(frame[name].notna(),0).to_numpy('float32')
                else:
                    values=pd.to_numeric(frame[name]).to_numpy(dtype='float32',na_value=np.nan)
                    values[~np.isfinite(values)]=np.nan
                    if 'screening_230/data/interim/features' not in path.as_posix() and 'sequence_flatten' not in str(path):
                        values[np.isnan(values)]=-999999
                matrix[rows,positions[name]]=values
        for name in names:
            counts[positions[name]]+=int(seen.sum())
        validation.guard()
    assert np.all(counts==len(ids))
    return matrix
