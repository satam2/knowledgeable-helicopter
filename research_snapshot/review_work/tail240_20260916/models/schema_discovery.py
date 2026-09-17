"""Bound metadata discovery memory without changing frozen feature selection."""
from pathlib import Path
import pyarrow.parquet as pq


def cached_discovery(discover, columns):
    original=pq.read_schema
    cache={}
    counts={'requests':0,'physical_reads':0}

    def read_once(path, *args, **kwargs):
        if args or kwargs:
            raise ValueError('Reviewed discovery contract uses local path-only read_schema')
        key=str(Path(path).resolve())
        counts['requests']+=1
        if key not in cache:
            cache[key]=original(path)
            counts['physical_reads']+=1
        return cache[key]

    # Discovery runs synchronously before model fitting; always restore Arrow.
    pq.read_schema=read_once
    try:
        result=discover(columns)
    finally:
        pq.read_schema=original
    return result,dict(counts,paths=list(cache))
