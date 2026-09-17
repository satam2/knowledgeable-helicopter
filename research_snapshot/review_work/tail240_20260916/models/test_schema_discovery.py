import pytest
import schema_discovery as subject


def test_same_selection_one_read_per_file_and_restore(monkeypatch):
    calls=[]
    class Schema:
        names=['first','third']
    def native(path):
        calls.append(path)
        return Schema()
    monkeypatch.setattr(subject.pq,'read_schema',native)
    def select(columns):
        return [(path,[name for name in columns if name in subject.pq.read_schema(path).names]) for path in ['a.parquet','b.parquet']]
    result,receipt=subject.cached_discovery(select,['first','second','third'])
    assert result==[('a.parquet',['first','third']),('b.parquet',['first','third'])]
    assert calls==['a.parquet','b.parquet']
    assert receipt['requests']==6 and receipt['physical_reads']==2
    assert subject.pq.read_schema is native


def test_restore_after_error(monkeypatch):
    native=lambda path: None
    monkeypatch.setattr(subject.pq,'read_schema',native)
    def failed(columns):
        subject.pq.read_schema('a.parquet')
        raise RuntimeError('probe')
    with pytest.raises(RuntimeError,match='probe'):
        subject.cached_discovery(failed,[])
    assert subject.pq.read_schema is native
