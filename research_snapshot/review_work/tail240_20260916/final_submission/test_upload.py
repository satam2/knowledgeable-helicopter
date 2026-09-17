import hashlib
import io
import json
from datetime import datetime,timezone
import pytest
import upload


class Client:
    def __init__(self,payload,existing=False):
        self.payload=payload
        self.existing=existing
        self.puts=[]
        self.gets=[]

    def get_paginator(self,name):
        assert name=='list_objects_v2'
        return self

    def paginate(self,**kwargs):
        assert kwargs=={'Bucket':upload.BUCKET}
        return [dict(Contents=[dict(Key=upload.KEY,Size=len(self.payload),LastModified=datetime.now(timezone.utc))] if self.existing else [])]

    def put_object(self,**kwargs):
        assert kwargs['Bucket']==upload.BUCKET and kwargs['Key']==upload.KEY
        assert kwargs['IfNoneMatch']=='*'
        assert kwargs['Body']==self.payload
        self.puts.append(kwargs['Key'])
        return dict(ResponseMetadata=dict(HTTPStatusCode=200,RequestId='test'),ETag='test')

    def get_object(self,**kwargs):
        assert kwargs['Bucket']==upload.BUCKET
        self.gets.append(kwargs['Key'])
        if kwargs['Key']==upload.KEY:return dict(Body=io.BytesIO(self.payload))
        assert kwargs['Key']==upload.KEY+'_result.json'
        value=dict(status='Succeeded',file=upload.KEY,bucket=upload.BUCKET,used_pairs=344841,score=270.,
                   inputs={'test_set_file':'do-not-read'})
        return dict(Body=io.BytesIO(json.dumps(value).encode()))


def prepared(tmp_path,monkeypatch,wrong_hash=False,existing=False):
    payload=b'local-test-only'
    digest=hashlib.sha256(payload).hexdigest()
    monkeypatch.setattr(upload,'OUT',tmp_path)
    (tmp_path/upload.KEY).write_bytes(payload)
    (tmp_path/'submission_ready.json').write_text(json.dumps(dict(submission=dict(sha256=digest,rows=344841))))
    (tmp_path/'independent_verification.json').write_text(json.dumps(dict(status='passed',submission_sha256='wrong' if wrong_hash else digest)))
    client=Client(payload,existing)
    monkeypatch.setattr(upload.boto3,'client',lambda *a,**kw:client)
    return client


def test_one_verified_upload_and_score_only_reads(tmp_path,monkeypatch):
    client=prepared(tmp_path,monkeypatch)
    upload.run('dummy-access','dummy-secret','upload')
    assert client.puts==[upload.KEY]
    assert client.gets==[upload.KEY,upload.KEY+'_result.json']
    saved=json.loads((tmp_path/'organizer_result.json').read_text())
    assert saved['score']==270. and 'inputs' not in saved
    for path in tmp_path.iterdir():
        if path.suffix=='.json':assert 'dummy-secret' not in path.read_text()


@pytest.mark.parametrize('wrong_hash,existing',[(True,False),(False,True)])
def test_rejects_changed_file_or_existing_destination(tmp_path,monkeypatch,wrong_hash,existing):
    client=prepared(tmp_path,monkeypatch,wrong_hash,existing)
    with pytest.raises(AssertionError):upload.run('dummy-access','dummy-secret','upload')
    assert not client.puts


def test_file_mutation_after_verification_cannot_change_uploaded_bytes(tmp_path,monkeypatch):
    client=prepared(tmp_path,monkeypatch)
    original=client.put_object
    def mutate_then_upload(**kwargs):
        (tmp_path/upload.KEY).write_bytes(b'changed-after-verification')
        return original(**kwargs)
    client.put_object=mutate_then_upload
    upload.run('dummy-access','dummy-secret','upload')
    receipt=json.loads((tmp_path/'upload_receipt.json').read_text())
    remote=json.loads((tmp_path/'remote_verification.json').read_text())
    assert receipt['sha256']==remote['sha256']==hashlib.sha256(client.payload).hexdigest()


@pytest.mark.parametrize('change',[{'bucket':'wrong-bucket'},{'score':None},{'score':float('nan')}])
def test_invalid_organizer_result_cannot_be_reported_as_success(tmp_path,monkeypatch,change):
    client=prepared(tmp_path,monkeypatch)
    original=client.get_object
    def changed_result(**kwargs):
        response=original(**kwargs)
        if kwargs['Key'].endswith('_result.json'):
            value=json.loads(response['Body'].read());value.update(change)
            return dict(Body=io.BytesIO(json.dumps(value).encode()))
        return response
    client.get_object=changed_result
    with pytest.raises((AssertionError,ValueError)):upload.run('dummy-access','dummy-secret','upload')
    assert client.puts==[upload.KEY]
