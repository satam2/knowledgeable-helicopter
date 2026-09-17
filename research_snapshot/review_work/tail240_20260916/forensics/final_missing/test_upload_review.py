"""Network-free upload review probes; credentials and payloads are synthetic."""
from datetime import datetime,timezone
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[4]
SOURCE=ROOT/'review_work/tail240_20260916/final_submission/upload.py'
spec=importlib.util.spec_from_file_location('reviewed_final_upload',SOURCE)
upload=importlib.util.module_from_spec(spec)
spec.loader.exec_module(upload)
from botocore.exceptions import ClientError


class FakeClient:
    def __init__(self,fail=False):
        self.payload=None
        self.calls=[]
        self.fail=fail
    def get_paginator(self,name):
        assert name=='list_objects_v2'
        return self
    def paginate(self,**kwargs):
        return [{'Contents':[]}]
    def put_object(self,**kwargs):
        self.calls.append(('put',kwargs['Key']))
        assert kwargs['IfNoneMatch']=='*'
        self.payload=kwargs['Body'].read()
        if self.fail:
            raise ClientError({'Error':{'Code':'InternalError','Message':'synthetic'},'ResponseMetadata':{'HTTPStatusCode':500}},'PutObject')
        return {'ResponseMetadata':{'HTTPStatusCode':200},'ETag':'synthetic'}
    def get_object(self,**kwargs):
        self.calls.append(('get',kwargs['Key']))
        if kwargs['Key']==upload.KEY:
            return {'Body':io.BytesIO(self.payload)}
        if kwargs['Key']==upload.KEY+'_result.json':
            value={'status':'Succeeded','file':upload.KEY,'used_pairs':344841,'score':1.}
            return {'Body':io.BytesIO(json.dumps(value).encode())}
        raise AssertionError('Unexpected remote key')


class UploadReviewTests(unittest.TestCase):
    def fixture(self,folder):
        path=folder/upload.KEY
        path.write_bytes(b'approved synthetic prediction payload')
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        docs={'submission_ready.json':{'submission':{'sha256':digest,'rows':344841}},
              'independent_verification.json':{'status':'passed','submission_sha256':digest}}
        for name,value in docs.items():
            (folder/name).write_text(json.dumps(value))
        return path,digest

    def test_http_error_preserves_attempt_and_does_not_retry(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)
            self.fixture(folder)
            client=FakeClient(fail=True)
            with patch.object(upload,'OUT',folder),patch.object(upload.boto3,'client',return_value=client):
                with self.assertRaises(ClientError):
                    upload.run('synthetic_access','synthetic_secret','upload')
                self.assertTrue((folder/'upload_attempt.json').exists())
                self.assertFalse((folder/'upload_receipt.json').exists())
                with self.assertRaisesRegex(AssertionError,'Prior attempt'):
                    upload.run('synthetic_access','synthetic_secret','upload')
            self.assertEqual(client.calls,[('put',upload.KEY)])

    def test_changed_payload_after_hash_is_currently_not_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)
            path,approved=self.fixture(folder)
            client=FakeClient()
            original=upload.write_json
            def write_then_mutate(destination,value):
                original(destination,value)
                if destination.name=='upload_attempt.json':
                    path.write_bytes(b'changed after approval')
            with patch.object(upload,'OUT',folder),patch.object(upload.boto3,'client',return_value=client),patch.object(upload,'write_json',side_effect=write_then_mutate):
                upload.run('synthetic_access','synthetic_secret','upload')
            self.assertNotEqual(hashlib.sha256(client.payload).hexdigest(),approved)
            self.assertEqual(json.loads((folder/'upload_receipt.json').read_text())['sha256'],approved)
            self.assertTrue((folder/'remote_verification.json').exists())
            self.assertEqual([key for action,key in client.calls if action=='get'],[upload.KEY,upload.KEY+'_result.json'])


if __name__=='__main__':
    unittest.main()
