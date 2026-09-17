"""Independent network-free tests of corrected immutable upload and retry rules."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import test_upload_review as prior
from botocore.exceptions import ClientError
upload=prior.upload


class FixedClient(prior.FakeClient):
    def put_object(self,**kwargs):
        self.calls.append(('put',kwargs['Key']))
        assert kwargs['IfNoneMatch']=='*' and isinstance(kwargs['Body'],bytes)
        self.payload=kwargs['Body']
        if self.fail:
            raise ClientError({'Error':{'Code':'InternalError','Message':'synthetic'},'ResponseMetadata':{'HTTPStatusCode':500}},'PutObject')
        return {'ResponseMetadata':{'HTTPStatusCode':200},'ETag':'synthetic'}


class FixedUploadTests(unittest.TestCase):
    def test_immutable_bytes_and_result_reconciliation_ignore_local_mutation(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)
            path,digest=prior.UploadReviewTests().fixture(folder)
            client=FixedClient()
            original=upload.write_json
            def mutate_after_attempt(destination,value):
                original(destination,value)
                if destination.name=='upload_attempt.json':
                    path.write_bytes(b'changed after approval')
            with patch.object(upload,'OUT',folder),patch.object(upload.boto3,'client',return_value=client),patch.object(upload,'write_json',side_effect=mutate_after_attempt):
                upload.run('synthetic_access','synthetic_secret','upload')
                upload.run('synthetic_access','synthetic_secret','result')
            self.assertEqual(hashlib.sha256(client.payload).hexdigest(),digest)
            self.assertEqual(sum(action=='put' for action,_ in client.calls),1)
            self.assertEqual(json.loads((folder/'remote_verification.json').read_text())['sha256'],digest)
            for document in folder.glob('*.json'):
                self.assertNotIn('synthetic_secret',document.read_text())

    def test_http500_preserves_attempt_and_refuses_second_put(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)
            prior.UploadReviewTests().fixture(folder)
            client=FixedClient(fail=True)
            with patch.object(upload,'OUT',folder),patch.object(upload.boto3,'client',return_value=client):
                with self.assertRaises(ClientError):
                    upload.run('synthetic_access','synthetic_secret','upload')
                self.assertTrue((folder/'upload_attempt.json').exists())
                self.assertFalse((folder/'upload_receipt.json').exists())
                with self.assertRaisesRegex(AssertionError,'Prior attempt'):
                    upload.run('synthetic_access','synthetic_secret','upload')
            self.assertEqual(client.calls,[('put',upload.KEY)])


if __name__=='__main__':
    unittest.main()
