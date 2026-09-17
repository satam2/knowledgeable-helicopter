"""Interactive credentials stay in memory; upload only the verified prediction artifact."""

import base64
import getpass
import hashlib
import json
from pathlib import Path

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'private_runs/submission_v2'
BUCKET = 'prc-2026-knowledgeable-helicopter'
ENDPOINT = 'https://s3.opensky-network.org'
NAME = 'knowledgeable-helicopter_v2.parquet'


def save(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/name).write_text(json.dumps(value,indent=2,default=str),encoding='utf-8')


def main():
    access = getpass.getpass('Access key (hidden): ')
    secret = getpass.getpass('Secret key (hidden): ')
    client = boto3.client('s3',endpoint_url=ENDPOINT,aws_access_key_id=access,
        aws_secret_access_key=secret,region_name='us-east-1',
        config=Config(signature_version='s3v4',s3={'addressing_style':'path'},
                      connect_timeout=15,read_timeout=45,retries={'max_attempts':2}))
    del access,secret
    verified = False
    while True:
        action = input('Action [check/upload/verify/result/quit]: ').strip()
        if action == 'quit':
            return
        try:
            if action == 'check':
                contents=[]
                for page in client.get_paginator('list_objects_v2').paginate(Bucket=BUCKET):
                    contents.extend({'key':o['Key'],'bytes':o['Size'],'modified':str(o['LastModified'])} for o in page.get('Contents',[]))
                version_free = not any(o['key']==NAME for o in contents)
                result={'endpoint':ENDPOINT,'bucket':BUCKET,'objects':contents,
                        'v2_available':version_free,'total_bytes':sum(o['bytes'] for o in contents)}
                verified = version_free
                save('bucket_check.json',result)
                print(json.dumps(result,indent=2),flush=True)
            elif action == 'upload':
                if not verified:
                    raise ValueError('Run check first; version must be unused')
                receipt=json.loads((OUT/'submission_ready.json').read_text())
                file=OUT/NAME
                data=file.read_bytes()
                digest=hashlib.sha256(data).hexdigest()
                if receipt['sha256']!=digest or not receipt['passed'] or receipt['rows']!=344841:
                    raise ValueError('Submission receipt mismatch')
                if receipt.get('pipeline_verified') is not True:
                    raise ValueError('Final pipeline not verified')
                audit=json.loads((OUT/'independent_verification.json').read_text())
                if not audit['passed'] or audit['submission']['sha256']!=digest:
                    raise ValueError('Independent submission verification failed')
                response=client.put_object(Bucket=BUCKET,Key=NAME,Body=data,IfNoneMatch='*',
                    ContentType='application/octet-stream',
                    ContentMD5=base64.b64encode(hashlib.md5(data).digest()).decode(),
                    Metadata={'sha256':digest})
                result={'bucket':BUCKET,'key':NAME,'bytes':len(data),'sha256':digest,
                        'etag':response.get('ETag'),'http_status':response['ResponseMetadata']['HTTPStatusCode'],
                        'request_id':response['ResponseMetadata']['RequestId']}
                save('upload_receipt.json',result)
                verified=False
                print(json.dumps(result,indent=2),flush=True)
            elif action == 'verify':
                expected=json.loads((OUT/'upload_receipt.json').read_text())
                response=client.get_object(Bucket=BUCKET,Key=NAME)
                digest=hashlib.sha256(response['Body'].read()).hexdigest()
                if digest!=expected['sha256']:
                    raise ValueError('Remote bytes differ from local submission')
                result={'bucket':BUCKET,'key':NAME,'remote_sha256':digest,'matches_local':True,
                        'bytes':response['ContentLength'],'modified':str(response['LastModified'])}
                save('remote_verification.json',result)
                print(json.dumps(result,indent=2),flush=True)
            elif action == 'result':
                for version in ['v1','v2']:
                    key=f'knowledgeable-helicopter_{version}.parquet_result.json'
                    try:
                        response=client.get_object(Bucket=BUCKET,Key=key)
                        value=json.loads(response['Body'].read())
                        save(f'organizer_{version}_result.json',value)
                        print(json.dumps({'key':key,'result':value},indent=2),flush=True)
                    except ClientError as exc:
                        if exc.response['Error']['Code'] in ['NoSuchKey','404']:
                            print(version+' organizer result not available yet',flush=True)
                        else:
                            raise
            else:
                print('Unknown action',flush=True)
        except ClientError as exc:
            print(json.dumps({'error':exc.response['Error']['Code'],
                'http_status':exc.response['ResponseMetadata']['HTTPStatusCode']}),flush=True)
        except Exception as exc:
            print(type(exc).__name__+': '+str(exc),flush=True)


if __name__=='__main__':
    main()
