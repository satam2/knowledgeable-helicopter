"""Single verified prediction upload; credentials supplied only in process memory."""
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'knowledgeable-helicopter-screening/src'))
from taxiout.artifacts import read_json,write_json,sha256
from taxiout.paths import external_path
OUT=external_path(ROOT/'private_runs/tail240_20260916/final_submission_v3_v2')
BUCKET='prc-2026-knowledgeable-helicopter'
KEY='knowledgeable-helicopter_v3.parquet'
ENDPOINT='https://s3.opensky-network.org'


def objects(client):
    rows=[]
    for page in client.get_paginator('list_objects_v2').paginate(Bucket=BUCKET):
        rows.extend(dict(key=o['Key'],bytes=o['Size'],modified=o['LastModified'].isoformat()) for o in page.get('Contents',[]))
    today=datetime.now(timezone.utc).date()
    submitted=sum(r['key'].endswith('.parquet') and datetime.fromisoformat(r['modified']).date()==today for r in rows)
    return dict(endpoint=ENDPOINT,bucket=BUCKET,checked_utc=datetime.now(timezone.utc).isoformat(),
                utc_day_submissions=submitted,objects=rows,total_bytes=sum(r['bytes'] for r in rows))


def retrieve_result(client):
    try:
        response=client.get_object(Bucket=BUCKET,Key=KEY+'_result.json')
    except ClientError as exc:
        if exc.response['Error']['Code'] in ['NoSuchKey','404']:return None
        raise
    value=json.loads(response['Body'].read())
    # Organizer result includes truth-data references. Preserve only score receipt fields.
    receipt={name:value[name] for name in ['status','file','bucket','used_pairs','score','error','message'] if name in value}
    receipt['retrieved_utc']=datetime.now(timezone.utc).isoformat()
    receipt['result_key']=KEY+'_result.json'
    write_json(OUT/'organizer_result.json',receipt)
    print('ORGANIZER_RESULT',json.dumps(receipt),flush=True)
    return receipt


def run(access,secret,action='check'):
    client=boto3.client('s3',endpoint_url=ENDPOINT,aws_access_key_id=access,aws_secret_access_key=secret,
                       config=Config(connect_timeout=20,read_timeout=40,retries={'max_attempts':0}))
    access=secret=None
    inventory=objects(client)
    OUT.mkdir(parents=True,exist_ok=True)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
    write_json(OUT/f'bucket_check_{stamp}.json',inventory)
    print('BUCKET_CHECK',json.dumps(inventory),flush=True)
    if action=='check':return
    if action=='upload':
        ready=read_json(OUT/'submission_ready.json')
        verification=read_json(OUT/'independent_verification.json')
        path=OUT/KEY
        payload=path.read_bytes()
        digest=hashlib.sha256(payload).hexdigest()
        assert ready['submission']['sha256']==digest
        assert verification['status']=='passed' and verification['submission_sha256']==digest
        assert ready['submission']['rows']==344841
        assert all(r['key']!=KEY for r in inventory['objects']),'Existing destination: never upload twice'
        assert inventory['utc_day_submissions']<5
        assert inventory['total_bytes']+len(payload)<1_000_000_000
        assert not (OUT/'upload_attempt.json').exists(),'Prior attempt requires remote reconciliation, never blind retry'
        attempt=dict(bucket=BUCKET,key=KEY,sha256=digest,bytes=len(payload),
                     independent_verification_sha256=sha256(OUT/'independent_verification.json'),
                     utc=datetime.now(timezone.utc).isoformat())
        write_json(OUT/'upload_attempt.json',attempt)
        response=client.put_object(Bucket=BUCKET,Key=KEY,Body=payload,ContentType='application/octet-stream',IfNoneMatch='*')
        receipt={**attempt,'http_status':response['ResponseMetadata']['HTTPStatusCode'],
                 'etag':response.get('ETag'),'request_id':response['ResponseMetadata'].get('RequestId')}
        write_json(OUT/'upload_receipt.json',receipt)
        assert receipt['http_status']==200
        print('UPLOADED',json.dumps(receipt),flush=True)
    elif action=='result':
        attempt=read_json(OUT/'upload_attempt.json')
        assert attempt['bucket']==BUCKET and attempt['key']==KEY
        digest=attempt['sha256']
    else:raise ValueError(action)
    remote=client.get_object(Bucket=BUCKET,Key=KEY)
    body=remote['Body'].read()
    assert hashlib.sha256(body).hexdigest()==digest and len(body)==attempt['bytes']
    write_json(OUT/'remote_verification.json',dict(status='passed',sha256=hashlib.sha256(body).hexdigest(),bytes=len(body)))
    for _ in range(45):
        result=retrieve_result(client)
        if result:
            assert result['status']=='Succeeded',result
            assert result['used_pairs']==344841 and result['file']==KEY
            assert result['bucket']==BUCKET
            assert type(result.get('score')) in (int,float) and math.isfinite(result['score']) and result['score']>=0
            return
        print('AWAIT_ORGANIZER',datetime.now(timezone.utc).isoformat(),flush=True)
        time.sleep(20)
    raise TimeoutError('Uploaded once and remotely verified; organizer result still pending. Poll only, do not upload again.')
