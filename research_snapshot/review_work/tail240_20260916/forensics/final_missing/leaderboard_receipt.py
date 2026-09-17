"""Public leaderboard receipt only after accepted V3 appears; no credentials."""
from datetime import datetime,timezone
import json
import math
from pathlib import Path
import sys
import urllib.parse
import urllib.request

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'knowledgeable-helicopter-screening/src'))
from taxiout.artifacts import read_json,write_json,sha256
from taxiout.paths import external_path
OUT=external_path(ROOT/'private_runs/tail240_20260916/final_submission_v3_v2')
URL='https://datacomp.opensky-network.org/api/competitions/bb3693e1-26bc-4a9e-8619-4fe78b4eab0c/leaderboard'
TEAM='knowledgeable-helicopter'
KEY='knowledgeable-helicopter_v3.parquet'
PUBLIC_FIELDS=['submissionId','teamId','teamName','filename','score','usedPairs','processedAt']


def summarize(items):
    if len({row['submissionId'] for row in items})!=len(items):
        raise ValueError('Leaderboard changed across page boundaries; retry a fresh snapshot')
    best={}
    ours=[]
    accepted=[]
    for original in items:
        row={name:original[name] for name in PUBLIC_FIELDS if name in original}
        score=row.get('score')
        if row['teamName']==TEAM:
            ours.append(row)
        if score is None:
            continue
        if not isinstance(score,(int,float)) or isinstance(score,bool) or not math.isfinite(score):
            raise ValueError('Leaderboard contains a nonfinite or nonnumeric score')
        name=row['teamName']
        if name not in best or score<best[name]['score']:
            best[name]=row
        if name==TEAM and row.get('filename')==KEY and row.get('usedPairs')==344841:
            accepted.append(row)
    if len(accepted)!=1:
        raise ValueError('Accepted V3 is absent or ambiguous; do not report a current V3 team rank yet')
    current=best[TEAM]
    rank=1+sum(row['score']<current['score'] for row in best.values())
    return dict(team=TEAM,v3=accepted[0],best=current,team_rank_by_best_score=rank,
                teams=len(best),submissions=ours,total_submission_rows=len(items))


def main():
    target=external_path(OUT/'leaderboard_receipt.json')
    if target.exists():
        raise FileExistsError('Preserve the existing leaderboard receipt; request a separately reviewed refresh')
    accepted_path=OUT/'organizer_result.json'
    accepted=read_json(accepted_path)
    if accepted.get('status')!='Succeeded' or accepted.get('file')!=KEY or accepted.get('used_pairs')!=344841:
        raise ValueError('Matching organizer acceptance is required before leaderboard access')
    started=datetime.now(timezone.utc).isoformat()
    items=[]
    cursor=None
    seen=set()
    pages=0
    while True:
        address=URL+('?' + urllib.parse.urlencode({'cursor':cursor}) if cursor else '')
        with urllib.request.urlopen(address,timeout=25) as response:
            data=json.load(response)
        if not isinstance(data,dict) or not isinstance(data.get('items'),list):
            raise ValueError('Unexpected public leaderboard response schema')
        items.extend(data['items'])
        pages+=1
        cursor=data.get('nextCursor')
        if not cursor:
            break
        if cursor in seen or pages>=100:
            raise ValueError('Unexpected pagination cycle/limit')
        seen.add(cursor)
    result=summarize(items)
    result.update(snapshot_started_utc=started,retrieved_utc=datetime.now(timezone.utc).isoformat(),
        source=URL,pages=pages,source_sha256=sha256(__file__),
        preserved_reader_sha256=sha256(ROOT/'review_work/read_submission_rank.py'),
        organizer_result_sha256=sha256(accepted_path),
        method='One plus number of teams with strictly lower best score. Rank reported only with exactly one accepted V3 entry.',
        snapshot_limitation='Public paginated snapshot is not atomic; duplicate boundary IDs are rejected, but concurrent missing or updated rows cannot be fully excluded.',
        no_credentials=True,no_truth_access=True,no_upload=True)
    write_json(target,result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    main()
