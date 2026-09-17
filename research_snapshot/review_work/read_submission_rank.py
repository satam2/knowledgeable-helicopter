"""Read the public leaderboard to report this team's accepted submission, never tune."""

import json
import urllib.request
import urllib.parse
from datetime import datetime,timezone
from pathlib import Path

OUT=Path(__file__).resolve().parents[1]/'private_runs/submission_v2'
URL='https://datacomp.opensky-network.org/api/competitions/bb3693e1-26bc-4a9e-8619-4fe78b4eab0c/leaderboard'
TEAM='knowledgeable-helicopter'


def main():
    items=[]
    cursor=None
    seen=set()
    pages=0
    while True:
        address=URL+('?' + urllib.parse.urlencode({'cursor':cursor}) if cursor else '')
        with urllib.request.urlopen(address,timeout=25) as response:
            data=json.load(response)
        items.extend(data['items'])
        pages+=1
        cursor=data.get('nextCursor')
        if not cursor:
            break
        if cursor in seen or pages>=100:
            raise ValueError('Unexpected pagination cycle/limit')
        seen.add(cursor)
    by_id={v['submissionId']:v for v in items}
    if len(by_id)!=len(items):
        raise ValueError('Leaderboard changed across page boundaries; retry snapshot')
    best={}
    for row in items:
        if row.get('score') is None:
            continue
        name=row['teamName']
        if name not in best or row['score']<best[name]['score']:
            best[name]=row
    current=best.get(TEAM)
    ours=[v for v in items if v['teamName']==TEAM]
    rank=(1+sum(r['score']<current['score'] for r in best.values())) if current else None
    result={'retrieved_utc':datetime.now(timezone.utc).isoformat(),'source':URL,
        'team':TEAM,'best':current,'team_rank_by_best_score':rank,'teams':len(best),
        'submissions':ours,'pages':pages,'total_submission_rows':len(items),
        'method':'One plus number of teams with strictly lower best score; public paginated snapshot, not atomic.'}
    (OUT/'leaderboard_receipt.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
