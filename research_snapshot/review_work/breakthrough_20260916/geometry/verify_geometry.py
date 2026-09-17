"""Independent heap shortest paths and direct raw/cache alignment checks."""
from collections import defaultdict
from datetime import datetime,timezone
import hashlib
import heapq
import json
import math
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/"knowledgeable-helicopter-screening/src"))
from taxiout.paths import external_path
from taxiout.schema import ID,PHASE
OUT=external_path(ROOT/"private_runs/breakthrough_20260916/geometry")
PUBLIC=external_path(ROOT/"output/breakthrough_20260916/geometry/public")
RAW=external_path(ROOT/"data/09-15-2026-18-55-03_files_list")


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1048576),b""):h.update(b)
    return h.hexdigest()


def independent_map(airport):
    elements=json.loads((PUBLIC/(airport+".json")).read_text())["elements"]
    nodes={e["id"]:(e["lat"],e["lon"]) for e in elements if e["type"]=="node"}
    adjacency=defaultdict(dict);taxinodes=set();stands=defaultdict(set);runways=defaultdict(set)
    for e in elements:
        tags=e.get("tags",{});kind=tags.get("aeroway")
        if e["type"]=="way" and kind in ("taxiway","taxilane","parking_position") and tags.get("area")!="yes":
            path=e["nodes"]
            if kind!="parking_position":taxinodes.update(path)
            for a,b in zip(path,path[1:]):
                lat1,lon1=map(math.radians,nodes[a]);lat2,lon2=map(math.radians,nodes[b])
                h=math.sin((lat2-lat1)/2)**2+math.cos(lat1)*math.cos(lat2)*math.sin((lon2-lon1)/2)**2
                weight=2*6371008.8*math.atan2(math.sqrt(h),math.sqrt(max(0,1-h)))
                adjacency[a][b]=min(weight,adjacency[a].get(b,float("inf")))
                adjacency[b][a]=min(weight,adjacency[b].get(a,float("inf")))
        if kind=="parking_position" and tags.get("ref"):
            node=e["id"] if e["type"]=="node" else e.get("nodes",[None])[-1]
            if node in nodes:stands[tags["ref"].strip().upper()].add(node)
        if kind=="runway" and e["type"]=="way" and tags.get("ref"):
            for r in re.split(r"[/;]",tags["ref"]):runways[r.strip().upper()].update(e["nodes"])
    return adjacency,stands,{r:set(nodes)&taxinodes&set(adjacency) for r,nodes in runways.items()}


def shortest(adjacency,start,destinations):
    queue=[(0,start)];done=set()
    while queue:
        d,n=heapq.heappop(queue)
        if n in done:continue
        if n in destinations:return d
        done.add(n)
        for target,weight in adjacency.get(n,{}).items():
            if target not in done:heapq.heappush(queue,(d+weight,target))
    return np.nan


def main():
    manifest=json.loads((OUT/"manifest.json").read_text())
    for name,digest in manifest["outputs"].items():assert sha(OUT/name)==digest
    maps={a:independent_map(a) for a in ("LIRF","LFPG")}
    receipts=[]
    for data_name,feature_name in [("training_2025-07-01_2025-08-01.parquet","training_features.parquet"),("ranking.parquet","ranking_features.parquet")]:
        raw=pq.read_table(RAW/data_name,columns=[ID,PHASE,"ADEP_mvt","STAND_mvt","RUNWAY_mvt"],use_threads=False).to_pandas()
        raw=raw.loc[raw[PHASE].eq("DEP")]
        feature=pd.read_parquet(OUT/feature_name).set_index(ID)
        if data_name=="ranking.parquet":assert np.array_equal(raw[ID],feature.index)
        count=0
        for airport,(adj,stands,runways) in maps.items():
            relevant=raw.loc[raw.ADEP_mvt.eq(airport)]
            selected=pd.concat([relevant.sample(n=80,random_state=20260916),relevant.loc[relevant[ID].map(feature.histgeom_graph_route_available).eq(0)].head(20)]).drop_duplicates(ID)
            for _,row in selected.iterrows():
                s="" if pd.isna(row.STAND_mvt) else str(row.STAND_mvt).strip().upper()
                r="" if pd.isna(row.RUNWAY_mvt) else str(row.RUNWAY_mvt).strip().upper()
                candidates=stands.get(s,set());node=next(iter(candidates)) if len(candidates)==1 else None
                actual=feature.loc[row[ID]]
                assert actual.histgeom_airport_covered==1
                assert actual.histgeom_stand_ref_count==len(candidates)
                assert actual.histgeom_stand_unambiguous==(len(candidates)==1)
                assert actual.histgeom_stand_graph_attached==(node in adj)
                assert actual.histgeom_runway_ref_mapped==(r in runways)
                expected=shortest(adj,node,runways.get(r,set())) if node is not None else np.nan
                assert np.isclose(actual.histgeom_mapped_graph_shortest_m,expected,rtol=1e-6,atol=.002,equal_nan=True)
                assert actual.histgeom_graph_route_available==np.isfinite(expected)
                count+=1
        other=feature.loc[raw.loc[~raw.ADEP_mvt.isin(maps),ID]]
        assert other.histgeom_mapped_graph_shortest_m.isna().all()
        assert other.histgeom_airport_covered.eq(0).all()
        receipts.append({"dataset":data_name,"queries":count,"all7features_checked":True})
    verification={"status":"passed","created_utc":datetime.now(timezone.utc).isoformat(),"verifier_sha256":sha(__file__),"manifest_sha256":sha(OUT/"manifest.json"),"checks":receipts,"ranking_id_order_verified":True,"no_labels_read":True,"no_network_calls":True,"method":"Independent adjacency and heap Dijkstra, haversine atan2 distances; saved feature hashes checked"}
    (OUT/"verification.json").write_text(json.dumps(verification,indent=2),encoding="utf-8")
    print(json.dumps(verification,indent=2),flush=True)


if __name__=="__main__":main()
