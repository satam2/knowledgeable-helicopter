"""Audit historical OSM topology, then privately match exact stand/runway refs."""
import hashlib
import json
import math
from collections import Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path
import re
import sys

import networkx as nx
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial import cKDTree

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/"knowledgeable-helicopter-screening/src"))
from taxiout.paths import external_path
from taxiout.schema import ID,PHASE

PUBLIC=external_path(ROOT/"output/breakthrough_20260916/geometry/public")
OUT=external_path(ROOT/"private_runs/breakthrough_20260916/geometry")
RAW=external_path(ROOT/"data/09-15-2026-18-55-03_files_list")


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
    return h.hexdigest()


def distance(a,b):
    p1,p2=math.radians(a[0]),math.radians(b[0])
    dp,dl=p2-p1,math.radians(b[1]-a[1])
    return 6371008.8*2*math.asin(min(1,math.sqrt(math.sin(dp/2)**2+math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2)))


def norm(value):return str(value).strip().upper() if pd.notna(value) else ""


def airport_map(airport):
    data=json.loads((PUBLIC/(airport+".json")).read_text())
    if data.get("remark"):raise ValueError(data["remark"])
    elements=data["elements"]
    nodes={e["id"]:(e["lat"],e["lon"]) for e in elements if e["type"]=="node"}
    graph=nx.Graph();taxi_nodes=set();stand_nodes=defaultdict(set);runways=defaultdict(set)
    counts=Counter();restrictions=Counter();endpoints=[];missing_nodes=0
    for e in elements:
        tags=e.get("tags",{});kind=tags.get("aeroway")
        if kind:counts[kind]+=1
        if kind in ("taxiway","taxilane","parking_position") and e["type"]=="way":
            if tags.get("area")=="yes":continue
            path=e.get("nodes",[])
            if kind in ("taxiway","taxilane"):
                taxi_nodes.update(path)
                for key in ("oneway","access","aircraft","service"):
                    if key in tags:restrictions[key+"="+tags[key]]+=1
            for a,b in zip(path,path[1:]):
                if a not in nodes or b not in nodes:missing_nodes+=1;continue
                length=distance(nodes[a],nodes[b])
                if not graph.has_edge(a,b) or graph[a][b]["weight"]>length:graph.add_edge(a,b,weight=length)
        if kind=="parking_position" and tags.get("ref"):
            terminal=e["id"] if e["type"]=="node" else (e.get("nodes") or [None])[-1]
            if terminal in nodes:stand_nodes[norm(tags["ref"])].add(terminal)
        if kind=="runway" and e["type"]=="way" and tags.get("ref"):
            path=e.get("nodes",[])
            refs=[norm(r) for r in re.split(r"[/;]",tags["ref"])]
            for r in refs:runways[r].update(path)
            endpoints.append({"ref":tags["ref"],"way_id":e["id"],"first_node":path[0],"last_node":path[-1],"first_latlon":nodes.get(path[0]),"last_latlon":nodes.get(path[-1]),"tags":tags,"verified_operational_threshold":False})
    route_lengths={};runway_entries={}
    for runway,pathnodes in runways.items():
        entries=pathnodes & taxi_nodes & set(graph)
        runway_entries[runway]=len(entries)
        route_lengths[runway]=nx.multi_source_dijkstra_path_length(graph,entries,weight="weight") if entries else {}
    unique={k:next(iter(v)) for k,v in stand_nodes.items() if len(v)==1}
    attached={k:n for k,n in unique.items() if n in graph}
    components=sorted((len(c) for c in nx.connected_components(graph)),reverse=True)
    # Proximity is a diagnostic only: it never inserts an inferred graph edge.
    reference_lat=np.mean([nodes[n][0] for n in taxi_nodes if n in nodes])
    def xy(node):
        lat,lon=nodes[node]
        return (math.radians(lon)*6371008.8*math.cos(math.radians(reference_lat)),math.radians(lat)*6371008.8)
    tree=cKDTree([xy(n) for n in taxi_nodes if n in nodes])
    gaps=[float(tree.query(xy(n))[0]) for n in unique.values()]
    rec={"airport":airport,"requested_snapshot":"2024-12-31T23:59:59Z","public_sha256":sha(PUBLIC/(airport+".json")),
         "server_database_base_timestamp":data.get("osm3s",{}).get("timestamp_osm_base"),"element_counts":dict(counts),
         "graph_nodes":graph.number_of_nodes(),"graph_edges":graph.number_of_edges(),"connected_components":len(components),"largest_components":components[:10],
         "stand_refs":len(stand_nodes),"unambiguous_stand_refs":len(unique),"attached_unambiguous_stand_refs":len(attached),
         "unambiguous_refs_with_any_runway_route":sum(any(n in lengths for lengths in route_lengths.values()) for n in unique.values()),
         "nearest_taxi_node_gap_m_quantiles":dict(zip(["q0","q25","q50","q75","q95","q100"],np.quantile(gaps,[0,.25,.5,.75,.95,1]).tolist())),
         "refs_with_nearest_taxi_node_within_m":{str(m):sum(g<=m for g in gaps) for m in (1,10,30,100)},
         "runway_shared_taxi_nodes":runway_entries,"runway_endpoints":endpoints,"ignored_operational_restriction_tags":dict(restrictions),"missing_way_node_edges":missing_nodes,
         "graph_policy":"Undirected mapped taxiway/taxilane/parking lead-in ways, exact shared nodes only; no runway traversal, no snapping, no claim of operational direction/access clearance or actual path"}
    return {"refs":stand_nodes,"unique":unique,"attached":attached,"routes":route_lengths,"report":rec}


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/"manifest.json").exists():raise ValueError("Completed result preserved")
    maps={a:airport_map(a) for a in ("LIRF","LFPG")}
    (OUT/"topology.json").write_text(json.dumps({a:m["report"] for a,m in maps.items()},indent=2),encoding="utf-8")
    protocol={"created_utc":datetime.now(timezone.utc).isoformat(),"source_sha256":sha(__file__),"snapshot":"2024-12-31T23:59:59Z",
      "features":"Static historical map feasibility; exact uppercase/trim refs only; unmapped paths remain NaN; all departure IDs preserved",
      "information":"No labels/clocks read; no network calls in this process; private stand/runway values never sent externally",
      "warning":"Mapped graph distance to any entry of physical runway is not actual taxi route, travel time, threshold distance, clearance or verified operational lower bound",
      "raw_columns":[ID,PHASE,"ADEP_mvt","STAND_mvt","RUNWAY_mvt"]}
    (OUT/"protocol.json").write_text(json.dumps(protocol,indent=2),encoding="utf-8")
    frozen=json.loads((ROOT/"private_runs/submission_v2/protocol.json").read_text())
    records=[];writer=None
    for path in [*sorted(RAW.glob("training_*.parquet")),RAW/"ranking.parquet"]:
        assert sha(path)==frozen["raw_hashes"][path.name]
        raw=pq.read_table(path,columns=protocol["raw_columns"],use_threads=False).to_pandas()
        raw=raw.loc[raw[PHASE].eq("DEP")].reset_index(drop=True)
        feature=pd.DataFrame({ID:raw[ID]})
        names=["airport_covered","stand_ref_count","stand_unambiguous","stand_graph_attached","runway_ref_mapped","graph_route_available","mapped_graph_shortest_m"]
        for name in names:feature["histgeom_"+name]=np.nan if name=="mapped_graph_shortest_m" else np.float32(0)
        record={"file":path.name,"rows":len(raw),"airports":{}}
        for airport,mapping in maps.items():
            index=raw.index[raw.ADEP_mvt.eq(airport)]
            counts=np.array([len(mapping["refs"].get(norm(v),())) for v in raw.loc[index,"STAND_mvt"]])
            standkeys=[norm(v) for v in raw.loc[index,"STAND_mvt"]]
            runwaykeys=[norm(v) for v in raw.loc[index,"RUNWAY_mvt"]]
            attached=np.array([v in mapping["attached"] for v in standkeys])
            knownrunway=np.array([v in mapping["routes"] for v in runwaykeys])
            routes=np.array([mapping["routes"].get(r,{}).get(mapping["unique"].get(s),np.nan) for s,r in zip(standkeys,runwaykeys)],dtype=float)
            values=[np.ones(len(index)),counts,counts==1,attached,knownrunway,np.isfinite(routes),routes]
            for name,value in zip(names,values):feature.loc[index,"histgeom_"+name]=np.asarray(value,dtype=np.float32)
            record["airports"][airport]={"rows":len(index),"nonnull_stands":int(raw.loc[index,"STAND_mvt"].notna().sum()),"exact_stand_ref_rows":int((counts>0).sum()),"unambiguous_stand_rows":int((counts==1).sum()),"graph_attached_rows":int(attached.sum()),"runway_ref_rows":int(knownrunway.sum()),"route_rows":int(np.isfinite(routes).sum()),"route_m_quantiles":dict(zip(["q0","q25","q50","q75","q95","q100"],np.nanquantile(routes,[0,.25,.5,.75,.95,1]).tolist())) if np.isfinite(routes).any() else None}
        for name in names:feature["histgeom_"+name]=feature["histgeom_"+name].astype(np.float32)
        table=pa.Table.from_pandas(feature,preserve_index=False)
        if path.name=="ranking.parquet":feature.to_parquet(OUT/"ranking_features.parquet",index=False)
        else:
            if writer is None:writer=pq.ParquetWriter(OUT/"training_features.parquet",table.schema,compression="zstd")
            writer.write_table(table)
        records.append(record);print(record,flush=True)
    writer.close()
    expected=pd.read_parquet(ROOT/"private_runs/screening_230/data/interim/audit/departures.parquet",columns=[ID])
    actual=pd.read_parquet(OUT/"training_features.parquet",columns=[ID])
    assert np.array_equal(expected[ID],actual[ID])
    manifest={"status":"complete","source_sha256":sha(__file__),"records":records,"training_id_order_verified":True,
              "features":list(feature.columns)[1:],"outputs":{p.name:sha(p) for p in OUT.glob("*.parquet")},"accuracy_tested":False}
    (OUT/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")


if __name__=="__main__":main()
