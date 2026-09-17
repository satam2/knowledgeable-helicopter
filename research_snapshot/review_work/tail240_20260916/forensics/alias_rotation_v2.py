"""Outcome-free pilot of alias-aware strict public-aircraft previous legs."""
import os
for key in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"]:
    os.environ[key]="1"
import argparse
import sys
import gc
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import psutil

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT / "review_work/tail240_20260916/source_distinctions"))
import opdi_alias_dates as dates
from taxiout.artifacts import read_json,write_json,sha256,utc_now
ID,TIME=dates.ID,dates.TIME
BASE=dates.BASE
MAPS=ROOT / "private_runs/tail240_20260916/source_distinctions/opdi_dates_v1"
OUT=ROOT / "private_runs/tail240_20260916/forensics/alias_rotation/pilot_v2"
SUFFIXES=["match","match_ambiguous","match_offset_sec","previous_leg_available","previous_same_airport","ground_interval_sec","previous_leg_duration_sec","previous_leg_age_at_takeoff_sec"]
FIELDS=[ID,TIME,"PHASE_mvt","FLIGHT_mvt","ADEP_mvt","ADES_mvt","AOBT_3_flt"]
pa.set_cpu_count(1)
pa.set_io_thread_count(1)


def ts(series):
    return pd.to_datetime(series,utc=True,errors="coerce").astype("datetime64[ns, UTC]")


def prefix_codes(flights,mapping):
    raw=dates.normalize(flights)
    parts=raw.str.extract(r"^([A-Z]+)([0-9]+)([A-Z]*)$")
    prefix,number,suffix=parts[0],parts[1],parts[2]
    learned=(prefix.map(mapping).fillna(prefix)+number+suffix).fillna(raw)
    lexical=(prefix.map(dates.lexical_prefix)+number+suffix).fillna(raw)
    return pd.DataFrame({"raw":raw,"learned":learned,"lexical":lexical},index=flights.index)


def build_month(query,public,mapping,month):
    query=query.copy()
    public=public.copy()
    query[TIME]=ts(query[TIME])
    public["first_seen"],public["last_seen"]=ts(public.first_seen),ts(public.last_seen)
    for column in ["adep","ades","icao24","flt_id"]:
        public[column]=dates.normalize(public[column])
    public["id"]=public["id"].astype("string")
    public=public.loc[public.first_seen.dt.strftime("%Y%m").eq(month) & public.last_seen.gt(public.first_seen) & public.icao24.ne("")].copy()
    for column in ["ADEP_mvt","ADES_mvt"]:
        query[column]=dates.normalize(query[column])
    codes=prefix_codes(query.FLIGHT_mvt,mapping)
    wanted=set(pd.concat([codes[c] for c in codes]))-{ "" }
    groups={key:g for key,g in public.loc[public.flt_id.isin(wanted)].groupby(["flt_id","adep","ades"],sort=False)}
    columns=[f"alias_{mode}_{name}" for mode in ["raw","learned","lexical"] for name in SUFFIXES]
    result=pd.DataFrame(np.nan,index=pd.Index(query[ID],name=ID),columns=columns,dtype="float32")
    for mode in ["raw","learned","lexical"]:
        result[[f"alias_{mode}_{name}" for name in ["match","match_ambiguous","previous_leg_available","previous_same_airport"]]]=np.float32(0)
    matches=[]
    for pos,row in enumerate(query.itertuples(index=False)):
        fields=row._asdict()
        if fields[TIME].strftime("%Y%m")!=month:
            continue
        for mode in ["raw","learned","lexical"]:
            code=codes.iloc[pos][mode]
            key=(code,fields["ADEP_mvt"],fields["ADES_mvt"])
            if not all(key): continue
            group=groups.get(key)
            if group is None: continue
            hits=group.loc[(fields[TIME]-group.first_seen).dt.total_seconds().abs().le(600)]
            prefix=f"alias_{mode}_"
            if len(hits)>1:
                result.loc[fields[ID],prefix+"match_ambiguous"]=1
            elif len(hits)==1:
                hit=hits.iloc[0]
                result.loc[fields[ID],prefix+"match"]=1
                result.loc[fields[ID],prefix+"match_offset_sec"]=(fields[TIME]-hit.first_seen).total_seconds()
                matches.append((fields,mode,code,hit))
    tails={hit.icao24 for _,_,_,hit in matches}
    by_tail={key:g for key,g in public.loc[public.icao24.isin(tails)].groupby("icao24",sort=False)}
    witnesses=[]
    for fields,mode,code,hit in matches:
        prefix=f"alias_{mode}_"
        witness={ID:fields[ID],"mode":mode,"query_flight":fields["FLIGHT_mvt"],"matched_code":code,
            "public_id":str(hit.id),"icao24":hit.icao24,"public_first_seen":hit.first_seen,"public_last_seen":hit.last_seen,
            "query_time":fields[TIME],"previous_public_id":None,"previous_last_seen":pd.NaT,"rejection":None}
        before=by_tail[hit.icao24].loc[lambda x:x.first_seen.lt(hit.first_seen)]
        if before.last_seen.ge(hit.first_seen).any():
            witness["rejection"]="overlapping_earlier_leg"
        else:
            eligible=before.loc[before.last_seen.lt(hit.first_seen) & before.last_seen.lt(fields[TIME])]
            latest=eligible.loc[eligible.last_seen.eq(eligible.last_seen.max())]
            if len(latest)!=1:
                witness["rejection"]="no_unique_previous_leg"
            else:
                previous=latest.iloc[0]
                result.loc[fields[ID],prefix+"previous_leg_available"]=1
                witness.update(previous_public_id=str(previous.id),previous_last_seen=previous.last_seen)
                if previous.ades!=fields["ADEP_mvt"]:
                    witness["rejection"]="previous_destination_mismatch"
                else:
                    result.loc[fields[ID],prefix+"previous_same_airport"]=1
                    result.loc[fields[ID],prefix+"ground_interval_sec"]=(hit.first_seen-previous.last_seen).total_seconds()
                    result.loc[fields[ID],prefix+"previous_leg_duration_sec"]=(previous.last_seen-previous.first_seen).total_seconds()
                    result.loc[fields[ID],prefix+"previous_leg_age_at_takeoff_sec"]=(fields[TIME]-previous.last_seen).total_seconds()
        witnesses.append(witness)
    return result,pd.DataFrame(witnesses)


def declare():
    definition={"scope":"Coverage pilot: every Rome missing-NM training DEP, all2025 logicalUTCmonths; no ranking or labels.",
        "query_fields":FIELDS,"public_fields":dates.PUBLIC_FIELDS,"forbidden_fields_loaded":[],
        "modes":"raw exactnormalized control; observed earlierlogicalUTCmonth PREFIX-ONLY mapping; lexicalrepeatedsuffix PREFIX-ONLY collapse separately. Preserve numeric leadingzeros and suffix exactly. Unparsed code uses raw normalized spelling.",
        "mapping":"Frozen opdi_dates_v1 monthlymapping: earlier known-observedprefixpairs,20distinctFLIGHTIDs,98%dominance, samedigit/suffix. No targets.",
        "identity":"Strict unique normalizedcode+bothairports within600sec publicfirst_seen ofownsuppliedmovement. No datehypotheses. Publicid exactstring beforeanyconcat; blankICAO24excluded; duplicates remainambiguous.",
        "previous_leg":"SameICAO24 completedpublicleg; actualfirst_seen querycalendar month only. Anyearlier overlappingleg invalidates. Previouslast_seen beforebothmatchedfirst_seen andquerymovement, latestlast_seen mustunique. Durationfeatures only previousdestination==queryorigin.",
        "control":"Recompute raw8features and compare bitexactfloat32/equalNaN to frozen oldOPDIcache onpilotIDs. Additional aliascoverage only recognized aftercontrolparity.",
        "gate":"If fewer than10additionalRome same-day matches versusraw, nofullmodel. Anyfullcache/model needsseparatedeclaration.",
        "licensing":"Existing local noncommercial attributedpublicresearch; prize/commercialeligibility unresolved. No downloads.",
        "resources":"2CPUmax4GiB,minimum8GiBhostavailable",
        "source_sha256":sha256(__file__),"dependency_sources":{str(Path(p).relative_to(ROOT)):sha256(p) for p in [dates.__file__,dates.alias.__file__]},
        "mapping_manifest_sha256":sha256(MAPS / "manifest.json"),"old_manifest_sha256":sha256(BASE / "manifest.json"),"acquisition_sha256":sha256(BASE / "acquisition.json")}
    path=OUT / "protocol.json"
    if path.exists(): assert read_json(path)["declaration"]==definition
    else:
        OUT.mkdir(parents=True,exist_ok=False)
        write_json(path,{"created_utc":utc_now(),"declaration":definition})


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--declare-only",action="store_true")
    args=parser.parse_args()
    declare()
    if args.declare_only:
        print("Declared aliasrotation Romepilot; no build",flush=True)
        return
    frozen=read_json(ROOT / "private_runs/submission_v2/protocol.json")
    frames=[]
    for path in sorted(dates.common.RAW.glob("training_*.parquet")):
        assert sha256(path)==frozen["raw_hashes"][path.name]
        q=pd.read_parquet(path,columns=FIELDS,filters=[("PHASE_mvt","=","DEP"),("ADEP_mvt","=","LIRF")],dtype_backend="pyarrow")
        q=q.loc[ts(q.AOBT_3_flt).isna()].copy()
        q[TIME]=ts(q[TIME])
        frames.append(q)
    query=pd.concat(frames,ignore_index=True)
    assert query[ID].is_unique
    query["month"]=query[TIME].dt.strftime("%Y%m")
    mappings=read_json(MAPS / "manifest.json")["months"]
    sources={r["month"]:r for r in read_json(BASE / "acquisition.json")["sources"]}
    oldmanifest=read_json(BASE / "manifest.json")
    assert sha256(BASE / "training_features.parquet")==oldmanifest["outputs"]["training_features.parquet"]
    old=pd.read_parquet(BASE / "training_features.parquet").set_index(ID)
    chunks,links,stats=[],[],[]
    for month,rows in query.groupby("month",sort=True):
        source=sources[month]
        path=Path(source["local_path"])
        assert sha256(path)==source["sha256"]
        public=pd.read_parquet(path,columns=dates.PUBLIC_FIELDS,dtype_backend="pyarrow")
        public["id"]=public["id"].astype("string")
        features,witnesses=build_month(rows,public,mappings[month]["mapping"],month)
        raw=features[[f"alias_raw_{s}" for s in SUFFIXES]].to_numpy()
        expected=old.loc[features.index,[f"opdi_{s}" for s in SUFFIXES]].to_numpy()
        np.testing.assert_array_equal(raw,expected)
        counts={"month":month,"queries":len(rows),"raw_matches":int(features.alias_raw_match.sum()),"raw_rotation":int(features.alias_raw_previous_same_airport.sum())}
        for mode in ["learned","lexical"]:
            counts[mode+"_new_matches"]=int((features[f"alias_{mode}_match"].eq(1) & features.alias_raw_match.eq(0)).sum())
            counts[mode+"_new_rotations"]=int((features[f"alias_{mode}_previous_same_airport"].eq(1) & features.alias_raw_previous_same_airport.eq(0)).sum())
        chunks.append(features)
        links.append(witnesses)
        stats.append(counts)
        print("ALIAS_ROTATION",counts,flush=True)
        peak=getattr(psutil.Process().memory_info(),"peak_wset",psutil.Process().memory_info().rss)
        assert peak<4*1024**3 and psutil.virtual_memory().available>8*1024**3
        del public
        gc.collect()
    pd.concat(chunks).reset_index().to_parquet(OUT / "features.parquet",index=False)
    pd.concat(links,ignore_index=True).to_parquet(OUT / "witnesses.parquet",index=False)
    query.to_parquet(OUT / "queries.parquet",index=False)
    write_json(OUT / "manifest.json",{"status":"complete","source_sha256":sha256(__file__),"protocol_sha256":sha256(OUT / "protocol.json"),
        "queries":len(query),"raw_control_parity":True,"months":stats,"peak_bytes":peak,
        "gate_additional_matches":{mode:int(sum(r[mode+"_new_matches"] for r in stats)) for mode in ["learned","lexical"]},
        "outputs":{p.name:sha256(p) for p in OUT.iterdir() if p.is_file()}})


if __name__=="__main__":
    main()
