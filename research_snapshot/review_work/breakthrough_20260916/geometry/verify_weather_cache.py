"""Read-only independent weather cache verification and coverage report."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/"knowledgeable-helicopter-screening/src"))
from taxiout.paths import external_path
from taxiout.schema import ID,PHASE,MOVEMENT
PUBLIC=external_path(ROOT/"output/breakthrough_20260916/public_weather")
CACHE=external_path(ROOT/"private_runs/breakthrough_20260916/weather")
OUT=external_path(ROOT/"private_runs/breakthrough_20260916/geometry/weather_review")
RAW=external_path(ROOT/"data/09-15-2026-18-55-03_files_list")


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1048576),b""):h.update(block)
    return h.hexdigest()


def number(row,column):
    value=row[column]
    return float(value) if value not in ("M","T","") and pd.notna(value) else np.nan


def observed_values(row):
    temp,dew,wind,direction=number(row,"tmpf"),number(row,"dwpf"),number(row,"sknt"),number(row,"drct")
    layers=[number(row,c) for c in ("skyl1","skyl2","skyl3")]
    layers=[v for v in layers if np.isfinite(v)]
    code=str(row.wxcodes) if row.wxcodes not in ("M","") else ""
    return {"temperature_c":(temp-32)*5/9,"dewpoint_depression_c":(temp-dew)*5/9,
      "wind_east_kt":-wind*np.sin(direction*np.pi/180),"wind_north_kt":-wind*np.cos(direction*np.pi/180),
      "wind_direction_missing":float(not np.isfinite(direction)),"visibility_reported_miles":number(row,"vsby"),
      "pressure_inhg":number(row,"alti"),"gust_reported_kt":number(row,"gust"),"wind_speed_kt":wind,
      "humidity_pct":number(row,"relh"),"lowest_cloud_reported_ft":min(layers) if layers else np.nan,
      **{name+"_reported":float(any(c in code for c in tokens)) for name,tokens in {"rain":["RA","DZ"],"snow":["SN","SG","PL"],"fog":["FG","BR"],"thunder":["TS"],"freezing":["FZ"]}.items()}}


def main():
    manifest=json.loads((CACHE/"manifest.json").read_text());assert manifest["status"]=="complete"
    for name,digest in manifest["outputs"].items():assert sha(CACHE/name)==digest
    receipts=json.loads((PUBLIC/"manifest.json").read_text());assert receipts["status"]=="complete"
    frames=[]
    for rec in receipts["receipts"]:
        path=PUBLIC/f"{rec['station']}_{rec['period']}.csv"
        assert sha(path)==rec["sha256"]
        frames.append(pd.read_csv(path,keep_default_na=False))
    weather=pd.concat(frames,ignore_index=True)
    weather["valid"]=pd.to_datetime(weather.valid,utc=True)
    duplicates=int(weather.duplicated(["station","valid"]).sum())
    weather=weather.sort_values(["station","valid"],kind="stable").drop_duplicates(["station","valid"],keep="last")
    observations={s:f.reset_index(drop=True) for s,f in weather.groupby("station")}
    observation_coverage=[]
    for (station,month),frame in weather.groupby(["station",weather.valid.dt.strftime("%Y-%m")]):
        observation_coverage.append({"station":station,"month":month,"rows":len(frame),"max_internal_gap_sec":float(frame.valid.diff().dt.total_seconds().max()),"wind_missing_pct":float(100*frame.drct.isin(["M","T",""]).mean()),"wxcode_empty_or_M_pct":float(100*frame.wxcodes.isin(["M",""]).mean())})
    training=pd.read_parquet(CACHE/"training_features.parquet").set_index(ID)
    ranking=pd.read_parquet(CACHE/"ranking_features.parquet").set_index(ID)
    cover=[];checked=[];all_training_ids=[]
    frozen=json.loads((ROOT/"private_runs/submission_v2/protocol.json").read_text())
    for path in [*sorted(RAW.glob("training_*.parquet")),RAW/"ranking.parquet"]:
        assert sha(path)==frozen["raw_hashes"][path.name]
        raw=pq.read_table(path,columns=[ID,PHASE,"ADEP_mvt",MOVEMENT,"AOBT_3_flt"],use_threads=False).to_pandas()
        q=raw.loc[raw[PHASE].eq("DEP")].reset_index(drop=True)
        frame=ranking if path.name=="ranking.parquet" else training
        assert q[ID].isin(frame.index).all()
        if path.name=="ranking.parquet":assert np.array_equal(q[ID],frame.index)
        else:all_training_ids.extend(q[ID].tolist())
        features=frame.loc[q[ID]].reset_index()
        for (airport,month),positions in q.groupby(["ADEP_mvt",q[MOVEMENT].dt.strftime("%Y-%m")]).indices.items():
            item={"file":path.name,"airport":airport,"month":month,"rows":len(positions)}
            for stage in ("T","N"):
                part=features.iloc[positions]
                item[stage+"_available_pct"]=float(100*part["weather_"+stage+"_available"].mean())
                item[stage+"_age_p99_sec"]=float(part["weather_"+stage+"_age_sec"].quantile(.99))
                item[stage+"_stale_rows"]=int(part["weather_"+stage+"_age_sec"].gt(10800).sum())
            item["missing_N_rows"]=int(q.iloc[positions].AOBT_3_flt.isna().sum())
            item["N_after_T_rows"]=int(q.iloc[positions].AOBT_3_flt.gt(q.iloc[positions][MOVEMENT]).sum())
            cover.append(item)
        if path.name!="ranking.parquet" and not any(f"2025-{m}-01" in path.name for m in ("06","07","10","11")):continue
        selected=[]
        for airport,part in q.groupby("ADEP_mvt"):
            selected.append(part.sample(n=min(8,len(part)),random_state=20260916))
            selected.append(part.head(2))
            selected.append(part.loc[part.AOBT_3_flt.isna()].head(2))
            selected.append(part.loc[part.AOBT_3_flt.gt(part[MOVEMENT])].head(2))
            stale_ids=features.loc[features.weather_T_available.eq(0),ID]
            selected.append(part.loc[part[ID].isin(stale_ids)].head(2))
        sample=pd.concat(selected).drop_duplicates(ID)
        for _,row in sample.iterrows():
            actual=frame.loc[row[ID]]
            station=observations.get(str(row.ADEP_mvt))
            for stage,timefield in (("T",MOVEMENT),("N","AOBT_3_flt")):
                when=row[timefield];prefix="weather_"+stage+"_"
                past=station.loc[station.valid.lt(when)] if station is not None and pd.notna(when) else pd.DataFrame()
                chosen=past.iloc[-1] if len(past) else None
                age=(when-chosen.valid).total_seconds() if chosen is not None else np.nan
                accepted=np.isfinite(age) and 0<age<=10800
                assert np.isclose(actual[prefix+"age_sec"],age,rtol=1e-6,atol=.001,equal_nan=True)
                assert actual[prefix+"available"]==accepted
                columns=[c for c in actual.index if c.startswith(prefix) and c not in (prefix+"age_sec",prefix+"available")]
                expected=observed_values(chosen) if accepted else {c[len(prefix):]:np.nan for c in columns}
                for c in columns:
                    if not np.isclose(actual[c],expected[c[len(prefix):]],rtol=1e-5,atol=.001,equal_nan=True):raise AssertionError((path.name,stage,c,actual[c],expected[c[len(prefix):]]))
        checked.append({"file":path.name,"queries":len(sample),"all36features":True})
        print("VERIFIED",checked[-1],flush=True)
    assert np.array_equal(np.array(all_training_ids),training.index)
    OUT.mkdir(parents=True,exist_ok=True)
    report={"status":"passed","created_utc":datetime.now(timezone.utc).isoformat(),"manifest_sha256":sha(CACHE/"manifest.json"),"verifier_sha256":sha(__file__),"checks":checked,"total_queries":sum(c["queries"] for c in checked),"feature_coverage":cover,"observation_coverage":observation_coverage,"duplicate_observations_deduplicated":duplicates,"training_ranking_id_order_verified":True,"hidden_labels_read":False,"network_calls":False,"publication_time_verified":False}
    (OUT/"verification.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print("WEATHER_VERIFIED",report["total_queries"],flush=True)


if __name__=="__main__":main()
