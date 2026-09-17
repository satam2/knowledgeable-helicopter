"""Independent saved wind/density checks using public tables and verified weather."""
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/"knowledgeable-helicopter-screening/src"))
from taxiout.paths import external_path
from taxiout.schema import ID,PHASE
PUBLIC=external_path(ROOT/"output/breakthrough_20260916/weather_wind/public")
OUT=external_path(ROOT/"private_runs/breakthrough_20260916/weather_wind")
WEATHER=external_path(ROOT/"private_runs/breakthrough_20260916/weather")
RAW=external_path(ROOT/"data/09-15-2026-18-55-03_files_list")


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1048576),b""):h.update(b)
    return h.hexdigest()


def text(value):return str(value).strip().upper() if pd.notna(value) else ""


def expected(row,wx,runways,airports):
    runway=text(row.RUNWAY_mvt)
    candidates=[]
    for side,other in (("le","he"),("he","le")):
        selected=runways.loc[runways.airport_ident.eq(row.ADEP_mvt)&runways[side+"_ident"].map(text).eq(runway)]
        candidates.extend((r,side,other) for _,r in selected.iterrows())
    result={"wxrunway_mapping_available":int(len(candidates)==1)}
    heading=coord=closed=error=reciprocal=np.nan
    if len(candidates)==1:
        match,side,other=candidates[0]
        heading=match[side+"_heading_degT"]
        if not 0<=heading<=360:heading=np.nan
        closed=match.closed
        lon1,lon2=match[side+"_longitude_deg"],match[other+"_longitude_deg"]
        lat1,lat2=map(math.radians,[match[side+"_latitude_deg"],match[other+"_latitude_deg"]])
        delta=math.radians(lon2-lon1)
        coord=math.degrees(math.atan2(math.sin(delta)*math.cos(lat2),math.cos(lat1)*math.sin(lat2)-math.sin(lat1)*math.cos(lat2)*math.cos(delta)))%360
        error=abs((heading-coord+180)%360-180)
        reciprocal=abs((heading-match[other+"_heading_degT"])%360-180)
    result.update(wxrunway_heading_true_deg=heading,wxrunway_heading_coordinate_true_deg=coord,wxrunway_heading_disagreement_deg=error,wxrunway_heading_reciprocal_error_deg=reciprocal,wxrunway_source_current_closed=closed)
    speed=wx.weather_T_wind_speed_kt
    east,north=wx.weather_T_wind_east_kt,wx.weather_T_wind_north_kt
    calm=np.isfinite(speed) and speed==0
    fromangle=math.degrees(math.atan2(-east,-north)) if np.isfinite(east) and np.isfinite(north) else np.nan
    for suffix,angle in (("",heading),("coord_",coord)):
        head=cross=np.nan
        if calm and np.isfinite(angle):head=cross=0.
        elif np.isfinite(angle) and np.isfinite(fromangle):
            head=speed*math.cos(math.radians(fromangle-angle));cross=speed*math.sin(math.radians(fromangle-angle))
        prefix="wxrunway_T_"+suffix
        result[prefix+"headwind_kt"]=head;result[prefix+"crosswind_from_right_kt"]=cross
        result[prefix+"abs_crosswind_kt"]=abs(cross)
        result[prefix+"tailwind_kt"]=max(-head,0)
        result[prefix+"projection_available"]=int(np.isfinite(head))
    result["wxrunway_T_calm_direction_resolved"]=int(calm)
    altitude=float(airports.loc[airports.ident.eq(row.ADEP_mvt),"elevation_ft"].iloc[0])*.3048
    qnh=wx.weather_T_pressure_inhg*33.8638866667
    base=qnh**.190284-1013.25**.190284*.0065*altitude/288.15
    pressure=base**(1/.190284)+.3 if qnh>0 and base>0 else np.nan
    temp=wx.weather_T_temperature_c+273.15
    density=100*pressure/(287.05*temp) if temp>0 else np.nan
    result.update(wxair_airport_elevation_m=altitude,wxair_T_isa_temperature_departure_c=wx.weather_T_temperature_c-(15-.0065*altitude),
        wxair_T_est_station_pressure_hpa=pressure,wxair_T_est_dry_density_kg_m3=density,wxair_T_est_dry_density_ratio=density/1.225,wxair_T_density_available=int(np.isfinite(density)))
    return result


def main():
    manifest=json.loads((OUT/"manifest.json").read_text());assert manifest["status"]=="complete"
    for name,digest in manifest["outputs"].items():assert sha(OUT/name)==digest
    runways=pd.read_csv(PUBLIC/"runways.csv");airports=pd.read_csv(PUBLIC/"airports.csv")
    checks=[]
    for month in ("07","11","ranking"):
        name="ranking.parquet" if month=="ranking" else f"training_2025-{month}-01_2025-{int(month)+1:02d}-01.parquet"
        feature_name="ranking_features.parquet" if month=="ranking" else "training_features.parquet"
        raw=pq.read_table(RAW/name,columns=[ID,PHASE,"ADEP_mvt","RUNWAY_mvt"],use_threads=False).to_pandas()
        query=raw.loc[raw[PHASE].eq("DEP")]
        features=pd.read_parquet(OUT/feature_name).set_index(ID)
        wx=pd.read_parquet(WEATHER/feature_name).set_index(ID)
        assert np.array_equal(features.index,wx.index)
        if month=="ranking":assert np.array_equal(query[ID],features.index)
        assert not np.isinf(features.to_numpy()).any()
        selected=[]
        for airport,part in query.groupby("ADEP_mvt"):
            selected.append(part.sample(n=min(20,len(part)),random_state=20260916))
            scores=features.loc[part[ID]]
            selected.append(part.loc[part[ID].isin(scores.loc[scores.wxrunway_mapping_available.eq(0)].index)].head(3))
            selected.append(part.loc[part[ID].isin(scores.loc[scores.wxrunway_T_projection_available.eq(0)].index)].head(3))
            selected.append(part.loc[part[ID].isin(scores.loc[scores.wxrunway_T_calm_direction_resolved.eq(1)].index)].head(3))
        samples=pd.concat(selected).drop_duplicates(ID)
        for _,row in samples.iterrows():
            result=expected(row,wx.loc[row[ID]],runways,airports)
            assert set(result)==set(features.columns)
            for key,value in result.items():
                actual=features.loc[row[ID],key]
                if not np.isclose(actual,value,rtol=1e-5,atol=.0001,equal_nan=True):raise AssertionError((key,float(actual),float(value)))
        checks.append({"cohort":month,"queries":len(samples),"features_checked":len(features.columns)})
        print("VERIFIED",checks[-1],flush=True)
    report={"status":"passed","created_utc":datetime.now(timezone.utc).isoformat(),"source_sha256":sha(__file__),"manifest_sha256":sha(OUT/"manifest.json"),"checks":checks,"total_queries":sum(v["queries"] for v in checks),"independent_method":"Public CSV lookup, scalar trigonometry using wind-from direction vs builder vectorprojection; scalar density units","hidden_labels_loaded":False,"network_calls":False,"historical_validity_verified":False}
    (OUT/"verification.json").write_text(json.dumps(report,indent=2),encoding="utf-8")


if __name__=="__main__":main()
