"""Local runway wind interactions with explicit current-source and density limits."""
import hashlib
import json
from pathlib import Path
import sys
from datetime import datetime,timezone

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/"knowledgeable-helicopter-screening/src"))
from taxiout.paths import external_path
from taxiout.schema import ID,PHASE
PUBLIC=external_path(ROOT/"output/breakthrough_20260916/weather_wind/public")
OUT=external_path(ROOT/"private_runs/breakthrough_20260916/weather_wind")
WEATHER=external_path(ROOT/"private_runs/breakthrough_20260916/weather")
RAW=external_path(ROOT/"data/09-15-2026-18-55-03_files_list")
AIRPORTS=["EDDF","EDDM","EGLL","EHAM","LEBL","LEMD","LFPG","LIRF","LTFM","LSZH"]
INHG_TO_HPA=33.8638866667
GAS_DRY=287.05
STANDARD_TEMP_K=288.15
STANDARD_PRESSURE_HPA=1013.25
LAPSE_K_PER_M=.0065
EXPONENT=.190284


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1048576),b""):h.update(b)
    return h.hexdigest()


def norm(series):return series.astype("string").fillna("").str.strip().str.upper()


def bearing(lat1,lon1,lat2,lon2):
    p1,p2,d=np.deg2rad(lat1),np.deg2rad(lat2),np.deg2rad(lon2-lon1)
    return np.rad2deg(np.arctan2(np.sin(d)*np.cos(p2),np.cos(p1)*np.sin(p2)-np.sin(p1)*np.cos(p2)*np.cos(d)))%360


def public_map():
    receipts=json.loads((PUBLIC/"receipt.json").read_text())
    for name in ("airports.csv","runways.csv"):assert sha(PUBLIC/name)==receipts[name]["sha256"]
    airports=pd.read_csv(PUBLIC/"airports.csv");runways=pd.read_csv(PUBLIC/"runways.csv")
    airports=airports.loc[airports.ident.isin(AIRPORTS)].copy()
    assert len(airports)==10 and airports.ident.is_unique
    elevations=dict(zip(airports.ident,pd.to_numeric(airports.elevation_ft,errors="coerce")*.3048))
    runways=runways.loc[runways.airport_ident.isin(AIRPORTS)].copy()
    values=[]
    for side,other in (("le","he"),("he","le")):
        rows=pd.DataFrame({"airport":runways.airport_ident,"runway":norm(runways[side+"_ident"]),"runway_record_id":runways.id,"closed_current":runways.closed,
            "supplied_heading":pd.to_numeric(runways[side+"_heading_degT"],errors="coerce"),
            "other_heading":pd.to_numeric(runways[other+"_heading_degT"],errors="coerce")})
        rows["coordinate_heading"]=bearing(runways[side+"_latitude_deg"],runways[side+"_longitude_deg"],runways[other+"_latitude_deg"],runways[other+"_longitude_deg"])
        rows["heading_disagreement"]=np.abs((rows.supplied_heading-rows.coordinate_heading+180)%360-180)
        rows["reciprocal_error"]=np.abs((rows.supplied_heading-rows.other_heading)%360-180)
        values.append(rows)
    mapping=pd.concat(values,ignore_index=True)
    mapping=mapping.loc[mapping.runway.ne("")]
    mapping["duplicate_key"]=mapping.duplicated(["airport","runway"],keep=False)
    mapping["supplied_heading"]=mapping.supplied_heading.where(mapping.supplied_heading.between(0,360))
    return mapping,elevations,receipts


def project(east,north,heading):
    angle=np.deg2rad(heading)
    return -east*np.sin(angle)-north*np.cos(angle),-east*np.cos(angle)+north*np.sin(angle)


def density_proxy(qnh_inhg,temp_c,elevation_m):
    qnh=qnh_inhg*INHG_TO_HPA
    base=qnh**EXPONENT-STANDARD_PRESSURE_HPA**EXPONENT*LAPSE_K_PER_M*elevation_m/STANDARD_TEMP_K
    pressure=np.where((qnh>0)&(base>0),base**(1/EXPONENT)+.3,np.nan)
    kelvin=temp_c+273.15
    density=np.where(kelvin>0,pressure*100/(GAS_DRY*kelvin),np.nan)
    return pressure,density


def transform(query,weather,mapping,elevations):
    keys=query[["ADEP_mvt","RUNWAY_mvt"]].copy()
    keys["airport"]=norm(keys.ADEP_mvt);keys["runway"]=norm(keys.RUNWAY_mvt)
    mapped=keys.merge(mapping.loc[~mapping.duplicate_key],on=["airport","runway"],how="left",sort=False,validate="many_to_one")
    assert len(mapped)==len(query)
    out=pd.DataFrame({ID:query[ID].to_numpy()})
    for name,source in {"heading_true_deg":"supplied_heading","heading_coordinate_true_deg":"coordinate_heading","heading_disagreement_deg":"heading_disagreement","heading_reciprocal_error_deg":"reciprocal_error","source_current_closed":"closed_current"}.items():out["wxrunway_"+name]=mapped[source].to_numpy(float)
    out["wxrunway_mapping_available"]=mapped.runway_record_id.notna().to_numpy(float)
    east=weather.weather_T_wind_east_kt.to_numpy(float);north=weather.weather_T_wind_north_kt.to_numpy(float)
    speed=weather.weather_T_wind_speed_kt.to_numpy(float)
    calm=np.isfinite(speed)&(speed==0)
    east=np.where(calm,0,east);north=np.where(calm,0,north)
    for suffix,heading in (("",mapped.supplied_heading.to_numpy(float)),("coord_",mapped.coordinate_heading.to_numpy(float))):
        head,cross=project(east,north,heading)
        prefix="wxrunway_T_"+suffix
        out[prefix+"headwind_kt"]=head
        out[prefix+"crosswind_from_right_kt"]=cross
        out[prefix+"abs_crosswind_kt"]=np.abs(cross)
        out[prefix+"tailwind_kt"]=np.maximum(-head,0)
        out[prefix+"projection_available"]=np.isfinite(head).astype(float)
    out["wxrunway_T_calm_direction_resolved"]=calm.astype(float)
    elevation=keys.airport.map(elevations).to_numpy(float)
    temp=weather.weather_T_temperature_c.to_numpy(float)
    pressure,density=density_proxy(weather.weather_T_pressure_inhg.to_numpy(float),temp,elevation)
    out["wxair_airport_elevation_m"]=elevation
    out["wxair_T_isa_temperature_departure_c"]=temp-(15-LAPSE_K_PER_M*elevation)
    out["wxair_T_est_station_pressure_hpa"]=pressure
    out["wxair_T_est_dry_density_kg_m3"]=density
    out["wxair_T_est_dry_density_ratio"]=density/1.225
    out["wxair_T_density_available"]=np.isfinite(density).astype(float)
    for c in out:
        if c!=ID:out[c]=out[c].astype(np.float32)
    return out


def tests():
    head,cross=project(np.array([0.,-10.,0.,10.]),np.array([-10.,0.,10.,0.]),np.zeros(4))
    np.testing.assert_allclose(head,[10.,0.,-10.,0.],atol=1e-8)
    np.testing.assert_allclose(cross,[0.,10.,0.,-10.],atol=1e-8)
    hp,cp=project(np.array([-10.]),np.array([0.]),np.array([90.]))
    np.testing.assert_allclose(hp,[10.],atol=1e-8);np.testing.assert_allclose(cp,[0.],atol=1e-8)
    p,r=density_proxy(np.array([1013.25/INHG_TO_HPA]),np.array([15.]),np.array([0.]))
    np.testing.assert_allclose(p,[1013.55],atol=1e-7)
    np.testing.assert_allclose(r,[101355/(GAS_DRY*288.15)],atol=1e-7)
    print("TEST_PASS projection_signs_cardinals,station_pressure_units,dry_density",flush=True)


def main():
    tests()
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/"manifest.json").exists():raise ValueError("Completed cache preserved")
    wm=json.loads((WEATHER/"manifest.json").read_text())
    verified=json.loads((ROOT/"private_runs/breakthrough_20260916/geometry/weather_review/verification.json").read_text())
    assert wm["status"]=="complete" and verified["status"]=="passed" and verified["manifest_sha256"]==sha(WEATHER/"manifest.json")
    for name,digest in wm["outputs"].items():assert sha(WEATHER/name)==digest
    mapping,elevations,receipts=public_map()
    mapping.to_parquet(OUT/"public_runway_map.parquet",index=False)
    protocol={"created_utc":datetime.now(timezone.utc).isoformat(),"source_sha256":sha(__file__),"weather_manifest_sha256":sha(WEATHER/"manifest.json"),"weather_verification_sha256":sha(ROOT/"private_runs/breakthrough_20260916/geometry/weather_review/verification.json"),
       "ourairports_revision":receipts["runways.csv"]["revision"],"ourairports_commit_date":receipts["runways.csv"]["commit_date"],"source_vintage":"Current public2026 metadata; historical2025 validity NOT established",
       "matching":"Exact airport_ident plus runway end identifier after uppercase/trim; duplicates excluded; closedcurrentrunways retained with flag; no numericrunway heading inference",
       "heading":"Both supplied degreestrue and endpointcoordinate initialbearing separately; discrepancies retained, no learned correction",
       "wind":"Verified strictprior observation-valid weather_T only; headwind positive opposingtravel; crosswindpositive arrivingfromright; tailwindmax(-head,0); calm speed0 resolvesmissingdirection tozero",
       "density":"Approximate dryair density only: Smithsonian/MetPy altimeter->station pressure algebra with airport(notweatherstation)elevation; idealgas dryR287.05; humidityignored; no operationalperformanceclaim",
       "hidden_columns_loaded":[],"redundancy":"Feature interactions of existing weather/airport/runway data plus public heading/elevation, not new weather evidence","training":"No model training or evaluation"}
    (OUT/"protocol.json").write_text(json.dumps(protocol,indent=2),encoding="utf-8")
    training_weather=pd.read_parquet(WEATHER/"training_features.parquet").set_index(ID)
    ranking_weather=pd.read_parquet(WEATHER/"ranking_features.parquet").set_index(ID)
    frozen=json.loads((ROOT/"private_runs/submission_v2/protocol.json").read_text())
    records=[];writer=None;training_ids=[]
    for path in [*sorted(RAW.glob("training_*.parquet")),RAW/"ranking.parquet"]:
        assert sha(path)==frozen["raw_hashes"][path.name]
        raw=pq.read_table(path,columns=[ID,PHASE,"ADEP_mvt","RUNWAY_mvt"],use_threads=False).to_pandas()
        query=raw.loc[raw[PHASE].eq("DEP")].reset_index(drop=True)
        w=ranking_weather if path.name=="ranking.parquet" else training_weather
        assert w.index.is_unique and query[ID].isin(w.index).all()
        feature=transform(query,w.loc[query[ID]].reset_index(),mapping,elevations)
        assert np.array_equal(feature[ID],query[ID])
        if path.name=="ranking.parquet":feature.to_parquet(OUT/"ranking_features.parquet",index=False)
        else:
            training_ids.extend(query[ID].tolist())
            table=pa.Table.from_pandas(feature,preserve_index=False)
            if writer is None:writer=pq.ParquetWriter(OUT/"training_features.parquet",table.schema,compression="zstd")
            writer.write_table(table)
        record={"file":path.name,"rows":len(query),"airports":{}}
        for airport,indices in query.groupby("ADEP_mvt").indices.items():
            part=feature.iloc[indices]
            record["airports"][airport]={"rows":len(part),"mapped_rows":int(part.wxrunway_mapping_available.sum()),"heading_rows":int(part.wxrunway_heading_true_deg.notna().sum()),"wind_rows":int(part.wxrunway_T_projection_available.sum()),"coordinate_wind_rows":int(part.wxrunway_T_coord_projection_available.sum()),"density_rows":int(part.wxair_T_density_available.sum()),"heading_disagreement_gt3deg_rows":int(part.wxrunway_heading_disagreement_deg.gt(3).sum())}
        records.append(record);print(record,flush=True)
    writer.close()
    assert np.array_equal(training_ids,training_weather.index)
    assert np.array_equal(feature[ID],ranking_weather.index)
    manifest={"status":"complete","created_utc":datetime.now(timezone.utc).isoformat(),"source_sha256":sha(__file__),"protocol_sha256":sha(OUT/"protocol.json"),"training_rows":len(training_ids),"ranking_rows":len(feature),"features":list(feature.columns)[1:],"records":records,"training_ranking_id_order_verified":True,"outputs":{p.name:sha(p) for p in OUT.glob("*.parquet")},"prediction_gain_tested":False}
    (OUT/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")


if __name__=="__main__":main()
