"""Independently stream official XLSX source tables; never alter source workbooks."""
import os
for key in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"]:os.environ[key]="1"
import argparse
import itertools
import json
import posixpath
import sys
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime,timedelta
import numpy as np
import pandas as pd
import psutil
from acquire import ROOT,OUT as ACQUISITION,digest,write,now

OUT=ROOT / "private_runs/tail240_20260916/forensics/atfm/inspection_v1"
AIRPORTS=["EDDF","EDDM","EGLL","EHAM","LEBL","LEMD","LFPG","LIRF","LTAI","LTFM","LSZH"]
N="{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
R="{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"


def shared_strings(archive):
    if "xl/sharedStrings.xml" not in archive.namelist():return []
    root=ET.fromstring(archive.read("xl/sharedStrings.xml"))
    return ["".join(node.itertext()) for node in root.findall(N+"si")]


def sheets(archive):
    root=ET.fromstring(archive.read("xl/workbook.xml"))
    props=root.find(N+"workbookPr")
    assert props is None or props.get("date1904","0") not in ("1","true")
    rels=ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    paths={node.attrib["Id"]:posixpath.normpath(posixpath.join("xl",node.attrib["Target"])) for node in rels}
    return {node.attrib["name"]:paths[node.attrib[R+"id"]] for node in root.find(N+"sheets")}


def rows(archive,sheet_path,strings):
    parent=None
    with archive.open(sheet_path) as stream:
        for event,node in ET.iterparse(stream,events=("start","end")):
            if event=="start" and node.tag==N+"sheetData":parent=node
            elif event=="end" and node.tag==N+"row":
                cells={}
                formulas={}
                for cell in node.findall(N+"c"):
                    position=cell.attrib["r"]
                    raw=cell.find(N+"v")
                    kind=cell.get("t")
                    value=None if raw is None else raw.text
                    if kind=="s" and value is not None:value=strings[int(value)]
                    elif kind=="inlineStr":value="".join(cell.find(N+"is").itertext())
                    elif value is not None and kind not in ("str","e"):
                        try:value=float(value)
                        except ValueError:pass
                    cells[position]=value
                    formula=cell.find(N+"f")
                    if formula is not None:formulas[position]=formula.text
                yield int(node.attrib["r"]),cells,formulas
                if parent is not None:parent.clear()


def col(address):return "".join(c for c in address if c.isalpha())


def excel_date(value):
    if isinstance(value,(float,int)):
        if not np.isfinite(value):return None
        return (datetime(1899,12,30)+timedelta(days=float(value))).date().isoformat()
    if value is None:return None
    return pd.Timestamp(value).date().isoformat()


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--metadata-only",action="store_true");args=parser.parse_args()
    acquisition=json.loads((ACQUISITION / "manifest.json").read_text())
    assert acquisition["status"]=="complete"
    directory=OUT / ("metadata" if args.metadata_only else "tables")
    directory.mkdir(parents=True,exist_ok=False)
    write(directory / "protocol.json",{"created_utc":now(),"source_sha256":digest(__file__),
        "acquisition_manifest_sha256":digest(ACQUISITION / "manifest.json"),"source_workbooks_unchanged":True,
        "parser":"StdlibZipFile+ElementTree streaming row elements. Sharedstrings decoded; explicit sparsecelladdresses. Cached sourcevalues read and formula texts retained in metadata. Excel1900dates only;1904 rejected.",
        "selection":"Bothofficial DATA sheets; allrows scanned. Retain all11declaredairports overallpublisheddates; airport/date filters do not dependonprivateoutcomes.",
        "scope":"Actualmetadata/fielddefinitions/sourcecoverage, no prediction and no taxiout indicator data.","airports":AIRPORTS})
    reports=[]
    for source in acquisition["sources"]:
        if not source["filename"].endswith(".xlsx"):continue
        path=ACQUISITION / source["filename"]
        assert digest(path)==source["sha256"]
        with zipfile.ZipFile(path) as archive:
            strings=shared_strings(archive);names=sheets(archive)
            metadata={name:[{"row":n,"cells":cells,"formulas":formulas} for n,cells,formulas in rows(archive,member,strings)] for name,member in names.items() if name in ["META","Copyright notice and disclaimer","Change_log"]}
            write(directory / (path.stem+"_metadata.json"),{"sheets":names,"metadata":metadata,"core":archive.read("docProps/core.xml").decode("utf-8")})
            reader=rows(archive,names["DATA"],strings)
            number,header,_=next(reader)
            labels={col(address):value for address,value in header.items() if value is not None}
            assert len(set(labels.values()))==len(labels)
            print("HEADERS",path.name,labels,flush=True)
            if args.metadata_only:
                write(directory / (path.stem+"_data_preview.json"),{"header_row":number,"labels":labels,"first_rows":[{"row":n,"cells":c,"formulas":f} for n,c,f in itertools.islice(reader,5)]})
                print("METADATA",path.name,json.dumps(metadata,ensure_ascii=True),flush=True)
                continue
            retained=[];scanned=0;formulas_count=0;first_date=None;last_date=None
            for number,cells,formulas in reader:
                if not any(v is not None for v in cells.values()):continue
                row={labels[col(address)]:value for address,value in cells.items() if col(address) in labels}
                if row.get("APT_ICAO") is None:continue
                scanned+=1;formulas_count+=len(formulas)
                date=excel_date(row.get("FLT_DATE"))
                if date is not None:
                    first_date=date if first_date is None else min(first_date,date)
                    last_date=date if last_date is None else max(last_date,date)
                if row["APT_ICAO"] in AIRPORTS:
                    row["source_row"]=number;row["date_utc"]=date;retained.append(row)
                if scanned%100000==0:
                    peak=getattr(psutil.Process().memory_info(),"peak_wset",psutil.Process().memory_info().rss)
                    assert peak<6*1024**3 and psutil.virtual_memory().available>8*1024**3
                    print("SCANNED",path.name,scanned,"retained",len(retained),flush=True)
            frame=pd.DataFrame(retained)
            assert not frame.duplicated(["APT_ICAO","date_utc"]).any()
            frame.to_parquet(directory / (path.stem+"_airports.parquet"),index=False)
            coverage=[]
            for airport,part in frame.groupby("APT_ICAO",sort=True):
                dates=pd.to_datetime(part.date_utc)
                for month in [f"2025-{m:02}" for m in range(1,13)]+["2026-01","2026-07"]:
                    expected=pd.date_range(month+"-01",pd.Timestamp(month+"-01")+pd.offsets.MonthEnd(1),freq="D")
                    available=pd.DatetimeIndex(dates[dates.dt.strftime("%Y-%m").eq(month)])
                    coverage.append({"airport":airport,"month":month,"days":len(available),"expected_days":len(expected),"missing_dates":[d.date().isoformat() for d in expected.difference(available)]})
            report={"file":path.name,"sha256":source["sha256"],"headers":list(labels.values()),"scanned_rows":scanned,
                "all_source_min_date":first_date,"all_source_max_date":last_date,"retained_rows":len(frame),"airports":sorted(frame.APT_ICAO.unique()),
                "retained_missing_cells":{c:int(frame[c].isna().sum()) for c in frame},"source_formula_cells_count":formulas_count,"coverage":coverage}
            reports.append(report)
            write(directory / (path.stem+"_coverage.json"),report)
            print("SOURCE_DONE",path.name,scanned,len(frame),first_date,last_date,flush=True)
    peak=getattr(psutil.Process().memory_info(),"peak_wset",psutil.Process().memory_info().rss)
    assert peak<6*1024**3 and psutil.virtual_memory().available>8*1024**3
    write(directory / "manifest.json",{"status":"complete","source_sha256":digest(__file__),"protocol_sha256":digest(directory / "protocol.json"),
        "peak_bytes":peak,"reports":reports,"outputs":{p.name:digest(p) for p in directory.iterdir() if p.is_file()}})


if __name__=="__main__":main()
