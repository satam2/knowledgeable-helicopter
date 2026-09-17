"""Public research discovery and primary documentation; never reads challenge data."""
from datetime import datetime,timezone
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request,urlopen

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/"output/breakthrough_20260916/research_gap/sources"
SOURCES={
 "mit_publications":"https://web.mit.edu/hamsa/www/pubs.html",
 "mit_home":"https://web.mit.edu/hamsa/www/",
 "eurocontrol_ddr":"https://www.eurocontrol.int/model/demand-data-repository",
 "eurocontrol_ddr_manual":"https://www.eurocontrol.int/publication/ddr2-reference-manual",
 "eurocontrol_rnd":"https://www.eurocontrol.int/dashboard/rnd-data-archive",
 "eurocontrol_cdm":"https://www.eurocontrol.int/concept/airport-collaborative-decision-making",
 "ans_methodology":"https://ansperformance.eu/methodology/",
 "crossref_queue":"https://api.crossref.org/works?"+urlencode({"query.bibliographic":"Simaiakis Balakrishnan queuing model airport departure process","rows":5}),
 "crossref_graph":"https://api.crossref.org/works?"+urlencode({"query.bibliographic":"airport taxi out time graph neural network aircraft surface trajectory","rows":7}),
 "nasa_surface":"https://ntrs.nasa.gov/api/citations/search?"+urlencode({"query":"taxi out time prediction surface surveillance","page.size":10}),
}


class Text(HTMLParser):
    def __init__(self):super().__init__();self.parts=[];self.links=[]
    def handle_data(self,data):self.parts.append(data)
    def handle_starttag(self,tag,attrs):
        if tag=="a":self.links.extend(v for k,v in attrs if k=="href")


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    receipts={}
    for name,url in SOURCES.items():
        rec={"url":url,"fetched_utc":datetime.now(timezone.utc).isoformat(),"private_data_read":False}
        try:
            with urlopen(Request(url,headers={"User-Agent":"PRC2026-public-input-observability-research/1.0"}),timeout=40) as response:
                body=response.read();ctype=response.headers.get("content-type","")
            (OUT/(name+".raw")).write_bytes(body)
            rec.update(status=200,content_type=ctype,sha256=hashlib.sha256(body).hexdigest(),bytes=len(body))
            if "html" in ctype:
                parsed=Text();parsed.feed(body.decode("utf-8",errors="replace"))
                (OUT/(name+".txt")).write_text("\n".join(x.strip() for x in parsed.parts if x.strip()),encoding="utf-8")
                (OUT/(name+"_links.json")).write_text(json.dumps(parsed.links,indent=2),encoding="utf-8")
            else:(OUT/(name+".txt")).write_bytes(body)
        except Exception as exc:rec["error"]=repr(exc)
        receipts[name]=rec
        (OUT.parent/"receipts.json").write_text(json.dumps(receipts,indent=2),encoding="utf-8")
        print(name,rec.get("status"),rec.get("error"),flush=True)


if __name__=="__main__":main()
