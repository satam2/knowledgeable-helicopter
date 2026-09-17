"""Compose a six-page private planning brief from verified aggregate evidence."""
import json
import math
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.graphics import renderPDF
from reportlab.graphics.shapes import Circle, Drawing, Line, Polygon, Rect, String
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph

ROOT = Path(__file__).resolve().parents[1]
REPO = (ROOT / "knowledgeable-helicopter").resolve()
OUT = (ROOT / "output/PRC_2026_Data_Informed_Plan.pdf").resolve()
PRIVATE = (ROOT / "private_analysis").resolve()
assert not OUT.is_relative_to(REPO) and not PRIVATE.is_relative_to(REPO)
P = json.loads((PRIVATE / "profile.json").read_text(encoding="utf-8"))
F = json.loads((PRIVATE / "focused_checks.json").read_text(encoding="utf-8"))
E = json.loads((ROOT / "review_work/evidence.json").read_text(encoding="utf-8"))

for name, file in [("Arial", "arial.ttf"), ("Arial-Bold", "arialbd.ttf"), ("Arial-Italic", "ariali.ttf")]:
    pdfmetrics.registerFont(TTFont(name, str(Path("C:/Windows/Fonts") / file)))
pdfmetrics.registerFontFamily("Arial", normal="Arial", bold="Arial-Bold", italic="Arial-Italic", boldItalic="Arial-Bold")

INK = colors.HexColor("#202629")
TEAL = colors.HexColor("#087E78")
TEAL_PALE = colors.HexColor("#E6F3F1")
GRAY = colors.HexColor("#747E84")
LIGHT = colors.HexColor("#F0F3F4")
GRID = colors.HexColor("#D6DEDF")
WARM = colors.HexColor("#B64D36")
WARM_PALE = colors.HexColor("#FAEFEB")
WHITE = colors.white
W, H, M, CONTENT = 612, 792, 42, 528
LAYOUT = []
FIGURES = []
FIGURE_DIR = PRIVATE / "figures"
FIGURE_DIR.mkdir(exist_ok=True)

C = canvas.Canvas(str(OUT), pagesize=(W, H))
C.setTitle("PRC 2026 Data Informed Experiment Plan")
C.setAuthor("Technical Review")
C.setSubject("Six-page private plan revised after profiling the authorized PRC dataset")


def text(value, x, top, size=10.5, bold=False, color=INK, align="left"):
    C.setFillColor(color)
    C.setFont("Arial-Bold" if bold else "Arial", size)
    y = H - top - size
    if align == "right":
        C.drawRightString(x, y, value)
    elif align == "center":
        C.drawCentredString(x, y, value)
    else:
        C.drawString(x, y, value)


def para(value, top, x=M, width=CONTENT, size=10.5, leading=14.5, color=INK, max_h=100):
    style = ParagraphStyle("body", fontName="Arial", fontSize=size, leading=leading, textColor=color,
                           allowWidows=0, allowOrphans=0, splitLongWords=True)
    p = Paragraph(value, style)
    _, height = p.wrap(width, H)
    assert height <= max_h, (C.getPageNumber(), value[:80], height, max_h)
    assert top + height < 735, ("body overflow", C.getPageNumber(), top, height)
    p.drawOn(C, x, H-top-height)
    LAYOUT.append({"page":C.getPageNumber(),"x":x,"top":top,"width":width,"height":height,"text":value})
    return top + height


def rule(top, x=M, width=CONTENT):
    C.setStrokeColor(GRID)
    C.setLineWidth(.5)
    C.line(x, H-top, x+width, H-top)


def band(top, height, fill=TEAL_PALE, x=M, width=CONTENT):
    C.setFillColor(fill)
    C.rect(x, H-top-height, width, height, stroke=0, fill=1)


def page(number, section, title, subtitle, source):
    if number > 1:
        C.showPage()
    C.bookmarkPage(f"page-{number}")
    C.addOutlineEntry(section, f"page-{number}", 0)
    text("PRC 2026  /  PRIVATE EXPERIMENT PLAN", M, 25, 8.5, True, TEAL)
    text(f"{number:02d}  /  {section.upper()}", W-M, 25, 8.5, False, GRAY, "right")
    if title:
        para(title, 67, size=24, leading=28, max_h=58)
        para(subtitle, 108, size=10.8, leading=14.5, color=GRAY, max_h=46)
    rule(744)
    text(source, M, 753, 7.7, color=GRAY)
    text(f"15 SEP 2026   |   {number} / 6", W-M, 770, 7.7, color=GRAY, align="right")
    text("Private aggregates. No flight-level records reproduced.", M, 770, 7.7, color=GRAY)


def label(d, value, x, y, size=9, bold=False, color=INK, anchor="start"):
    d.add(String(x,y,value,fontName="Arial-Bold" if bold else "Arial",fontSize=size,fillColor=color,textAnchor=anchor))


def figure(name, drawing, top, caption=None):
    assert drawing.width <= CONTENT and top + drawing.height < 735
    renderPDF.draw(drawing,C,M,H-top-drawing.height)
    renderPDF.drawToFile(drawing,str(FIGURE_DIR/(name+".pdf")))
    FIGURES.append({"name":name,"page":C.getPageNumber(),"top":top,"height":drawing.height,"caption":caption})
    if caption:
        para(caption,top+drawing.height+5,size=8.6,leading=11.5,color=GRAY,max_h=37)


def bar_axes(d, left, right, bottom, top, maximum, ticks):
    for tick in ticks:
        xx=left+(right-left)*tick/maximum
        d.add(Line(xx,bottom,xx,top,strokeColor=GRID,strokeWidth=.45))
        label(d,f"{tick:g}%",xx,bottom-15,size=8,color=GRAY,anchor="middle")


def budget_chart():
    weight=192122/344841
    total=sum(w*E["folds"][f]["overall"]["sse"]/E["folds"][f]["overall"]["n"] for f,w in [("F1",weight),("F3",1-weight)])
    missing=sum(w*E["folds"][f]["missing"]["sse"]/E["folds"][f]["overall"]["n"] for f,w in [("F1",weight),("F3",1-weight)])
    share=missing/total
    d=Drawing(CONTENT,78)
    available_w=CONTENT*(1-share)
    d.add(Rect(0,23,available_w,47,fillColor=GRAY,strokeColor=None))
    d.add(Rect(available_w,23,CONTENT-available_w,47,fillColor=TEAL,strokeColor=None))
    label(d,f"{100*(1-share):.1f}%  observed-clock flights",12,49,11,True,WHITE)
    label(d,f"{100*share:.1f}%  missing clocks",available_w+11,49,10.6,True,WHITE)
    label(d,"0%",0,7,8,color=GRAY)
    label(d,"Share of historical seasonal mean squared error",CONTENT/2,7,8.4,color=GRAY,anchor="middle")
    label(d,"100%",CONTENT,7,8,color=GRAY,anchor="end")
    return d,math.sqrt(total-missing),100*(1-250**2/total)


def missing_airport_chart():
    d=Drawing(CONTENT,294)
    left,right,maximum=124,471,3.5
    bar_axes(d,left,right,20,270,maximum,[0,1,2,3])
    label(d,"2025 training",0,280,8.8,color=GRAY)
    d.add(Circle(92,283,3,fillColor=GRAY,strokeColor=None))
    label(d,"2026 ranking",205,280,8.8,color=TEAL)
    d.add(Circle(293,283,3,fillColor=TEAL,strokeColor=None))
    order=sorted(P["airports"],key=lambda a:P["ranking_airports"][a]["missing_pct"],reverse=True)
    for i,a in enumerate(order):
        y=255-i*24
        train,rank=P["airports"][a]["missing_pct"],P["ranking_airports"][a]["missing_pct"]
        xx1=left+train/maximum*(right-left)
        xx2=left+rank/maximum*(right-left)
        label(d,a,0,y-2,10,bold=a=="LIRF")
        d.add(Line(xx1,y,xx2,y,strokeColor=GRID,strokeWidth=2))
        d.add(Circle(xx1,y,3.5,fillColor=GRAY,strokeColor=None))
        d.add(Circle(xx2,y,3.5,fillColor=TEAL,strokeColor=None))
        label(d,f"{train:.2f}",xx1-5,y+6,8.3,color=GRAY,anchor="end")
        label(d,f"{rank:.2f}",xx2+6,y-9,8.3,bold=True,color=TEAL)
    return d


def tail_bar():
    total=P["diagnostics"]["missing_tail_concentration"]["total"]
    rome=P["diagnostics"]["missing_tail_concentration"]["by_airport"]["LIRF"]
    d=Drawing(CONTENT,60)
    split=CONTENT*rome/total
    d.add(Rect(0,23,split,29,fillColor=TEAL,strokeColor=None))
    d.add(Rect(split,23,CONTENT-split,29,fillColor=GRAY,strokeColor=None))
    label(d,f"Rome (LIRF): {rome} flights / {100*rome/total:.1f}%",10,33,10.5,True,WHITE)
    label(d,str(total-rome),split+(CONTENT-split)/2,33,9.5,True,WHITE,anchor="middle")
    label(d,"2025 missing-clock flights with raw taxi-out above two hours",0,6,8.5,color=GRAY)
    label(d,"Other airports",CONTENT,6,8.3,color=GRAY,anchor="end")
    return d


def support_chart():
    d=Drawing(CONTENT,199)
    left,right=190,480
    bar_axes(d,left,right,19,174,100,[0,25,50,75,100])
    key="ADEP_mvt+STAND_mvt+RUNWAY_mvt"
    specs=[("All ranking departures",151,[("250k shared sample",P["category_support"]["ranking_250k"][key]["support_lt20_pct"],GRAY),
                                             ("All training departures",P["category_support"]["ranking_all_training"][key]["support_lt20_pct"],TEAL)]),
           ("Missing-clock ranking only",72,[("Missing-only training",P["category_support"]["ranking_missing_specialist"][key]["support_lt20_pct"],GRAY),
                                              ("All training departures",P["category_support"]["ranking_missing_all_training"][key]["support_lt20_pct"],TEAL)])]
    for title,y,rows in specs:
        label(d,title,0,y+28,10.2,True)
        for j,(name,value,color) in enumerate(rows):
            yy=y-j*25
            label(d,name,0,yy+2,9.4,color=color)
            d.add(Rect(left,yy,value/100*(right-left),12,fillColor=color,strokeColor=None))
            label(d,f"{value:.1f}%",left+value/100*(right-left)+6,yy+2,9.3,True,color)
    return d


def prefix_chart():
    d=Drawing(CONTENT,144)
    left,right=190,468
    bar_axes(d,left,right,18,128,25,[0,5,10,15,20,25])
    for title,key,y in [("All ranking", "ranking",109),("Missing-clock ranking", "ranking_missing",55)]:
        label(d,title,0,y+14,10,True)
        for j,(name,value,color) in enumerate([
            ("Full flight code",F["flight_prefix"][key]["exact_designator_new_or_missing_pct"],GRAY),
            ("Coarse prefix",F["flight_prefix"][key]["new_or_unparsed_pct"],TEAL)]):
            yy=y-j*21
            label(d,name,0,yy-1,9.2,color=color)
            d.add(Rect(left,yy-2,value/25*(right-left),11,fillColor=color,strokeColor=None))
            label(d,f"{value:.1f}%",left+value/25*(right-left)+5,yy-1,9.2,True,color)
    return d


def schedule_chart():
    d=Drawing(CONTENT,178)
    left,right=207,477
    bar_axes(d,left,right,20,164,100,[0,25,50,75,100])
    rows=[("Rome: taxi-out >2h",439,F["schedule_regimes"]["lirf"]["schedule_within300_tail_pct"],TEAL),
          ("Rome: other durations",1049,F["schedule_regimes"]["lirf"]["schedule_within300_nontail_pct"],GRAY),
          ("Elsewhere: taxi-out >2h",42,F["schedule_regimes"]["other_airports"]["schedule_within300_tail_pct"],TEAL),
          ("Elsewhere: other durations",20940,F["schedule_regimes"]["other_airports"]["schedule_within300_nontail_pct"],GRAY)]
    for i,(name,n,value,color) in enumerate(rows):
        y=146-i*36
        label(d,name,0,y+7,9.6,True)
        label(d,f"n = {n:,}",0,y-6,8.4,color=GRAY)
        d.add(Rect(left,y-2,value/100*(right-left),15,fillColor=color,strokeColor=None))
        label(d,f"{value:.1f}%",left+value/100*(right-left)+5,y+1,9.2,True,color)
    return d


def arrow(x,y,length=16,color=GRAY):
    C.setStrokeColor(color)
    C.setFillColor(color)
    C.setLineWidth(.9)
    C.line(x,H-y,x+length,H-y)
    p=C.beginPath()
    p.moveTo(x+length,H-y)
    p.lineTo(x+length-4,H-y+3)
    p.lineTo(x+length-4,H-y-3)
    p.close()
    C.drawPath(p,stroke=0,fill=1)


def fold_timeline():
    d=Drawing(CONTENT,110)
    left=64
    cw=38
    months=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    for i,name in enumerate(months):
        label(d,name,left+i*cw+cw/2,94,8.5,color=GRAY,anchor="middle")
    for row,(fold,tune,score) in enumerate([("F1",5,6),("F3",9,10)]):
        y=62-row*31
        label(d,fold,0,y+6,10.5,True)
        for i in range(12):
            fill=LIGHT if i<tune else TEAL_PALE if i==tune else TEAL if i==score else WHITE
            d.add(Rect(left+i*cw,y,cw-3,22,fillColor=fill,strokeColor=GRID,strokeWidth=.4))
            if i==tune:
                label(d,"tune",left+i*cw+(cw-3)/2,y+7,8.2,color=TEAL,anchor="middle")
            if i==score:
                label(d,"score",left+i*cw+(cw-3)/2,y+7,8.2,True,WHITE,anchor="middle")
    label(d,"Gray = initial fit. Refit includes fit + tune. Scoring rows are never sampled.",0,6,8.4,color=GRAY)
    return d


def build():
    page(1,"Decision","","","[1] Verified private pack. [2] Historical validation. Target of 250s supplied by user.")
    text("PRC 2026",M,69,33,True)
    text("Data-informed experiment plan",M,115,24,True)
    para("A focused next step toward the 250-second benchmark",156,size=12,leading=16,color=GRAY,max_h=20)
    para("Keep the existing CatBoost and clock-correction architecture. The actual data points to two priorities: <b>better support for ordinary predictions</b> and <b>more informative missing-clock predictions</b>.",199,size=11.2,leading=16,max_h=53)
    for x,value,title in [(M,"2.09M","TRAINING DEPARTURES"),(M+182,"14 / 14","INPUT HASHES MATCH"),(M+364,"344,841","RANKING DEPARTURES")]:
        text(value,x,278,27,True,TEAL if x==M else INK)
        text(title,x,316,8.2,True,GRAY)
    rule(347)
    text("Both prediction groups need to improve",M,367,15,True)
    chart,floor,reduction=budget_chart()
    figure("01_error_budget",chart,400)
    para(f"The published local seasonal score is <b>331.2 seconds RMSE</b>. Even perfect missing-clock predictions would leave <b>{floor:.1f}s</b> if other errors stayed fixed. Reaching 250 locally would need about <b>{reduction:.0f}% less mean squared error</b>.",492,size=10.7,leading=14.8,max_h=60)
    for top,num,title,body in [(570,"01","Scale the useful signal","Test more training rows and a higher tree ceiling separately."),
                                (618,"02","Recover missing context","Test a coarse flight prefix; then consider shared movement-only learning."),
                                (666,"03","Target the schedule hypothesis","Test an extra correction for Rome missing-clock flights, with the current fallback.")]:
        text(num,M,top,13,True,TEAL)
        text(title,M+34,top,11,True)
        para(body,top+18,x=M+34,width=CONTENT-34,size=9.8,leading=13,max_h=29)

    page(2,"What changed","Missing clocks hide different populations",
         "A missing NM clock is a missing information bundle. Airport context matters.",
         "[1] Full 2025 training and Jan/Jul 2026 ranking departures. Descriptive aggregates.")
    text("Missing-clock rate by airport",M,155,12,True)
    figure("02_airport_missingness",missing_airport_chart(),181)
    para("Training covers the full year; ranking covers January and July. These rates describe the supplied populations, not a season-adjusted drift estimate.",483,size=8.8,leading=11.8,color=GRAY,max_h=28)
    band(530,67)
    para("<b>Eight NM fields disappear together.</b> Every missing-AOBT departure also lacks the other seven checked NM clock/flight fields: 22,470 training rows and 5,290 ranking rows. A single-clock imputation will not restore that context.",542,x=M+12,width=CONTENT-24,size=10.1,leading=14,max_h=46)
    text("Rome contains most of the missing-clock extreme tail",M,620,12,True)
    figure("03_rome_tail",tail_bar(),652)

    page(3,"Recover context","Use more data where support is thin",
         "Location coverage supports a scaling test. Coarse flight text may transfer better than exact codes.",
         "[1] Counts in the original pack. Support and novelty are diagnostics, not accuracy results.")
    text("Ranking rows with fewer than 20 training examples",M,156,12,True)
    para("Support is counted for the exact airport / stand / runway combination.",176,size=9.2,leading=12,color=GRAY,max_h=15)
    figure("04_category_support",support_chart(),203)
    para("The full pool gives much better support, but matched and unmatched flights have different labels. A shared movement-only model with missing-group calibration is a <b>second-wave test</b>, not an assumed improvement.",410,size=10.1,leading=14,max_h=44)
    text("A reusable prefix avoids much of the exact-code novelty",M,479,12,True)
    para("Share of ranking rows whose value is new or unavailable in 2025 training.",499,size=9.2,leading=12,color=GRAY,max_h=15)
    figure("05_prefix_novelty",prefix_chart(),521)
    para("Test the observed two- or three-letter prefix before the numeric suffix, plus an unparsed flag. Treat it as a category, not a verified airline identity. Prefix novelty includes unparsed strings; exact-code novelty includes absent values. Verify performance on unseen groups.",681,size=9.1,leading=12.3,max_h=51)

    page(4,"Target the tail","Use the schedule signal selectively",
         "Take-off minus scheduled time can track long taxi-out at Rome, but is unreliable as a universal proxy.",
         "[1] Missing-clock training rows only. True-duration groups are diagnostic, never routing inputs.")
    text("When is the schedule offset within five minutes of the label?",M,155,12,True)
    figure("06_schedule_agreement",schedule_chart(),184)
    para("These are raw clock-agreement rates, not trained-model accuracy. The airport difference motivates a test; it does not establish the cause of the long durations.",373,size=9.1,leading=12.3,color=GRAY,max_h=29)
    rule(421)
    text("A long schedule offset is not a reliable global tail detector",M,440,11.5,True)
    for x,key,title in [(M,"lirf","ROME / LIRF"),(M+280,"other_airports","OTHER AIRPORTS")]:
        r=F["schedule_regimes"][key]
        text(f"{r['tail_precision_pct']:.1f}%",x,467,30,True,TEAL if key=="lirf" else WARM)
        text(title,x+105,475,9.1,True,GRAY)
        text(f"{r['high_anchor_true_tail_n']:,} of {r['high_anchor_n']:,} flagged flights",x,507,9.5,color=GRAY)
    para("Among missing-clock rows with observed schedule offset above two hours, these shares actually have a taxi-out label above two hours. A blanket rule would make many false assignments.",533,size=9.8,leading=13.3,max_h=43)
    band(595,125)
    text("PROPOSED TEST",M+12,606,8.4,True,TEAL)
    para("For Rome missing-clock flights: <b>schedule offset + learned correction</b>. Fit the correction using the eligible missing cohort and airport context; retain the direct specialist everywhere else and when the schedule is unusable.",626,x=M+12,width=CONTENT-24,size=10.1,leading=14,max_h=47)
    para("Rome has only 1,488 missing-clock training rows. Check ordinary cases, timing conventions and leave-day-out stability. Do not gate on the unknown true duration or repair raw labels.",678,x=M+12,width=CONTENT-24,size=9.1,leading=12.3,max_h=31)

    page(5,"First batch","Run eight focused comparisons",
         "Four candidates on the same two screening folds. Recover the baseline first; change one idea at a time.",
         "[3] Proposed work only. No new CatBoost models or validation scores were produced in this review.")
    experiments=[
        ("A","More training rows","250k to 500k; keep the 180-tree limit.","Measure error and rare-category support. Scale further only if the curve improves."),
        ("B","More boosting capacity","180 to 400 trees; keep the 250k sample.","Keep tuning-month early stopping. Record selected trees; a larger cap is not a result."),
        ("C","Coarse flight-prefix feature","Add prefix + unparsed flag to the existing feature pipeline.","Use a training-only vocabulary. Check missing-clock and unseen-category errors separately."),
        ("D","Rome schedule-residual route","Add a missing-clock correction head; apply it to Rome only.","Keep other routes fixed. Reject gains that depend on one day or badly hurt ordinary Rome rows.")
    ]
    top=157
    for code,title,knob,readout in experiments:
        rule(top)
        text(code,M,top+13,23,True,TEAL)
        text(title,M+45,top+11,12,True)
        para(knob,top+31,x=M+45,width=CONTENT-45,size=10.2,leading=14,max_h=30)
        para(readout,top+63,x=M+45,width=CONTENT-45,size=9.4,leading=12.5,color=GRAY,max_h=29)
        top+=100
    rule(top)
    text("Original time splits; complete scoring cohorts",M,574,12,True)
    figure("07_screening_timeline",fold_timeline(),601)
    para("A/B use configuration changes; C/D need code. Keep four threads and one large fit until resources are measured.",718,size=8.5,leading=11,max_h=12)

    page(6,"Execution","Keep the work private and the gains credible",
         "Build the file boundary first. Then reproduce, screen and validate before combining changes.",
         "[1] Private profile + focused checks. [2] Repository comparison.json / release.yaml. [3] Proposed plan.")
    steps=[("01","External paths"),("02","Baseline"),("03","8 screens"),("04","2 finalists")]
    for i,(n,title) in enumerate(steps):
        x=M+i*137
        band(157,57,fill=LIGHT,x=x,width=117)
        text(n,x+10,165,9,True,TEAL)
        text(title,x+10,184,11,True)
        if i<3:
            arrow(x+121,185,12)
    text("Promote only when the evidence survives scrutiny",M,236,12,True)
    gates=[("Useful gain","At least max(2s, 0.5%) better seasonal RMSE; keep the existing fold-regression limits."),
           ("Stable gain","Check route errors and leave-day-out sensitivity; a single extreme day cannot carry the claim."),
           ("Broader support","Run the best two on F2/G1, then repeat finalists with three seeds before an ensemble.")]
    y=266
    for title,body in gates:
        text(title,M,y,10.5,True,TEAL)
        para(body,y-1,x=M+104,width=CONTENT-104,size=10,leading=13.8,max_h=31)
        y+=47
    para("December is already exposed. Additional folds remain development evidence, not a fresh independent test.",395,size=8.8,leading=11.5,color=GRAY,max_h=13)
    rule(414)
    text("The repository contains source, never the private pack",M,434,12,True)
    paths=[("data/","Existing private input folder"),("private_runs/","Proposed caches, models, reports and predictions"),("knowledgeable-helicopter/","Code, tests and existing public reports")]
    y=466
    for name,note in paths:
        text(name,M,y,10.4,True,TEAL if name!="knowledgeable-helicopter/" else INK)
        text(note,M+178,y,9.5,color=GRAY)
        y+=27
    para("These are sibling folders. No copying, hardlinks or junctions into the checkout. The stock commands still write there: introduce an external artifact root and reject paths inside the repo <b>before running audit or training</b>. Raw input paths alone are not enough.",558,size=10,leading=13.8,max_h=59)
    band(637,85,fill=TEAL_PALE)
    para("<b>After the first batch:</b> consider a shared movement-only baseline plus missing-group correction if sparse support still dominates. Test runway context, local time and quality features next; combine only changes that earn their place.",648,x=M+12,width=CONTENT-24,size=9.8,leading=13.3,max_h=44)
    para("Measured now: all 14 hashes, full-pack descriptive profiling and aggregate checks. The 331.2s score is historical; 250s is the user-reported target. New model gains remain unmeasured.",694,x=M+12,width=CONTENT-24,size=8.5,leading=11.2,max_h=24)
    C.save()
    (PRIVATE/"visual_plan_layout.json").write_text(json.dumps({"pages":6,"paragraphs":LAYOUT,"figures":FIGURES},indent=2),encoding="utf-8")
    print(json.dumps({"pdf":str(OUT),"pages":6,"figures":len(FIGURES)},indent=2))


if __name__=="__main__":
    build()
