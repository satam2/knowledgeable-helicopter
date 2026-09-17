"""Five-page private visual proposal; all numerical results come from saved receipts."""

import json
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'private_runs/aviation_spike'
AN = json.loads((DATA/'analysis.json').read_text())
ARR = json.loads((DATA/'arrival_availability.json').read_text())
PDF = ROOT/'output/PRC_2026_Aviation_Research_Proposal.pdf'
for name, file in [('Arial', 'arial.ttf'), ('Arial-Bold', 'arialbd.ttf')]:
    pdfmetrics.registerFont(TTFont(name, str(Path('C:/Windows/Fonts')/file)))
pdfmetrics.registerFontFamily('Arial', normal='Arial', bold='Arial-Bold')
INK, TEAL, BLUE, RED, GRAY, GRID, PALE = [colors.HexColor(s) for s in
    ['#20282B','#007F78','#387FA6','#BE543D','#647279','#D7E1E3','#EFF5F5']]
W,H,M = 612,792,42
C = canvas.Canvas(str(PDF), pagesize=(W,H))
C.setTitle('PRC 2026 | Aviation-Informed Research Proposal')
C.setAuthor('Private Local Research')
LAYOUT = []


def text(value,x,top,size=10,bold=False,color=INK,right=False):
    value = str(value)
    font = 'Arial-Bold' if bold else 'Arial'
    width = pdfmetrics.stringWidth(value,font,size)
    assert (x-width if right else x)>=M-1 and (x if right else x+width)<=W-M+1, value
    C.setFont(font,size)
    C.setFillColor(color)
    (C.drawRightString if right else C.drawString)(x,H-top-size,value)


def para(value,top,x=M,width=528,size=10.5,leading=15,max_h=75,color=INK):
    p=Paragraph(value,ParagraphStyle('p',fontName='Arial',fontSize=size,leading=leading,textColor=color))
    _,h=p.wrap(width,H)
    assert h<=max_h and top+h<735,(C.getPageNumber(),value[:60],top,h,max_h)
    p.drawOn(C,x,H-top-h)
    LAYOUT.append({'page':C.getPageNumber(),'text':value,'top':top,'height':h,'x':x,'width':width})


def line(top):
    C.setStrokeColor(GRID)
    C.setLineWidth(.6)
    C.line(M,H-top,W-M,H-top)


def page(n,title,subtitle):
    if n>1:
        C.showPage()
    text('PRC 2026 / PRIVATE RESEARCH PROPOSAL',M,26,8.5,True,TEAL)
    text(f'0{n} / 05',W-M,26,8.5,True,GRAY,True)
    para(title,65,size=25,leading=29,max_h=60)
    para(subtitle,107,size=10.2,leading=14,max_h=44,color=GRAY)
    line(748)
    text('16 SEPTEMBER 2026  |  Proposal only. No new campaign launched.',M,761,7.7,color=GRAY)
    text(str(n),W-M,761,8,color=GRAY,right=True)


def node(title,detail,x,top,width=156,height=66,color=TEAL):
    C.setFillColor(PALE)
    C.rect(x,H-top-height,width,height,fill=1,stroke=0)
    C.setFillColor(color)
    C.rect(x,H-top-height,3,height,fill=1,stroke=0)
    para('<b>'+title+'</b>',top+10,x+11,width-20,size=10,leading=13,max_h=28)
    para(detail,top+32,x+11,width-20,size=9,leading=12,max_h=height-33,color=GRAY)


def arrow(x1,y1,x2,y2):
    C.setStrokeColor(TEAL)
    C.setLineWidth(1.2)
    C.line(x1,H-y1,x2,H-y2)
    C.line(x2,H-y2,x2-4,H-y2+3)
    C.line(x2,H-y2,x2-4,H-y2-3)


best=AN['candidates']['control']['seasonal_rmse_sec']
oracle=AN['fixed_expert_oracle_gate']['seasonal_rmse_sec']
page(1,'Aviation first. Two processes.',
     'Recommendation: add a model of surface operations and strengthen the model of timestamp reliability.')
for x,value,title in [(42,f'{best:.2f}s','CURRENT LOCAL REFERENCE'),(239,f'{oracle:.2f}s','IDEAL EXISTING GATE*'),(470,'230s','OBJECTIVE')]:
    text(value,x,165,28,True,TEAL if x!=239 else BLUE)
    text(title,x,204,7.1,True,GRAY)
para('*A label-aware oracle between the two frozen experts, with all other routes fixed. It is a diagnostic bound, not an achievable score. Refining this gate alone cannot reach 230.',238,size=10,leading=14,max_h=46)
line(301)
text('The residual is not additional taxi delay',M,321,14,True)
para('Takeoff minus airport off-block is the target. Takeoff minus NM off-block is our clock proxy. Subtracting them cancels takeoff:',350,max_h=46)
C.setFillColor(PALE)
C.rect(M,H-443,528,40,fill=1,stroke=0)
text('target - proxy = NM off-block - airport off-block',56,413,15,True,TEAL)
node('Surface operations','Route time + queues + operating conditions',42,473,247,78,TEAL)
node('Timestamp reliability','Which clock or recording convention is trustworthy?',321,473,249,78,BLUE)
para('The new physical expert should predict total taxi time without depending on NM actual off-block. Keep the clock and schedule experts, then combine different estimates using chronological out-of-sample evidence.',581,max_h=61)
para(f'<b>Scale of the task:</b> reaching 230 from {best:.2f} requires 38.3% less mean squared error. The reported competitor range is not independently verified against our local cohorts. There is no supported promise that 230 is attainable.',672,size=9.5,leading=13,max_h=54)

page(2,'What the small tests tell us',
     'Two additions to the same 300-tree gate. Complete, unchanged July and November score cohorts.')
text('Gate input',M,165,9,True,GRAY)
for name,x in [('July',340),('November',446),('Seasonal',570)]:
    text(name,x,165,9,True,GRAY,True)
for i,(arm,title) in enumerate([('control','Current reference'),('geometry','Stand/runway reference times'),('operations','Surface operations context')]):
    top=195+i*33
    item=AN['candidates'][arm]
    text(title,M,top,10,arm=='control')
    for val,x in [(item['folds']['F1']['rmse_sec'],340),(item['folds']['F3']['rmse_sec'],446),(item['seasonal_rmse_sec'],570)]:
        text(f'{val:.2f}',x,top,10,arm=='control',TEAL if arm=='control' else INK,True)
para('<b>Decision: promote neither.</b> Geometry adds 0.056 seconds; operations adds 0.025. The tests assess added information in the current gate, not a new physical expert, arrival-duration features, or a real queue model.',302,size=10,leading=14,max_h=46)
line(367)
text('Where the remaining squared error sits',M,385,14,True)
text('Diagnostic slices; percentages are seasonal error shares',M,410,9,color=GRAY)
rows=[('NM missing','NM clock missing',TEAL),('NM target gap <=60s','Clock / target gap <=60s',BLUE),
      ('NM target gap 60..300s','Gap 1-5 minutes',BLUE),('NM target gap 300..1800s','Gap 5-30 minutes',BLUE),
      ('NM target gap >1800s','Gap >30 minutes',RED)]
for i,(key,label,color) in enumerate(rows):
    top=440+i*33
    value=AN['diagnostics']['error_budget'][key]['sse_pct']
    text(label,M,top,9.4)
    C.setFillColor(color)
    C.rect(243,H-top-14,value/35*260,12,fill=1,stroke=0)
    text(f'{value:.1f}%',570,top,9.5,True,color,True)
para('Missing clocks and gaps above 30 minutes: <b>2.19% of rows, 47.14% of error.</b> These label-dependent slices diagnose the problem; they cannot select an expert at inference.',626,size=10,leading=14,max_h=45)
para('The largest 1% of errors account for 54.4% of squared error. Day-bootstrap intervals show no reliable gain from either operations probe; November geometry worsens slightly. One fixed seed; existing development folds.',686,size=9,leading=12.5,max_h=42)

page(3,'A useful signal already in the data',
     'Arrival taxi-in outcomes remain visible. The current feature pipeline discards them for both phases.')
for x,value,title in [(42,'344,693','RANKING ARRIVALS'),(259,'100%','TAXI-IN LABEL COVERAGE'),(473,'>99.5%','RECENT COVERAGE*')]:
    text(value,x,165,27,True,TEAL)
    text(title,x,204,7.2,True,GRAY)
para('*Departures with a completed arrival in the previous 30 minutes, in both score months. Median support: 16 arrivals in July and 15 in November. Availability is proven; predictive benefit has not been tested.',239,size=10,leading=14,max_h=46)
text('Turn arrivals into a surface-condition signal',M,312,14,True)
node('Arrival reference','Fit-only stand/runway taxi-in baseline',42,348,156,79)
node('Recent arrival excess','Observed duration minus arrival reference',229,348,156,79)
node('Physical taxi-out head','Combine state, route and departure demand',414,348,156,79)
arrow(200,387,227,387)
arrow(387,387,412,387)
para('Use recent excess, variability and sample support by airport/runway/stand. Require arrival in-block before the query in the causal version; landing alone is too early to know the completed taxi-in duration.',458,max_h=60)
line(535)
text('The domain distinction that counts miss',M,555,14,True)
para('<b>Throughput is not a queue.</b> Many takeoffs may mean either high demand or an efficiently clearing runway. Model pushback demand, aircraft still taxiing, and runway discharge separately. Arrival/departure mix and wake sequencing change capacity.',585,max_h=65)
para('Taxi-in is not taxi-out: runway exits, route direction, pushback and gate waits differ. Use arrivals as a contextual signal and a weak geometry prior. No aircraft registration is supplied; callsigns cannot establish exact aircraft rotations.',677,size=9.3,leading=13,max_h=52)

page(4,'The proposed first wave',
     'Four independent tracks after a common availability contract. Test information and targets before library size.')
items=[
('A','Arrival surface state','First new-signal test',
 'Arrival-specific references, recent excess and dispersion. Compare duration features against count-only context in a direct physical head. Keep support and missingness explicit.'),
('B','Independent physical expert','New target and error pattern',
 'Predict taxi time minus a shrunk stand/runway q10 reference, without NM actual off-block. Compare direct-target and q20 controls with the same inputs; cross-fit every label-derived prior.'),
('C','Demand and runway capacity','Describe the operating state',
 'Separate pushbacks, taxiing inventory and discharge. Add active runway configuration, changes and wake sequence. Evaluate conditions along the taxi interval, with explicit time availability.'),
('D','Clock and exceptional regimes','Address concentrated failures',
 'Learn soft reliability from clock precision, inter-clock differences, local-date rollover and airport/operator conventions. Keep missing and rare-duration cases; never route by their hidden labels.')]
for i,(letter,title,tag,body) in enumerate(items):
    top=165+i*119
    text(letter,M,top,22,True,TEAL)
    text(title,81,top,12.5,True)
    text(tag,81,top+22,8.7,True,BLUE)
    para(body,top+43,x=81,width=489,size=10,leading=14,max_h=57)
    if i<3:
        line(top+106)
line(650)
para('<b>Parallel feasibility checks:</b> openly licensed historical weather and stand/taxiway geometry at Rome and Paris first. Together they contribute 52.2% of error. Check coverage and historical accuracy before building models. LightGBM follows a useful feature set.',671,size=9.7,leading=13.5,max_h=58)

page(5,'Execution and decision gates',
     'A proposed campaign with clear dependencies. The completed spike leaves the current reference unchanged.')
node('1 / Fix the contract','Retrospective batch vs causal event-time inputs',42,163,156,76)
node('2 / Parallel screens','A-D own separate modules; E-F audit data coverage',229,163,156,76)
node('3 / Integrate evidence','One owner aligns results, blends and verifies',414,163,156,76)
arrow(200,201,227,201)
arrow(387,201,412,201)
text('Availability first',M,270,12.5,True)
para('The challenge supplies takeoff and retrospective logs. Strict-prior context is our current policy, not a restriction established by the data page. Review the full rules before separately testing label-free two-sided context. Final NM event timestamps do not prove real-time publication availability.',295,size=10,leading=14,max_h=61)
text('A result must survive the same cohort',M,377,12.5,True)
para('Use the original purges and complete raw-label cohorts. Advance at >=2 seconds seasonal gain, improvement in both months, every day-removal check, and <5% missing-clock regression; or a predeclared blend meeting those gates. Fit priors and blend weights only inside fit/tune data.',402,size=10,leading=14,max_h=64)
para('Then check F2/G1 and full-pipeline seeds. F2/G1 reuse October; all current score folds were exposed. A new prospective labeled period is needed for independent confirmation. Keep every failure. Do not trim score tails to manufacture a gain.',478,size=9.8,leading=13.7,max_h=58)
line(550)
text('Grounded in the task and aviation methodology',M,568,12,True)
sources=[
 ('[1] Challenge fields: departure outcomes hidden; arrival outcomes retained.', 'https://prc-data-challenge-2026.netlify.app/data.html'),
 ('[2] Eligibility: external data must be open and documented.', 'https://prc-data-challenge-2026.netlify.app/eligibility.html'),
 ('[3] EUROCONTROL taxi-out methodology, March 2023, sections 3.5-5.', 'https://ansperformance.eu/library/ATXOT_indicator_documentation_mar23.pdf')]
for i,(label,url) in enumerate(sources):
    para(f'<link href="{escape(url)}" color="#007F78">{escape(label)}</link>',599+i*24,size=9,leading=12,max_h=22)
para('EUROCONTROL uses a stand/runway q10 reference. Its monitoring filters and same-month labels must not be copied into predictive validation. The q20 probe here was a simpler, separately declared test.',674,size=8.9,leading=12.5,max_h=40)
text('Private data and all new derived artifacts remain outside Git. No upload or submission.',M,721,8.1,True,GRAY)
C.save()
(DATA/'proposal_layout.json').write_text(json.dumps(LAYOUT,indent=2),encoding='utf-8')
print(PDF)
