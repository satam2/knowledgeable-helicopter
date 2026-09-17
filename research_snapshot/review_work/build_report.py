"""Render the review from parsed Markdown with ReportLab and verified public data."""
import json
import re
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab import rl_config
from reportlab.graphics.shapes import Drawing, Line, Polygon, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate, Frame, KeepTogether, PageBreak, PageTemplate, Paragraph,
    Spacer, Table, TableStyle,
)
from reportlab.platypus.tableofcontents import TableOfContents

BASE = Path(__file__).resolve().parents[1]
WORK = BASE / "review_work"
OUT = BASE / "output/PRC_2026_Deep_Review.pdf"
WIDTH = 7.0 * inch
rl_config.warnOnMissingFontGlyphs = 1
for face, name in [("Arial", "arial.ttf"), ("Arial-Bold", "arialbd.ttf"), ("Arial-Italic", "ariali.ttf"), ("Arial-BoldItalic", "arialbi.ttf")]:
    pdfmetrics.registerFont(TTFont(face, str(Path("C:/Windows/Fonts") / name)))
pdfmetrics.registerFontFamily("Arial", normal="Arial", bold="Arial-Bold", italic="Arial-Italic", boldItalic="Arial-BoldItalic")
styles = getSampleStyleSheet()
styles.add(ParagraphStyle("BodyReview", fontName="Arial", fontSize=10.3, leading=14.2, spaceAfter=7.5, textColor=colors.black, splitLongWords=True, allowWidows=0, allowOrphans=0))
styles.add(ParagraphStyle("TitleReview", parent=styles["BodyReview"], fontName="Arial-Bold", fontSize=25, leading=29, spaceAfter=15))
styles.add(ParagraphStyle("SubtitleReview", parent=styles["BodyReview"], fontSize=14, leading=19, spaceAfter=14))
styles.add(ParagraphStyle("SectionReview", parent=styles["BodyReview"], fontName="Arial-Bold", fontSize=16, leading=20, spaceBefore=17, spaceAfter=10, keepWithNext=True))
styles.add(ParagraphStyle("SubsectionReview", parent=styles["BodyReview"], fontName="Arial-Bold", fontSize=11.7, leading=15, spaceBefore=10, spaceAfter=6, keepWithNext=True))
styles.add(ParagraphStyle("SmallReview", parent=styles["BodyReview"], fontSize=8.4, leading=11.2, spaceAfter=7))
styles.add(ParagraphStyle("CellReview", parent=styles["BodyReview"], fontSize=8.5, leading=11.4, spaceAfter=0))
styles.add(ParagraphStyle("CellHeaderReview", parent=styles["CellReview"], fontName="Arial-Bold", textColor=colors.white))
styles.add(ParagraphStyle("BulletReview", parent=styles["BodyReview"], leftIndent=12, firstLineIndent=-9, bulletIndent=0))
styles.add(ParagraphStyle("TOCReview", parent=styles["BodyReview"], fontSize=11, leading=16, spaceBefore=4, spaceAfter=4))
TEAL = colors.HexColor("#16675D")
GRAY = colors.HexColor("#5B6165")
LIGHT = colors.HexColor("#F3F5F5")


def inline(tokens):
    parts = []
    for token in tokens:
        kind = token["type"]
        text = token.get("text", token.get("raw", ""))
        if kind in {"text", "escape"}:
            parts.append(inline(token["tokens"]) if "tokens" in token else escape(text))
        elif kind == "codespan":
            parts.append('<font name="Arial">' + escape(text) + '</font>')
        elif kind in {"strong", "em"}:
            tag = "b" if kind == "strong" else "i"
            parts.append(f"<{tag}>" + inline(token["tokens"]) + f"</{tag}>")
        elif kind == "link":
            parts.append('<link href="' + escape(token["href"], {'"': '&quot;'}) + '" color="#16675D">' + inline(token["tokens"]) + '</link>')
        elif kind == "br":
            parts.append("<br/>")
        else:
            raise ValueError(f"Unsupported inline token: {kind}")
    return "".join(parts)


class ReviewDocument(BaseDocTemplate):
    def __init__(self):
        super().__init__(str(OUT), pagesize=(8.5*inch, 11*inch), leftMargin=.75*inch,
                         rightMargin=.75*inch, topMargin=.65*inch, bottomMargin=.65*inch,
                         title="PRC 2026 Taxi Out Prediction Review", author="Technical Review",
                         subject="Evidence review and improvement roadmap for taxi-out estimation")
        frame = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id="body", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
        self.addPageTemplates(PageTemplate(id="review", frames=frame, onPage=self.page_decoration))
        self.sections = []

    def beforeDocument(self):
        self.sections = []

    def page_decoration(self, canvas, doc):
        canvas.saveState()
        canvas.setFont("Arial", 8)
        canvas.setFillColor(GRAY)
        if doc.page > 1:
            canvas.drawString(self.leftMargin, 11*inch - .34*inch, "PRC 2026  |  Technical review")
        canvas.drawString(self.leftMargin, .32*inch, "15 September 2026  |  Revision 9e5b9e6")
        canvas.drawRightString(7.75*inch, .32*inch, str(doc.page))
        canvas.restoreState()

    def afterFlowable(self, flowable):
        if isinstance(flowable, Paragraph) and flowable.style.name == "SectionReview":
            title = flowable.getPlainText()
            key = "section-" + title.split()[0]
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(title, key, 0)
            self.notify("TOCEntry", (0, title, self.page, key))
            self.sections.append({"heading": title, "page": self.page})


def table_from_token(token):
    headings = [h["text"] for h in token["header"]]
    n = len(headings)
    if headings[0] == "Stage":
        widths = [65, 230, 209]
    elif headings[0] == "Observed proxy condition":
        widths = [181, 255, 68]
    elif headings[0] == "Fold" and headings[1] == "Initial fitting":
        widths = [37, 116, 108, 130, 113]
    elif headings[0] == "Candidate":
        widths = [208, 74, 74, 74, 74]
    elif headings[0] == "Fold" and headings[1] == "Missing rows":
        widths = [40, 70, 101, 108, 185]
    elif headings[0] == "Priority":
        widths = [50, 207, 247]
    else:
        widths = [WIDTH/n] * n
    numeric = token.get("align", [None]*n)
    cell_styles = [ParagraphStyle(f"Cell-{i}", parent=styles["CellReview"], alignment=TA_RIGHT if numeric[i] == "right" else TA_LEFT) for i in range(n)]
    header_styles = [ParagraphStyle(f"Header-{i}", parent=styles["CellHeaderReview"], alignment=TA_RIGHT if numeric[i] == "right" else TA_LEFT) for i in range(n)]
    data = [[Paragraph(inline(h["tokens"]), header_styles[i]) for i,h in enumerate(token["header"])]]
    for row in token["rows"]:
        data.append([Paragraph(inline(cell["tokens"]), cell_styles[i]) for i,cell in enumerate(row)])
    table = Table(data, colWidths=widths, hAlign="LEFT", repeatRows=1, spaceBefore=3, spaceAfter=11)
    settings = [
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#3F474A")),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, LIGHT]),
        ("GRID", (0,0), (-1,-1), .45, colors.HexColor("#D9D9D9")),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("LEFTPADDING", (0,0), (-1,-1), 8), ("RIGHTPADDING", (0,0), (-1,-1), 8),
        ("TOPPADDING", (0,0), (-1,-1), 7), ("BOTTOMPADDING", (0,0), (-1,-1), 7),
    ]
    table.setStyle(TableStyle(settings))
    return table


def concentration_chart(evidence):
    drawing = Drawing(WIDTH, 218)
    left, scale = 47, 10.4
    drawing.add(String(0, 204, "Missing records occupy few rows but contribute substantial error", fontName="Arial-Bold", fontSize=10.5))
    for tick in [0, 10, 20, 30, 40]:
        x = left + tick*scale
        drawing.add(Line(x, 31, x, 174, strokeColor=colors.HexColor("#D9D9D9"), strokeWidth=.4))
        drawing.add(String(x, 17, f"{tick}%", fontName="Arial", fontSize=8, textAnchor="middle"))
    for i, fold in enumerate(["F1", "F2", "F3", "G1", "C1"]):
        item = evidence["folds"][fold]
        y = 158-i*27
        drawing.add(String(0, y+2, fold, fontName="Arial-Bold", fontSize=9))
        for j, (value, color) in enumerate([(item["missing_row_pct"], GRAY), (item["missing_sse_pct"], TEAL)]):
            yy = y+7-j*10
            drawing.add(Rect(left, yy, value*scale, 7, fillColor=color, strokeColor=None))
            drawing.add(String(left+value*scale+4, yy-.2, f"{value:.2f}%", fontName="Arial", fontSize=7.8))
    drawing.add(Rect(270, 185, 9, 6, fillColor=GRAY, strokeColor=None))
    drawing.add(String(283, 184, "Row share", fontName="Arial", fontSize=8))
    drawing.add(Rect(353, 185, 9, 6, fillColor=TEAL, strokeColor=None))
    drawing.add(String(366, 184, "Squared-error share", fontName="Arial", fontSize=8))
    return drawing


def method_diagram():
    drawing = Drawing(WIDTH, 141)
    boxes = [(0, 85, "Observed take-off T", "Available at inference"),
             (179, 85, "NM off-block N", "Observed proxy clock"),
             (358, 85, "Airport off-block B", "Training labels only")]
    for x, y, title, note in boxes:
        drawing.add(Rect(x, y, 146, 44, fillColor=LIGHT, strokeColor=colors.HexColor("#CAD1D0"), strokeWidth=.5))
        drawing.add(String(x+73, y+27, title, fontName="Arial-Bold", fontSize=8.8, textAnchor="middle"))
        drawing.add(String(x+73, y+12, note, fontName="Arial", fontSize=8, textAnchor="middle"))
    drawing.add(String(0, 58, "Proxy p = T - N", fontName="Arial-Bold", fontSize=10))
    drawing.add(String(179, 58, "Target y = T - B", fontName="Arial-Bold", fontSize=10))
    drawing.add(String(358, 58, "Correction r = N - B", fontName="Arial-Bold", fontSize=10))
    drawing.add(Rect(0, 2, WIDTH, 36, fillColor=colors.HexColor("#EAF3F1"), strokeColor=None))
    drawing.add(String(WIDTH/2, 23, "Prediction = observed proxy + learned clock correction", fontName="Arial-Bold", fontSize=10, textAnchor="middle"))
    drawing.add(String(WIDTH/2, 9, "Negative and missing proxies use separate direct prediction routes", fontName="Arial", fontSize=8.5, textAnchor="middle"))
    return drawing


def main():
    tokens = json.loads((WORK / "report_tokens.json").read_text(encoding="utf-8"))
    evidence = json.loads((WORK / "evidence.json").read_text(encoding="utf-8"))
    story = []
    toc_added = False
    active_subsection = ""
    diagram_added = False
    for token in tokens:
        kind = token["type"]
        if kind == "space":
            continue
        if kind == "heading":
            depth, text = token["depth"], token["text"]
            if depth == 2 and text.startswith("1 ") and not toc_added:
                story.append(PageBreak())
                story.append(Paragraph("Contents", styles["SubtitleReview"]))
                toc = TableOfContents()
                toc.levelStyles = [styles["TOCReview"]]
                story.extend([toc, Spacer(1, 20), Paragraph("The report separates measured results, code findings and proposed experiments. Source references R1-R18 are collected in Section 12.", styles["SmallReview"]), PageBreak()])
                toc_added = True
            style = "TitleReview" if depth == 1 else "SubtitleReview" if not re.match(r"\d", text) and depth == 2 else "SectionReview" if depth == 2 else "SubsectionReview"
            story.append(Paragraph(inline(token["tokens"]), styles[style]))
            if depth == 3:
                active_subsection = text
        elif kind == "paragraph":
            style = "SmallReview" if token["text"].startswith("Review date:") else "BodyReview"
            story.append(Paragraph(inline(token["tokens"]), styles[style]))
            if active_subsection.startswith("2.3 ") and "900-second" in token["text"] and not diagram_added:
                story.extend([Spacer(1, 4), method_diagram(), Paragraph("Figure 1. The residual task estimates disagreement between the NM and airport clocks. [R9, R10]", styles["SmallReview"])])
                diagram_added = True
        elif kind == "table":
            story.append(table_from_token(token))
            if token["header"][1]["text"] == "Missing rows":
                story.extend([concentration_chart(evidence), Paragraph("Figure 2. Shares use each fold's own complete scoring cohort. F2 and G1 share scoring rows. [R4, R6]", styles["SmallReview"])])
        elif kind == "list":
            for item in token["items"]:
                children = item["tokens"]
                body = " ".join(inline(child.get("tokens", [{"type":"text", "text":child.get("text","")}])) for child in children if child["type"] != "space")
                story.append(Paragraph("- " + body, styles["BulletReview"]))
        else:
            raise ValueError(f"Unsupported block token: {kind}")
    doc = ReviewDocument()
    doc.multiBuild(story)
    (WORK / "pdf_sections.json").write_text(json.dumps(doc.sections, indent=2), encoding="utf-8")
    print(json.dumps({"pdf": str(OUT), "pages": doc.page, "sections": doc.sections}, indent=2))


if __name__ == "__main__":
    main()
