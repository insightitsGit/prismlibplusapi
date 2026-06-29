"""
Render novelty_article.md as a formal, academic-style PDF (serif, single column,
justified, numbered sections — like an arXiv/technical-report paper).
Markdown -> HTML -> xhtml2pdf (pisa). Uses the built-in Times-Roman serif font.
"""

import re
from pathlib import Path

import markdown
from xhtml2pdf import pisa

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "novelty_article.md"
OUT = ROOT / "PrismLib_Novelty_CacheAsFailover.pdf"

raw = SRC.read_text(encoding="utf-8")
lines = raw.splitlines()

# Title block: first H1, the bold subtitle line, the affiliation/date line.
title = next((l[2:].strip() for l in lines if l.startswith("# ")), "Technical Note")
subtitle = ""
affil = ""
for l in lines[:8]:
    s = l.strip()
    if s.startswith("**") and not subtitle:
        subtitle = s.strip("* ").strip()
    elif "·" in s and not s.startswith("#") and not affil and "**" not in s:
        affil = s

# Body = everything after the first horizontal rule (drops the title block).
body_md = raw.split("\n---\n", 1)[1] if "\n---\n" in raw else raw
html_body = markdown.markdown(body_md, extensions=["tables", "fenced_code", "sane_lists"])

# Render the leading "## Abstract" specially: italicise its paragraph.
CSS = """
@page {
    size: letter;
    margin: 2.6cm 2.4cm 2.6cm 2.4cm;
    @frame footer { -pdf-frame-content: footerContent; bottom: 1.2cm; left: 2.4cm; right: 2.4cm; height: 1cm; }
}
body { font-family: "Times-Roman", serif; font-size: 10.5pt; color: #111111; line-height: 1.42; }
h1 { font-family: "Times-Roman", serif; font-size: 15pt; font-weight: bold; text-align: center;
     margin: 0 0 2pt 0; line-height: 1.25; }
h2 { font-family: "Times-Roman", serif; font-size: 11.5pt; font-weight: bold; margin: 15pt 0 5pt 0; }
h3 { font-family: "Times-Roman", serif; font-size: 10.5pt; font-weight: bold; font-style: italic; margin: 11pt 0 3pt 0; }
p  { margin: 0 0 6pt 0; text-align: justify; }
ul, ol { margin: 4pt 0 8pt 0; }
li { margin: 2pt 0; text-align: justify; }
strong { font-weight: bold; }
em { font-style: italic; }
code { font-family: "Courier", monospace; font-size: 9.5pt; }
a { color: #111111; text-decoration: none; }
hr { border: none; border-top: 0.5px solid #999999; margin: 12pt 0; }
.title { text-align: center; font-size: 15pt; font-weight: bold; margin: 0 0 6pt 0; line-height: 1.25; }
.subtitle { text-align: center; font-size: 11pt; font-style: italic; margin: 0 0 4pt 0; }
.author { text-align: center; font-size: 11.5pt; margin: 8pt 0 2pt 0; }
.affil { text-align: center; font-size: 10pt; margin: 0 0 4pt 0; }
.rule { border-top: 1px solid #333333; margin: 12pt 0 14pt 0; }
"""

HTML = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
  <meta name="author" content="Amin Parva">
  <style>{CSS}</style></head>
<body>
  <div id="footerContent" style="font-size:8pt; color:#777777; text-align:center;">
    {title} &middot; <pdf:pagenumber>
  </div>

  <div class="title">{title}</div>
  <div class="subtitle">{subtitle}</div>
  <div class="author">Amin Parva</div>
  <div class="affil">{affil}</div>
  <div class="rule"></div>

  {html_body}
</body></html>
"""

with open(OUT, "wb") as f:
    result = pisa.CreatePDF(HTML, dest=f, encoding="utf-8")
if result.err:
    raise SystemExit(f"PDF generation failed with {result.err} error(s)")
print(f"OK wrote {OUT}  ({OUT.stat().st_size // 1024} KB)")
