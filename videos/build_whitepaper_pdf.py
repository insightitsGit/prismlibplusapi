"""
Render a markdown doc to a styled, typeset PDF.
Uses python-markdown (tables + fenced code) -> HTML -> xhtml2pdf (pisa).

Usage:
    python build_whitepaper_pdf.py                       # default whitepaper
    python build_whitepaper_pdf.py <src.md> <out.pdf>    # any doc
"""

import re
import sys
from pathlib import Path

import markdown
from xhtml2pdf import pisa

ROOT = Path(__file__).resolve().parent.parent          # C:\code\PrismLib
if len(sys.argv) >= 3:
    SRC = Path(sys.argv[1])
    OUT = Path(sys.argv[2])
else:
    SRC = ROOT / "whitepaper_chorus_mesh.md"
    OUT = ROOT / "PrismLib_CHORUS_Mesh_Whitepaper.pdf"

raw = SRC.read_text(encoding="utf-8")

# --- Pull the title block (first H1 + the two lines under the rule) ----------
# Title = first level-1 heading; subtitle = the bold line; meta = the byline.
lines = raw.splitlines()
title = "PrismLib Micro & the CHORUS Mesh"
subtitle = ""
meta = ""
for ln in lines[:10]:
    if ln.startswith("# "):
        title = ln[2:].strip()
    elif ln.startswith("**") and not subtitle:
        subtitle = ln.strip("* ").strip()
    elif "Version" in ln and not meta:
        meta = ln.strip("* ").strip()

# Drop the original title block from the body (everything up to the first '---')
body_md = raw.split("\n---\n", 1)[1] if "\n---\n" in raw else raw

html_body = markdown.markdown(
    body_md,
    extensions=["tables", "fenced_code", "sane_lists"],
)

# Technical-report style: Times serif, single column, formal numbered sections,
# clean ruled tables, no marketing color.
CSS = """
@page {
    size: letter;
    margin: 2.4cm 2.3cm 2.5cm 2.3cm;
    @frame footer {
        -pdf-frame-content: footerContent;
        bottom: 1.2cm; left: 2.3cm; right: 2.3cm; height: 1cm;
    }
}
body { font-family: "Times-Roman", serif; font-size: 10.3pt;
       color: #111111; line-height: 1.4; }
h1 { font-family: "Times-Roman", serif; font-size: 14pt; font-weight: bold;
     color: #111111; margin: 16pt 0 4pt 0; padding-bottom: 3pt;
     border-bottom: 0.75px solid #555555; }
h2 { font-family: "Times-Roman", serif; font-size: 11.5pt; font-weight: bold;
     color: #111111; margin: 14pt 0 4pt 0; }
h3 { font-family: "Times-Roman", serif; font-size: 10.3pt; font-weight: bold;
     font-style: italic; color: #111111; margin: 11pt 0 3pt 0; }
p  { margin: 0 0 6pt 0; text-align: justify; }
ul, ol { margin: 4pt 0 8pt 0; }
li { margin: 2pt 0; text-align: justify; }
a  { color: #111111; text-decoration: none; }
strong { font-weight: bold; }
em { font-style: italic; }
code { font-family: "Courier", monospace; font-size: 9pt; color: #111111; }
pre  { font-family: "Courier", monospace; font-size: 8.4pt;
       background: #f4f4f4; border: 0.5px solid #cccccc;
       padding: 7pt; line-height: 1.3; }
table { -pdf-keep-with-next: true; margin: 8pt 0; width: 100%;
        border-top: 1px solid #333333; border-bottom: 1px solid #333333; }
th { background: #ededed; color: #111111; font-size: 8.8pt; font-weight: bold;
     padding: 4pt 6pt; text-align: left; border-bottom: 0.75px solid #777777; }
td { font-size: 8.8pt; padding: 3.5pt 6pt; border-bottom: 0.4px solid #dddddd;
     vertical-align: top; }
blockquote { border-left: 2px solid #888888; padding: 2pt 10pt; margin: 8pt 0;
             color: #333333; font-style: italic; font-size: 9.8pt; }
hr { border: none; border-top: 0.5px solid #bbbbbb; margin: 12pt 0; }
.titlepage { text-align: center; padding-top: 4.5cm; }
.titlepage .t  { font-size: 21pt; font-weight: bold; color: #111111; line-height: 1.25; }
.titlepage .s  { font-size: 12.5pt; font-style: italic; color: #222222; margin-top: 14pt; }
.titlepage .byline { font-size: 11pt; color: #222222; margin-top: 28pt; line-height: 1.7; }
.titlepage .badge { font-size: 8.5pt; color: #777777; margin-top: 10pt; }
"""

HTML = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
  <meta name="author" content="Amin Parva">
  <style>{CSS}</style></head>
<body>
  <div id="footerContent" style="font-size:8pt; color:#888888; text-align:center;">
    {title} &middot; InsightIts &middot; <pdf:pagenumber>
  </div>

  <div class="titlepage">
    <div class="t">{title}</div>
    <div class="s">{subtitle}</div>
    <div class="byline">
      <b>Amin Parva</b><br/>
      {meta}
    </div>
    <div class="badge">Apache 2.0 &middot; github.com/insightitsGit/prismlib &middot; insightits.info@gmail.com</div>
  </div>
  <pdf:nextpage/>

  {html_body}
</body></html>
"""

with open(OUT, "wb") as f:
    result = pisa.CreatePDF(HTML, dest=f, encoding="utf-8")

if result.err:
    raise SystemExit(f"PDF generation failed with {result.err} error(s)")
print(f"OK wrote {OUT}  ({OUT.stat().st_size // 1024} KB)")
