"""
Convert docs/paper.md to docs/paper.pdf using markdown + xhtml2pdf.
"""
import markdown
from xhtml2pdf import pisa
import os

DOCS_DIR = os.path.join(os.path.dirname(__file__), '..', 'docs')
MD_PATH = os.path.join(DOCS_DIR, 'paper.md')
PDF_PATH = os.path.join(DOCS_DIR, 'paper.pdf')

with open(MD_PATH, 'r', encoding='utf-8') as f:
    md_text = f.read()

# Convert Markdown to HTML with tables and code highlighting
html_body = markdown.markdown(md_text, extensions=['tables', 'fenced_code'])

# Wrap in full HTML with professional styling
html_full = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
@page {{
    size: A4;
    margin: 2cm 2.5cm;
}}
body {{
    font-family: Georgia, 'Times New Roman', serif;
    font-size: 11pt;
    line-height: 1.5;
    color: #222;
}}
h1 {{
    font-size: 18pt;
    text-align: center;
    margin-bottom: 0.3em;
    color: #1a1a2e;
}}
h2 {{
    font-size: 14pt;
    border-bottom: 1px solid #ccc;
    padding-bottom: 4px;
    margin-top: 1.5em;
    color: #16213e;
}}
h3 {{
    font-size: 12pt;
    margin-top: 1.2em;
    color: #0f3460;
}}
p {{
    text-align: justify;
    margin: 0.5em 0;
}}
table {{
    border-collapse: collapse;
    width: 100%;
    margin: 1em 0;
    font-size: 10pt;
}}
th, td {{
    border: 1px solid #bbb;
    padding: 6px 10px;
    text-align: left;
}}
th {{
    background-color: #e8eaf6;
    font-weight: bold;
}}
code {{
    background-color: #f0f0f0;
    padding: 1px 4px;
    font-size: 10pt;
    font-family: Courier, monospace;
}}
pre {{
    background-color: #f8f8f8;
    border: 1px solid #ddd;
    padding: 10px;
    font-size: 9pt;
}}
pre code {{
    background: none;
    padding: 0;
}}
img {{
    max-width: 100%;
    margin: 1em auto;
}}
hr {{
    border: none;
    border-top: 1px solid #ccc;
    margin: 1.5em 0;
}}
a {{
    color: #1565c0;
    text-decoration: none;
}}
</style>
</head>
<body>
{html_body}
</body>
</html>
"""

# Write PDF using xhtml2pdf (resolves images relative to docs/ folder)
with open(PDF_PATH, 'wb') as pdf_file:
    status = pisa.CreatePDF(html_full, dest=pdf_file,
                            path=os.path.abspath(DOCS_DIR) + os.sep)

if status.err:
    print(f"Errors occurred during PDF generation: {status.err}")
else:
    size_kb = os.path.getsize(PDF_PATH) / 1024
    print(f"PDF saved to {PDF_PATH} ({size_kb:.0f} KB)")
