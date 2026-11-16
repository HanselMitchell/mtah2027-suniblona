"""Render TECHNICAL_DOCUMENT.md to submission/suniblona.pdf (markdown + headless Chrome/Edge)."""
import os
import shutil
import subprocess
import sys

import markdown

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSS = """body{font-family:'Segoe UI',Arial,sans-serif;font-size:10pt;line-height:1.38;margin:0 6mm;color:#111}
h1{font-size:17pt;margin:0 0 4px}h2{font-size:12.5pt;margin:14px 0 4px;border-bottom:1px solid #ccc}
table{border-collapse:collapse;font-size:8.6pt;margin:6px 0}td,th{border:1px solid #bbb;padding:2px 5px}
th{background:#f0f0f0}code{font-size:8.8pt;background:#f4f4f4}pre{background:#f4f4f4;padding:6px;font-size:8.3pt;white-space:pre-wrap}
li{margin:2px 0}@page{size:A4;margin:14mm 12mm}"""


def main():
    src = os.path.join(ROOT, "TECHNICAL_DOCUMENT.md")
    out_dir = os.path.join(ROOT, "submission")
    os.makedirs(out_dir, exist_ok=True)
    html = markdown.markdown(open(src, encoding="utf-8").read(), extensions=["tables", "fenced_code"])
    html_path = os.path.join(out_dir, "suniblona.html")
    with open(html_path, "w", encoding="utf-8") as fh:
        fh.write("<!doctype html><html><head><meta charset='utf-8'><style>{}</style></head><body>{}</body></html>"
                 .format(CSS, html))
    candidates = [r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                  r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                  shutil.which("chromium") or "", shutil.which("google-chrome") or ""]
    browser = next((c for c in candidates if c and os.path.isfile(c)), None)
    if not browser:
        sys.exit("no Chrome/Edge found; open {} and print to PDF".format(html_path))
    pdf = os.path.join(out_dir, "suniblona.pdf")
    subprocess.run([browser, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    "--print-to-pdf=" + pdf, "file:///" + html_path.replace("\\", "/")], check=True)
    os.remove(html_path)
    print("wrote", pdf)


if __name__ == "__main__":
    main()
