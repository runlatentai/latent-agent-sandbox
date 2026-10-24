"""Build-time: write PEP 503 simple-index pages for the wheels under /sandbox/index/simple/<project>/."""

import html
import os
import sys

root = sys.argv[1]
projects = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
for p in projects:
    files = sorted(f for f in os.listdir(os.path.join(root, p)) if f.endswith(".whl"))
    links = "".join(f'<a href="{html.escape(f)}">{html.escape(f)}</a><br>\n' for f in files)
    open(os.path.join(root, p, "index.html"), "w").write(f"<!DOCTYPE html><html><body>\n{links}</body></html>\n")
links = "".join(f'<a href="{p}/">{p}</a><br>\n' for p in projects)
open(os.path.join(root, "index.html"), "w").write(f"<!DOCTYPE html><html><body>\n{links}</body></html>\n")
