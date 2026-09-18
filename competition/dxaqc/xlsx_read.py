"""Minimal dependency-free .xlsx reader (first sheet -> list of rows of strings)."""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def _col_index(ref: str) -> int:
    n = 0
    for ch in re.match(r"[A-Z]+", ref).group(0):
        n = n * 26 + ord(ch) - 64
    return n - 1


def read_first_sheet(path: str | Path) -> list[list[str]]:
    with zipfile.ZipFile(path) as z:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in z.namelist():
            for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall("m:si", NS):
                shared.append("".join(t.text or "" for t in si.iter("{%s}t" % NS["m"])))
        sheet = sorted(n for n in z.namelist() if n.startswith("xl/worksheets/sheet"))[0]
        rows: list[list[str]] = []
        for row in ET.fromstring(z.read(sheet)).iter("{%s}row" % NS["m"]):
            cells: dict[int, str] = {}
            for c in row.findall("m:c", NS):
                v = c.find("m:v", NS)
                inline = c.find("m:is", NS)
                if c.get("t") == "s" and v is not None:
                    text = shared[int(v.text)]
                elif inline is not None:
                    text = "".join(t.text or "" for t in inline.iter("{%s}t" % NS["m"]))
                else:
                    text = v.text if v is not None and v.text is not None else ""
                cells[_col_index(c.get("r"))] = text
            rows.append([cells.get(i, "") for i in range(max(cells) + 1)] if cells else [])
    return rows
