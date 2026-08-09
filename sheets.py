"""
Google-Sheet feed reader — pulls the WHEEL and GROWTH ticker columns from a *published* CSV
tab (File → Share → Publish to web → one tab → CSV). Read-only, one-way: the URL serves only
that tab's cell values, so the master planning sheet stays private and there's no path back to
any brokerage. No API key, no OAuth — just an HTTPS GET the app and the cron can both do.

Robust by design: it locates the WHEEL and GROWTH columns by their HEADER text (not fixed
positions), so rearranging rows/columns in the sheet won't break it. Returns {} on any failure
so callers fall back to the committed wheel_universe.json.
"""

import csv
import io
import re

import requests

_TIMEOUT = 15
_TICKER = re.compile(r"^[A-Z][A-Z0-9.\-]{0,5}$")     # plausible US ticker
_SKIP = {"WHEEL", "GROWTH", "STOCK", "TICKER", "CASH", "TOTAL", "SUM", "SIZE", "NA", "N/A"}


def _norm(s):
    return (s or "").strip().upper()


def parse_universe(csv_text):
    """Extract {'wheel': [...], 'growth': [...]} from published-CSV text. Finds the row that
    contains BOTH a 'WHEEL' and a 'GROWTH' header cell, then reads down those two columns."""
    rows = list(csv.reader(io.StringIO(csv_text)))
    wheel_col = growth_col = header_i = None
    for i, row in enumerate(rows):
        cells = [_norm(c) for c in row]
        wc = next((j for j, c in enumerate(cells) if c == "WHEEL"), None)
        gc = next((j for j, c in enumerate(cells) if c.startswith("GROWTH")), None)
        if wc is not None and gc is not None:
            wheel_col, growth_col, header_i = wc, gc, i
            break
    if header_i is None:
        return {}

    def _col(idx):
        out, blanks = [], 0
        for row in rows[header_i + 1:]:
            val = _norm(row[idx]) if idx < len(row) else ""
            if not val:
                blanks += 1
                if blanks >= 4:            # a gap of 4 blank rows = end of the list
                    break
                continue
            blanks = 0
            if val in _SKIP or not _TICKER.match(val):
                continue
            if val not in out:
                out.append(val)
        return out

    return {"wheel": _col(wheel_col), "growth": _col(growth_col)}


def fetch_universe(url):
    """GET the published CSV and parse it. Returns {} on any error (network, non-CSV, no headers)
    so the caller can fall back to the committed lists."""
    if not url:
        return {}
    try:
        r = requests.get(url, timeout=_TIMEOUT)
        r.raise_for_status()
        u = parse_universe(r.text)
        return u if u.get("wheel") else {}
    except Exception:
        return {}
