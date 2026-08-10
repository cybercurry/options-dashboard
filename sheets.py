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
_SKIP = {"WHEEL", "GROWTH", "STOCK", "TICKER", "CASH", "TOTAL", "SUM", "SIZE", "NA", "N/A",
         # position-block keywords — so a positions table on the same tab can't bleed into the
         # universe columns and be mistaken for tickers.
         "TYPE", "STRATEGY", "LEG", "STRIKE", "EXPIRY", "EXP", "EXPIRATION", "CONTRACTS", "QTY",
         "QUANTITY", "POSITION", "POSITIONS", "OPEN", "SYMBOL", "NAME",
         "CSP", "CC", "LEAP", "PMCC", "PUT", "CALL"}


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


# ── open positions block (for the Positions / trade-management view) ──────────────
_POS_TICKER = {"TICKER", "SYMBOL", "POSITION", "NAME"}
_POS_TYPE   = {"TYPE", "STRATEGY", "LEG"}
_POS_STRIKE = {"STRIKE"}
_POS_EXPIRY = {"EXPIRY", "EXP", "EXPIRATION"}
_POS_QTY    = {"CONTRACTS", "QTY", "QUANTITY", "CTR", "#"}


def parse_positions(csv_text):
    """Extract open positions from the published CSV. Finds the header row that contains an
    EXPIRY cell (unique to the positions block — the universe block has none), then reads the
    Ticker / Type / Strike / Expiry / Contracts columns by their header text. Returns a list of
    dicts (strings kept raw; the app parses dates/numbers). [] if no positions block is present."""
    rows = list(csv.reader(io.StringIO(csv_text)))
    hdr_i = cols = None
    for i, row in enumerate(rows):
        cells = [_norm(c) for c in row]
        if not any(c in _POS_EXPIRY for c in cells):
            continue
        c = {}
        for j, cell in enumerate(cells):
            if cell in _POS_TICKER and "ticker" not in c: c["ticker"] = j
            elif cell in _POS_TYPE and "type" not in c:   c["type"] = j
            elif cell in _POS_STRIKE and "strike" not in c: c["strike"] = j
            elif cell in _POS_EXPIRY and "expiry" not in c: c["expiry"] = j
            elif cell in _POS_QTY and "qty" not in c:      c["qty"] = j
        if "ticker" in c and "expiry" in c:      # a real positions header row
            hdr_i, cols = i, c
            break
    if hdr_i is None:
        return []

    def _cell(row, key):
        j = cols.get(key)
        return row[j].strip() if (j is not None and j < len(row)) else ""

    out, blanks = [], 0
    for row in rows[hdr_i + 1:]:
        tk = _norm(_cell(row, "ticker"))
        if not tk:
            blanks += 1
            if blanks >= 3:
                break
            continue
        blanks = 0
        if not _TICKER.match(tk) or tk in _SKIP:
            continue
        out.append({
            "ticker": tk,
            "type":   _cell(row, "type").upper() or "—",
            "strike": _cell(row, "strike"),
            "expiry": _cell(row, "expiry"),
            "contracts": _cell(row, "qty"),
        })
    return out


def fetch_positions(url):
    """GET the published CSV and parse the positions block. [] on any error."""
    if not url:
        return []
    try:
        r = requests.get(url, timeout=_TIMEOUT)
        r.raise_for_status()
        return parse_positions(r.text)
    except Exception:
        return []
