"""
Macro / bond-market data — keyless public sources, so no API key to manage.

  • Treasury yield curve + Fed funds rate  → FRED CSV export (fredgraph.csv?id=…), which is
    public and needs NO API key. Daily series, ~1 business-day lag (fine for macro context).
  • This-week economic calendar            → ForexFactory's own weekly JSON feed
    (nfs.faireconomy.media/ff_calendar_thisweek.json), rendered natively in the app.

Everything degrades quietly to None/[] so a source hiccup never breaks the Market Stats tab.
"""

import csv
import io

import requests

_TIMEOUT = 12
_FRED = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"
_FF_CAL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
_UA = {"User-Agent": "options-dashboard (macro fetch)"}

# Treasury constant-maturity yields, short → long. Label : FRED series id.
_TENORS = [("3M", "DGS3MO"), ("6M", "DGS6MO"), ("1Y", "DGS1"), ("2Y", "DGS2"),
           ("5Y", "DGS5"), ("7Y", "DGS7"), ("10Y", "DGS10"), ("30Y", "DGS30")]


def _fred_latest(series):
    """Most recent numeric value of a FRED series (missing days are '.'). None on any failure."""
    try:
        r = requests.get(_FRED.format(series), headers=_UA, timeout=_TIMEOUT)
        r.raise_for_status()
        rows = list(csv.reader(io.StringIO(r.text)))
        for row in reversed(rows[1:]):          # walk back to the last real print
            try:
                return float(row[-1])
            except (ValueError, IndexError):
                continue
    except Exception:
        return None
    return None


def yield_curve():
    """[(tenor_label, yield_pct), …] short→long, skipping any tenor FRED couldn't return."""
    out = []
    for lbl, sid in _TENORS:
        v = _fred_latest(sid)
        if v is not None:
            out.append((lbl, v))
    return out


def fed_funds_rate():
    """Effective federal funds rate (DFF), latest print, in percent. None on failure."""
    return _fred_latest("DFF")


def curve_spread_2s10s(curve):
    """10Y − 2Y in basis points from a yield_curve() list, or None if either tenor is missing.
    Negative = inverted (classic recession lead)."""
    d = dict(curve)
    if "2Y" in d and "10Y" in d:
        return round((d["10Y"] - d["2Y"]) * 100)
    return None


def econ_calendar():
    """This week's economic events from ForexFactory's public JSON feed. Returns a list of dicts:
    {title, country, date, impact, forecast, previous}. [] on any failure."""
    try:
        r = requests.get(_FF_CAL, headers=_UA, timeout=_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, list):
            return data
    except Exception:
        return []
    return []
