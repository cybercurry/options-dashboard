"""
Daily S&P-500 fundamentals builder → site/data/fundamentals.json (the Fundamentals tab's
"research any ticker" coverage set).

Why separate from build_json.py:
  • Filings change quarterly, not intraday — a once-a-day refresh is plenty, and it keeps
    ~500 SEC requests OUT of the 30-minute options scan (protects that pipeline's runtime and
    stays well within SEC's fair-access limit).
  • ~500 entries would add ~2 MB to signals.json (fetched on every page load); a standalone
    file is lazy-loaded only when the Fundamentals tab is opened.

Lean on purpose: SEC XBRL metrics + verdicts + red flags + SEC industry + a live price for
valuation. NO yfinance business summary and NO AI summary (both are per-ticker and would be
slow/costly at 500×) — those stay watchlist-only, produced by build_json.py.
"""

import json
import time
import pathlib
import datetime

import requests

import fundamentals
import tradier

OUT = pathlib.Path(__file__).with_name("site") / "data" / "fundamentals.json"

# Maintained S&P-500 constituents (Symbol,Name,Sector,…). Try main then master.
_SP500_CSV = [
    "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv",
    "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv",
]
_MIN_OK = 300          # never overwrite the file with a partial/failed run
_SLEEP = 0.15          # between SEC GETs — fair-access throttle (~7 req/s at 2 GETs/ticker)


def _sp500_symbols():
    """The current S&P-500 tickers from the maintained dataset CSV. [] on failure."""
    for url in _SP500_CSV:
        try:
            r = requests.get(url, headers=fundamentals._HEADERS, timeout=20)
            r.raise_for_status()
            lines = r.text.strip().splitlines()
            syms = []
            for ln in lines[1:]:                      # skip header
                sym = ln.split(",", 1)[0].strip().strip('"').upper()
                if sym and sym not in syms:
                    syms.append(sym)
            if len(syms) >= _MIN_OK:
                return syms
        except Exception as e:
            print("  S&P list fetch failed (%s): %s" % (url, e))
    return []


def _cik_map():
    """SEC ticker→(cik10, title) — fetched ONCE for the whole batch (analyze() would re-pull
    this ~1 MB map per ticker, so we cache it and resolve locally instead)."""
    data = fundamentals._get("https://www.sec.gov/files/company_tickers.json")
    m = {}
    for row in data.values():
        t = str(row.get("ticker", "")).upper()
        if t:
            m[t] = (str(row["cik_str"]).zfill(10), row.get("title", t))
    return m


def _resolve(sym, cikmap):
    """S&P symbol → SEC (cik, title), tolerating dotted class tickers (BRK.B → BRK-B / BRKB)."""
    for cand in (sym, sym.replace(".", "-"), sym.replace(".", "")):
        hit = cikmap.get(cand.upper())
        if hit:
            return hit
    return (None, None)


def _prices(symbols):
    """Batch last prices via Tradier (comma-separated, chunked). {SYM: last}. {} if no token."""
    out = {}
    if not tradier.is_configured():
        print("  Tradier not configured — valuation (P/E, mkt cap) will be blank.")
        return out
    for i in range(0, len(symbols), 100):
        chunk = symbols[i:i + 100]
        try:
            for q in tradier.get_quotes(chunk):
                sym = str(q.get("symbol", "")).upper()
                last = q.get("last")
                if sym and isinstance(last, (int, float)):
                    out[sym] = float(last)
        except Exception as e:
            print("  price chunk %d failed: %s" % (i // 100, e))
    return out


def _lean_analyze(sym, title, cik, price):
    """SEC-only fundamentals entry (no yfinance, no AI). Same shape the app renders."""
    facts = fundamentals.company_facts(cik)                 # SEC GET
    m = fundamentals._metrics_from_sec(facts, price)
    flags, groups = fundamentals._assess(m)
    sic = fundamentals._sec_sic(cik)                        # SEC GET
    prof = {}
    if sic:
        prof = {"sic": sic, "industry": sic}
    return {"ok": True, "error": None, "ticker": sym.upper(), "company": title,
            "cik": cik, "source": "SEC EDGAR — 10-K / 10-Q filings",
            "profile": prof, "metrics": m, "groups": groups, "flags": flags,
            "price": price}


def main():
    syms = _sp500_symbols()
    if not syms:
        print("No S&P-500 list — leaving fundamentals.json untouched.")
        return
    print("S&P-500 symbols:", len(syms))
    cikmap = _cik_map()
    prices = _prices(syms)
    print("prices fetched:", len(prices))

    out = {}
    for n, sym in enumerate(syms, 1):
        cik, title = _resolve(sym, cikmap)
        if not cik:
            continue
        try:
            out[sym] = _lean_analyze(sym, title, cik, prices.get(sym))
        except Exception as e:
            # one bad filer never sinks the batch
            print("  %s failed: %s" % (sym, e))
        time.sleep(_SLEEP)
        if n % 50 == 0:
            print("  …%d/%d (%d ok)" % (n, len(syms), len(out)))

    if len(out) < _MIN_OK:
        print("Only %d ok entries (< %d) — leaving fundamentals.json untouched." % (len(out), _MIN_OK))
        return

    payload = {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "count": len(out),
        "source": "SEC EDGAR (10-K/10-Q XBRL) · S&P 500",
        "fundamentals": out,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, separators=(",", ":"), default=str))
    print("wrote %s — %d fundamentals (%.0f KB)" % (OUT, len(out), OUT.stat().st_size / 1024))


if __name__ == "__main__":
    main()
