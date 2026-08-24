#!/usr/bin/env python3
"""build_json.py — headless engine runner for the optionintel.app static site.

Reuses the SAME engine as the Streamlit app (signals.py → tradier.py); no logic
rewrite. Meant to run on a schedule (GitHub Actions) with TRADIER_TOKEN in the
environment, or locally with the Streamlit secret. Writes site/data/signals.json,
which the static site fetches. Public/anonymous — watchlist only, never IBKR data.
"""
import json
import datetime
import pathlib

import requests

import signals
import sheets

HERE = pathlib.Path(__file__).parent
OUT = HERE / "site" / "data"


def load_universe():
    """Same source as the app: published-sheet CSV (tickers only), falling back to
    the committed wheel_universe.json so a Google hiccup never blanks the scan."""
    cfg = {}
    try:
        cfg = json.loads((HERE / "wheel_universe.json").read_text())
    except Exception:
        pass
    live = sheets.fetch_universe(cfg.get("source_url")) if cfg.get("source_url") else {}
    if live.get("wheel"):
        return {"wheel": live.get("wheel", []), "growth": live.get("growth", []), "_source": "sheet"}
    return {"wheel": cfg.get("wheel", []), "growth": cfg.get("growth", []), "_source": "fallback"}


def _fred_last(series):
    """Latest value of a FRED series via the keyless CSV endpoint."""
    try:
        r = requests.get("https://fred.stlouisfed.org/graph/fredgraph.csv?id=" + series, timeout=15)
        r.raise_for_status()
        for row in reversed(r.text.strip().splitlines()[1:]):
            parts = row.split(",")
            if len(parts) == 2 and parts[1] not in ("", ".", None):
                return float(parts[1])
    except Exception:
        return None
    return None


def _vix():
    try:
        import yfinance as yf
        h = yf.Ticker("^VIX").history(period="5d")
        if not h.empty:
            return round(float(h["Close"].dropna().iloc[-1]), 2)
    except Exception:
        return None
    return None


def _cnn_fng():
    try:
        r = requests.get(
            "https://production.dataviz.cnn.io/index/fearandgreed/graphdata",
            headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                                   "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
                     "Referer": "https://www.cnn.com/markets/fear-and-greed",
                     "Origin": "https://www.cnn.com"}, timeout=15)
        r.raise_for_status()
        fg = r.json().get("fear_and_greed", {})
        return round(float(fg.get("score"))), (fg.get("rating") or "").title()
    except Exception:
        return None, None


def _btc_fng():
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=1", timeout=15)
        r.raise_for_status()
        d = (r.json().get("data") or [{}])[0]
        return int(d.get("value")), d.get("value_classification")
    except Exception:
        return None, None


def _yf_quote(sym):
    """(price, pct-change) from the last two daily closes via yfinance."""
    try:
        import yfinance as yf
        c = yf.Ticker(sym).history(period="5d")["Close"].dropna()
        if len(c) >= 2:
            p, pv = float(c.iloc[-1]), float(c.iloc[-2])
            return p, ((p / pv - 1) * 100 if pv else None)
        if len(c) == 1:
            return float(c.iloc[-1]), None
    except Exception:
        pass
    return None, None


def _yf_last(sym):
    p, _ = _yf_quote(sym)
    return round(p, 2) if p is not None else None


# The app's PULSE_TICKERS / SECTOR_TICKERS, verbatim.
PULSE = [("^GSPC", "S&P 500", "", False), ("^NDX", "Nasdaq 100", "", False), ("^DJI", "Dow Jones", "", False),
         ("^RUT", "R2000", "", False), ("DX-Y.NYB", "DXY", "", False), ("CL=F", "Crude Oil", "$", False),
         ("GC=F", "Gold", "$", False), ("BTC-USD", "Bitcoin", "$", False), ("^TNX", "10Y Yield", "", True),
         ("^IRX", "3M Yield", "", True)]
SECTORS = [("XLK", "Technology"), ("XLF", "Financials"), ("XLV", "Health Care"), ("XLE", "Energy"),
           ("XLI", "Industrials"), ("XLC", "Comm. Services"), ("XLY", "Consumer Disc."),
           ("XLP", "Consumer Staples"), ("XLU", "Utilities"), ("XLRE", "Real Estate"),
           ("XLB", "Materials"), ("BTC-USD", "Digital Assets")]


def fetch_pulse():
    out = []
    for sym, label, prefix, is_yield in PULSE:
        p, pct = _yf_quote(sym)
        out.append({"label": label, "price": p, "pct": pct, "prefix": prefix, "is_yield": is_yield})
    return out


def fetch_sectors():
    out = []
    for sym, label in SECTORS:
        p, pct = _yf_quote(sym)
        out.append({"label": label, "ticker": ("BTC" if sym == "BTC-USD" else sym), "pct": pct, "price": p})
    return out


def fetch_market():
    """Keyless market snapshot — VIX + F&G + curve + Fed + the Overview's macro-signal extras.
    Each source degrades to None independently; never aborts the build."""
    vix = _vix()
    fng, fng_lbl = _cnn_fng()
    btc, btc_lbl = _btc_fng()
    t10, t2, fed = _fred_last("DGS10"), _fred_last("DGS2"), _fred_last("DFF")
    three_m = _yf_last("^IRX")               # 3M yield (for the 10Y−3M curve on Overview)
    skew = _yf_last("^SKEW")                  # CBOE SKEW (tail risk)
    vix9d = _yf_last("^VIX9D")               # 9-day VIX (vs 30-day → contango/backwardation)
    curve = None
    if t10 is not None and t2 is not None:
        spr = t10 - t2
        curve = "steepening" if spr > 0.5 else "flat" if spr > -0.1 else "inverted"
    vix_lbl = None
    if vix is not None:
        vix_lbl = "calm" if vix < 15 else "normal" if vix < 20 else "elevated" if vix < 30 else "high"
    return {"vix": vix, "vix_label": vix_lbl, "fng": fng, "fng_label": fng_lbl,
            "btc_fng": btc, "btc_fng_label": btc_lbl,
            "ten_year": t10, "two_year": t2, "three_month": three_m, "curve": curve, "fed_funds": fed,
            "skew": skew, "vix9d": vix9d}


def fetch_fundamentals_all(names, prices):
    """SEC-filing fundamentals per name (the app's Fundamentals tab), via the same pure
    `fundamentals.analyze()` module. Each ticker degrades to ok:False on its own error."""
    try:
        import fundamentals
    except Exception:
        return {}
    out = {}
    for t in names:
        try:
            out[t] = fundamentals.analyze(t, prices.get(t))
        except Exception as e:
            out[t] = {"ok": False, "error": "%s: %s" % (type(e).__name__, e), "ticker": t}
    return out


def main():
    uni = load_universe()
    data = signals.scan(uni)                       # {"signals": [...], "leaps": [...], "params": {...}}
    data["market"] = fetch_market()
    data["pulse"] = fetch_pulse()
    data["sectors"] = fetch_sectors()
    _names = [o.get("ticker") for o in data.get("overview", []) if o.get("ticker")]
    _prices = {o.get("ticker"): o.get("price") for o in data.get("overview", [])}
    data["fundamentals"] = fetch_fundamentals_all(_names, _prices)
    data["generated_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    data["universe"] = {"wheel": len(uni.get("wheel", [])),
                        "growth": len(uni.get("growth", [])),
                        "source": uni.get("_source"),
                        "wheel_names": [str(t).upper() for t in uni.get("wheel", [])],
                        "growth_names": [str(t).upper() for t in uni.get("growth", [])]}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "signals.json").write_text(json.dumps(data, indent=2, default=str))
    print("wrote {} — {} signals, {} leaps (universe: {})".format(
        OUT / "signals.json", len(data.get("signals", [])), len(data.get("leaps", [])), data["universe"]))


if __name__ == "__main__":
    main()
