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

import math
import statistics

import requests

import macro
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
            res = fundamentals.analyze(t, prices.get(t))
            try:
                res["news"] = fundamentals.company_news(t)
            except Exception:
                res["news"] = []
            out[t] = res
        except Exception as e:
            out[t] = {"ok": False, "error": "%s: %s" % (type(e).__name__, e), "ticker": t}
    return out


def _yf_hist(sym, period="1y"):
    try:
        import yfinance as yf
        return [float(x) for x in yf.Ticker(sym).history(period=period)["Close"].dropna()]
    except Exception:
        return []


def _rv20(closes):
    """20-day annualised realised vol (%), for the S&P VRP calc."""
    c = closes[-21:]
    if len(c) < 21:
        return None
    rets = [math.log(c[i] / c[i - 1]) for i in range(1, len(c)) if c[i - 1] > 0]
    return statistics.pstdev(rets) * math.sqrt(252) * 100 if len(rets) >= 2 else None


def fetch_market_stats():
    """Everything the app's 📊 Market Stats tab shows — via the pure `macro` module + yfinance."""
    ms = {}
    try:
        yc = macro.yield_curve()
        ms["yield_curve"] = yc
        ms["fed"] = macro.fed_funds_rate()
        ms["spread_2s10s"] = macro.curve_spread_2s10s(yc)
    except Exception:
        pass
    try:
        import yfinance as yf
        # Yahoo drops trailingPE/forwardPE for an S&P ETF intermittently (esp. forwardPE),
        # so walk the big S&P trackers until both are filled — one usually carries the field
        # when another doesn't. Same source, just more reliable.
        for _sym in ("SPY", "IVV", "VOO"):
            if ms.get("pe_trailing") is not None and ms.get("pe_forward") is not None:
                break
            try:
                info = yf.Ticker(_sym).info or {}
            except Exception:
                continue
            if ms.get("pe_trailing") is None and info.get("trailingPE") is not None:
                ms["pe_trailing"] = info.get("trailingPE")
            if ms.get("pe_forward") is None and info.get("forwardPE") is not None:
                ms["pe_forward"] = info.get("forwardPE")
    except Exception:
        pass
    vh = _yf_hist("^VIX", "1y")
    if vh:
        now, prev = vh[-1], (vh[-2] if len(vh) > 1 else vh[-1])
        ms.update({"vix": round(now, 1), "vix_chg": round(now - prev, 2),
                   "vix_hi": round(max(vh), 1), "vix_lo": round(min(vh), 1),
                   "vix_avg": round(sum(vh) / len(vh), 1)})
    sh = _yf_hist("SPY", "3mo")
    rv = _rv20(sh) if sh else None
    if rv is not None:
        ms["spy_realized"] = round(rv, 1)
        if ms.get("vix") is not None:
            ms["vrp"] = round(ms["vix"] - rv, 1)
    for key, sym in (("ovx", "^OVX"), ("gvz", "^GVZ")):
        h = _yf_hist(sym, "5d")
        if h:
            now, prev = h[-1], (h[-2] if len(h) > 1 else h[-1])
            ms[key] = round(now, 1); ms[key + "_chg"] = round(now - prev, 2)
    try:
        ms["calendar"] = macro.econ_calendar()
    except Exception:
        ms["calendar"] = []
    return ms


# ── Options-chain embed ──────────────────────────────────────────────────────────
# The static site has no live backend, so the browsable Options Chain must be baked into
# the JSON. Bounded (expiries + strike band) to keep the file light and the scan fast.
CHAIN_TARGET_DTES  = [7, 21, 30, 45, 60, 90]   # embed the expiry nearest each (deduped)
CHAIN_DTE_MAX      = 120    # only consider expiries within ~4 months
CHAIN_STRIKE_BAND  = 0.25   # strikes within ±25% of spot …
CHAIN_STRIKES_SIDE = 28     # … and at most this many each side of ATM


def _fnum(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def fetch_chains(names, prices):
    """Per name: a bounded set of expiries, each with the calls/puts ladder (strike, delta,
    bid, ask, IV%, OI). Compact keys keep the JSON small. Never raises for one bad name."""
    import tradier
    import datetime as _dt
    import bisect
    if not tradier.is_configured():
        return {}
    today = _dt.date.today()
    out = {}
    for t in names:
        try:
            spot = _fnum(prices.get(t))
            exps = list(dict.fromkeys(tradier.get_expirations(t) or []))
            avail = []
            for e in exps:
                try:
                    d = (_dt.date.fromisoformat(e) - today).days
                except Exception:
                    continue
                if 0 <= d <= CHAIN_DTE_MAX:
                    avail.append((e, d))
            if not avail:
                continue
            # Spread the embedded expiries across target DTEs so the ~30-day working window
            # (what CSP/CC actually trade) is always present, not just the front weeklies.
            chosen = {}
            for tgt in CHAIN_TARGET_DTES:
                e, d = min(avail, key=lambda x: abs(x[1] - tgt))
                chosen[e] = d
            picked = sorted(chosen.items(), key=lambda x: x[1])
            lo = spot * (1 - CHAIN_STRIKE_BAND) if spot else None
            hi = spot * (1 + CHAIN_STRIKE_BAND) if spot else None
            exp_out = []
            for e, d in picked:
                opts = tradier.get_option_chain(t, e, greeks=True) or []
                calls, puts = [], []
                for o in opts:
                    k = _fnum(o.get("strike"))
                    if k is None or (lo is not None and (k < lo or k > hi)):
                        continue
                    g = o.get("greeks") or {}
                    iv = g.get("mid_iv") or g.get("smv_vol")
                    dv = g.get("delta")
                    row = {"k": k,
                           "d": round(_fnum(dv), 3) if dv is not None else None,
                           "b": _fnum(o.get("bid")), "a": _fnum(o.get("ask")),
                           "iv": round(_fnum(iv) * 100, 1) if iv else None,
                           "oi": int(o.get("open_interest") or 0)}
                    (calls if o.get("option_type") == "call" else puts).append(row)

                def _trim(rows):
                    rows.sort(key=lambda r: r["k"])
                    if spot and len(rows) > 2 * CHAIN_STRIKES_SIDE:
                        i = bisect.bisect_left([r["k"] for r in rows], spot)
                        rows = rows[max(0, i - CHAIN_STRIKES_SIDE): i + CHAIN_STRIKES_SIDE]
                    return rows

                calls, puts = _trim(calls), _trim(puts)
                if calls or puts:
                    exp_out.append({"exp": e, "dte": d, "calls": calls, "puts": puts})
            if exp_out:
                out[t] = {"spot": round(spot, 2) if spot else None, "expiries": exp_out}
        except Exception:
            continue
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
    data["chains"] = fetch_chains(_names, _prices)
    data["market_stats"] = fetch_market_stats()
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
