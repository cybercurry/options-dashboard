import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta, timezone
import math
import re
import requests
from scipy.stats import norm
from urllib.parse import quote
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import html
import tradier   # Tradier API access door (real quotes / chains / IV / Greeks)
import fundamentals   # SEC EDGAR fundamentals door (real 10-K/10-Q XBRL, red flags)
import signals   # headless wheel-signal scan engine (CSP/CC premium opportunities)
import sheets   # published Google-Sheet CSV reader (live wheel/growth universe)
import macro   # keyless macro data — Treasury yield curve, Fed rate, econ calendar (FRED/FF)
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Options Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

try:
    from streamlit_autorefresh import st_autorefresh
    HAS_AUTOREFRESH = True
except ImportError:
    HAS_AUTOREFRESH = False

# Baked-in default so even the bare link (no ?tickers=…) loads Jay's list (25 July).
DEFAULT_WATCHLIST = ["NVDA", "TSLA", "GLD", "BE", "VST", "AMZN", "SPCX", "NVTS", "VRT", "SLV", "PLTR", "AAPL", "GOOG", "IREN", "NBIS", "NOW", "WMT"]

VIX_ZONES = [
    (0,  15, "#16a34a", "LOW — Ideal LEAP buying zone"),
    (15, 20, "#ca8a04", "NORMAL — Balanced regime"),
    (20, 30, "#ea580c", "ELEVATED — CC writing premium rich"),
    (30, 99, "#dc2626", "HIGH — Aggressive premium selling regime"),
]

# 23 July — the equity tiles were ETF proxies (SPY/QQQ/DIA/IWM) but LABELLED as the indices,
# so "S&P 500" read $747 (SPY) instead of ~7499 (the actual index) — a persistent ~10x mismatch
# that looked like bad/stale data. Point them at the real Yahoo index symbols so the tile matches
# what you see quoted everywhere (^GSPC = S&P 500, ^NDX = Nasdaq 100, ^DJI = Dow, ^RUT = Russell
# 2000). Prefix "" because index levels are points, not dollars. Commodities/crypto stay in $.
PULSE_TICKERS = [
    ("^GSPC",    "S&P 500",    "",   False),
    ("^NDX",     "Nasdaq 100", "",   False),
    ("^DJI",     "Dow Jones",  "",   False),
    ("^RUT",     "R2000",      "",   False),
    ("DX-Y.NYB", "DXY",        "",   False),
    ("CL=F",     "Crude Oil",  "$",  False),
    ("GC=F",     "Gold",       "$",  False),
    ("BTC-USD",  "Bitcoin",    "$",  False),
    ("^TNX",     "10Y Yield",  "",   True),
    ("^IRX",     "3M Yield",   "",   True),
]

VIX_TERM_TICKERS = [
    ("^VIX9D", "9-Day"),
    ("^VIX",   "30-Day"),
    ("^VIX3M", "3-Month"),
    ("^VIX6M", "6-Month"),
]

# ── Sector heatmap — SPDR ETFs + BTC as 12th sector ──────────────────────────
# (ticker, full name, short label for tile)
SECTOR_TICKERS = [
    ("XLK",     "Technology",       "XLK"),
    ("XLF",     "Financials",       "XLF"),
    ("XLV",     "Health Care",      "XLV"),
    ("XLE",     "Energy",           "XLE"),
    ("XLI",     "Industrials",      "XLI"),
    ("XLC",     "Comm. Services",   "XLC"),
    ("XLY",     "Consumer Disc.",   "XLY"),
    ("XLP",     "Consumer Staples", "XLP"),
    ("XLU",     "Utilities",        "XLU"),
    ("XLRE",    "Real Estate",      "XLRE"),
    ("XLB",     "Materials",        "XLB"),
    ("BTC-USD", "Digital Assets",   "BTC"),
]

# ── Screener constants ─────────────────────────────────────────────────────────
NIS_FLOOR = 0.00157
NIS_CEIL  = 0.01253

# CSP delta=30, DTE=30 (updated from 18/37)
STRATEGY_PARAMS = {
    "CSP": {
        "delta_opt": 30, "delta_lo": 20, "delta_hi": 45,
        "dte_opt":   30, "dte_lo":   21, "dte_hi":   45,
        "w_iv": 0.50,    "w_dte": 0.30,  "w_delta": 0.20,
        "iv_dir": 1,     "option_type": "put",
    },
    "CC": {
        "delta_opt": 30, "delta_lo": 20, "delta_hi": 50,
        "dte_opt":   30, "dte_lo":   21, "dte_hi":   45,
        "w_iv": 0.50,    "w_dte": 0.30,  "w_delta": 0.20,
        "iv_dir": 1,     "option_type": "call",
    },
    "LEAP": {
        "delta_opt": 80, "delta_lo": 60, "delta_hi": 95,
        "dte_opt":  542, "dte_lo":  180, "dte_hi":  900,
        "w_iv": 0.30,    "w_dte": 0.40,  "w_delta": 0.30,
        "iv_dir": -1,    "option_type": "call",
    },
}

RISK_FREE_RATE = 0.045

# ══════════════════════════════════════════════════════════════════════════════
# WATCHLIST — kept in watchlist.json, a simple committed list that is the app's home for the
# watchlist. No tickers in the URL. To change the list permanently, edit that one file (it's
# just symbols). In-app add/remove are session-only tweaks — Streamlit's host has no writable
# storage, so runtime edits can't be saved back to the file; edit the file to make it stick.
# ══════════════════════════════════════════════════════════════════════════════
_WATCHLIST_FILE = Path(__file__).with_name("watchlist.json")

def load_watchlist_file():
    try:
        ticks = [str(t).strip().upper() for t in json.loads(_WATCHLIST_FILE.read_text()) if str(t).strip()]
        if ticks:
            seen = set()
            return [t for t in ticks if not (t in seen or seen.add(t))]
    except Exception:
        pass
    return DEFAULT_WATCHLIST.copy()

if "watchlist" not in st.session_state:
    st.session_state.watchlist = load_watchlist_file()
    try:
        st.query_params.clear()   # keep the URL clean — the saved list is the source
    except Exception:
        pass

# ══════════════════════════════════════════════════════════════════════════════
# DATA FETCHERS
# ══════════════════════════════════════════════════════════════════════════════
def _quote_single(ticker):
    # 21 July — Market Pulse tiles were showing stale / day-behind numbers. Root cause:
    # we read the *daily* 5d bars and compared last close vs prior close. Outside regular
    # US hours (pre/post-market, weekends, holidays) that surfaces yesterday's price and
    # yesterday's % move, and futures/yields/crypto (GC=F, CL=F, ^TNX, BTC-USD) roll on
    # different clocks than SPY/QQQ, so the tiles disagreed on their "as of" time.
    #
    # Fix: prefer fast_info, which gives the latest intraday trade (updates through the
    # session and in extended hours) plus the correct prior-session close as the day-change
    # baseline. Fall back to the original daily-bar method if fast_info is unavailable, so
    # a source hiccup never regresses below today's behaviour.
    #
    # Uncached on purpose — callers reach it via the cached fetch_quote (single) or
    # fetch_quotes (batched, concurrent) wrappers below.
    try:
        fi    = yf.Ticker(ticker).fast_info
        price = getattr(fi, "last_price", None)
        prev  = getattr(fi, "previous_close", None)
        if price is not None and prev not in (None, 0):
            price = float(price); prev = float(prev)
            if price > 0 and prev > 0:
                return {"price": price, "pct": (price / prev - 1) * 100}
    except Exception:
        pass
    try:
        df = yf.download(ticker, period="5d", auto_adjust=True, progress=False)
        if df is None or df.empty:
            return None
        cl = df["Close"].squeeze().dropna()
        if len(cl) < 2:
            return None
        curr = float(cl.iloc[-1]); prev = float(cl.iloc[-2])
        return {"price": curr, "pct": (curr/prev - 1)*100}
    except Exception:
        return None

@st.cache_data(ttl=60, show_spinner=False)
def fetch_quote(ticker):
    return _quote_single(ticker)

@st.cache_data(ttl=60, show_spinner=False)
def fetch_quotes(tickers):
    # 21 July — the Overview fires a quote per pulse + sector tile. Run sequentially that's
    # ~20 blocking round-trips every render, and painfully slow now that auto-refresh is on.
    # Fetch them concurrently instead (I/O-bound, so threads help a lot) and cache the whole
    # batch as one entry. `tickers` is a tuple so it stays hashable for st.cache_data.
    tickers = tuple(tickers)
    if not tickers:
        return {}
    workers = min(8, len(tickers))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = ex.map(_quote_single, tickers)
    return {t: q for t, q in zip(tickers, results)}

@st.cache_data(ttl=60, show_spinner=False)
def fetch_sector_quotes(tickers):
    """Sector-tile quotes, Tradier-first. The SPDR sector ETFs (XLK…XLB) are all Tradier
    symbols, so pull them in ONE real quote call — reliable, unlike yfinance, which was
    surfacing stale / day-behind sector tiles (same root cause we already fixed for the
    Market Pulse and the whole data layer). Crypto (BTC-USD, which Tradier can't quote) and
    any Tradier miss fall back to yfinance. Returns {ticker: {price, pct}}."""
    tickers = tuple(tickers)
    if not tickers:
        return {}
    out = {}
    # Tradier can't quote crypto ("-USD") or Yahoo index ("^") symbols — route only real
    # equities/ETFs to it; everything else falls through to the yfinance path below.
    trad = [t for t in tickers if tradier.is_configured()
            and "-" not in t and not t.startswith("^")]
    if trad:
        try:
            for q in tradier.get_quotes(trad):
                sym  = q.get("symbol")
                last = q.get("last")
                pct  = q.get("change_percentage")
                if pct is None and last and q.get("prevclose"):
                    pct = (float(last) / float(q["prevclose"]) - 1) * 100
                if sym and last is not None and pct is not None:
                    out[sym] = {"price": float(last), "pct": float(pct)}
        except Exception:
            pass  # fall through to yfinance for anything Tradier didn't return
    # yfinance fallback for whatever Tradier didn't cover (crypto, or a miss).
    missing = [t for t in tickers if t not in out]
    for m in missing:
        q = _quote_single(m)
        if q:
            out[m] = q
    return out

@st.cache_data(ttl=300, show_spinner=False)
def fetch_cnn_fg():
    """
    Fetch CNN Fear & Greed stock market index.
    Tries the CNN dataviz endpoint with browser-like headers.
    Falls back to None if unavailable.
    """
    try:
        r = requests.get(
            "https://production.dataviz.cnn.io/index/fearandgreed/graphdata",
            headers={
                "User-Agent":  "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                               "AppleWebKit/537.36 (KHTML, like Gecko) "
                               "Chrome/124.0.0.0 Safari/537.36",
                "Referer":     "https://www.cnn.com/markets/fear-and-greed",
                "Origin":      "https://www.cnn.com",
                "Accept":      "application/json, text/plain, */*",
            },
            timeout=10,
        )
        d   = r.json()
        fg  = d.get("fear_and_greed", d)
        score  = round(float(fg.get("score",  fg.get("value", 50))), 1)
        rating = str(fg.get("rating", fg.get("value_classification", "neutral"))
                     ).replace("_", " ").title()
        return score, rating
    except Exception:
        return None, None

@st.cache_data(ttl=300, show_spinner=False)
def fetch_crypto_fg():
    """
    Fetch Crypto Fear & Greed from Alternative.me (free, no API key).
    This is the same index shown on CoinMarketCap and CoinStats.
    """
    try:
        r   = requests.get("https://api.alternative.me/fng/?limit=1", timeout=8)
        d   = r.json()
        entry  = d["data"][0]
        score  = int(entry["value"])
        rating = entry["value_classification"].replace("_", " ").title()
        return score, rating
    except Exception:
        return None, None

def _parse_epoch_or_iso(v):
    # CNN stamps its score with either epoch-ms/epoch-s or an ISO8601 string depending on
    # the field — accept all three and hand back a naive-UTC datetime (None if unparseable).
    if v is None:
        return None
    try:
        if isinstance(v, (int, float)) or (isinstance(v, str) and v.strip().isdigit()):
            n = float(v)
            if n > 1e12:      # milliseconds → seconds
                n /= 1000.0
            return datetime.utcfromtimestamp(n)
        dt = datetime.fromisoformat(str(v).replace("Z", "+00:00"))
        if dt.tzinfo:
            dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
        return dt
    except Exception:
        return None

@st.cache_data(ttl=60, show_spinner=False)
def fetch_data_health():
    # 23 July — "everything on the Overview has been frozen for weeks" spans THREE independent
    # pipelines (Yahoo, CNN, Alternative.me), which don't fail identically by chance — so we
    # need to see, per source, whether it connected and HOW OLD its data is. This probe runs in
    # the deployment (where the data is actually reachable) and reports exactly that, so we can
    # tell "a source is down / serving stale data" apart from "the app/tab just wasn't
    # refreshing". Each check is isolated; one failure never blanks the rest of the report.
    now  = datetime.utcnow()
    rows = []

    def _age(dt):
        secs = (now - dt).total_seconds()
        if secs < 0:      return "future?"
        if secs < 3600:   return f"{secs/60:.0f} min"
        if secs < 86400:  return f"{secs/3600:.1f} h"
        return f"{secs/86400:.1f} d"

    # Yahoo backbone — a daily pull exposes the last bar's DATE, i.e. how old Yahoo's data is.
    try:
        df = yf.download("SPY", period="5d", auto_adjust=True, progress=False)
        if df is not None and not df.empty:
            last     = pd.to_datetime(df.index[-1]).to_pydatetime().replace(tzinfo=None)
            age_days = (now.date() - last.date()).days
            price    = float(df["Close"].squeeze().iloc[-1])
            rows.append({"Source": "Yahoo Finance (indices · VIX · sectors · watchlist)",
                         "Status": "🟢 live" if age_days <= 4 else "🟠 STALE",
                         "Data as of": last.strftime("%Y-%m-%d"), "Age": _age(last),
                         "Sample value": f"SPY close ${price:,.2f}"})
        else:
            rows.append({"Source": "Yahoo Finance (indices · VIX · sectors · watchlist)",
                         "Status": "🔴 NO DATA", "Data as of": "—", "Age": "—",
                         "Sample value": "empty response"})
    except Exception as e:
        rows.append({"Source": "Yahoo Finance (indices · VIX · sectors · watchlist)",
                     "Status": "🔴 ERROR", "Data as of": "—", "Age": "—",
                     "Sample value": f"{type(e).__name__}: {e}"[:70]})

    # Yahoo fast_info — the live intraday path the pulse tiles use.
    try:
        lp = getattr(yf.Ticker("SPY").fast_info, "last_price", None)
        rows.append({"Source": "Yahoo fast_info (live pulse quotes)",
                     "Status": "🟢 live" if lp else "🔴 NO DATA",
                     "Data as of": "realtime" if lp else "—", "Age": "~now" if lp else "—",
                     "Sample value": f"SPY ${float(lp):,.2f}" if lp else "no last_price"})
    except Exception as e:
        rows.append({"Source": "Yahoo fast_info (live pulse quotes)",
                     "Status": "🔴 ERROR", "Data as of": "—", "Age": "—",
                     "Sample value": f"{type(e).__name__}"})

    # CNN Fear & Greed — Stocks gauge.
    try:
        r = requests.get(
            "https://production.dataviz.cnn.io/index/fearandgreed/graphdata",
            headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                                   "Chrome/124.0.0.0 Safari/537.36",
                     "Referer": "https://www.cnn.com/markets/fear-and-greed",
                     "Origin":  "https://www.cnn.com",
                     "Accept":  "application/json, text/plain, */*"},
            timeout=10)
        fg    = r.json().get("fear_and_greed", {})
        score = fg.get("score", fg.get("value"))
        dt    = _parse_epoch_or_iso(fg.get("timestamp"))
        if dt is not None:
            age_h = (now - dt).total_seconds() / 3600
            rows.append({"Source": "CNN Fear & Greed (Stocks gauge)",
                         "Status": "🟢 live" if age_h <= 48 else "🟠 STALE",
                         "Data as of": dt.strftime("%Y-%m-%d %H:%M"), "Age": _age(dt),
                         "Sample value": f"score {float(score):.0f}" if score is not None else "—"})
        else:
            rows.append({"Source": "CNN Fear & Greed (Stocks gauge)",
                         "Status": "🟢 live" if score is not None else "🔴 NO DATA",
                         "Data as of": "(no timestamp in feed)", "Age": "—",
                         "Sample value": f"score {float(score):.0f}" if score is not None else "parse failed"})
    except Exception as e:
        rows.append({"Source": "CNN Fear & Greed (Stocks gauge)",
                     "Status": "🔴 ERROR", "Data as of": "—", "Age": "—",
                     "Sample value": f"{type(e).__name__}"})

    # Alternative.me — Crypto gauge.
    try:
        entry = requests.get("https://api.alternative.me/fng/?limit=1", timeout=8).json()["data"][0]
        dt    = datetime.utcfromtimestamp(int(entry["timestamp"]))
        age_h = (now - dt).total_seconds() / 3600
        rows.append({"Source": "Alternative.me (Crypto gauge)",
                     "Status": "🟢 live" if age_h <= 48 else "🟠 STALE",
                     "Data as of": dt.strftime("%Y-%m-%d %H:%M"), "Age": _age(dt),
                     "Sample value": f"score {entry['value']}"})
    except Exception as e:
        rows.append({"Source": "Alternative.me (Crypto gauge)",
                     "Status": "🔴 ERROR", "Data as of": "—", "Age": "—",
                     "Sample value": f"{type(e).__name__}"})

    return rows, now

@st.cache_data(ttl=60, show_spinner=False)
def fetch_vix_term():
    quotes = fetch_quotes(tuple(ticker for ticker, _ in VIX_TERM_TICKERS))
    out = {}
    for ticker, label in VIX_TERM_TICKERS:
        q = quotes.get(ticker)
        if q: out[label] = q["price"]
    return out

@st.cache_data(ttl=60, show_spinner=False)
def fetch_skew():
    q = fetch_quote("^SKEW")
    return q["price"] if q else None

_PERIOD_DAYS = {"6mo": 195, "1y": 370, "2y": 740}

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_prices(ticker, period="1y"):
    # 25 July — prefer Tradier daily history (reliable on cloud) over yfinance, which flakes on
    # shared hosting and was the real cause of watchlist tickers intermittently not loading.
    # Falls back to yfinance so nothing regresses. Index symbols (^VIX etc.) stay on yfinance —
    # Tradier uses different index symbology and yfinance handles those fine.
    if tradier.is_configured() and not ticker.startswith("^"):
        try:
            end   = datetime.utcnow().date()
            start = end - timedelta(days=_PERIOD_DAYS.get(period, 370))
            days  = tradier.get_history(ticker, interval="daily",
                                        start=start.isoformat(), end=end.isoformat())
            if days and len(days) >= 30:
                df = pd.DataFrame(days)
                df["date"] = pd.to_datetime(df["date"])
                df = (df.set_index("date").sort_index()
                        .rename(columns={"open":"Open","high":"High","low":"Low",
                                         "close":"Close","volume":"Volume"}))
                df = df[["Open","High","Low","Close","Volume"]].apply(pd.to_numeric, errors="coerce")
                df = df.dropna(subset=["Close"])
                if len(df) >= 30:
                    return df
        except Exception:
            pass  # fall through to yfinance
    try:
        df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
        if df.empty or len(df) < 30: return None
        return df
    except Exception:
        return None

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_vix(period="1y"):
    return fetch_prices("^VIX", period)

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_ovx(period="1y"):
    return fetch_prices("^OVX", period)

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_gvz(period="1y"):
    return fetch_prices("^GVZ", period)

@st.cache_data(ttl=21600, show_spinner=False)   # PE barely moves intraday — refresh every 6h
def fetch_sp500_pe():
    """S&P 500 trailing & forward P/E via the SPY ETF (its PE tracks the index). Valuation
    context — rich market = more downside risk to a CSP seller. (yfinance; None if it hiccups.)"""
    try:
        info = yf.Ticker("SPY").info
        return info.get("trailingPE"), info.get("forwardPE")
    except Exception:
        return None, None

@st.cache_data(ttl=1800, show_spinner=False)
def _fetch_all_expiries_raw(ticker):
    # Raises on failure/empty so st.cache_data does NOT cache a bad result (25 June fix) —
    # previously a transient Yahoo block/rate-limit got cached for 30 min via a swallowed
    # exception returning [], which silently "stuck" the whole watchlist until TTL expired.
    tk = yf.Ticker(ticker)
    opts = list(tk.options)
    if not opts:
        raise RuntimeError("yfinance returned no option expiries (empty .options)")
    return opts

def fetch_all_expiries(ticker):
    """Uncached wrapper — converts the raise back to the old ([], err) shape so callers
    don't crash, while keeping the underlying cache un-poisoned by failures."""
    # 25 July — prefer Tradier's expiration list (reliable) over yfinance's .options, with a
    # yfinance fallback so it can't regress.
    if tradier.is_configured():
        try:
            exps = tradier.get_expirations(ticker)
            if exps:
                # Tradier can return the SAME date more than once (one row per option root
                # when includeAllRoots is on) — dedupe so the Options Chain shows each expiry
                # tile once, not twice. Preserves chronological order.
                return list(dict.fromkeys(exps)), None
        except Exception:
            pass
    try:
        return _fetch_all_expiries_raw(ticker), None
    except Exception as e:
        return [], f"{type(e).__name__}: {e}"

@st.cache_data(ttl=1800, show_spinner=False)
def _fetch_chain_cached_raw(ticker, expiry):
    # Same fix as above (25 June) — raise instead of swallowing, so a transient failure
    # isn't cached for 30 minutes.
    tk    = yf.Ticker(ticker)
    chain = tk.option_chain(expiry)
    dte   = (datetime.strptime(expiry, "%Y-%m-%d") - datetime.utcnow()).days
    calls_empty = chain.calls is None or chain.calls.empty
    puts_empty  = chain.puts  is None or chain.puts.empty
    if calls_empty and puts_empty:
        raise RuntimeError(f"empty option chain for {ticker} {expiry} (calls and puts both empty)")
    return chain.calls, chain.puts, dte

def fetch_chain_cached(ticker, expiry):
    """Uncached wrapper — converts the raise back to the old (None,None,None) shape so
    callers don't crash, while keeping the underlying cache un-poisoned by failures."""
    try:
        calls, puts, dte = _fetch_chain_cached_raw(ticker, expiry)
        return calls, puts, dte, None
    except Exception as e:
        return None, None, None, f"{type(e).__name__}: {e}"

# ── Tradier-backed chain — REAL vendor IV & Greeks (ORATS), same shape as above ─────
# 25 July — Jay funded Tradier; the Options Chain tab now prefers real chains over
# yfinance's calculated/absent Greeks. Returns DataFrames with the SAME column names the
# display/chart code already expects (strike, impliedVolatility, lastPrice, bid, ask,
# volume, openInterest, delta, …) so nothing downstream changes. Raises on empty so a
# transient failure isn't cached (same discipline as the yfinance path).
@st.cache_data(ttl=60, show_spinner=False)
def _fetch_chain_tradier_raw(ticker, expiry):
    opts = tradier.get_option_chain(ticker, expiry, greeks=True)
    if not opts:
        raise RuntimeError(f"Tradier returned no contracts for {ticker} {expiry}")
    rows = []
    for o in opts:
        g  = o.get("greeks") or {}
        iv = g.get("mid_iv")
        if iv in (None, 0):
            iv = g.get("smv_vol")           # ORATS surface vol fallback
        rows.append({
            "option_type":      o.get("option_type"),
            "strike":           o.get("strike"),
            "lastPrice":        o.get("last"),
            "bid":              o.get("bid"),
            "ask":              o.get("ask"),
            "volume":           o.get("volume"),
            "openInterest":     o.get("open_interest"),
            "impliedVolatility": iv,          # already a fraction, like yfinance
            "delta":            g.get("delta"),
            "gamma":            g.get("gamma"),
            "theta":            g.get("theta"),
            "vega":             g.get("vega"),
        })
    df    = pd.DataFrame(rows)
    calls = df[df["option_type"] == "call"].drop(columns=["option_type"]).reset_index(drop=True)
    puts  = df[df["option_type"] == "put"].drop(columns=["option_type"]).reset_index(drop=True)
    dte   = (datetime.strptime(expiry, "%Y-%m-%d") - datetime.utcnow()).days
    return calls, puts, dte

def fetch_chain_tradier(ticker, expiry):
    """Uncached wrapper — same (calls, puts, dte, err) contract as fetch_chain_cached."""
    try:
        calls, puts, dte = _fetch_chain_tradier_raw(ticker, expiry)
        return calls, puts, dte, None
    except Exception as e:
        return None, None, None, f"{type(e).__name__}: {e}"

@st.cache_data(ttl=30, show_spinner=False)
def fetch_underlying_last(ticker):
    """Real-time last trade for the underlying from Tradier — the actual market price of
    the asset (None if unavailable). Used to show a live price next to the chain's DTE."""
    try:
        qs = tradier.get_quotes(ticker)
        if qs:
            last = qs[0].get("last")
            if last is not None:
                return float(last)
    except Exception:
        pass
    return None

def fetch_fundamentals(ticker, price):
    """Fundamentals read (SEC EDGAR → Yahoo fallback). Caches ONLY successful reads, in
    session — so a failed lookup (bad ticker, transient SEC blip) is never sticky: the next
    attempt re-fetches instead of being pinned to the miss for hours. Filings change quarterly,
    so a session-lived cache of successes is plenty; a full page reload refetches. Price is part
    of the key so a big move refreshes P/E."""
    key = f"{ticker.upper()}|{round(price, 2) if price else 0}"
    cache = st.session_state.setdefault("_fx_cache", {})
    if key in cache:
        return cache[key]
    res = fundamentals.analyze(ticker, price)
    if res.get("ok"):
        cache[key] = res          # cache successes only — misses always retry
    return res

@st.cache_data(ttl=1800, show_spinner=False)   # 30 min — headlines refresh a few times a day
def fetch_company_news(ticker):
    """Recent real headlines (Yahoo Finance). Cached separately from the fundamentals so news
    stays fresher than the 6h filings data."""
    return fundamentals.company_news(ticker)

@st.cache_data(ttl=1800, show_spinner=False)
def load_signal_universe():
    """Signal universe (wheel + growth). LIVE from the published Google-Sheet CSV each scan;
    falls back to the committed wheel_universe.json (then the watchlist) if the sheet is
    unreachable, so a Google hiccup never blanks the scan."""
    cfg = {}
    try:
        cfg = json.loads((Path(__file__).parent / "wheel_universe.json").read_text())
    except Exception:
        pass
    live = sheets.fetch_universe(cfg.get("source_url")) if cfg.get("source_url") else {}
    if live.get("wheel"):
        return {**live, "_source": "sheet"}
    if cfg.get("wheel"):
        return {**cfg, "_source": "file"}
    return {"wheel": st.session_state.get("watchlist", []), "growth": [], "_source": "watchlist"}

@st.cache_data(ttl=1800, show_spinner=False)
def run_signal_scan(nonce):
    """Live wheel-signal scan (same engine the cron uses). `nonce` lets the Scan-now button force
    a refresh past the 30-min cache."""
    return signals.scan(load_signal_universe())

@st.cache_data(ttl=21600, show_spinner=False)   # 6h — FRED daily series
def fetch_yield_curve():
    return macro.yield_curve()

@st.cache_data(ttl=21600, show_spinner=False)
def fetch_fed_rate():
    return macro.fed_funds_rate()

@st.cache_data(ttl=3600, show_spinner=False)     # 1h — this-week econ calendar
def fetch_econ_calendar():
    return macro.econ_calendar()

def fetch_chain(ticker, expiry):
    """Unified chain fetch for the screener / analyse / deep-dive: Tradier real IV & Greeks
    when a token is configured (find_target_strike then reads the real delta/theta directly
    instead of Black-Scholes), else yfinance. Same (calls, puts, dte, err) shape either way,
    and it falls back to yfinance if a Tradier fetch fails so it can never regress."""
    if tradier.is_configured():
        calls, puts, dte, err = fetch_chain_tradier(ticker, expiry)
        if calls is not None:
            return calls, puts, dte, err
    return fetch_chain_cached(ticker, expiry)

# ══════════════════════════════════════════════════════════════════════════════
# BLACK-SCHOLES GREEKS
# ══════════════════════════════════════════════════════════════════════════════
def _bs_greeks(S, K, T, sigma, r=RISK_FREE_RATE, option_type="call"):
    try:
        if any(x is None for x in (S, K, T, sigma)): return None, None, None
        S, K, T, sigma = float(S), float(K), float(T), float(sigma)
        if any(pd.isna(x) for x in (S,K,T,sigma)): return None, None, None
        if T<=0 or sigma<=0 or S<=0 or K<=0:         return None, None, None
        sigma  = max(0.05, min(sigma, 3.0))
        T_yr   = T/365.0; sqrtT = math.sqrt(T_yr)
        d1     = (math.log(S/K) + (r + 0.5*sigma**2)*T_yr)/(sigma*sqrtT)
        d2     = d1 - sigma*sqrtT
        delta  = norm.cdf(d1) if option_type=="call" else norm.cdf(d1)-1.0
        pdf_d1 = norm.pdf(d1)
        tc     = -(S*pdf_d1*sigma)/(2*sqrtT) - r*K*math.exp(-r*T_yr)*norm.cdf(d2)
        theta  = (tc if option_type=="call" else tc + r*K*math.exp(-r*T_yr))/365.0
        # d2 is returned so callers can derive POP = N(d2) (put) / N(-d2) (call) — §9.2
        return round(abs(delta)*100.0, 2), round(abs(theta), 4), round(d2, 4)
    except Exception:
        return None, None, None

# ── Technical indicators ───────────────────────────────────────────────────────
def calc_hv(close, window=20):
    return np.log(close/close.shift(1)).rolling(window).std()*np.sqrt(252)*100

def calc_iv_rank(hv_series, lookback=252):
    s = hv_series.dropna().tail(lookback)
    if len(s) < 30: return None
    lo, hi, cur = s.min(), s.max(), s.iloc[-1]
    return round((cur-lo)/(hi-lo)*100, 1) if hi!=lo else 50.0

def calc_iv_percentile(hv_series, lookback=252):
    s = hv_series.dropna().tail(lookback)
    if len(s) < 30: return None
    return round((s<s.iloc[-1]).sum()/len(s)*100, 1)

def calc_rsi(close, window=14):
    delta = close.diff()
    ag = delta.clip(lower=0).ewm(alpha=1/window, min_periods=window).mean()
    al = (-delta).clip(lower=0).ewm(alpha=1/window, min_periods=window).mean()
    return 100 - 100/(1 + ag/al.replace(0, np.nan))

def calc_atr(df, window=14):
    hi=df["High"].squeeze(); lo=df["Low"].squeeze(); cl=df["Close"].squeeze()
    tr = pd.concat([hi-lo,(hi-cl.shift(1)).abs(),(lo-cl.shift(1)).abs()],axis=1).max(axis=1)
    return tr.rolling(window).mean()

def calc_bb_width(close, window=20):
    mid=close.rolling(window).mean(); std=close.rolling(window).std()
    return ((mid+2*std)-(mid-2*std))/mid*100

def calc_bb_bands(close, window=20):
    mid=close.rolling(window).mean(); std=close.rolling(window).std()
    return mid+2*std, mid, mid-2*std

def calc_bb_pctb(close, window=20):
    upper,_,lower=calc_bb_bands(close,window)
    return (close-lower)/(upper-lower+1e-10)

def calc_atm_iv(chain, price):
    # Return None (never NaN) when a strike's IV is missing — a NaN is truthy in Python, so it
    # would slip past `if c_iv` checks and surface as "nan%" in the Overview chart/table.
    def _iv(v):
        try:
            v = float(v)
            return round(v*100, 1) if math.isfinite(v) and v > 0 else None
        except (TypeError, ValueError):
            return None
    try:
        c_atm = chain.calls.iloc[(chain.calls["strike"]-price).abs().argsort()[:1]]
        p_atm = chain.puts.iloc[(chain.puts["strike"]-price).abs().argsort()[:1]]
        return _iv(c_atm["impliedVolatility"].values[0]), _iv(p_atm["impliedVolatility"].values[0])
    except Exception:
        return None, None

def calc_pcr(chain):
    try:
        pv=chain.puts["volume"].sum(); cv=chain.calls["volume"].sum()
        return round(pv/cv,2) if cv>0 else None
    except Exception:
        return None

# ── Screener helpers ───────────────────────────────────────────────────────────
def find_target_strike(chain_df, target_delta_abs, option_type, price, dte, hv_pct=None):
    if chain_df is None or chain_df.empty or not price or price<=0 or not dte or dte<=0:
        return None

    iv_raw_col = chain_df.get("impliedVolatility", pd.Series(dtype=float))
    iv_clean   = pd.to_numeric(iv_raw_col, errors="coerce").dropna() if iv_raw_col is not None else pd.Series(dtype=float)
    valid_ivs  = iv_clean[(iv_clean>0.01)&(iv_clean<5.0)]
    median_iv  = float(valid_ivs.median()) if len(valid_ivs)>=3 else None

    best=None; min_score=float("inf"); inspected=0; scored=0

    for _, row in chain_df.iterrows():
        inspected += 1
        try:
            K      = float(row.get("strike",0) or 0)
            oi_raw = row.get("openInterest",0)
            oi     = 0.0 if pd.isna(oi_raw) else float(oi_raw or 0)
            vol_raw= row.get("volume",0)
            vol    = 0.0 if pd.isna(vol_raw) else float(vol_raw or 0)
            iv_raw = row.get("impliedVolatility",0)
            iv     = 0.0 if pd.isna(iv_raw) else float(iv_raw or 0)
            bid    = float(row.get("bid",0) or 0)
            ask    = float(row.get("ask",0) or 0)
        except (TypeError, ValueError):
            continue
        if K<=0: continue

        d_raw=row.get("delta",None); t_raw=row.get("theta",None); yahoo_ok=False
        if d_raw is not None and t_raw is not None:
            try: yahoo_ok = (not pd.isna(d_raw)) and (not pd.isna(t_raw))
            except Exception: pass

        d2_val=None; sigma=None
        if yahoo_ok:
            # Real vendor Greeks came in on the chain (Tradier/ORATS delta+theta, per-day
            # theta like _bs_greeks) — use them directly instead of Black-Scholes.
            d_abs=abs(float(d_raw))*100.0; theta=abs(float(t_raw)); gs="tradier"
            # no d1/d2 available from the raw chain field — POP falls back to None below
        else:
            if 0.01<iv<5.0:              sigma=iv;          src="strike"
            elif median_iv is not None:  sigma=median_iv;   src="chain_median"
            elif hv_pct and hv_pct>0:   sigma=float(hv_pct); src="hv20"
            else:                        sigma=0.30;         src="default"
            d_abs, theta, d2_val = _bs_greeks(price, K, dte, sigma, option_type=option_type)
            if d_abs is None or theta is None: continue
            gs = f"bs_{src}"

        if pd.isna(d_abs) or pd.isna(theta): continue
        scored += 1

        if   oi>=100: pen=0
        elif oi>=10:  pen=0.5
        elif oi>=1:   pen=2
        else:         pen=4
        score = abs(d_abs-target_delta_abs)+pen

        if score < min_score:
            min_score=score
            mid=(bid+ask)/2.0
            eff_iv=(round(iv*100,1) if (gs=="tradier" and iv>0.01)
                    else round(sigma*100,1) if sigma is not None else None)
            # POP (§9.2) — probability the short option expires OTM (favorable), via N(d2).
            # Put: favorable if S_T>K -> N(d2). Call: favorable if S_T<K -> N(-d2)=1-N(d2).
            if d2_val is not None:
                pop = round(norm.cdf(d2_val)*100,1) if option_type=="put" else round((1-norm.cdf(d2_val))*100,1)
            else:
                pop = None
            # Liquidity score (§9.2) — 60/40 blend of OI/volume, capped at 100.
            # OI>=100 or volume>=50 contracts/day already scores "fully liquid" (0 score
            # penalty above); these caps are a reasonable first pass, revisit if it misranks live names.
            liq_oi  = min(100.0, (oi/100.0)*100.0)
            liq_vol = min(100.0, (vol/50.0)*100.0)
            liquidity_score = round(0.6*liq_oi + 0.4*liq_vol, 1)
            best={"strike":K,"delta":round(d_abs,1),"theta":round(theta,4),
                  "iv":eff_iv,"oi":int(oi),"volume":int(vol),"bid":bid,"ask":ask,"mid":round(mid,2),
                  "spread_pct":round((ask-bid)/mid*100,1) if mid>0 else None,
                  "pop":pop,"liquidity_score":liquidity_score,
                  "greek_source":gs,"_inspected":inspected,"_scored":scored}
    return best

def calc_nis(theta, dte, strike):
    # Denominator fixed 24 June: strike, not spot price (§9.1) — theta is earned against
    # the capital at risk on the contract, which is sized off strike, not the moving spot.
    # NOTE: NIS_FLOOR/NIS_CEIL were originally calibrated with price as the denominator;
    # since strike sits close to (but not exactly at) price for ~30Δ contracts, the raw
    # NIS scale shifts slightly. Flagging in case the 0-100 spread looks off once tested
    # against real chains — floor/ceil may need a re-calibration pass.
    if theta<=0 or dte<=0 or strike<=0: return 0.0
    raw = theta*math.sqrt(dte)/strike
    return min(100.0, max(0.0,(raw-NIS_FLOOR)/(NIS_CEIL-NIS_FLOOR)*100.0))

def _tri_score(value, optimal, lo, hi):
    if value<lo or value>hi: return 0.0
    half = max(abs(optimal-lo),abs(hi-optimal))
    return max(0.0, 100.0*(1.0-abs(value-optimal)/half)) if half>0 else 100.0

def calc_suitability(nis, dte, delta_abs, strategy):
    p=STRATEGY_PARAMS[strategy]
    a=nis if p["iv_dir"]==1 else 100.0-nis
    b=_tri_score(dte,       p["dte_opt"],  p["dte_lo"],  p["dte_hi"])
    c=_tri_score(delta_abs, p["delta_opt"],p["delta_lo"],p["delta_hi"])
    return round(p["w_iv"]*a+p["w_dte"]*b+p["w_delta"]*c, 1)

def score_color(s):
    if s>=80: return "#22c55e"
    if s>=60: return "#eab308"
    if s>=40: return "#f97316"
    return "#ef4444"

def calc_four_gates(r, bb_veto_mode="Hard", soft_penalty=10, leg="csp",
                     leap_intrinsic=None, leap_extrinsic=None):
    """bb_veto_mode (§9.3, 24 June): 'Hard' (default, original behavior — walking the lower
    band 2+ sessions fails G3 and blocks all_pass), 'Soft' (-soft_penalty points off each leg's
    score instead of blocking Status — see get_screener_row), or 'Off' (informational only,
    never gates or penalizes). Recommended over a one-off override rule since a hand-written
    exception is just a second hard-coded rule with the same brittleness.

    leg (22 June, per Jay; direction corrected same day): 'csp', 'cc', or 'leap'. CSP and CC
    each get a 4th gate — G4 Median — with OPPOSITE pass conditions, because gates are no
    longer one shared computation across all three tables (this was previously an open "next
    step, your call" — Jay locked it in by asking for leg-specific median checks). Direction:
    CSP wants to catch a setup right after a reversal off the low, with more upside runway left
    before the next reversal — that means price at/BELOW the median; once price is already
    above the median, most of that runway is used up (this is the BE case — riding the upper
    band, all gates green, no room left), so G4 FAILS when price is above the median for CSP.
    CC is the mirror image: catch a setup near the top with room to fall, so G4 FAILS when
    price is below the median for CC.

    leg='leap' (26 June, per Jay): a different 4th gate — G4 Premium Mix — since a LEAP buyer's
    "runway" question isn't price-vs-median, it's "am I paying mostly for real (intrinsic)
    value or mostly for rented time (extrinsic)?" Pass requires extrinsic to be at most 60% of
    the contract's mid premium (the 50/50 split Jay proposed, plus a 10-point buffer so a
    contract sitting a percent or two over the line doesn't flip needlessly). Needs
    leap_intrinsic/leap_extrinsic passed in — these are computed in get_screener_row() from the
    real ~80Δ/542-DTE contract, not derived from r."""
    df=r.get("df"); cl=r.get("cl"); price=r.get("price",0); pct=r.get("pct",0)
    gates={}

    if cl is not None and len(cl.dropna())>=50:
        ma20=float(cl.rolling(20).mean().dropna().iloc[-1]); ma50=float(r.get("ma50",0))
        g1=(ma20>ma50) or (price>ma20)
        gates["G1"]={"pass":g1,"label":"Trend (MA)",
            "reason":f"20MA={ma20:.2f} {'>' if ma20>ma50 else '<'} 50MA={ma50:.2f}  |  Price {'>' if price>ma20 else '<'} 20MA"}
    else:
        gates["G1"]={"pass":False,"label":"Trend (MA)","reason":"Insufficient history"}

    g3=pct>-2.5
    gates["G2"]={"pass":g3,"label":"Session","reason":f"Today:{pct:+.2f}% ({'OK' if g3 else 'FAIL — down >2.5%'})"}

    bb_penalty=0; walking=False
    if cl is not None and len(cl.dropna())>=22:
        _,_,lower=calc_bb_bands(cl); lo_c=lower.dropna(); cl_a=cl.loc[lo_c.index]
        if len(lo_c)>=2:
            walking=(float(cl_a.iloc[-1])<float(lo_c.iloc[-1])) and (float(cl_a.iloc[-2])<float(lo_c.iloc[-2]))
            band_txt=f"Lower band:{float(lo_c.iloc[-1]):.2f}  |  {'Walking lower ❌' if walking else 'Inside bands ✓'}"
            if bb_veto_mode=="Hard":
                gates["G3"]={"pass":not walking,"label":"BB Veto","reason":band_txt}
            elif bb_veto_mode=="Soft":
                bb_penalty = soft_penalty if walking else 0
                gates["G3"]={"pass":True,"label":"BB Veto",
                    "reason":band_txt+(f"  (soft: −{soft_penalty} pts)" if walking else "")}
            else:  # "Off"
                gates["G3"]={"pass":True,"label":"BB Veto",
                    "reason":band_txt+("  (veto off — informational only)" if walking else "")}
        else:
            gates["G3"]={"pass":True,"label":"BB Veto","reason":"Sparse — pass"}
    else:
        gates["G3"]={"pass":True,"label":"BB Veto","reason":"Insufficient — pass"}

    # G4 Median (22 June, Jay's request; direction corrected same day) — CSP fails ABOVE the
    # median band (catch the bounce early, before the up-move's runway is used up); CC fails
    # BELOW it (catch the topping setup early, before the down-move's runway is used up).
    # Only applies to leg in {"csp","cc"} — leap gets its own G4 below instead.
    if leg in ("csp","cc") and cl is not None and len(cl.dropna())>=20:
        median=float(cl.rolling(20).mean().dropna().iloc[-1])
        if leg=="csp":
            g4=price<=median
            gates["G4"]={"pass":g4,"label":"Median (CSP)",
                "reason":f"Price ${price:.2f} {'≤' if g4 else '>'} median ${median:.2f}"
                          + ("" if g4 else "  — FAIL: above median, runway used up")}
        else:  # cc
            g4=price>=median
            gates["G4"]={"pass":g4,"label":"Median (CC)",
                "reason":f"Price ${price:.2f} {'≥' if g4 else '<'} median ${median:.2f}"
                          + ("" if g4 else "  — FAIL: below median, runway used up")}
    elif leg in ("csp","cc"):
        gates["G4"]={"pass":True,"label":f"Median ({leg.upper()})","reason":"Insufficient — pass"}
    elif leg=="leap":
        # G4 Premium Mix (26 June, Jay's request) — replaces the "no G4 for LEAP" gap. Fails
        # when extrinsic (time value) is more than 60% of the contract's premium — paying
        # mostly to rent time rather than for the moneyness already baked into the strike.
        # 60% = the 50/50 split Jay proposed, plus a 10-point buffer.
        total=(leap_intrinsic or 0.0)+(leap_extrinsic or 0.0)
        if leap_intrinsic is not None and leap_extrinsic is not None and total>0:
            ext_share=leap_extrinsic/total
            g4=ext_share<=0.60
            gates["G4"]={"pass":g4,"label":"Premium Mix (LEAP)",
                "reason":f"Extrinsic ${leap_extrinsic:.2f} of ${total:.2f} premium ({ext_share*100:.0f}%)"
                          + ("" if g4 else "  — FAIL: paying mostly for time value")}
        else:
            gates["G4"]={"pass":True,"label":"Premium Mix (LEAP)","reason":"No LEAP contract — pass"}

    return {"gates":gates,"all_pass":all(g["pass"] for g in gates.values()),
            "bb_walking":walking,"bb_penalty":bb_penalty,"bb_veto_mode":bb_veto_mode}

# 22 June fix — schema-version stamp for the cached screener rows below. Streamlit's file-
# watcher reruns an already-open session's script on a code push WITHOUT clearing
# st.session_state, so a browser tab open before this edit still held rows built under the
# old single-"gate_result" schema; reading them back under the new per-leg
# "gate_result_csp"/"gate_result_cc"/"gate_result_leap" keys crashed with a KeyError. Bump
# this whenever get_screener_row's returned dict shape changes, so a stale cache is detected
# and discarded (forcing a re-click of "Run Screener") instead of crashing the page.
# v3 (26 June) — gate_result_leap now carries a G4 entry (Premium Mix); Deep Dive also reads
# this gate straight out of screener_results, so a stale v2 cache (no G4) needs discarding
# rather than silently showing a 3-gate LEAP result next to a 4-gate label.
_SCREENER_SCHEMA_VERSION = 3

def get_screener_row(ticker, result, bb_veto_mode="Hard", soft_penalty=10,
                      target_delta_csp=30.0, target_dte_csp=30, target_delta_cc=30.0,
                      target_delta_leap=80.0, target_dte_leap=542):
    # 22 June correction — CC default delta and LEAP default DTE realigned to Jay's stated
    # defaults (CC Δ30, LEAP 542 DTE ≈ 18 months). Was previously Δ35 for CC, 547 for LEAP.
    # 26 June — target delta/DTE are now caller-supplied (default Δ30/30DTE for CSP, Δ30/30DTE
    # for CC, Δ80/542DTE for LEAP) instead of hard-coded, per Jay: keep Δ30/30DTE as the
    # default, but let a trader manually dial it elsewhere on the chain (chart/support-resistance call)
    # rather than baking one fixed target into the code. The 21-45 / 180-900 DTE *windows*
    # stay fixed — they're sanity bounds on what counts as "CSP-ish" / "LEAP-ish" at all, not
    # the tunable target itself.
    price=result.get("price"); all_exps=result.get("all_exps",[]); hv_raw=result.get("hv20")
    if not all_exps or not price or price<=0: return None

    today=datetime.utcnow(); exp_csp=None; dte_csp=None; min_diff=999
    for exp in all_exps:
        try:
            dte=(datetime.strptime(exp,"%Y-%m-%d")-today).days
            diff=abs(dte-target_dte_csp)
            if 21<=dte<=45 and diff<min_diff:
                min_diff=diff; exp_csp=exp; dte_csp=dte
        except Exception:
            continue
    if exp_csp is None: return None

    calls_df,puts_df,_,_=fetch_chain(ticker,exp_csp)
    if puts_df is None or puts_df.empty: return None

    hv_sigma=(hv_raw/100.0) if (hv_raw and hv_raw>0) else None
    csp=find_target_strike(puts_df, target_delta_csp,"put", price,dte_csp,hv_sigma)
    if csp is None: return None

    cc=None
    if calls_df is not None and not calls_df.empty:
        cc=find_target_strike(calls_df,target_delta_cc,"call",price,dte_csp,hv_sigma)

    nis=calc_nis(csp["theta"],dte_csp,csp["strike"])
    csp_score=calc_suitability(nis,dte_csp,csp["delta"],"CSP")
    cc_nis = calc_nis(cc["theta"],dte_csp,cc["strike"]) if cc else 0.0
    cc_score =(calc_suitability(cc_nis,dte_csp,cc["delta"],"CC") if cc else 0.0)

    # §8/§9.4 LEAP fix (24 June) — fetch a real long-dated contract instead of reusing the
    # CSP's ~30Δ/30DTE numbers. _tri_score() hard-zeroes DTE-fit/delta-fit outside LEAP's
    # 180-900 DTE / 60-95 delta window, so reusing CSP's ~30DTE/~30delta capped LEAP Score
    # at ~30/100 for every ticker regardless of actual LEAP suitability. Fetch the expiry
    # closest to 542 DTE within the 180-900 window, find its ~80-delta call, score that
    # contract on its own theta/delta/DTE — same find_target_strike pattern as CC.
    exp_leap=None; dte_leap=None; min_diff_leap=999999
    for exp in all_exps:
        try:
            dte_l=(datetime.strptime(exp,"%Y-%m-%d")-today).days
            diff_l=abs(dte_l-target_dte_leap)
            if 180<=dte_l<=900 and diff_l<min_diff_leap:
                min_diff_leap=diff_l; exp_leap=exp; dte_leap=dte_l
        except Exception:
            continue

    leap=None; leap_nis=None; leap_score=None
    leap_intrinsic=None; leap_extrinsic=None; leap_extrinsic_per_day=None
    if exp_leap is not None:
        leap_calls_df,_,_,_=fetch_chain(ticker,exp_leap)
        if leap_calls_df is not None and not leap_calls_df.empty:
            leap=find_target_strike(leap_calls_df,target_delta_leap,"call",price,dte_leap,hv_sigma)
    if leap is not None:
        # NOTE: NIS_FLOOR/NIS_CEIL were hand-calibrated to the CSP's ~30delta/30DTE shape;
        # theta-to-IV scaling differs at LEAP's ~80delta/542DTE shape (§8 side note), so
        # leap_nis/leap_score may need their own floor/ceil once tested on real LEAP chains.
        leap_nis=calc_nis(leap["theta"],dte_leap,leap["strike"])
        leap_score=calc_suitability(leap_nis,dte_leap,leap["delta"],"LEAP")
        # 26 June — LEAP-buyer cost metrics (Jay's request): how much of the mid premium is
        # pure time value (extrinsic) above intrinsic, and what that works out to per day on
        # average over the life of the contract. This is a flat average (extrinsic ÷ DTE),
        # not the instantaneous θ/day already shown — theta accelerates as expiry nears, so
        # the average and the instantaneous rate will differ (average < current θ/day is
        # normal early in a long-dated contract's life).
        leap_mid_val=leap.get("mid") or 0.0
        leap_intrinsic=round(max(0.0, price-leap["strike"]), 2)
        leap_extrinsic=round(max(0.0, leap_mid_val-leap_intrinsic), 2)
        leap_extrinsic_per_day=round(leap_extrinsic/dte_leap, 4) if dte_leap and dte_leap>0 else None
    # else: no expiry in the 180-900 DTE window (or no usable chain) for this ticker —
    # leap_score stays None rather than silently reusing CSP's numbers.

    # 22 June — Gates are now per-leg, not one shared computation (see calc_four_gates'
    # leg= param). CSP and CC each get their own G4 Median check with opposite pass
    # conditions. 26 June — LEAP now gets its own G4 too: Premium Mix (intrinsic vs
    # extrinsic of the real LEAP contract computed just above).
    gate_result_csp =calc_four_gates(result, bb_veto_mode=bb_veto_mode, soft_penalty=soft_penalty, leg="csp")
    gate_result_cc  =calc_four_gates(result, bb_veto_mode=bb_veto_mode, soft_penalty=soft_penalty, leg="cc")
    gate_result_leap=calc_four_gates(result, bb_veto_mode=bb_veto_mode, soft_penalty=soft_penalty, leg="leap",
                                      leap_intrinsic=leap_intrinsic, leap_extrinsic=leap_extrinsic)

    # §9.3 BB veto Soft mode (24 June) — apply the points penalty to each leg's score instead
    # of blocking Status. Hard/Off modes carry bb_penalty==0, so this is a no-op for them.
    pen_csp=gate_result_csp.get("bb_penalty",0)
    pen_cc =gate_result_cc.get("bb_penalty",0)
    pen_leap=gate_result_leap.get("bb_penalty",0)
    if pen_csp: csp_score=max(0.0,csp_score-pen_csp)
    if pen_cc and cc: cc_score=max(0.0,cc_score-pen_cc)
    if pen_leap and leap_score is not None: leap_score=max(0.0,leap_score-pen_leap)

    # §9.2 free-data-win metrics — Annualized Return %, Breakeven, POP, Liquidity.
    # Ann. return is yield-on-strike (collateral/notional), annualized off the actual DTE.
    csp_mid=csp.get("mid") or 0.0
    csp_ann_return = round((csp_mid/csp["strike"])*(365.0/dte_csp)*100.0,2) if csp["strike"]>0 and dte_csp>0 else None
    csp_breakeven  = round(csp["strike"]-csp_mid,2)
    csp_breakeven_pct = round((price-csp_breakeven)/price*100.0,2) if price>0 else None
    cc_mid = cc.get("mid") if cc else None
    cc_ann_return = (round((cc_mid/cc["strike"])*(365.0/dte_csp)*100.0,2)
                      if cc and cc_mid and cc["strike"]>0 and dte_csp>0 else None)

    # §10 mean-reversion timing signal — already computed once per ticker in analyse(),
    # reused here rather than recomputed. Flag/score, not a filter (revised §10.4.1).
    cc_lbl,cc_tsc,cc_treasons = result.get("cc", ("—",0,[]))
    csp_lbl,csp_tsc,csp_treasons = result.get("csp", ("—",0,[]))
    # LEAP entry signal (leap_signal — HV-rank/RSI/MA) surfaced so the LEAP view stands on
    # its own criteria like CSP/CC's mean-reversion timing does (added for CSP/CC/LEAP parity).
    leap_lbl2,leap_tsc,leap_treasons = result.get("leap", ("—",0,[]))

    return {"ticker":ticker,"price":price,"expiry":exp_csp,"dte":dte_csp,
            "csp_strike":csp["strike"],"csp_delta":csp["delta"],"csp_theta":csp["theta"],
            "csp_iv":csp["iv"],"csp_oi":csp["oi"],"csp_volume":csp.get("volume"),"csp_spread":csp["spread_pct"],
            "csp_mid":csp_mid,"csp_pop":csp.get("pop"),"csp_liquidity":csp.get("liquidity_score"),
            "csp_ann_return":csp_ann_return,"csp_breakeven":csp_breakeven,"csp_breakeven_pct":csp_breakeven_pct,
            "cc_strike":cc["strike"] if cc else None,"cc_delta":cc["delta"] if cc else None,
            "cc_theta":cc["theta"] if cc else None,"cc_iv":cc["iv"] if cc else None,
            "cc_spread":cc["spread_pct"] if cc else None,
            "cc_mid":cc_mid,"cc_pop":cc.get("pop") if cc else None,"cc_liquidity":cc.get("liquidity_score") if cc else None,
            "cc_oi":cc.get("oi") if cc else None,"cc_volume":cc.get("volume") if cc else None,"cc_ann_return":cc_ann_return,
            # 26 June — added so the CC table can carry the same column set as CSP (NIS, Greeks).
            "cc_nis":round(cc_nis,1) if cc else None,"cc_greek_source":cc.get("greek_source") if cc else None,
            "nis":round(nis,1),"csp_score":csp_score,"cc_score":cc_score,
            "leap_expiry":exp_leap,"leap_dte":dte_leap,
            "leap_strike":leap["strike"] if leap else None,"leap_delta":leap["delta"] if leap else None,
            "leap_theta":leap["theta"] if leap else None,"leap_iv":leap["iv"] if leap else None,
            "leap_oi":leap.get("oi") if leap else None,"leap_mid":leap.get("mid") if leap else None,
            "leap_nis":round(leap_nis,1) if leap_nis is not None else None,
            "leap_greek_source":leap.get("greek_source") if leap else None,
            # 26 June — added so the LEAP table can carry the same column set as CSP/CC.
            "leap_volume":leap.get("volume") if leap else None,"leap_spread":leap.get("spread_pct") if leap else None,
            "leap_pop":leap.get("pop") if leap else None,"leap_liquidity":leap.get("liquidity_score") if leap else None,
            # 26 June — LEAP-buyer cost metrics (extrinsic premium, avg $/day to hold it).
            "leap_intrinsic":leap_intrinsic,"leap_extrinsic":leap_extrinsic,
            "leap_extrinsic_per_day":leap_extrinsic_per_day,
            "leap_score":leap_score,
            "gate_result_csp":gate_result_csp,"gate_result_cc":gate_result_cc,
            "gate_result_leap":gate_result_leap,
            "greek_source":csp.get("greek_source","unknown"),
            "_inspected":csp.get("_inspected"),"_scored":csp.get("_scored"),
            "cc_timing_label":cc_lbl,"cc_timing_score":cc_tsc,"cc_timing_reasons":cc_treasons,
            "csp_timing_label":csp_lbl,"csp_timing_score":csp_tsc,"csp_timing_reasons":csp_treasons,
            "leap_timing_label":leap_lbl2,"leap_timing_score":leap_tsc,"leap_timing_reasons":leap_treasons}

# ── Signal engines ─────────────────────────────────────────────────────────────
def leap_signal(iv_ratio,rsi_val,above_50ma,above_200ma):
    # 26 June — dropped the VIX bullet (Jay: too generic, index-wide, not ticker-specific).
    # Deep Dive now appends a Premium Mix tick (real ~80Δ/542-DTE contract, intrinsic vs
    # extrinsic) sourced from the Screener tab's data instead — see tab_dive. vix_lvl is no
    # longer a param here; analyse() still accepts vix_current but no longer threads it in.
    score=0; reasons=[]
    if iv_ratio is not None:   # LEAP buyer wants CHEAP premium → low IV vs realized
        if iv_ratio<1.0:    score+=3; reasons.append("✅ IV cheap vs realized — good entry")
        elif iv_ratio<1.25: score+=2; reasons.append("🟡 IV fair vs realized")
        else:               score-=1; reasons.append("❌ IV rich vs realized — expensive entry")
    if rsi_val is not None:
        if 33<=rsi_val<=52:   score+=2; reasons.append("✅ RSI ideal recovery zone (33-52)")
        elif 52<rsi_val<=65:  score+=1; reasons.append("🟡 RSI extended, not overbought")
        elif rsi_val<30:      score+=1; reasons.append("🟡 RSI oversold — wait for turn")
        else:                 score-=1; reasons.append("❌ RSI overbought — avoid chasing")
    if above_200ma: score+=2; reasons.append("✅ Above 200MA — long term trend intact")
    else:           score-=1; reasons.append("❌ Below 200MA — trend broken")
    if above_50ma:  score+=1; reasons.append("✅ Above 50MA — medium term OK")
    label=("🟢 STRONG ENTRY" if score>=7 else "🟡 DECENT ENTRY" if score>=4
           else "🟠 MARGINAL" if score>=2 else "🔴 AVOID")
    return label, score, reasons

# ── Candle reversal patterns (§10.3a) — OR logic, any one pattern is enough ─────
def _candle_reversal(df, direction, lookback=2):
    """direction: 'bearish' (CC trigger) or 'bullish' (CSP mirror)."""
    try:
        o=df["Open"].squeeze(); h=df["High"].squeeze(); l=df["Low"].squeeze(); c=df["Close"].squeeze()
    except Exception:
        return False, None
    n=len(c)
    if n<4: return False, None
    for off in range(lookback):
        t=n-1-off; t1=t-1; t2=t-2
        if t1<0: continue
        o_t,h_t,l_t,c_t=float(o.iloc[t]),float(h.iloc[t]),float(l.iloc[t]),float(c.iloc[t])
        o1,h1,l1,c1=float(o.iloc[t1]),float(h.iloc[t1]),float(l.iloc[t1]),float(c.iloc[t1])
        body_t=abs(c_t-o_t); body1=abs(c1-o1); rng1=max(h1-l1,1e-9)
        uw1=h1-max(o1,c1); lw1=min(o1,c1)-l1
        if direction=="bearish":
            if c1>o1 and c_t<o_t and o_t>=c1 and c_t<=o1 and body_t>body1*0.9:
                return True,"Bearish engulfing"
            mid1=(o1+c1)/2
            if c1>o1 and o_t>c1 and c_t<o_t and o1<c_t<mid1:
                return True,"Dark cloud cover"
            if abs(h_t-h1)/rng1<0.015 and c_t<o_t:
                return True,"Tweezer top"
            if uw1>=2*body1 and lw1<=body1*0.3 and c_t<o_t and c_t<c1:
                return True,"Shooting star + confirmation"
            if t2>=0:
                o2,h2,l2,c2=float(o.iloc[t2]),float(h.iloc[t2]),float(l.iloc[t2]),float(c.iloc[t2])
                body2=abs(c2-o2); rngt=max(h_t-l_t,1e-9)
                if c2>o2 and body2>rngt*0.5 and body1<rng1*0.3 and c_t<o_t and c_t<(o2+c2)/2:
                    return True,"Evening star"
        else:
            if c1<o1 and c_t>o_t and o_t<=c1 and c_t>=o1 and body_t>body1*0.9:
                return True,"Bullish engulfing"
            mid1=(o1+c1)/2
            if c1<o1 and o_t<c1 and c_t>o_t and mid1<c_t<o1:
                return True,"Piercing line"
            if abs(l_t-l1)/rng1<0.015 and c_t>o_t:
                return True,"Tweezer bottom"
            if lw1>=2*body1 and uw1<=body1*0.3 and c_t>o_t and c_t>c1:
                return True,"Hammer + confirmation"
            if t2>=0:
                o2,h2,l2,c2=float(o.iloc[t2]),float(h.iloc[t2]),float(l.iloc[t2]),float(c.iloc[t2])
                body2=abs(c2-o2); rngt=max(h_t-l_t,1e-9)
                if c2<o2 and body2>rngt*0.5 and body1<rng1*0.3 and c_t>o_t and c_t>(o2+c2)/2:
                    return True,"Morning star"
    return False, None

# ── Mean-reversion timing trigger (§10.1 CC / §10.2 CSP) — score, not a gate ────
def _mean_reversion_score(pctb_s, rsi_s, df, direction):
    """direction: 'cc' (fade overbought) or 'csp' (fade oversold). Returns (score, reasons, pattern)."""
    pctb_c=pctb_s.dropna(); rsi_c=rsi_s.dropna()
    if len(pctb_c)<3 or len(rsi_c)<5: return 0,["Insufficient history for mean-reversion read"],None
    pctb_today=float(pctb_c.iloc[-1]); pctb_prev=float(pctb_c.iloc[-2]) if len(pctb_c)>=2 else pctb_today
    pctb_3=pctb_c.iloc[-3:]
    rsi_today=float(rsi_c.iloc[-1]); rsi_prev=float(rsi_c.iloc[-2]) if len(rsi_c)>=2 else rsi_today
    rsi_5=rsi_c.iloc[-5:]
    score=0; reasons=[]

    if direction=="cc":
        fired,pattern=_candle_reversal(df,"bearish",2)
        if pctb_today>=0.85: score+=2; reasons.append(f"✅ Near/touching upper BB ({pctb_today:.2f})")
        if pctb_3.max()>=0.95 and pctb_today<pctb_prev: score+=3; reasons.append("✅ Spiked then rolled over")
        if pctb_today>0.5:   score+=1; reasons.append("✅ Above midline")
        if rsi_5.max()>70 and rsi_today<rsi_prev: score+=3; reasons.append(f"✅ RSI exceeded 70, turning down ({rsi_today:.0f})")
        if fired: score+=3; reasons.append(f"✅ {pattern}")
    else:
        fired,pattern=_candle_reversal(df,"bullish",2)
        if pctb_today<=0.15: score+=2; reasons.append(f"✅ Near/touching lower BB ({pctb_today:.2f})")
        if pctb_3.min()<=0.05 and pctb_today>pctb_prev: score+=3; reasons.append("✅ Dropped then bounced")
        if pctb_today<0.5:   score+=1; reasons.append("✅ Below midline")
        if rsi_5.min()<30 and rsi_today>rsi_prev: score+=3; reasons.append(f"✅ RSI dropped below 30, turning up ({rsi_today:.0f})")
        if fired: score+=3; reasons.append(f"✅ {pattern}")
    return score, reasons, (pattern if fired else None)

def _pctb_now(pctb_s):
    """Latest Bollinger %B. 0.5 == the 20-day midline (the same 'median' the CC/CSP G4 gate
    uses), so %B<0.5 = below median, %B>0.5 = above median."""
    c=pctb_s.dropna()
    return float(c.iloc[-1]) if len(c)>=1 else None

def _setup_label(score, kind, blocked=False):
    # 25 July — median rule made consistent with the Screener's G4 gate: a CC/CSP timing read
    # can only go GREEN when price is on the correct side of the median (CC above, CSP below).
    # If the mean-reversion read is strong but price is on the wrong side, cap at yellow with a
    # clear "wrong side of median" — never green. Mean-reversion stays the judgement layer.
    verb = "write now" if kind=="cc" else "sell put"
    tier = "full" if score>=10 else "partial" if score>=6 else "early" if score>=3 else "none"
    if blocked and tier in ("full","partial"):
        return "🟡 Timing ok — wrong side of median"
    return {"full":f"🟢 FULL SETUP — {verb}","partial":"🟡 PARTIAL SETUP",
            "early":"🟠 BUILDING UP / WAIT","none":"🔴 NO SETUP"}[tier]

def median_chip(pctb, leg):
    """Side-by-side median indicator (Signal 1) shown next to the timing label (Signal 2).
    Price vs its 20-day median (Bollinger midline, %B 0.5). CC needs above, CSP needs below."""
    if pctb is None:
        return "Median: —"
    above = pctb >= 0.5
    pos   = "🔼 above" if above else "🔽 below"
    ok    = above if leg == "cc" else (not above)
    return f"Median: {pos} " + ("✅" if ok else "❌ needs " + ("above" if leg == "cc" else "below"))

def iv_richness(c_iv, p_iv, hv20):
    """Premium richness for a theta seller: ATM IV vs 20-day realized vol (HV20). IV usually
    sits above realized (the variance risk premium), so 'rich' means it's meaningfully above —
    more decay to harvest and an IV-crush tailwind. Returns a short tag (or '—')."""
    ivs = [v for v in (c_iv, p_iv) if v]
    if not ivs or not hv20 or hv20 <= 0:
        return "—"
    ratio = (sum(ivs)/len(ivs)) / hv20
    if ratio >= 1.25: return "🟢 Rich"
    if ratio >= 1.00: return "⚪ Fair"
    return "🔴 Cheap"

def cc_signal(iv_ratio,pctb_s,rsi_s,df):
    score,reasons,_=_mean_reversion_score(pctb_s,rsi_s,df,"cc")
    if iv_ratio is not None:   # CC seller wants RICH premium → high IV vs realized
        if iv_ratio>=1.25:  score+=2; reasons.append("✅ IV rich vs realized — premium fat")
        elif iv_ratio>=1.0: score+=1; reasons.append("🟡 IV fair vs realized")
        else:               reasons.append("❌ IV below realized — thin premium even if setup fires")
    _pb=_pctb_now(pctb_s); below_median=_pb is not None and _pb<0.5
    if below_median:
        reasons.append(f"⛔ Price below median (%B {_pb:.2f}) — CC needs above median; no green")
    label=_setup_label(score,"cc",blocked=below_median)
    return label, score, reasons

def csp_signal(iv_ratio,pctb_s,rsi_s,df,walking=False):
    score,reasons,_=_mean_reversion_score(pctb_s,rsi_s,df,"csp")
    if walking:
        score=max(0,score-4); reasons.append("❌ Still walking the lower band — breakdown, not a bounce (veto)")
    if iv_ratio is not None:   # CSP seller wants RICH premium → high IV vs realized
        if iv_ratio>=1.25:  score+=2; reasons.append("✅ IV rich vs realized — CSP premium fat")
        elif iv_ratio>=1.0: score+=1; reasons.append("🟡 IV fair vs realized")
    _pb=_pctb_now(pctb_s); above_median=_pb is not None and _pb>0.5
    if above_median:
        reasons.append(f"⛔ Price above median (%B {_pb:.2f}) — CSP needs below median; no green")
    label=_setup_label(score,"csp",blocked=above_median)
    return label, score, reasons

def analyse(ticker, period, vix_current):
    df=fetch_prices(ticker,period)
    if df is None: return None
    cl=df["Close"].squeeze()
    # Use the last VALID close, not literally the last row: when the market is closed / a
    # holiday / just before a fresh print, yfinance often returns a trailing NaN bar, which
    # would surface as "$nan" in every price column. Drop trailing NaNs first.
    _clv=cl.dropna()
    if len(_clv)<2: return None
    curr=float(_clv.iloc[-1]); prev=float(_clv.iloc[-2]); pct_chg=(curr/prev-1)*100 if prev else 0.0
    hv20_s=calc_hv(cl,20); hv60_s=calc_hv(cl,60)
    hv_cur=float(hv20_s.dropna().iloc[-1]) if not hv20_s.dropna().empty else None
    hvr=calc_iv_rank(hv20_s); hvpct=calc_iv_percentile(hv20_s)
    rsi_s=calc_rsi(cl)
    rsi_cur=float(rsi_s.dropna().iloc[-1]) if not rsi_s.dropna().empty else None
    atr_s=calc_atr(df)
    atr_cur=float(atr_s.dropna().iloc[-1]) if not atr_s.dropna().empty else None
    bbw_s=calc_bb_width(cl)
    bbw_cur=float(bbw_s.dropna().iloc[-1]) if not bbw_s.dropna().empty else None
    ma50=float(cl.rolling(50).mean().iloc[-1]); ma200=float(cl.rolling(200).mean().iloc[-1])
    ab50=curr>ma50; ab200=curr>ma200
    pctb_s=calc_bb_pctb(cl); pctb_c=pctb_s.dropna()
    pctb_cur=float(pctb_c.iloc[-1]) if len(pctb_c)>=1 else None
    walking_lower=bool(len(pctb_c)>=2 and pctb_c.iloc[-1]<=0.2 and pctb_c.iloc[-2]<=0.2)
    all_exps,exp_err=fetch_all_expiries(ticker)
    c_iv=p_iv=pcr_val=chain=exp=dte=None; chain_err=None
    if all_exps:
        today=datetime.utcnow()
        valid=[e for e in all_exps if (datetime.strptime(e,"%Y-%m-%d")-today).days>14]
        if valid:
            exp=valid[0]
            calls_df,puts_df,dte,chain_err=fetch_chain(ticker,exp)
            if calls_df is not None:
                chain=type("_C",(),{"calls":calls_df,"puts":puts_df})()
                c_iv,p_iv=calc_atm_iv(chain,curr); pcr_val=calc_pcr(chain)
    # Premium richness for the signals = ATM IV ÷ 20-day realized vol (replaces HV Rank, which
    # Jay found meaningless — IV vs realized is what actually matters to a premium seller/buyer).
    _ivs=[v for v in (c_iv,p_iv) if v]
    iv_ratio=((sum(_ivs)/len(_ivs))/hv_cur) if (_ivs and hv_cur and hv_cur>0) else None
    leap_lbl,leap_sc,leap_r=leap_signal(iv_ratio,rsi_cur,ab50,ab200)
    cc_lbl,cc_sc,cc_r=cc_signal(iv_ratio,pctb_s,rsi_s,df)
    csp_lbl,csp_sc,csp_r=csp_signal(iv_ratio,pctb_s,rsi_s,df,walking_lower)
    return {"ticker":ticker,"price":curr,"pct":pct_chg,
            "hv20":hv_cur,"hvr":hvr,"hvpct":hvpct,"hv20_s":hv20_s,"hv60_s":hv60_s,
            "rsi":rsi_cur,"rsi_s":rsi_s,"atr":atr_cur,"bbw":bbw_cur,"bbw_s":bbw_s,
            "ma50":ma50,"ma200":ma200,"ab50":ab50,"ab200":ab200,
            "c_iv":c_iv,"p_iv":p_iv,"pcr":pcr_val,"exp":exp,"dte":dte,
            "all_exps":all_exps,"df":df,"cl":cl,
            # 25 June fix — real fetch-error text instead of a swallowed exception, so the
            # Screener debug log can show *why* (rate limit, empty chain, etc.) instead of
            # a generic "no expiries"/"chain failed".
            "fetch_error":exp_err or chain_err,
            "pctb":pctb_cur,"walking_lower":walking_lower,
            "leap":(leap_lbl,leap_sc,leap_r),
            "cc":(cc_lbl,cc_sc,cc_r),
            "csp":(csp_lbl,csp_sc,csp_r)}

def vix_zone(v):
    for lo,hi,color,label in VIX_ZONES:
        if lo<=v<hi: return color,label
    return "#6b7280","Unknown"

def fmt(v,fs=".1f",su=""):
    # None- AND NaN/inf-safe: a NaN is not None, so it would otherwise print as "nan".
    try:
        if v is None or (isinstance(v,float) and not math.isfinite(v)):
            return "—"
        return f"{v:{fs}}{su}"
    except (TypeError, ValueError):
        return "—"

def greek_source_label(gs):
    return {"tradier":"📡 Tradier (real)","yahoo":"📡 Yahoo","bs_strike":"📐 BS (strike IV)",
            "bs_chain_median":"📐 BS (chain med IV)","bs_hv20":"📐 BS (HV20)",
            "bs_default":"📐 BS (30% def)"}.get(gs, gs or "—")

# Reusable pure-CSS hover tooltip for plain labels/controls (buttons, section intros) —
# same :hover technique as the screener column-header and nav-tab tooltips, just packaged
# as a one-liner so call sites don't need a click-based help= icon.
def _hover_tip(label, text):
    st.markdown(f"""<style>
    .jay-tip{{position:relative;display:inline-block;cursor:help;
        font-size:0.82rem;color:#6b7280;border-bottom:1px dotted #6b7280;}}
    .jay-tip .jay-tip-text{{
        visibility:hidden;opacity:0;transition:opacity 0.15s ease;
        position:absolute;bottom:135%;left:0;
        background:#1f2937;color:#f9fafb;text-align:left;border-radius:6px;
        padding:6px 10px;font-size:0.78rem;font-weight:400;line-height:1.35;
        white-space:normal;width:max-content;max-width:260px;
        box-shadow:0 4px 14px rgba(0,0,0,0.4);z-index:9999;pointer-events:none;
    }}
    .jay-tip:hover .jay-tip-text{{visibility:visible;opacity:1;}}
    </style><span class="jay-tip">{label}<span class="jay-tip-text">{text}</span></span>""",
        unsafe_allow_html=True)

# ── Gauge helpers ──────────────────────────────────────────────────────────────
def fg_color(score):
    if score is None: return "#6b7280"
    if score<25:  return "#dc2626"
    if score<45:  return "#ea580c"
    if score<55:  return "#ca8a04"
    if score<75:  return "#16a34a"
    return "#15803d"

# Sector zone config (shared by both stock and crypto gauges)
# Zone definitions: (score_lo, score_hi, bright_color, dim_color, label)
_FG_ZONES = [
    (0,  25, "#ef4444", "rgba(239,68,68,0.18)",  "EXTREME\nFEAR"),
    (25, 45, "#f97316", "rgba(249,115,22,0.18)", "FEAR"),
    (45, 55, "#eab308", "rgba(234,179,8,0.18)",  "NEUTRAL"),
    (55, 75, "#22c55e", "rgba(34,197,94,0.18)",  "GREED"),
    (75,100, "#15803d", "rgba(21,128,61,0.18)",  "EXTREME\nGREED"),
]

def fg_color(score):
    if score is None: return "#6b7280"
    for lo, hi, bright, *_ in _FG_ZONES:
        if score < hi: return bright
    return "#15803d"

def semicircle_gauge(score, title, rating, source_label=""):
    """
    True top-half-only semicircle gauge.
    Built from filled scatter polygons (arc segments) so there is zero
    bottom-half bleed.  Coordinate system: centre=(0,0), arc from 180°
    (left=score 0) to 0° (right=score 100).  y-axis clipped at -0.5 so
    the bottom half is simply never rendered.
    """
    R_OUT = 1.00   # outer ring radius
    R_IN  = 0.58   # inner hole radius
    N     = 80     # polygon resolution

    def s2a(s):
        """Score 0→180°, score 100→0°  (math angles, 0=right 90=top)"""
        return 180.0 - s * 1.8

    def arc_pts(a1_deg, a2_deg, r):
        angs = np.linspace(np.radians(a1_deg), np.radians(a2_deg), N)
        return r * np.cos(angs), r * np.sin(angs)

    # Active zone index
    active = 4
    for i, (lo, hi, *_) in enumerate(_FG_ZONES):
        if (score or 0) <= hi:
            active = i
            break

    fig = go.Figure()

    # ── Coloured arc sectors ───────────────────────────────────────────────
    for i, (slo, shi, bright, dim, lbl) in enumerate(_FG_ZONES):
        a1 = s2a(slo); a2 = s2a(shi)          # a1 > a2 (left → right)
        ox, oy = arc_pts(a1, a2, R_OUT)        # outer arc
        ix, iy = arc_pts(a2, a1, R_IN)         # inner arc (reversed)
        xs = np.concatenate([ox, ix, [ox[0]]]).tolist()
        ys = np.concatenate([oy, iy, [oy[0]]]).tolist()

        fig.add_trace(go.Scatter(
            x=xs, y=ys, fill="toself",
            fillcolor=bright if i == active else dim,
            line=dict(color="#0f172a", width=2),
            mode="lines", hoverinfo="skip", showlegend=False,
        ))

        # Zone label at arc midpoint
        ma  = math.radians(s2a((slo + shi) / 2))
        lr  = (R_OUT + R_IN) / 2
        fig.add_annotation(
            x=lr * math.cos(ma), y=lr * math.sin(ma),
            text=lbl.replace("\n", "<br>"),
            showarrow=False, align="center",
            xanchor="center", yanchor="middle",
            font=dict(size=8,
                      color="white" if i == active else "rgba(255,255,255,0.35)"),
        )

    # ── Tick marks at 0, 25, 50, 75, 100 ──────────────────────────────────
    for tv in [0, 25, 50, 75, 100]:
        ta = math.radians(s2a(tv))
        r0, r1, r2 = R_OUT + 0.02, R_OUT + 0.10, R_OUT + 0.21
        fig.add_shape(type="line",
            x0=r0*math.cos(ta), y0=r0*math.sin(ta),
            x1=r1*math.cos(ta), y1=r1*math.sin(ta),
            line=dict(color="rgba(255,255,255,0.45)", width=1.5))
        fig.add_annotation(
            x=r2*math.cos(ta), y=r2*math.sin(ta),
            text=str(tv), showarrow=False,
            font=dict(size=9, color="rgba(255,255,255,0.5)"),
            xanchor="center", yanchor="middle")

    # ── Needle ─────────────────────────────────────────────────────────────
    if score is not None:
        na  = math.radians(s2a(score))
        nlx = 0.76 * math.cos(na)
        nly = 0.76 * math.sin(na)
        # Forward shaft
        fig.add_shape(type="line",
            x0=0, y0=0, x1=nlx, y1=nly,
            line=dict(color="white", width=4))
        # Short tail
        fig.add_shape(type="line",
            x0=0, y0=0,
            x1=-0.10*math.cos(na), y1=-0.10*math.sin(na),
            line=dict(color="white", width=4))
        # Hub circle
        hw = 0.07
        fig.add_shape(type="circle",
            x0=-hw, y0=-hw, x1=hw, y1=hw,
            fillcolor="white", line_color="white")

    # ── Score + rating text ────────────────────────────────────────────────
    score_txt    = f"{score:.0f}" if score is not None else "—"
    rating_color = _FG_ZONES[active][2]

    # Score number just below the arc baseline
    fig.add_annotation(x=0, y=-0.08,
        text=f"<b>{score_txt}</b>",
        font=dict(size=46, color="white"),
        showarrow=False, xanchor="center", yanchor="top")
    # Rating label — pushed well below the number to avoid overlap
    if rating:
        fig.add_annotation(x=0, y=-0.46,
            text=f"<b>{rating}</b>",
            font=dict(size=14, color=rating_color),
            showarrow=False, xanchor="center", yanchor="top")

    # ── Title ──────────────────────────────────────────────────────────────
    t_html = f"<b>{title}</b>"
    if source_label:
        t_html += (f"<br><span style='font-size:10px;"
                   f"color:rgba(255,255,255,0.4)'>{source_label}</span>")

    fig.update_layout(
        # 22 June — title y nudged down from 0.99 to 0.90 (and top margin bumped 48→58).
        # y=0.99/yanchor=top positions the title in *paper* space, which spans the whole
        # figure including margins — so it sat almost flush against the absolute top edge
        # of the chart, overlapping the st.divider() line rendered just above it in
        # Streamlit. This pulls it down enough to clear that line.
        title=dict(text=t_html, font=dict(size=13, color="white"),
                   x=0.5, xanchor="center", y=0.90, yanchor="top"),
        height=330,
        # y range extended at bottom so score + rating both fit without clipping
        xaxis=dict(visible=False, range=[-1.45, 1.45]),
        yaxis=dict(visible=False, range=[-0.70, 1.28]),
        margin=dict(l=5, r=5, t=58, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    return fig

def vix_gauge(vix_val):
    color,label=vix_zone(vix_val)
    fig=go.Figure(go.Indicator(
        mode="gauge+number", value=vix_val,
        number={"font":{"size":34,"color":color}},
        title={"text":f"<b>VIX</b><br><span style='color:{color};font-size:0.8em'>{label.split('—')[0].strip()}</span>",
               "font":{"size":12}},
        gauge={"axis":{"range":[0,50],"tickvals":[0,15,20,30,50]},
               "bar":{"color":color,"thickness":0.25},
               "steps":[{"range":[0,15],"color":"rgba(22,163,74,0.12)"},
                         {"range":[15,20],"color":"rgba(202,138,4,0.12)"},
                         {"range":[20,30],"color":"rgba(234,88,12,0.12)"},
                         {"range":[30,50],"color":"rgba(220,38,38,0.12)"}],
               "threshold":{"line":{"color":color,"width":4},"thickness":0.75,"value":vix_val}}))
    fig.update_layout(height=210,template="plotly_dark",margin=dict(l=10,r=10,t=55,b=10))
    return fig

# ── Sector heatmap helpers ─────────────────────────────────────────────────────
def sector_tile_color(pct):
    """
    Map % change → tile fill color.
    Greens for positive, reds for negative.
    Shade intensity increases with move magnitude.
    """
    if pct is None:  return "#334155"   # no data — slate
    if pct >=  3.0:  return "#14532d"   # very deep green
    if pct >=  2.0:  return "#166534"
    if pct >=  1.0:  return "#15803d"
    if pct >=  0.3:  return "#16a34a"
    if pct >=  0.0:  return "#4ade80"   # light green (barely positive)
    if pct >= -0.3:  return "#f87171"   # light red (barely negative)
    if pct >= -1.0:  return "#dc2626"
    if pct >= -2.0:  return "#b91c1c"
    if pct >= -3.0:  return "#991b1b"
    return "#7f1d1d"                    # very deep red

def render_sector_heatmap(sector_data):
    """
    Draws a 4-column grid of equal-sized coloured tiles using Plotly
    shapes + annotations. Each tile shows sector name, ETF, and % change.

    sector_data: list of dicts — {label, ticker, pct}
    """
    N_COLS  = 4
    TILE_W  = 1.0
    TILE_H  = 0.9
    PAD_X   = 0.06
    PAD_Y   = 0.06
    n_rows  = math.ceil(len(sector_data) / N_COLS)

    fig = go.Figure()

    for i, s in enumerate(sector_data):
        row = i // N_COLS
        col = i  % N_COLS
        x0  = col * (TILE_W + PAD_X)
        x1  = x0 + TILE_W
        y0  = (n_rows - 1 - row) * (TILE_H + PAD_Y)
        y1  = y0 + TILE_H
        cx  = (x0 + x1) / 2
        cy  = (y0 + y1) / 2

        pct   = s.get("pct")
        color = sector_tile_color(pct)
        pct_txt = f"{pct:+.2f}%" if pct is not None else "—"

        # Tile background
        fig.add_shape(type="rect", x0=x0, y0=y0, x1=x1, y1=y1,
                      fillcolor=color, line=dict(color="#0f172a", width=2),
                      layer="below")

        # Sector name (top)
        fig.add_annotation(x=cx, y=cy+0.22, text=f"<b>{s['label']}</b>",
                           showarrow=False, font=dict(size=12, color="white"),
                           align="center", xanchor="center", yanchor="middle")

        # ETF ticker (middle)
        fig.add_annotation(x=cx, y=cy+0.02, text=s["ticker"],
                           showarrow=False, font=dict(size=10, color="rgba(255,255,255,0.75)"),
                           align="center", xanchor="center", yanchor="middle")

        # % change (bottom, larger, bold)
        fig.add_annotation(x=cx, y=cy-0.22, text=f"<b>{pct_txt}</b>",
                           showarrow=False, font=dict(size=14, color="white"),
                           align="center", xanchor="center", yanchor="middle")

    total_w = N_COLS * (TILE_W + PAD_X) - PAD_X
    total_h = n_rows  * (TILE_H + PAD_Y) - PAD_Y

    fig.update_layout(
        height=n_rows * 115,
        xaxis=dict(visible=False, range=[-PAD_X, total_w + PAD_X]),
        yaxis=dict(visible=False, range=[-PAD_Y, total_h + PAD_Y]),
        margin=dict(l=0, r=0, t=0, b=0),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig

# 22 June — moved up from inside the Screener tab so the Watchlist Overview table
# (Overview tab, rendered earlier in the script) can reuse the same in-header hover-
# tooltip table instead of a separate copy. Was previously a nested function only
# defined once tab_screener's block ran, which is too late for tab_dash to call it.
def _html_table(rows, legend, height):
    cols = [l for l, _ in legend]
    style = f"""<style>
    .jay-tbl-wrap{{max-height:{height}px;overflow:auto;border:1px solid #30363d;
        border-radius:6px;margin-bottom:0.6rem;}}
    .jay-tbl{{border-collapse:collapse;width:100%;font-size:0.85rem;color:#e6edf3;}}
    .jay-tbl th{{position:sticky;top:0;background:#161b22;text-align:left;
        padding:8px 10px;border-bottom:1px solid #30363d;white-space:nowrap;z-index:2;}}
    .jay-tbl td{{padding:6px 10px;border-bottom:1px solid #21262d;white-space:nowrap;background:#0d1117;}}
    .jay-tbl tbody tr:nth-child(odd) td{{background:#0d1117;}}
    .jay-tbl tbody tr:nth-child(even) td{{background:#1b222c;}}
    .jay-th-tt{{position:relative;display:inline-block;cursor:help;
        border-bottom:1px dotted #6b7280;}}
    .jay-th-tt .jay-tt-text{{
        visibility:hidden;opacity:0;transition:opacity 0.15s ease;
        position:absolute;top:135%;left:50%;transform:translateX(-50%);
        background:#1f2937;color:#f9fafb;text-align:center;border-radius:6px;
        padding:6px 10px;font-size:0.78rem;font-weight:400;line-height:1.35;
        white-space:normal;width:max-content;max-width:220px;
        box-shadow:0 4px 14px rgba(0,0,0,0.4);z-index:9999;pointer-events:none;
    }}
    .jay-th-tt .jay-tt-text::after{{
        content:"";position:absolute;bottom:100%;left:50%;margin-left:-5px;
        border-width:5px;border-style:solid;
        border-color:transparent transparent #1f2937 transparent;
    }}
    .jay-th-tt:hover .jay-tt-text{{visibility:visible;opacity:1;}}
    </style>"""
    head = "".join(
        f'<th><span class="jay-th-tt">{label}<span class="jay-tt-text">{text}</span></span></th>'
        for label, text in legend
    )
    body = "".join(
        "<tr>" + "".join(f"<td>{r.get(c, '—')}</td>" for c in cols) + "</tr>"
        for r in rows
    )
    html = (f'<div class="jay-tbl-wrap"><table class="jay-tbl"><thead><tr>{head}</tr>'
            f'</thead><tbody>{body}</tbody></table></div>')
    st.markdown(style + html, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ Settings")

    # The watchlist is the saved list in watchlist.json. Add/remove here are session-only
    # tweaks; "Reset to saved list" reloads the file. Edit the file to change it permanently.
    new_ticker=st.text_input("Add ticker (temporary)",placeholder="e.g. AMZN",key="new_ticker_input")
    if new_ticker:
        t=new_ticker.upper().strip()
        if t and t not in st.session_state.watchlist:
            st.session_state.watchlist.append(t)
            st.session_state["_wl_source"]="manual"
            st.rerun()

    if st.session_state.watchlist:
        remove=st.selectbox("Remove ticker (temporary)",["— select —"]+st.session_state.watchlist)
        if remove!="— select —":
            st.session_state.watchlist.remove(remove)
            st.session_state["_wl_source"]="manual"
            st.rerun()

    if st.session_state.get("_wl_source")=="manual":
        if st.button("↩️ Reset to saved list",use_container_width=True):
            for _k in ("watchlist","_wl_source"): st.session_state.pop(_k,None)
            st.rerun()

    period=st.selectbox("Price History",["6mo","1y","2y"],index=1)

    st.divider()
    # 21 July — default ON so the Market Pulse / sector tiles refresh on their own.
    # 5-min cadence (Jay found 60s too frequent). Toggle off to freeze the page.
    auto_refresh=st.toggle("🔄 Auto-refresh (5 min)",value=True)
    if auto_refresh and HAS_AUTOREFRESH:
        st_autorefresh(interval=300_000,key="pulse_refresh")
    elif auto_refresh and not HAS_AUTOREFRESH:
        st.warning("Add `streamlit-autorefresh` to requirements.txt")

    if st.button("🧹 Clear cached data",use_container_width=True):
        st.cache_data.clear()
        for key in ["screener_results", "screener_debug"]:
            if key in st.session_state:
                del st.session_state[key]
        st.success("Cache + screener results cleared.")
        st.rerun()

    st.divider()
    # ── Tradier connection — the real-data access door ──────────────────────────
    # Proves the token works and that real data flows even with the market closed
    # (reads the market clock, which is live 24/7). No token stored here — it lives
    # in the app's Secrets (TRADIER_TOKEN); see tradier.py for setup.
    with st.expander("🔌 Tradier data feed", expanded=not tradier.is_configured()):
        if not tradier.is_configured():
            st.warning("No token yet. Add **TRADIER_TOKEN** to the app's Secrets "
                       "(and optionally `TRADIER_ENV=\"production\"`), then Reboot.")
            st.caption("Streamlit Cloud: Manage app → Settings → Secrets. "
                       "Get the token in the Tradier dashboard → API Access.")
        elif st.button("🔌 Test Tradier connection", use_container_width=True):
            with st.spinner("Opening the door…"):
                res = tradier.ping()
            if res["ok"]:
                st.success(f"Connected ✅  ({res['env']})")
                st.caption(f"Market: **{res.get('market_state','?')}** — "
                           f"{res.get('market_desc','')}")
                if res.get("sample"):
                    st.caption(f"Live read: {res['sample']}")
                st.caption("Real data flows even while closed — this read came from a "
                           "closed-market REST call.")
            else:
                st.error(f"Not connected: {res['error']}")

    st.divider()
    vix_df=fetch_vix("1y"); vix_now=None; vix_chg=0
    if vix_df is not None and not vix_df.empty:
        vix_cl_s=vix_df["Close"].squeeze()
        vix_now=float(vix_cl_s.iloc[-1]); vix_prev=float(vix_cl_s.iloc[-2]); vix_chg=vix_now-vix_prev
        vc,vl=vix_zone(vix_now)
        st.markdown(f"### VIX: {vix_now:.1f} ({vix_chg:+.2f})")
        st.markdown(f"<span style='color:{vc};font-weight:700'>{vl}</span>",unsafe_allow_html=True)
        vix_52hi=float(vix_cl_s.max()); vix_52lo=float(vix_cl_s.min())
        vix_rank=(vix_now-vix_52lo)/(vix_52hi-vix_52lo)*100 if vix_52hi!=vix_52lo else 50
        st.progress(int(vix_rank),text=f"VIX 52wk Rank: {vix_rank:.0f}%")
    st.divider()
    st.caption("Pulse data: 60s TTL · Watchlist data: 30 min TTL")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
watchlist=st.session_state.watchlist
st.title("Options Intelligence Dashboard")

tab_dash,tab_dive,tab_chain,tab_vix,tab_signals,tab_fund=st.tabs(
    ["Overview","Deep Dive","Options Chain","📊 Market Stats","🎯 Signals","🔬 Fundamentals"])

# Hover explainers for first-time visitors — what each tab is for, in plain language.
# st.tabs() won't take custom HTML in its own labels, so (same pure-CSS :hover technique as
# the screener column tooltips, no click/no JS) this renders as a small legend strip directly
# under the tab bar instead of literally inside each tab button.
_TAB_LEGEND = [
    ("Overview", "Quick health-check across your whole watchlist — price, trend, and risk signals at a glance"),
    ("Deep Dive", "Zoom into one ticker — full technicals, fundamentals, and position-sizing guidance"),
    ("Options Chain", "Browse the live option chain for any ticker — pick an expiry and strike to inspect"),
    ("🌪️ Market Volatility", "Market-wide risk gauge — VIX level, term structure, and the current regime"),
    ("⚡ Screener", "Scans your whole watchlist for the best CSP / covered-call / LEAP candidates right now"),
]
st.markdown("""<style>
.jay-nav-tt-row{display:flex;flex-wrap:wrap;gap:0.5rem 1.3rem;margin:-0.7rem 0 0.9rem 0;}
.jay-nav-tt{position:relative;display:inline-block;cursor:help;
    font-size:0.78rem;color:#6b7280;border-bottom:1px dotted #6b7280;}
.jay-nav-tt .jay-nav-tt-text{
    visibility:hidden;opacity:0;transition:opacity 0.15s ease;
    position:absolute;top:135%;left:0;
    background:#1f2937;color:#f9fafb;text-align:left;border-radius:6px;
    padding:6px 10px;font-size:0.78rem;font-weight:400;line-height:1.35;
    white-space:normal;width:max-content;max-width:260px;
    box-shadow:0 4px 14px rgba(0,0,0,0.4);z-index:9999;pointer-events:none;
}
.jay-nav-tt:hover .jay-nav-tt-text{visibility:visible;opacity:1;}
</style>""", unsafe_allow_html=True)
# Tab-legend strip removed (per Jay) — it rendered below the tab container, i.e. at the bottom
# of every tab, which was just clutter. The CSS above stays: _tt() reuses .jay-nav-tt for the
# inline hover tooltips used throughout.

# 26 June — reusable inline hover-tooltip helper, module scope so every tab below can wrap a
# bare jargon term (SKEW, DXY, Contango, etc.) in markdown text without re-injecting CSS — the
# .jay-nav-tt/.jay-nav-tt-text classes are already on the page from the tab-legend strip above.
# Use inside any st.markdown(..., unsafe_allow_html=True) call. For st.metric labels (which
# can't render HTML), pass help= instead — that's a real Streamlit tooltip, confirmed working
# on the deployed app since requirements.txt now pins streamlit>=1.37.
def _tt(label, text):
    return f'<span class="jay-nav-tt">{label}<span class="jay-nav-tt-text">{text}</span></span>'

results={}
with st.spinner("Loading market data..."):
    for t in watchlist:
        r=analyse(t,period,vix_now)
        if r: results[t]=r

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab_dash:
    st.subheader("🌍 Market Pulse")
    st.caption(f"Updated: {datetime.utcnow().strftime('%H:%M:%S UTC')}  ·  ~15 min delayed  ·  Toggle 5-min refresh in sidebar")

    # ── Data Health — proves, per source, whether it connected and how old its data is ──────
    # Button-gated: the probe hits the network, so running it on every 60s auto-refresh would
    # add latency to every render. It runs only when clicked, and the result is stashed in
    # session_state so it survives the next auto-refresh rerun instead of vanishing.
    with st.expander("🩺 Data Health — is every source live & fresh? (open me if numbers look stale)"):
        st.caption("Probes every data source live and shows **how old each one's data is** — the "
                   "fastest way to tell 'a source is stale/down' from 'the page just wasn't refreshing'.")
        if st.button("🔄 Run data health check", key="run_health"):
            try:
                st.session_state["_health"] = fetch_data_health()
            except Exception as _e:
                st.session_state["_health"] = (f"{type(_e).__name__}: {_e}", None)
        _h = st.session_state.get("_health")
        if _h:
            _rows, _when = _h
            if _when is None:
                st.error(f"Data Health probe failed to run: {_rows}")
            else:
                st.caption(
                    f"Probed live at {_when.strftime('%Y-%m-%d %H:%M:%S')} UTC. "
                    "**How to read this:** if a row shows a *Data as of* from days or weeks ago, "
                    "that source is the problem. If every row says 🟢 live but the tiles still look "
                    "frozen, the app was serving a stale page — hard-reload with Ctrl+Shift+R "
                    "(Cmd+Shift+R on Mac). Sample values let you sanity-check against Google."
                )
                st.dataframe(pd.DataFrame(_rows), use_container_width=True, hide_index=True)

    pulse_data=fetch_quotes(tuple(ticker for ticker,*_ in PULSE_TICKERS))

    # 26 June — st.metric labels can't render HTML, so jargon-y ones get a real tooltip via
    # help= instead of the inline _tt() span used in markdown text below.
    _PULSE_HELP={"DXY":"US Dollar Index — dollar's value vs a basket of major currencies",
                 "R2000":"Russell 2000 — small-cap stock index",
                 "10Y Yield":"10-Year Treasury yield",
                 "3M Yield":"3-Month Treasury yield"}
    def render_pulse_col(col,ticker,label,prefix,is_yield):
        q=pulse_data.get(ticker)
        help_txt=_PULSE_HELP.get(label)
        if q:
            p=q["price"]; pct=q["pct"]
            if is_yield:    disp=f"{p:.2f}%"
            elif p>10000:   disp=f"{prefix}{p:,.0f}"
            elif p>100:     disp=f"{prefix}{p:,.2f}"
            else:           disp=f"{prefix}{p:.2f}"
            col.metric(label,disp,f"{pct:+.2f}%",help=help_txt)
        else:
            col.metric(label,"—","—",help=help_txt)

    cols1=st.columns(5)
    for col,(tk,lb,px,iy) in zip(cols1,PULSE_TICKERS[:5]):
        render_pulse_col(col,tk,lb,px,iy)
    cols2=st.columns(5)
    for col,(tk,lb,px,iy) in zip(cols2,PULSE_TICKERS[5:]):
        render_pulse_col(col,tk,lb,px,iy)

    st.divider()

    # Gauges
    stock_fg_score, stock_fg_rating   = fetch_cnn_fg()
    crypto_fg_score, crypto_fg_rating = fetch_crypto_fg()
    term_data = fetch_vix_term(); skew_val = fetch_skew()

    gcol1, gcol2, gcol3, gcol4 = st.columns([1.3, 1.3, 1.3, 1.1])

    with gcol1:
        if stock_fg_score is not None:
            st.plotly_chart(
                semicircle_gauge(stock_fg_score, "Stocks Fear & Greed",
                                 stock_fg_rating, "Source: CNN"),
                use_container_width=True)
        else:
            st.warning("CNN F&G unavailable")

    with gcol2:
        if crypto_fg_score is not None:
            st.plotly_chart(
                semicircle_gauge(crypto_fg_score, "Crypto Fear & Greed",
                                 crypto_fg_rating, "Source: Alternative.me"),
                use_container_width=True)
        else:
            st.warning("Crypto F&G unavailable")

    with gcol3:
        if vix_now is not None:
            st.plotly_chart(vix_gauge(vix_now), use_container_width=True)
        else:
            st.metric("VIX","—")

    with gcol4:
        st.markdown("**📊 Macro Signals**")
        tnx_q=pulse_data.get("^TNX"); irx_q=pulse_data.get("^IRX")
        _yc_tt=_tt("Yield Curve (10Y−3M)","10-year minus 3-month Treasury yield. Inverted "
                   "(negative) has historically preceded recessions — short-term debt yielding "
                   "more than long-term is the bond market pricing in future rate cuts.")
        if tnx_q and irx_q:
            spread=tnx_q["price"]-irx_q["price"]
            curve=("🟢 Normal" if spread>0.5 else "🟡 Flat" if spread>-0.3 else "🔴 Inverted")
            st.markdown(f"**{_yc_tt}:** {spread:+.2f}%  {curve}",unsafe_allow_html=True)
        else:
            st.markdown(f"**{_yc_tt}:** —",unsafe_allow_html=True)
        _skew_tt=_tt("SKEW","CBOE SKEW Index — prices the odds of a rare, sharp market drop "
                     "(tail risk) beyond what a normal distribution implies. Higher = more "
                     "tail-risk hedging demand priced into options.")
        if skew_val is not None:
            sk=("🔴 Elevated tail risk" if skew_val>145 else "🟡 Moderate" if skew_val>130 else "🟢 Low tail risk")
            st.markdown(f"**{_skew_tt}:** {skew_val:.1f}  {sk}",unsafe_allow_html=True)
        else:
            st.markdown(f"**{_skew_tt}:** —",unsafe_allow_html=True)
        if len(term_data)>=2:
            vals=list(term_data.values())
            if vals[-1]>vals[0]:
                shape="🟢 "+_tt("Contango","Longer-dated VIX futures pricier than near-term — "
                                "normal, calm-market shape")
            else:
                shape="🔴 "+_tt("Backwardation (stress)","Near-term VIX futures pricier than "
                                "longer-dated — the market is pricing immediate stress higher "
                                "than the future")
            st.markdown(f"**VIX Shape:** {shape}",unsafe_allow_html=True)

    st.divider()

    # ── SECTOR HEATMAP ─────────────────────────────────────────────────────────
    st.subheader("🟩 Sector Heatmap")
    st.caption("SPDR sector ETFs (live via Tradier) + Bitcoin as Digital Assets · colour intensity = move strength")

    sector_data = fetch_sector_quotes(tuple(ticker for ticker, *_ in SECTOR_TICKERS))
    sector_quotes = []
    for ticker, label, short in SECTOR_TICKERS:
        q = sector_data.get(ticker)
        sector_quotes.append({
            "label":  label,
            "ticker": short,
            "pct":    q["pct"] if q else None,
            "price":  q["price"] if q else None,
        })

    fig_sector = render_sector_heatmap(sector_quotes)
    st.plotly_chart(fig_sector, use_container_width=True)

    # Colour legend
    legend_cols = st.columns(10)
    legend_items = [
        ("#14532d", ">+3%"), ("#15803d", "+1–3%"), ("#4ade80", "0–+1%"),
        ("#334155", "n/a"),
        ("#f87171", "0–-1%"), ("#dc2626", "-1–-3%"), ("#7f1d1d", "<-3%"),
    ]
    for i, (col, (clr, lbl)) in enumerate(zip(legend_cols[1:-1], legend_items)):
        col.markdown(
            f"<div style='background:{clr};border-radius:4px;padding:3px 6px;"
            f"text-align:center;font-size:11px;color:white'>{lbl}</div>",
            unsafe_allow_html=True,
        )

    st.divider()

    # Watchlist table — height sized to show all rows without scrolling
    st.subheader("📋 Watchlist Overview")
    rows=[]
    for t,r in results.items():
        rows.append({"Ticker":t,"Price":f"${r['price']:.2f}","Chg %":f"{r['pct']:+.1f}%",
                     "HV%ile":fmt(r["hvpct"],".0f"),
                     "HV20":fmt(r["hv20"],".1f","%"),
                     "ATM IV C/P":f"{r['c_iv']:.0f}/{r['p_iv']:.0f}%" if (r.get("c_iv") and r.get("p_iv")) else "—",
                     "IV vs HV":iv_richness(r.get("c_iv"),r.get("p_iv"),r.get("hv20")),
                     "RSI":fmt(r["rsi"],".0f"),
                     "200MA":"✅" if r["ab200"] else "❌",
                     "PCR":fmt(r["pcr"],".2f"),
                     "Median":("🔼 Above" if r.get("pctb") is not None and r["pctb"]>=0.5
                               else "🔽 Below" if r.get("pctb") is not None else "—"),
                     "LEAP":r["leap"][0],"CC":r["cc"][0],"CSP":r["csp"][0]})
    if rows:
        tbl_height=38+len(rows)*35+4      # fits all rows exactly — no scrollbar
        # 22 June — same in-header hover-tooltip table as the Screener tab (st.dataframe's
        # header is a canvas-rendered grid and can't carry a real tooltip — see _html_table).
        _WATCH_LEGEND = [
            ("Ticker","Stock symbol"),
            ("Price","Current stock price"),
            ("Chg %","Today's percent change"),
            ("HV%ile","Historical volatility percentile vs its own 1-year range"),
            ("HV20","20-day historical (realized) volatility, annualized"),
            ("ATM IV C/P","At-the-money implied volatility — call / put"),
            ("IV vs HV","Is the premium worth selling? 🟢 Rich (fat premium for how much this "
                        "stock moves — good sell) · ⚪ Fair · 🔴 Cheap (underpaid — skip)"),
            ("RSI","Relative Strength Index (14) — momentum; <30 oversold, >70 overbought"),
            ("200MA","Price above (✅) or below (❌) its 200-day moving average — long-term trend"),
            ("PCR","Put/call volume ratio — elevated readings skew bearish"),
            ("Median","Price vs its 20-day median (midline) — Signal 1. CC needs 🔼 Above for a "
                      "green, CSP needs 🔽 Below. Shown next to the CC/CSP timing (Signal 2)."),
            ("LEAP","LEAP-buy timing signal (low-IV, oversold-leaning setup)"),
            ("CC","Covered-call timing signal (overbought-leaning setup to write calls)"),
            ("CSP","Cash-secured-put timing signal (oversold-bounce setup to sell puts)"),
        ]
        _html_table(rows, _WATCH_LEGEND, tbl_height)

    # ATM implied volatility per ticker (avg of call & put ATM IV) — hotter = richer premium
    # to sell, cheaper to buy. Real Tradier IV.
    def _fin(x): return isinstance(x,(int,float)) and math.isfinite(x) and x>0
    iv_data={t:(r["c_iv"]+r["p_iv"])/2 for t,r in results.items()
             if _fin(r.get("c_iv")) and _fin(r.get("p_iv"))}
    if iv_data:
        st.subheader("ATM Implied Volatility")
        ivv=list(iv_data.values())
        fig_iv=go.Figure(go.Bar(x=list(iv_data.keys()),y=ivv,
            marker=dict(color=ivv,colorscale="YlOrRd",showscale=False),
            text=[f"{v:.0f}%" for v in ivv],textposition="outside"))
        fig_iv.update_layout(height=320,template="plotly_dark",yaxis_title="ATM IV (%)",
                              margin=dict(l=0,r=0,t=20,b=0))
        st.plotly_chart(fig_iv,use_container_width=True)

    rsi_data={t:r["rsi"] for t,r in results.items() if r["rsi"] is not None}
    if rsi_data:
        st.subheader("RSI Snapshot")
        rsi_colors=["#22c55e" if 33<=v<=52 else "#eab308" if 52<v<=65 else "#f97316" if v>65 else "#94a3b8"
                    for v in rsi_data.values()]
        fig_rsi=go.Figure(go.Bar(x=list(rsi_data.keys()),y=list(rsi_data.values()),
            marker_color=rsi_colors,text=[f"{v:.0f}" for v in rsi_data.values()],textposition="outside"))
        for level,color,label in [(30,"#22c55e","30"),(50,"#94a3b8","50"),(70,"#ef4444","70")]:
            fig_rsi.add_hline(y=level,line_dash="dash",line_color=color,annotation_text=label)
        fig_rsi.update_layout(height=280,template="plotly_dark",yaxis_title="RSI (14)",
                               yaxis_range=[0,115],margin=dict(l=0,r=0,t=20,b=0))
        st.plotly_chart(fig_rsi,use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — DEEP DIVE
# ══════════════════════════════════════════════════════════════════════════════
with tab_dive:
    sel=st.selectbox("Select Ticker",list(results.keys()),key="dd_sel")
    if sel and sel in results:
        r=results[sel]; df=r["df"]; cl=r["cl"]
        # ── Childlike-simple: a row of chips (the verdicts). Every detail lives on mouseover;
        #    no tables, no dropdowns. Price · IV-vs-Realized (is the premium worth selling?) ·
        #    LEAP · CC · CSP (GO / WAIT). Hover any chip for the "why". ──
        st.markdown("""<style>
        .dd-chip{position:relative;display:inline-block;padding:11px 18px;border-radius:12px;
          font-weight:700;font-size:15px;color:#fff;margin:4px 8px 6px 0;cursor:default;}
        .dd-chip .dd-tip{visibility:hidden;opacity:0;position:absolute;left:0;top:calc(100% + 8px);
          z-index:9999;width:280px;background:#0f172a;color:#e2e8f0;border:1px solid #334155;
          border-radius:8px;padding:11px 13px;font-weight:400;font-size:13px;line-height:1.55;
          text-align:left;box-shadow:0 8px 24px rgba(0,0,0,.55);transition:opacity .12s;}
        .dd-chip:hover .dd-tip{visibility:visible;opacity:1;}
        </style>""", unsafe_allow_html=True)

        def _chip(text, color, tip):
            return (f'<span class="dd-chip" style="background:{color}">{text}'
                    f'<span class="dd-tip">{tip}</span></span>')

        _chips = []
        # Price (numbers you used to see as tiles now live in this chip's hover)
        _chips.append(_chip(
            f"{sel}  ${r['price']:.2f}  <span style='opacity:.8;font-weight:400'>{r['pct']:+.1f}%</span>",
            "#334155",
            f"ATM IV — call {fmt(r['c_iv'],'.0f','%')} / put {fmt(r['p_iv'],'.0f','%')}<br>"
            f"RSI(14) {fmt(r['rsi'],'.0f')} &nbsp;·&nbsp; PCR {fmt(r['pcr'],'.2f')} &nbsp;·&nbsp; "
            f"HV%ile {fmt(r['hvpct'],'.0f','%')}"))
        # IV vs Realized — the headline "worth selling?" read
        _prem = iv_richness(r.get("c_iv"), r.get("p_iv"), r.get("hv20"))
        _prem_color = "#16a34a" if "Rich" in _prem else ("#7f1d1d" if "Cheap" in _prem else "#475569")
        _chips.append(_chip(
            f"IV vs Realized · {_prem}", _prem_color,
            "Is the premium worth selling? Implied vol vs how much the stock actually moves. "
            "🟢 fat premium (good sell) · ⚪ ok · 🔴 thin (underpaid — skip)."))
        # Strategy verdicts
        for key, name in [("leap","LEAP"), ("cc","CC"), ("csp","CSP")]:
            label, _score, reasons = r[key]
            color, verdict = ("#16a34a","GO") if label.startswith("🟢") else \
                             ("#a16207","WAIT") if label.startswith("🟡") else ("#475569","WAIT")
            tip = f"<b>{name}: {label}</b><br>"
            if key in ("cc","csp"):
                tip += median_chip(r.get("pctb"), key) + "<br>"
            tip += "<br>".join(reasons) if reasons else "Not enough data yet."
            _chips.append(_chip(f"{name} · {verdict}", color, tip))
        st.markdown("<div style='margin:8px 0 4px'>" + "".join(_chips) + "</div>",
                    unsafe_allow_html=True)
        st.divider()
        bb_upper,bb_mid,bb_lower=calc_bb_bands(cl)
        vol=df["Volume"].squeeze()
        op=df["Open"].squeeze()
        vol_colors=["#26a69a" if cl.iloc[i]>=op.iloc[i] else "#ef5350" for i in range(len(cl))]
        fig=make_subplots(rows=4,cols=1,shared_xaxes=True,row_heights=[0.40,0.15,0.20,0.25],
            subplot_titles=[f"{sel} — Price, MAs & Bollinger Bands","Volume",
                            "HV20 / HV60 — Realized Vol","RSI (14)"],vertical_spacing=0.04)
        fig.add_trace(go.Scatter(x=df.index,y=bb_upper,name="BB Bands (20,2)",
            line=dict(color="#64748b",width=1,dash="dot"),showlegend=False),row=1,col=1)
        fig.add_trace(go.Scatter(x=df.index,y=bb_lower,name="BB Bands (20,2)",fill="tonexty",
            line=dict(color="#64748b",width=1,dash="dot"),fillcolor="rgba(100,116,139,0.08)"),row=1,col=1)
        fig.add_trace(go.Scatter(x=df.index,y=bb_mid,name="BB Mid (20MA)",
            line=dict(color="#cbd5e1",width=1,dash="dash")),row=1,col=1)
        fig.add_trace(go.Candlestick(x=df.index,open=op,high=df["High"].squeeze(),
            low=df["Low"].squeeze(),close=cl,name="Price",
            increasing_line_color="#26a69a",decreasing_line_color="#ef5350",
            increasing_fillcolor="#26a69a",decreasing_fillcolor="#ef5350"),row=1,col=1)
        fig.add_trace(go.Scatter(x=df.index,y=cl.rolling(50).mean(),name="50MA",
            line=dict(color="#f97316",width=1.4)),row=1,col=1)
        fig.add_trace(go.Scatter(x=df.index,y=cl.rolling(200).mean(),name="200MA",
            line=dict(color="#60a5fa",width=1.6)),row=1,col=1)
        fig.add_trace(go.Bar(x=df.index,y=vol,name="Volume",marker_color=vol_colors,
            opacity=0.7,showlegend=False),row=2,col=1)
        fig.add_trace(go.Scatter(x=df.index,y=vol.rolling(20).mean(),name="Vol 20MA",
            line=dict(color="#e2e8f0",width=1.2)),row=2,col=1)
        fig.add_trace(go.Scatter(x=df.index,y=r["hv20_s"],name="HV20",fill="tozeroy",
            line=dict(color="#a78bfa",width=1.5),fillcolor="rgba(167,139,250,0.12)"),row=3,col=1)
        fig.add_trace(go.Scatter(x=df.index,y=r["hv60_s"],name="HV60",
            line=dict(color="#7c3aed",width=1,dash="dot")),row=3,col=1)
        fig.add_trace(go.Scatter(x=df.index,y=r["rsi_s"],name="RSI",
            line=dict(color="#fbbf24",width=1.5),showlegend=False),row=4,col=1)
        for lvl,col in [(70,"#ef4444"),(50,"#94a3b8"),(30,"#22c55e")]:
            fig.add_hline(y=lvl,line_dash="dash",line_color=col,row=4,col=1)
        fig.update_layout(height=940,template="plotly_dark",xaxis_rangeslider_visible=False,
                          legend=dict(orientation="h",y=1.08,yanchor="bottom",x=0,font=dict(size=11)),
                          margin=dict(l=0,r=0,t=110,b=0))
        # 22 June — leave ~10 calendar days (≈7-8 trading days) of empty space after the last
        # candle so it's visually obvious nothing is cut off at the right edge (bumped 5→10
        # per Jay — 5 wasn't quite enough). update_xaxes with no row/col targets every xaxis
        # (all 4 stacked rows share this range since shared_xaxes=True), so price/volume/HV/
        # RSI stay aligned.
        fig.update_xaxes(range=[df.index.min(), df.index.max()+pd.Timedelta(days=10)])
        st.plotly_chart(fig,use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — OPTIONS CHAIN
# ══════════════════════════════════════════════════════════════════════════════
with tab_chain:
    st.caption("Click a ticker tile, then an expiry tile — no dropdowns.")
    tickers_avail=[t for t in results if results[t].get("all_exps")]
    if not tickers_avail:
        st.warning("No options data loaded.")
    else:
        st.markdown("**Ticker**")
        cur_tkr=st.session_state.get("chain_tkr")
        n_cols=8
        for i in range(0,len(tickers_avail),n_cols):
            cols=st.columns(n_cols)
            for c,tkr in zip(cols,tickers_avail[i:i+n_cols]):
                with c:
                    if st.button(tkr,key=f"tkrtile_{tkr}",use_container_width=True,
                                 type="primary" if tkr==cur_tkr else "secondary"):
                        st.session_state["chain_tkr"]=tkr
                        st.session_state.pop("chain_exp",None)
                        st.rerun()
        cur_tkr=st.session_state.get("chain_tkr")
        if not cur_tkr:
            st.info("Pick a ticker above.")
        else:
            r=results[cur_tkr]; price=r["price"]; all_exps=r.get("all_exps",[])
            # analyse()'s yfinance reference price can come back NaN (a Yahoo hiccup on this
            # ticker). That NaN would break moneyness, the chart vlines, and the header — so
            # fall back to Tradier's real-time last, which we already have on hand.
            if price is None or (isinstance(price,float) and math.isnan(price)):
                _t_last=fetch_underlying_last(cur_tkr) if tradier.is_configured() else None
                if _t_last is not None: price=_t_last
            today_p=datetime.utcnow()
            _price_txt=f"${price:,.2f}" if (price is not None and not (isinstance(price,float) and math.isnan(price))) else "n/a"
            st.markdown(f"**Expiry — {cur_tkr} ({_price_txt})**")
            cur_exp=st.session_state.get("chain_exp")
            n_cols_e=6
            for i in range(0,len(all_exps),n_cols_e):
                cols=st.columns(n_cols_e)
                for c,exp in zip(cols,all_exps[i:i+n_cols_e]):
                    with c:
                        try: dte_e=(datetime.strptime(exp,"%Y-%m-%d")-today_p).days
                        except Exception: dte_e=None
                        label=f"{exp} ({dte_e}d)" if dte_e is not None else exp
                        if st.button(label,key=f"exptile_{cur_tkr}_{exp}",use_container_width=True,
                                     type="primary" if exp==cur_exp else "secondary"):
                            st.session_state["chain_exp"]=exp
                            st.rerun()
            sel_c=cur_tkr; selected_exp=st.session_state.get("chain_exp")
            if not selected_exp:
                st.info("Pick an expiry above.")
            else:
                # Prefer Tradier's real IV/Greeks when the token is configured; fall back to
                # Yahoo (calculated IV, no Greeks) so the tab always works.
                if tradier.is_configured():
                    calls_df,puts_df,dte,chain_err=fetch_chain_tradier(sel_c,selected_exp)
                    _chain_src="🟢 **Tradier** — real IV & Greeks (ORATS)"
                    if calls_df is None:
                        calls_df,puts_df,dte,_fb_err=fetch_chain_cached(sel_c,selected_exp)
                        _chain_src=(f"🟡 **Yahoo** (Tradier fetch failed, fell back — {chain_err})")
                        chain_err=_fb_err
                else:
                    calls_df,puts_df,dte,chain_err=fetch_chain_cached(sel_c,selected_exp)
                    _chain_src="🟡 **Yahoo** — calculated IV, no Greeks · add a Tradier token for real data"
                st.caption(f"Data source: {_chain_src}")
                if calls_df is not None:
                    chain=type("_C",(),{"calls":calls_df,"puts":puts_df})()
                    # Single price shown left of DTE (the underlying's last, via analyse() with a
                    # Tradier real-time fallback set above). NOTE: dollar signs are escaped as
                    # "\$" — Streamlit markdown treats an unescaped "$…$" pair as LaTeX math and
                    # would swallow the HTML between them.
                    def _px(v):
                        return (f"\\${v:,.2f}" if v is not None and
                                not (isinstance(v,float) and math.isnan(v)) else "n/a")
                    st.markdown(f"<span style='font-size:1.9rem;font-weight:700;'>{_px(price)}</span>"
                                f"&nbsp;&nbsp;·&nbsp;&nbsp;"
                                f"<span style='font-size:1.9rem;font-weight:700;'>{dte} DTE</span>",
                                unsafe_allow_html=True)
                    # Lean, seller-focused columns (Jay's declutter): just what picks a strike
                    # to sell — Strike · Δ · Bid · Ask · IV% · OI. Dropped Last / Moneyness /
                    # Volume (Δ already tells you moneyness; you sell at the bid, not last).
                    def fmt_chain(df_raw):
                        df_raw=df_raw.copy()
                        df_raw["IV %"]=(df_raw["impliedVolatility"]*100).round(1)
                        cols=["strike","delta","bid","ask","IV %","openInterest"]
                        available=[c for c in cols if c in df_raw.columns]
                        return(df_raw[available]
                               .rename(columns={"strike":"Strike","delta":"Δ","bid":"Bid",
                                                 "ask":"Ask","openInterest":"OI"})
                               .sort_values("Strike").reset_index(drop=True))
                    _CHAIN_LEGEND=[("Strike","Option strike · 🎯 marks the ≈30-delta strike — the usual CSP/CC sell target"),
                                    ("Δ","Delta — roughly the chance it finishes in-the-money; ~0.30 is the common sell target"),
                                    ("Bid","What you'd collect selling here (the premium)"),
                                    ("Ask","Ask price — the Bid↔Ask gap is fill quality (tighter = easier fill)"),
                                    ("IV %","Implied volatility for this strike"),
                                    ("OI","Open interest — contracts outstanding; higher = more liquid, easier fills")]
                    def _target_strike(df_fmt):
                        if "Δ" not in df_fmt.columns: return None
                        d=(df_fmt["Δ"].abs()-0.30).abs()
                        return df_fmt.loc[d.idxmin(),"Strike"] if not d.dropna().empty else None
                    def _chain_html_rows(df_fmt, target=None):
                        cols=[l for l,_ in _CHAIN_LEGEND]
                        rows=[]
                        for _,row in df_fmt.iterrows():
                            d={}
                            for c in cols:
                                if c not in df_fmt.columns:
                                    d[c]="—"; continue
                                v=row[c]
                                if pd.isna(v):            d[c]="—"
                                elif c=="Strike":         d[c]=("🎯 " if (target is not None and v==target) else "")+f"${v:.2f}"
                                elif c in ("Bid","Ask"):  d[c]=f"${v:.2f}"
                                elif c=="IV %":           d[c]=f"{v:.1f}%"
                                elif c=="Δ":              d[c]=f"{v:.2f}"
                                elif c=="OI":             d[c]=f"{int(v):,}"
                                else:                     d[c]=v
                            rows.append(d)
                        return rows
                    col_c,col_p=st.columns(2)
                    with col_c:
                        st.subheader("Calls")
                        calls_fmt=fmt_chain(chain.calls)
                        _html_table(_chain_html_rows(calls_fmt,_target_strike(calls_fmt)),
                                    _CHAIN_LEGEND,min(38+len(calls_fmt)*35+12,520))
                    with col_p:
                        st.subheader("Puts")
                        puts_fmt=fmt_chain(chain.puts)
                        _html_table(_chain_html_rows(puts_fmt,_target_strike(puts_fmt)),
                                    _CHAIN_LEGEND,min(38+len(puts_fmt)*35+12,520))
                    # Optional context — folded away so the default view is just pickers + table.
                    with st.expander("📊 Skew & liquidity — IV smile · open interest (optional)"):
                        fig_smile=go.Figure()
                        fig_smile.add_trace(go.Scatter(x=chain.calls["strike"],y=chain.calls["impliedVolatility"]*100,
                            name="Calls IV",mode="lines+markers",line=dict(color="#26a69a",width=2),marker=dict(size=5)))
                        fig_smile.add_trace(go.Scatter(x=chain.puts["strike"], y=chain.puts["impliedVolatility"]*100,
                            name="Puts IV", mode="lines+markers",line=dict(color="#ef5350",width=2),marker=dict(size=5)))
                        fig_smile.add_vline(x=price,line_dash="dash",line_color="white",annotation_text=f"${price:.2f}")
                        fig_smile.update_layout(height=320,template="plotly_dark",xaxis_title="Strike",
                                                yaxis_title="IV (%)",title="IV Smile",margin=dict(l=0,r=0,t=30,b=0))
                        st.plotly_chart(fig_smile,use_container_width=True)
                        fig_oi=go.Figure()
                        fig_oi.add_trace(go.Bar(x=chain.calls["strike"],y=chain.calls["openInterest"],name="Call OI",marker_color="#26a69a",opacity=0.75))
                        fig_oi.add_trace(go.Bar(x=chain.puts["strike"], y=chain.puts["openInterest"],name="Put OI", marker_color="#ef5350",opacity=0.75))
                        fig_oi.add_vline(x=price,line_dash="dash",line_color="white")
                        fig_oi.update_layout(barmode="overlay",height=300,template="plotly_dark",title="Open Interest by Strike",
                                             xaxis_title="Strike",yaxis_title="OI",margin=dict(l=0,r=0,t=30,b=0))
                        st.plotly_chart(fig_oi,use_container_width=True)
                        st.markdown("""
**IV Smile**

What it shows: implied volatility (y-axis) by strike (x-axis) for this expiry — green line is
calls, red is puts, the dashed white line marks the current price.

What to look for: IV usually curves upward as strikes move away from the money in either
direction (the "smile"). On most equities the put side curves up faster than the call side —
the market pays more for downside protection than upside speculation. A steep put-side skew
(puts well above calls at the same distance from price) means crash insurance is expensive
right now — good news if you're selling CSPs (richer premium for the risk you're taking),
worse news if you're buying puts for protection. A flat smile means both tails are priced
similarly — calmer market, less skew to exploit. Where your target strike sits on the curve
tells you if you're selling rich or cheap relative to ATM, not just relative to the stock's
own history.

**Open Interest by Strike**

What it shows: total open contracts (y-axis) by strike (x-axis) — green bars are calls, red
bars are puts, dashed white line marks current price.

What to look for: large OI clusters mark strikes where a lot of positions are already
parked — these often act as informal support/resistance into expiry, since market makers
hedge those positions and that flow can pin price near a big strike as DTE shrinks ("gamma
pin"). A heavy put-OI wall below price can act like a floor; a heavy call-OI wall above price
can act like a lid on rallies. Separately, OI at your specific strike is a liquidity check —
thin OI usually means wide bid/ask spreads and harder fills, so all else equal favor strikes
with real OI behind them over a strike that's technically "perfect" on delta but empty on
interest.
                        """)
                else:
                    st.warning(f"Could not load chain for this expiry."
                               + (f" ({chain_err})" if chain_err else "")
                               + " Try again in a moment — usually a transient data-feed hiccup, "
                                 "not cached, so a retry helps.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — MARKET VOLATILITY
# ══════════════════════════════════════════════════════════════════════════════
with tab_vix:
    # ── Rates & the yield curve — the bond-market backdrop for everything else ──
    st.subheader("🏦 Rates & the Yield Curve")
    _curve=fetch_yield_curve()
    _fed=fetch_fed_rate()
    _cd=dict(_curve)
    _spread=macro.curve_spread_2s10s(_curve)
    rc1,rc2,rc3,rc4=st.columns(4)
    rc1.metric("Fed funds (effective)", f"{_fed:.2f}%" if _fed is not None else "—",
               help="The Fed's policy rate — the floor under all other rates. Hiking = tightening "
                    "(headwind for stocks, vol up); cutting = easing (tailwind).")
    rc2.metric("2Y Treasury", f"{_cd['2Y']:.2f}%" if '2Y' in _cd else "—",
               help="Short end — tracks where the market expects the Fed to be in ~2 years.")
    rc3.metric("10Y Treasury", f"{_cd['10Y']:.2f}%" if '10Y' in _cd else "—",
               help="The benchmark long rate — growth & inflation expectations, and the discount "
                    "rate equities are valued against. Rising 10Y pressures high-multiple stocks.")
    _move=fetch_quote("^MOVE")
    _move_v=_move.get("price") if _move else None
    _move_c=_move.get("pct") if _move else None
    rc4.metric("MOVE (bond VIX)", f"{_move_v:.0f}" if _move_v else "—",
               f"{_move_c:+.1f}%" if _move_c is not None else None,
               help="The MOVE index — implied volatility of US Treasury options, i.e. the bond "
                    "market's VIX. Rising = rate/bond-market stress, and it often LEADS equity vol "
                    "(VIX), so a climbing MOVE is an early warning for premium sellers. Rough bands: "
                    "calm <90 · normal 90–120 · elevated 120–150 · stress >150. (The 2s10s curve "
                    "read is still shown below the chart.)")
    if len(_curve)>=3:
        _xs=[t for t,_ in _curve]; _ys=[y for _,y in _curve]
        _line="#ef4444" if (_spread is not None and _spread<0) else "#3b82f6"
        fig_yc=go.Figure(go.Scatter(x=_xs,y=_ys,mode="lines+markers",
            line=dict(color=_line,width=2.5),marker=dict(size=8),
            text=[f"{y:.2f}%" for y in _ys],hovertemplate="%{x}: %{y:.2f}%<extra></extra>"))
        fig_yc.update_layout(height=280,template="plotly_dark",yaxis_title="Yield (%)",
                             xaxis_title="Maturity",margin=dict(l=0,r=0,t=20,b=0))
        st.plotly_chart(fig_yc,use_container_width=True)
    if _spread is not None:
        if _spread<0:
            st.caption("🔴 **Inverted curve** — short rates above long rates. The bond market is "
                       "pricing rate cuts / recession; a classic warning that leads downturns by "
                       "6–18 months. For a premium seller, inversions often precede vol spikes — "
                       "keep size modest.")
        elif _spread<25:
            st.caption("🟡 **Flat curve** — little gap between short and long rates. Late-cycle / "
                       "uncertain; watch for it tipping into inversion.")
        else:
            st.caption("🟢 **Normal (upward) curve** — long rates above short. Healthy growth "
                       "expectations, the market's default state.")
    else:
        st.caption("Yield-curve data unavailable right now (FRED source).")
    st.divider()

    # ── S&P 500 valuation (P/E) — how expensive is the market you're selling premium into ──
    st.subheader("S&P 500 Valuation")
    _pe_t,_pe_f=fetch_sp500_pe()
    pc1,pc2,pc3=st.columns(3)
    pc1.metric("S&P 500 P/E (trailing)", f"{_pe_t:.1f}" if _pe_t else "—",
               help="Price ÷ trailing 12-month earnings for the S&P 500 (via SPY). The market's "
                    "valuation — how much you pay per $1 of earnings.")
    pc2.metric("S&P 500 P/E (forward)", f"{_pe_f:.1f}" if _pe_f else "—",
               help="Price ÷ next-12-month expected earnings. Lower than trailing when earnings "
                    "are expected to grow.")
    if _pe_t:
        if   _pe_t>=25: _pe_short,_pe_full="🔴 Rich","Expensive — stretched vs the long-run average (~16–17)"
        elif _pe_t>=20: _pe_short,_pe_full="🟡 Above avg","Above the long-run average (~16–17)"
        else:           _pe_short,_pe_full="🟢 Fair","Around or below the long-run average (~16–17)"
        pc3.metric("Read", _pe_short, help=_pe_full)
        st.caption("Long-run average S&P 500 P/E is ~16–17. A rich market has less cushion — "
                   "relevant to a CSP seller who could end up owning the shares.")
    st.divider()
    # ── Variance Risk Premium: VIX (30-day implied) − S&P 20-day realized vol. The market-wide
    #    "are options overpriced vs reality?" edge a premium seller harvests. Kept simple. ──
    st.subheader("Variance Risk Premium — the seller's edge")
    _spy_df=fetch_prices("SPY","6mo")
    _sp_rv=None
    if _spy_df is not None and not _spy_df.empty:
        _rvs=calc_hv(_spy_df["Close"].squeeze(),20).dropna()
        _sp_rv=float(_rvs.iloc[-1]) if not _rvs.empty else None
    if vix_now is not None and _sp_rv:
        _vrp=vix_now-_sp_rv
        _vrp_read=("🟢 Rich — options overpriced vs reality; the edge is there to sell" if _vrp>=4 else
                   "🟡 Thin edge — premium only a little above actual movement" if _vrp>=0 else
                   "🔴 Negative — options cheaper than actual moves; careful selling")
        v1,v2,v3=st.columns(3)
        v1.metric("VIX (30-day implied)", f"{vix_now:.1f}",
                  help="What the market EXPECTS the S&P to move over the next 30 days "
                       "(annualized %), priced into option premiums.")
        v2.metric("S&P realized (20-day)", f"{_sp_rv:.1f}",
                  help="What the S&P ACTUALLY moved over the last 20 days (annualized %).")
        v3.metric("VRP (implied − realized)", f"{_vrp:+.1f}",
                  help="The gap, in volatility points. +6 means options price in 6 points more "
                       "movement than the S&P is actually delivering — that extra is the cushion "
                       "you pocket for selling. Bigger + = fatter edge. Negative = options cheaper "
                       "than real moves → you'd be underpaid, don't sell.")
        st.caption(f"{_vrp_read}.  VRP is in volatility points: it's how much *more* the market is "
                   "pricing in (VIX) than the S&P is *actually* moving. Positive = you're paid a "
                   "premium over reality to sell — the edge. Hover any number for detail.")
    else:
        st.caption("VRP unavailable (need both VIX and S&P realized vol).")
    st.divider()
    st.subheader("VIX — Volatility Regime")
    st.caption("VIX is implied volatility, not historical — it's priced off S&P 500 options "
               "and represents what the market expects annualized volatility to be over the "
               "**next 30 days specifically**. That's why it jumps before an event (Fed, "
               "earnings) even before anything's happened — the next 30 days' uncertainty is "
               "already baked into the option prices it's built from.")
    if vix_df is not None and not vix_df.empty:
        vix_cl=vix_df["Close"].squeeze()
        c1,c2,c3,c4=st.columns(4)
        c1.metric("Current VIX",f"{vix_now:.1f}",f"{vix_chg:+.2f}",
                   help="CBOE Volatility Index — implied volatility priced off S&P 500 options "
                        "for the next 30 days (see caption above)")
        c2.metric("52wk High",f"{vix_cl.max():.1f}",help="Highest VIX close in the last year")
        c3.metric("52wk Low",f"{vix_cl.min():.1f}",help="Lowest VIX close in the last year")
        c4.metric("52wk Avg",f"{vix_cl.mean():.1f}",help="Average VIX close over the last year")
        st.markdown("""
| VIX | Regime | LEAP | CC | CSP |
|---|---|---|---|---|
| <15 | Low | Best — cheapest premium | Thin | Good if trend up |
| 15–20 | Normal | Decent | Moderate | OK |
| 20–30 | Elevated | Expensive — selective | Rich | Rich |
| >30 | Fear | Very expensive — wait | Maximum | High risk |
        """)
    else:
        st.error("Could not load VIX data.")

    st.divider()
    st.subheader("Complementary Volatility Gauges")
    st.caption("VIX measures equity fear. These extend the picture to oil and gold — useful "
               "since stress can start outside equities and spread in, and gold vol directly "
               "prices your GLD premium.")

    def _vol_gauge(col,label,full_name,df_v,bands,note):
        with col:
            if df_v is not None and not df_v.empty:
                cl_v=df_v["Close"].squeeze()
                now_v=float(cl_v.iloc[-1]); prev_v=float(cl_v.iloc[-2]) if len(cl_v)>1 else now_v
                st.metric(label,f"{now_v:.1f}",f"{now_v-prev_v:+.2f}",help=full_name)
                band_lines="\n".join(f"| {b[0]} | {b[1]} |" for b in bands)
                st.markdown(f"*{full_name}*\n\n| Level | Regime |\n|---|---|\n{band_lines}")
                st.caption(note)
            else:
                st.warning(f"{label} data unavailable.")

    gcol1,gcol2=st.columns(2)
    ovx_df=fetch_ovx("1y"); gvz_df=fetch_gvz("1y")
    _vol_gauge(gcol1,"OVX","CBOE Crude Oil ETF Volatility Index",ovx_df,
        [("<25","Calm"),("25–40","Normal"),("40–60","Elevated"),(">60","Extreme — supply-shock territory")],
        "Spikes hard on supply shocks (2020 negative oil prices, 2022 invasion) — watch if energy names are on the watchlist.")
    _vol_gauge(gcol2,"GVZ","CBOE Gold ETF Volatility Index",gvz_df,
        [("<14","Calm"),("14–18","Normal"),("18–24","Elevated"),(">24","Extreme — usually a flight-to-safety spike")],
        "A GVZ spike means your GLD option premium just got richer — and usually signals something macro breaking.")

    st.divider()
    # ── This week's economic calendar (ForexFactory weekly JSON, rendered natively) ──
    st.subheader("📅 This Week — Economic Calendar")
    st.caption("Scheduled high-impact events (via ForexFactory) — Fed decisions, CPI, jobs — the "
               "ones that move the whole market. Don't open fresh premium right before a big red one.")
    _events=fetch_econ_calendar()
    if not _events:
        st.info("Calendar unavailable right now (ForexFactory feed).")
    else:
        _show_med=st.toggle("Include medium-impact",value=False,key="cal_med")
        _want={"High"} | ({"Medium"} if _show_med else set())
        _cal=[]
        for e in _events:
            if e.get("impact") not in _want:
                continue
            _dt=None
            try:
                _dt=datetime.fromisoformat(str(e.get("date","")).replace("Z","+00:00"))
            except Exception:
                pass
            _cal.append((_dt,e))
        _cal.sort(key=lambda t:(t[0] is None, t[0].timestamp() if t[0] else 9e18))
        if not _cal:
            st.caption("No high-impact events match this week.")
        else:
            _calrows=[]
            for _dt,e in _cal:
                _calrows.append({
                    "Day":_dt.strftime("%a %d") if _dt else "—",
                    "Time":_dt.strftime("%H:%M") if _dt else "",
                    "Cur":e.get("country","") or "",
                    "Impact":"🔴" if e.get("impact")=="High" else "🟠",
                    "Event":e.get("title","") or "",
                    "Forecast":e.get("forecast","") or "—",
                    "Previous":e.get("previous","") or "—"})
            st.dataframe(pd.DataFrame(_calrows),use_container_width=True,hide_index=True)
            st.caption("Times as provided by ForexFactory (US Eastern). 🔴 high · 🟠 medium impact. "
                       "Refreshes hourly.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB — FUNDAMENTALS (SEC EDGAR: real filings, no prices/technicals)
# ══════════════════════════════════════════════════════════════════════════════
with tab_fund:
    st.subheader("🔬 Fundamentals")
    st.caption("Straight from the company's SEC filings (10-K / 10-Q XBRL) — no prices, no "
               "technicals. Valuation · quality · balance-sheet health · red flags. US-listed filers.")

    # tiny formatters — keep the numbers human
    def _fx_money(v):
        if v is None: return "—"
        a=abs(v); s="-" if v<0 else ""
        if a>=1e12: return f"{s}${a/1e12:.2f}T"
        if a>=1e9:  return f"{s}${a/1e9:.2f}B"
        if a>=1e6:  return f"{s}${a/1e6:.1f}M"
        return f"{s}${a:,.0f}"
    def _fx_pct(v):  return "—" if v is None else f"{v*100:.1f}%"
    def _fx_x(v):    return "—" if v is None else f"{v:.2f}×"
    def _fx_num(v,d=2): return "—" if v is None else f"{v:.{d}f}"

    st.markdown("""<style>
    .fx-chip{position:relative;display:inline-block;padding:12px 20px;border-radius:12px;
      font-weight:700;font-size:15px;color:#fff;margin:4px 10px 6px 0;cursor:default;min-width:150px;
      text-align:center;box-shadow:0 4px 14px rgba(0,0,0,.28);}
    .fx-chip .fx-tip{visibility:hidden;opacity:0;position:absolute;left:0;top:calc(100% + 8px);
      z-index:9999;width:250px;background:#0f172a;color:#e2e8f0;border:1px solid #334155;
      border-radius:8px;padding:11px 13px;font-weight:400;font-size:13px;line-height:1.7;
      text-align:left;box-shadow:0 8px 24px rgba(0,0,0,.55);transition:opacity .12s;}
    .fx-chip:hover .fx-tip{visibility:visible;opacity:1;}

    /* search box */
    .st-key-fx_input input{background:#0f172a!important;border:1px solid #334155!important;
      border-radius:12px!important;color:#e2e8f0!important;font-size:16px!important;
      padding:12px 14px!important;}
    .st-key-fx_input input:focus{border-color:#3b82f6!important;
      box-shadow:0 0 0 3px rgba(59,130,246,.22)!important;}
    /* Analyse button — accent gradient */
    .st-key-fx_go button{background:linear-gradient(135deg,#3b82f6,#2563eb)!important;
      border:none!important;color:#fff!important;font-weight:700!important;border-radius:12px!important;
      height:46px!important;box-shadow:0 6px 18px rgba(37,99,235,.35)!important;}
    .st-key-fx_go button:hover{filter:brightness(1.09);transform:translateY(-1px);}
    /* watchlist ticker pills */
    [class*="st-key-fxq_"] button{background:#1e293b!important;border:1px solid #334155!important;
      color:#cbd5e1!important;border-radius:999px!important;font-weight:600!important;
      font-size:13px!important;padding:5px 6px!important;transition:all .12s!important;
      box-shadow:none!important;}
    [class*="st-key-fxq_"] button:hover{background:#273449!important;border-color:#3b82f6!important;
      color:#fff!important;transform:translateY(-2px);}

    /* company hero */
    .fx-hero{background:linear-gradient(135deg,#0b1220 0%,#152036 52%,#25324d 100%);
      border:1px solid #2c3a52;border-radius:18px;padding:20px 24px;margin:4px 0 14px;
      box-shadow:0 12px 34px rgba(0,0,0,.40);}
    .fx-hero-top{display:flex;justify-content:space-between;align-items:flex-start;gap:16px;
      flex-wrap:wrap;}
    .fx-hero-name{font-size:25px;font-weight:800;color:#f8fafc;line-height:1.15;}
    .fx-hero-sub{font-size:12.5px;color:#93a3b8;margin-top:6px;letter-spacing:.2px;}
    .fx-hero-right{text-align:right;white-space:nowrap;}
    .fx-hero-ticker{display:inline-block;background:#2563eb;color:#fff;font-weight:800;
      font-size:14px;letter-spacing:.6px;padding:4px 12px;border-radius:8px;}
    .fx-hero-price{font-size:21px;font-weight:700;color:#e2e8f0;margin-top:9px;}
    .fx-hero-dots{margin-top:9px;font-size:12px;color:#94a3b8;}
    .fx-hero-dots b{color:#e2e8f0;font-weight:600;}

    /* perma-visible short summary */
    .fx-summary{background:#0f172a;border:1px solid #24314a;border-radius:12px;
      padding:15px 18px;margin:0 0 12px;color:#dbe4f0;font-size:16.5px;line-height:1.62;}
    .fx-summary-tag{display:inline-block;background:#0b2740;color:#7dd3fc;font-size:11px;
      font-weight:700;letter-spacing:.4px;padding:2px 9px;border-radius:6px;margin-right:9px;
      vertical-align:middle;}

    /* news cards */
    .fx-news a{display:block;text-decoration:none;background:#0f172a;border:1px solid #1e2a3f;
      border-left:3px solid #3b82f6;border-radius:10px;padding:10px 14px;margin:7px 0;
      transition:all .12s;}
    .fx-news a:hover{background:#111c33;border-left-color:#60a5fa;transform:translateX(3px);
      box-shadow:0 6px 18px rgba(0,0,0,.30);}
    .fx-news .t{color:#e5edf7;font-weight:600;font-size:14px;line-height:1.4;}
    .fx-news .m{color:#64748b;font-size:12px;margin-top:4px;}

    /* overall verdict bar inside the hero */
    .fx-hero-verdict{margin-top:14px;display:flex;align-items:center;gap:14px;flex-wrap:wrap;
      padding:9px 16px;border-radius:12px;box-shadow:0 6px 18px rgba(0,0,0,.35);}
    .fx-ov{font-size:18px;font-weight:800;color:#fff;letter-spacing:.6px;}
    .fx-hero-vsub{font-size:13px;font-weight:600;color:rgba(255,255,255,.9);margin-left:auto;}
    /* section headers with accent bar */
    .fx-sec{font-size:16px;font-weight:800;color:#f1f5f9;margin:18px 0 8px;padding-left:11px;
      border-left:3px solid #3b82f6;}
    /* red-flag callout / clean callout */
    .fx-flags{background:rgba(153,27,27,.16);border:1px solid #7f1d1d;border-left:4px solid #ef4444;
      border-radius:10px;padding:8px 14px;margin:2px 0 4px;}
    .fx-flag{color:#fecaca;font-size:14px;font-weight:600;padding:4px 0;}
    .fx-clean{background:rgba(22,101,52,.16);border:1px solid #166534;border-left:4px solid #22c55e;
      border-radius:10px;padding:10px 14px;color:#bbf7d0;font-size:14px;font-weight:600;}
    </style>""", unsafe_allow_html=True)

    _wl=st.session_state.get("watchlist",[])
    _c1,_c2=st.columns([3,1])
    with _c1:
        _fx_in=st.text_input("Ticker",value=st.session_state.get("fx_ticker",""),
                             placeholder="e.g. AAPL",key="fx_input").strip().upper()
    with _c2:
        st.markdown("<div style='height:28px'></div>",unsafe_allow_html=True)
        _fx_go=st.button("Analyse",key="fx_go",use_container_width=True,type="primary")
    if _wl:
        st.caption("From your watchlist:")
        _qc=st.columns(min(len(_wl),9))
        for _i,_t in enumerate(_wl):
            with _qc[_i%len(_qc)]:
                if st.button(_t,key=f"fxq_{_t}",use_container_width=True):
                    st.session_state["fx_ticker"]=_t; st.rerun()

    _tkr=_fx_in if (_fx_go and _fx_in) else st.session_state.get("fx_ticker","")
    if _fx_go and _fx_in:
        st.session_state["fx_ticker"]=_fx_in

    if not _tkr:
        st.info("Type a ticker (or pick one from the watchlist) and hit **Analyse** — you'll get "
                "three verdicts and a red-flag list read straight from the latest filings.")
    else:
        _price=fetch_underlying_last(_tkr) if tradier.is_configured() else None
        with st.spinner(f"Reading {_tkr}'s SEC filings…"):
            _fd=fetch_fundamentals(_tkr,_price)
        if not _fd.get("ok"):
            st.error(_fd.get("error","Could not read fundamentals."))
        else:
            m=_fd["metrics"]; g=_fd["groups"]
            _srcid=f"SEC CIK {_fd['cik']}" if _fd.get("cik") else "Yahoo Finance"

            # ── Company profile — every line is either sourced verbatim or computed from the
            #    filings; nothing here is AI-written. ──
            _prof=_fd.get("profile") or {}
            _cls=[x for x in (_prof.get("sector"),_prof.get("industry"),_prof.get("country")) if x]
            if _prof.get("employees"):
                try: _cls.append(f"{int(_prof['employees']):,} employees")
                except Exception: pass

            # ── Hero card with an overall verdict badge (the headline decision) ──
            _verds=[g[k]["verdict"] for k in ("Valuation","Quality","Health")]
            if   "🔴" in _verds: _ov=("🔴","FRAGILE","#7f1d1d")
            elif "🟡" in _verds: _ov=("🟡","MIXED","#a16207")
            elif "🟢" in _verds: _ov=("🟢","SOUND","#166534")
            else:                _ov=("⚪","NO DATA","#475569")
            _price_str=f"${_price:,.2f}" if _price else "price n/a"
            _dots="&nbsp;&nbsp;".join(f"{g[k]['verdict']} {k}" for k in ("Valuation","Quality","Health"))
            _sub=html.escape("  ·  ".join(_cls)) if _cls else "US-listed filer"
            st.markdown(
                f"""<div class="fx-hero"><div class="fx-hero-top">
                <div><div class="fx-hero-name">{html.escape(_fd['company'])}</div>
                     <div class="fx-hero-sub">{_sub}</div></div>
                <div class="fx-hero-right"><div class="fx-hero-ticker">{html.escape(_fd['ticker'])}</div>
                     <div class="fx-hero-price">{_price_str}</div></div></div>
                <div class="fx-hero-verdict" style="background:{_ov[2]}">
                     <span class="fx-ov">{_ov[0]} {_ov[1]}</span>
                     <span class="fx-hero-vsub">{_dots}</span></div></div>""",
                unsafe_allow_html=True)
            st.caption(f"{_srcid}"+("" if _price else " · add Tradier token for P/E"))

            # ── Verdict chips (decision first) ──
            _vc={"🟢":"#16a34a","🟡":"#a16207","🔴":"#7f1d1d","⚪":"#475569"}
            def _fx_chip(name,verdict,tip):
                col=_vc.get(verdict,"#475569")
                return (f'<span class="fx-chip" style="background:{col}">{verdict} {name}'
                        f'<span class="fx-tip">{tip}</span></span>')
            _val_tip=(f"P/E: {_fx_num(m['pe'],1)}<br>PEG: {_fx_num(m['peg'],2)}<br>"
                      f"FCF yield: {_fx_pct(m['fcf_yield'])}<br>Market cap: {_fx_money(m['market_cap'])}")
            _qual_tip=(f"Gross margin: {_fx_pct(m['gross_margin'])}<br>Op margin: {_fx_pct(m['op_margin'])}<br>"
                       f"Net margin: {_fx_pct(m['net_margin'])}<br>ROE: {_fx_pct(m['roe'])}<br>"
                       f"Revenue YoY: {_fx_pct(m['rev_growth'])}<br>Net income YoY: {_fx_pct(m['ni_growth'])}")
            _hlth_tip=(f"Debt/Equity: {_fx_x(m['debt_to_equity'])}<br>Current ratio: {_fx_num(m['current_ratio'])}<br>"
                       f"Interest coverage: {_fx_x(m['interest_coverage'])}<br>Free cash flow: {_fx_money(m['fcf'])}<br>"
                       f"FCF margin: {_fx_pct(m['fcf_margin'])}<br>Share count YoY: {_fx_pct(m['share_change'])}")
            st.markdown("<div style='margin:8px 0 2px'>"
                        +_fx_chip("Valuation",g["Valuation"]["verdict"],_val_tip)
                        +_fx_chip("Quality",g["Quality"]["verdict"],_qual_tip)
                        +_fx_chip("Health",g["Health"]["verdict"],_hlth_tip)
                        +"</div><div style='font-size:12px;color:#64748b;margin-bottom:2px'>"
                        "Hover a chip for the numbers behind it.</div>",unsafe_allow_html=True)

            # ── Red flags — a callout box so problems pop ──
            st.markdown("<div class='fx-sec'>🚩 Red flags</div>",unsafe_allow_html=True)
            _flags=_fd["flags"]
            if not _flags:
                st.markdown("<div class='fx-clean'>✅ No red flags on the checks we run — "
                            "profitability, leverage, liquidity, coverage, cash flow, dilution, "
                            "valuation.</div>",unsafe_allow_html=True)
            else:
                _fb="<div class='fx-flags'>"
                for _f in _flags:
                    _fb+=f"<div class='fx-flag'>{_f['sev']} {html.escape(_f['text'])}</div>"
                st.markdown(_fb+"</div>",unsafe_allow_html=True)

            # ── Short summary (perma-visible) ──
            if _prof.get("summary_short"):
                _tag="✨ AI summary" if _prof.get("summary_ai") else "📝 In brief"
                st.markdown(f"<div class='fx-summary'><span class='fx-summary-tag'>{_tag}</span>"
                            f"{html.escape(_prof['summary_short'])}</div>",unsafe_allow_html=True)

            # ── Financial trend — color-coded (green = good, red = bad) ──
            _grn="#4ade80"; _red="#f87171"
            def _cspan(txt,good):
                return f"<span style='color:{_grn if good else _red};font-weight:700'>{txt}</span>"
            _tr=[]
            if m.get("rev_growth") is not None:
                _tr.append(_cspan(f"Revenue {m['rev_growth']*100:+.0f}% YoY", m["rev_growth"]>=0))
            if m.get("ni_growth") is not None:
                _tr.append(_cspan(f"Net income {m['ni_growth']*100:+.0f}% YoY", m["ni_growth"]>=0))
            if m.get("net_income_prev") and m.get("revenue_prev") and m.get("net_margin") is not None:
                _exp=m["net_margin"]>(m["net_income_prev"]/m["revenue_prev"])
                _tr.append(_cspan("margins expanding" if _exp else "margins compressing", _exp))
            if m.get("fcf") is not None:
                _tr.append(_cspan("FCF positive" if m["fcf"]>=0 else "FCF negative", m["fcf"]>=0))
            if _tr:
                st.markdown("📈 <b>Trend (from filings):</b> "+" · ".join(_tr),unsafe_allow_html=True)
            # What they invest in — real dollar figures from the filings.
            _inv=[]
            if m.get("rnd"):   _inv.append(f"R&D {_fx_money(m['rnd'])}/yr")
            if m.get("capex"): _inv.append(f"Capex {_fx_money(m['capex'])}/yr")
            if _inv:
                st.markdown("🔧 **Invests in:** "+" · ".join(_inv))
            # Business summary — quoted verbatim from the company's filing, never generated.
            if _prof.get("summary"):
                with st.expander("📄 Full business description (verbatim from filings)"):
                    # Break the single wall-of-text into balanced ~220-char paragraphs at sentence
                    # boundaries — words kept verbatim, only whitespace added — for readability.
                    _txt=_prof["summary"].strip()
                    _parts=re.split(r'(?<=[.!?])\s+', _txt)
                    _paras=[]; _buf=""
                    for _p in _parts:
                        _buf=(_buf+" "+_p).strip()
                        if len(_buf)>=220:
                            _paras.append(_buf); _buf=""
                    if _buf: _paras.append(_buf)
                    for _pa in (_paras or [_txt]):
                        st.markdown(
                            "<p style='margin:0 0 11px;line-height:1.65;color:#cbd5e1;"
                            f"padding-left:10px;border-left:2px solid #24314a'>{html.escape(_pa)}</p>",
                            unsafe_allow_html=True)
                    _wl_bits=[]
                    if _prof.get("website"): _wl_bits.append(_prof["website"])
                    if _prof.get("sic"):     _wl_bits.append(f"SEC industry: {_prof['sic']}")
                    if _wl_bits: st.caption("  ·  ".join(_wl_bits))
                    st.caption("Summary sourced verbatim from the company's own filing "
                               "(via Yahoo Finance) — not AI-generated.")

            # ── Recent headlines — real Yahoo Finance articles, links open the source ──
            _news=fetch_company_news(_tkr)
            if _news:
                st.markdown("<div class='fx-sec'>📰 Recent headlines</div>",unsafe_allow_html=True)
                _nh="<div class='fx-news'>"
                for _n in _news:
                    _meta=" · ".join([x for x in (_n.get("publisher"),_n.get("when")) if x])
                    _nh+=(f"<a href='{html.escape(_n['link'])}' target='_blank' rel='noopener'>"
                          f"<div class='t'>{html.escape(_n['title'])}</div>"
                          f"<div class='m'>{html.escape(_meta)}</div></a>")
                _nh+="</div>"
                st.markdown(_nh,unsafe_allow_html=True)
                st.caption("Headlines via Yahoo Finance — click to open the original article.")

            with st.expander("The numbers (latest annual filing)"):
                _rows=[
                    ("Revenue",_fx_money(m["revenue"]),"Prior yr",_fx_money(m["revenue_prev"])),
                    ("Net income",_fx_money(m["net_income"]),"Prior yr",_fx_money(m["net_income_prev"])),
                    ("EPS (diluted)",_fx_num(m["eps"]),"Shares out",_fx_money(m["shares"]).replace("$","")),
                    ("Gross margin",_fx_pct(m["gross_margin"]),"Op margin",_fx_pct(m["op_margin"])),
                    ("Net margin",_fx_pct(m["net_margin"]),"ROE",_fx_pct(m["roe"])),
                    ("Free cash flow",_fx_money(m["fcf"]),"FCF margin",_fx_pct(m["fcf_margin"])),
                    ("R&D spend",_fx_money(m.get("rnd")),"Capex",_fx_money(m.get("capex"))),
                    ("Debt / Equity",_fx_x(m["debt_to_equity"]),"Current ratio",_fx_num(m["current_ratio"])),
                    ("Interest coverage",_fx_x(m["interest_coverage"]),"Market cap",_fx_money(m["market_cap"])),
                    ("P/E",_fx_num(m["pe"],1),"FCF yield",_fx_pct(m["fcf_yield"])),
                ]
                st.dataframe(pd.DataFrame(_rows,columns=["Metric","Value","Metric ","Value "]),
                             use_container_width=True,hide_index=True)

            with st.expander("What makes each verdict 🟢 / 🟡 / 🔴"):
                _T=fundamentals.T
                st.markdown(
                    "Each chip shows the **worst** signal among its metrics — a single red "
                    "metric turns the whole chip red. (🟢 all clear · 🟡 caution · 🔴 problem · "
                    "⚪ no data.)\n\n"
                    "**Valuation**\n"
                    f"- 🔴 P/E > {_T['pe_high']:.0f}, or FCF yield < 0\n"
                    f"- 🟡 P/E > {_T['pe_rich']:.0f}, FCF yield < {_T['fcfy_thin']*100:.0f}%, or PEG > {_T['peg_high']:.0f}\n"
                    "- 🟢 none of the above\n\n"
                    "**Quality**\n"
                    "- 🔴 net margin < 0, or ROE < 0\n"
                    f"- 🟡 net margin < {_T['netmargin_thin']*100:.0f}%, or ROE < {_T['roe_thin']*100:.0f}%\n"
                    "- 🟢 profitable, with healthy margins and ROE\n\n"
                    "**Health**\n"
                    f"- 🔴 debt/equity > {_T['de_danger']:.1f}×, current ratio < {_T['cr_danger']:.1f}, "
                    f"interest coverage < {_T['cover_danger']:.0f}×, or negative free cash flow\n"
                    f"- 🟡 debt/equity > {_T['de_high']:.1f}×, current ratio < {_T['cr_thin']:.1f}, "
                    f"or interest coverage < {_T['cover_thin']:.0f}×\n"
                    "- 🟢 low leverage, comfortable liquidity and coverage, positive FCF\n\n"
                    f"The **🚩 Red-flag list** also calls out: revenue shrinking YoY, negative ROE, "
                    f"dilution (shares +{_T['dilution']*100:.0f}%/yr or more), and rich valuation (P/E > {_T['pe_rich']:.0f})."
                )
            st.caption(f"Data source: {_fd.get('source','—')}. "
                       "SEC path: balance-sheet items from the most recent 10-Q/10-K, flows "
                       "(revenue, income, cash flow) from the latest full fiscal year.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB — SIGNALS (wheel premium-selling opportunities across the watchlist)
# ══════════════════════════════════════════════════════════════════════════════
with tab_signals:
    st.subheader("🎯 Signals")
    st.caption("Cash-secured-put (and covered-call) premium opportunities across your watchlist, "
               "ranked by per-trade yield with the app's gates (Δ≈0.30 · 21–45 DTE · median rule · "
               "liquidity · earnings blackout). Suggestions only — you place them manually.")

    st.markdown("""<style>
    .sg-card{background:#0f172a;border:1px solid #24314a;border-left:4px solid #3b82f6;
      border-radius:12px;padding:12px 16px;margin:8px 0;box-shadow:0 6px 18px rgba(0,0,0,.25);}
    .sg-top{display:flex;justify-content:space-between;align-items:baseline;gap:10px;flex-wrap:wrap;}
    .sg-tkr{font-size:18px;font-weight:800;color:#f8fafc;}
    .sg-badge{display:inline-block;font-size:11px;font-weight:800;letter-spacing:.5px;
      padding:2px 9px;border-radius:6px;color:#fff;margin-left:8px;vertical-align:middle;}
    .sg-prem{font-size:20px;font-weight:800;}
    .sg-sub{color:#94a3b8;font-size:12.5px;margin-top:5px;}
    .sg-sub b{color:#dbe4f0;font-weight:600;}
    .sg-size{color:#7dd3fc;font-size:13px;font-weight:700;margin-top:6px;}
    .sg-earn{margin-top:7px;background:rgba(251,191,36,.14);border:1px solid #b45309;
      border-left:4px solid #fbbf24;border-radius:8px;padding:5px 10px;color:#fcd34d;
      font-size:12.5px;font-weight:700;}
    .sg-sec{font-size:16px;font-weight:800;color:#f1f5f9;margin:16px 0 6px;padding-left:11px;
      border-left:3px solid #3b82f6;}
    </style>""", unsafe_allow_html=True)

    _SIGNALS_FILE = Path(__file__).parent / "data" / "signals.json"

    def _load_signals_file():
        try:
            return json.loads(_SIGNALS_FILE.read_text())
        except Exception:
            return {"generated": None, "count": 0, "signals": [], "params": {}}

    # ── controls: capital + one clear action ──
    _sc1,_sc2 = st.columns([2,1])
    with _sc1:
        _cap = st.number_input("Available capital ($)", min_value=0.0,
                               value=float(st.session_state.get("sg_capital",0.0)), step=5000.0,
                               help="Used only to size contracts (90% deployed, 10% reserved for "
                                    "your manual longs). Leave 0 to just see the opportunities.")
        st.session_state["sg_capital"]=_cap
    with _sc2:
        st.markdown("<div style='height:28px'></div>",unsafe_allow_html=True)
        _live = st.button("🔄 Scan now", use_container_width=True, type="primary",
                          disabled=not tradier.is_configured())
    if not tradier.is_configured():
        st.info("Add your Tradier token to Streamlit secrets to run a scan.")

    if _live:
        st.session_state["sg_nonce"]=st.session_state.get("sg_nonce",0)+1
        st.session_state["sg_scanned"]=True

    _uni=load_signal_universe()
    _uni_src={"sheet":"Google Sheet (live)","file":"committed cache","watchlist":"watchlist"}.get(
        _uni.get("_source"),"—")
    st.caption(f"Universe: {len(_uni.get('wheel',[]))} wheel · {len(_uni.get('growth',[]))} growth "
               f"· source: {_uni_src}")

    if st.session_state.get("sg_scanned") and tradier.is_configured():
        with st.spinner("Scanning the wheel universe via Tradier… (~40–90s)"):
            _data = run_signal_scan(st.session_state.get("sg_nonce",0))
    else:
        _data = {"signals": [], "leaps": [], "params": {}, "count": 0}

    _sigs = _data.get("signals",[])
    _short = [s for s in _sigs if s.get("shortlist")]
    _params = _data.get("params",{})
    if _sigs:
        st.caption(f"Live scan  ·  {len(_sigs)} opportunities  ·  {len(_short)} on the shortlist")

    if not _sigs:
        if not st.session_state.get("sg_scanned"):
            st.info("Press **🔄 Scan now** to pull live CSP / CC / LEAP signals for your wheel "
                    "universe from Tradier. Enter your available capital first if you want "
                    "contract sizing.")
        else:
            st.warning("Scan ran, but nothing passed the gates right now (premium ≥1.2%, Δ≈0.30, "
                       "below median, liquid, no earnings before expiry). Markets are likely "
                       "closed — try again when they're open.")
    else:
        # ── capital sizing over the shortlist (CSP collateral = strike×100) ──
        _rows=[]; _deployed=0.0
        if _cap>0:
            _budget=_cap*0.90; _name_cap=_cap*0.10; _sector_cap=_cap*0.25
            _sec_used={}
            for s in _short:
                _coll_per=s["strike"]*100.0
                _sec=s.get("sector") or "—"
                _room=min(_name_cap,_budget-_deployed,_sector_cap-_sec_used.get(_sec,0.0))
                _n=int(_room//_coll_per) if _coll_per>0 else 0
                _coll=_n*_coll_per; _prem=_n*s["mid"]*100.0
                if _n>=1:
                    _deployed+=_coll; _sec_used[_sec]=_sec_used.get(_sec,0.0)+_coll
                _rows.append({**s,"contracts":_n,"collateral":_coll,"premium":_prem})
            _tot_prem=sum(r["premium"] for r in _rows)
            _m1,_m2,_m3,_m4=st.columns(4)
            _m1.metric("Deployable (90%)", f"${_budget:,.0f}")
            _m2.metric("Would deploy", f"${_deployed:,.0f}", f"{(_deployed/_cap*100):.0f}% of capital")
            _m3.metric("Est. premium / cycle", f"${_tot_prem:,.0f}")
            _m4.metric("Reserved (10%)", f"${_cap*0.10:,.0f}")
        else:
            _rows=[{**s,"contracts":None,"collateral":None,"premium":None} for s in _short]

        # ── shortlist cards ──
        st.markdown("<div class='sg-sec'>⭐ Shortlist — best cashflow entries (CSP)</div>",
                    unsafe_allow_html=True)
        if not _rows:
            st.warning("No CSP passed all gates (premium ≥1.2%, Δ≈0.30, below median, liquid, "
                       "no earnings before expiry) this scan. Covered-call ideas may still be below.")
        for r in _rows:
            _pc = r["premium_pct"]; _pcol = "#4ade80" if r.get("strong") else "#e2e8f0"
            _strong = " 🔥" if r.get("strong") else ""
            _size_line = ""
            if _cap>0:
                if r["contracts"]>=1:
                    _size_line=(f"<div class='sg-size'>➜ {r['contracts']} contract(s) · "
                                f"${r['collateral']:,.0f} collateral · ${r['premium']:,.0f} premium</div>")
                else:
                    _size_line="<div class='sg-size' style='color:#94a3b8'>➜ doesn't fit under the caps</div>"
            _ewarn = ("<div class='sg-earn'>⚠️ Earnings during the trade — assess carefully</div>"
                      if r.get("earnings_in_window") else "")
            st.markdown(
                f"""<div class='sg-card'><div class='sg-top'>
                <div><span class='sg-tkr'>{html.escape(r['ticker'])}</span>
                     <span class='sg-badge' style='background:#16a34a'>CSP</span></div>
                <div class='sg-prem' style='color:{_pcol}'>{_pc:.2f}%{_strong}</div></div>
                <div class='sg-sub'>Sell <b>${r['strike']:.1f}</b> put · <b>{r['expiry']}</b> "
                f"({r['dte']}d) · mid <b>${r['mid']:.2f}</b> · Δ {r.get('delta')} · "
                f"POP <b>{r.get('pop')}%</b> · IV {r.get('iv')}% · "
                f"<b>{html.escape(str(r.get('sector')))}</b> · {r.get('vol_bucket')}</div>"
                f"{_ewarn}{_size_line}</div>""", unsafe_allow_html=True)

        # ── covered-call ideas (only if you hold the shares) ──
        _ccs=[s for s in _sigs if s["strategy"]=="CC" and s.get("median_ok")]
        _ccs.sort(key=lambda s:s["premium_pct"],reverse=True)
        if _ccs:
            st.markdown("<div class='sg-sec'>📞 Covered-call ideas — only if you already hold ≥100 shares</div>",
                        unsafe_allow_html=True)
            for r in _ccs[:8]:
                _strong=" 🔥" if r.get("strong") else ""
                _ewarn = ("<div class='sg-earn'>⚠️ Earnings during the trade — assess carefully</div>"
                          if r.get("earnings_in_window") else "")
                st.markdown(
                    f"""<div class='sg-card' style='border-left-color:#2563eb'><div class='sg-top'>
                    <div><span class='sg-tkr'>{html.escape(r['ticker'])}</span>
                         <span class='sg-badge' style='background:#2563eb'>CC</span></div>
                    <div class='sg-prem' style='color:#e2e8f0'>{r['premium_pct']:.2f}%{_strong}</div></div>
                    <div class='sg-sub'>Sell <b>${r['strike']:.1f}</b> call · <b>{r['expiry']}</b> "
                    f"({r['dte']}d) · mid <b>${r['mid']:.2f}</b> · Δ {r.get('delta')} · "
                    f"POP <b>{r.get('pop')}%</b> · <b>{html.escape(str(r.get('sector')))}</b> · "
                    f"{r.get('vol_bucket')}</div>{_ewarn}</div>""", unsafe_allow_html=True)

        # ── LEAP / PMCC ideas (growth + covered-call basis) ──
        _leaps=_data.get("leaps",[])
        if _leaps:
            st.markdown("<div class='sg-sec'>🚀 LEAP ideas — growth &amp; PMCC basis (a BUY, not premium)</div>",
                        unsafe_allow_html=True)
            for r in _leaps[:8]:
                _pm=" · ✅ good PMCC basis" if r.get("good_pmcc") else ""
                st.markdown(
                    f"""<div class='sg-card' style='border-left-color:#7c3aed'><div class='sg-top'>
                    <div><span class='sg-tkr'>{html.escape(r['ticker'])}</span>
                         <span class='sg-badge' style='background:#7c3aed'>LEAP</span></div>
                    <div class='sg-prem' style='color:#e2e8f0;font-size:16px'>${r.get('cost',0):,.0f}</div></div>
                    <div class='sg-sub'>Buy <b>${r['strike']:.1f}</b> call · <b>{r['expiry']}</b> "
                    f"({r['dte']}d) · Δ <b>{r.get('delta')}</b> · mid ${r['mid']:.2f} · "
                    f"time value <b>{r.get('extrinsic_pct')}%</b>{_pm} · "
                    f"<b>{html.escape(str(r.get('sector')))}</b></div></div>""",
                    unsafe_allow_html=True)

        # ── full table for the curious ──
        with st.expander("All opportunities (full scan, incl. below-median & filtered)"):
            _tbl=[]
            for s in _sigs:
                _tbl.append({"Ticker":s["ticker"],"Type":s["strategy"],"Strike":s["strike"],
                    "Expiry":s["expiry"],"DTE":s["dte"],"Prem %":s["premium_pct"],
                    "Δ":s.get("delta"),"POP %":s.get("pop"),"IV %":s.get("iv"),
                    "Sector":s.get("sector"),"Vol":s.get("vol_bucket"),
                    "Median ok":"✅" if s.get("median_ok") else "—",
                    "OI":s.get("oi"),"Shortlist":"⭐" if s.get("shortlist") else ""})
            st.dataframe(pd.DataFrame(_tbl),use_container_width=True,hide_index=True)

        _pp=_params or {}
        st.caption(f"Gates: premium ≥{_pp.get('min_premium_pct',1.2)}% (🔥 ≥{_pp.get('strong_premium_pct',1.5)}%) · "
                   f"Δ≈{_pp.get('delta_opt',0.3)} · DTE {_pp.get('dte',[21,45])} · median rule · "
                   f"no new position within {_pp.get('earnings_blackout_days',7)}d of earnings. "
                   "Sizing: 90% deployed, 10% reserved · ≤10%/name · ≤25%/sector. "
                   "Not advice — verify each fill and place manually.")

