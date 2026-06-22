import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import math
import requests
from scipy.stats import norm
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

DEFAULT_WATCHLIST = ["NVDA", "META", "TSLA", "IBIT", "GLD", "GDXJ", "BE", "VST", "CRWV", "AMZN", "SPCX", "NVTS", "VRT", "SLV", "PLTR", "ORCL", "AAPL", "GOOG", "MSFT", "IREN", "NBIS", "URNM", "COPP", "COPJ", "PURR", "MSTR", "BMNR", "NOW", "CQQQ", "QANT.AS", "WMT"]

VIX_ZONES = [
    (0,  15, "#16a34a", "LOW — Ideal LEAP buying zone"),
    (15, 20, "#ca8a04", "NORMAL — Balanced regime"),
    (20, 30, "#ea580c", "ELEVATED — CC writing premium rich"),
    (30, 99, "#dc2626", "HIGH — Aggressive premium selling regime"),
]

PULSE_TICKERS = [
    ("SPY",      "S&P 500",    "$",  False),
    ("QQQ",      "Nasdaq 100", "$",  False),
    ("DIA",      "Dow Jones",  "$",  False),
    ("IWM",      "R2000",      "$",  False),
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
# PERSISTENT WATCHLIST via st.query_params
# The tickers are encoded in the URL — bookmark it to restore your watchlist.
# ══════════════════════════════════════════════════════════════════════════════
def _load_watchlist_from_params():
    try:
        raw = st.query_params.get("tickers", "")
        if raw:
            tickers = [t.strip().upper() for t in raw.split(",") if t.strip()]
            if tickers:
                return tickers
    except Exception:
        pass
    return DEFAULT_WATCHLIST.copy()

def _save_watchlist_to_params(watchlist):
    try:
        st.query_params["tickers"] = ",".join(watchlist)
    except Exception:
        pass

if "watchlist" not in st.session_state:
    st.session_state.watchlist = _load_watchlist_from_params()

_save_watchlist_to_params(st.session_state.watchlist)

# ══════════════════════════════════════════════════════════════════════════════
# DATA FETCHERS
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=60, show_spinner=False)
def fetch_quote(ticker):
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

@st.cache_data(ttl=60, show_spinner=False)
def fetch_vix_term():
    out = {}
    for ticker, label in VIX_TERM_TICKERS:
        q = fetch_quote(ticker)
        if q: out[label] = q["price"]
    return out

@st.cache_data(ttl=60, show_spinner=False)
def fetch_skew():
    q = fetch_quote("^SKEW")
    return q["price"] if q else None

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_prices(ticker, period="1y"):
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
    try:
        c_atm = chain.calls.iloc[(chain.calls["strike"]-price).abs().argsort()[:1]]
        p_atm = chain.puts.iloc[(chain.puts["strike"]-price).abs().argsort()[:1]]
        return round(c_atm["impliedVolatility"].values[0]*100,1), round(p_atm["impliedVolatility"].values[0]*100,1)
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

        d2_val=None
        if yahoo_ok:
            d_abs=abs(float(d_raw))*100.0; theta=abs(float(t_raw)); gs="yahoo"
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
            eff_iv=round(iv*100,1) if (gs=="yahoo" and iv>0.01) else round(sigma*100,1)
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

def calc_four_gates(r, bb_veto_mode="Hard", soft_penalty=10, leg="csp"):
    """bb_veto_mode (§9.3, 24 June): 'Hard' (default, original behavior — walking the lower
    band 2+ sessions fails G3 and blocks all_pass), 'Soft' (-soft_penalty points off each leg's
    score instead of blocking Status — see get_screener_row), or 'Off' (informational only,
    never gates or penalizes). Recommended over a one-off override rule since a hand-written
    exception is just a second hard-coded rule with the same brittleness.

    leg (22 June, per Jay; direction corrected same day): 'csp', 'cc', or anything else (e.g.
    'leap'). CSP and CC each get a 4th gate — G4 Median — with OPPOSITE pass conditions,
    because gates are no longer one shared computation across all three tables (this was
    previously an open "next step, your call" — Jay locked it in by asking for leg-specific
    median checks). Direction: CSP wants to catch a setup right after a reversal off the low,
    with more upside runway left before the next reversal — that means price at/BELOW the
    median; once price is already above the median, most of that runway is used up (this is
    the BE case — riding the upper band, all gates green, no room left), so G4 FAILS when
    price is above the median for CSP. CC is the mirror image: catch a setup near the top with
    room to fall, so G4 FAILS when price is below the median for CC. Any other leg value
    (LEAP) gets no G4 — not requested, left as 3 gates exactly as before rather than guessed
    at."""
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
    # Only applies to leg in {"csp","cc"}; any other leg (LEAP) skips this gate entirely.
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

    return {"gates":gates,"all_pass":all(g["pass"] for g in gates.values()),
            "bb_walking":walking,"bb_penalty":bb_penalty,"bb_veto_mode":bb_veto_mode}

# 22 June fix — schema-version stamp for the cached screener rows below. Streamlit's file-
# watcher reruns an already-open session's script on a code push WITHOUT clearing
# st.session_state, so a browser tab open before this edit still held rows built under the
# old single-"gate_result" schema; reading them back under the new per-leg
# "gate_result_csp"/"gate_result_cc"/"gate_result_leap" keys crashed with a KeyError. Bump
# this whenever get_screener_row's returned dict shape changes, so a stale cache is detected
# and discarded (forcing a re-click of "Run Screener") instead of crashing the page.
_SCREENER_SCHEMA_VERSION = 2

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

    calls_df,puts_df,_,_=fetch_chain_cached(ticker,exp_csp)
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
        leap_calls_df,_,_,_=fetch_chain_cached(ticker,exp_leap)
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
    # conditions; LEAP keeps the original 3 gates (no median check — not requested).
    gate_result_csp =calc_four_gates(result, bb_veto_mode=bb_veto_mode, soft_penalty=soft_penalty, leg="csp")
    gate_result_cc  =calc_four_gates(result, bb_veto_mode=bb_veto_mode, soft_penalty=soft_penalty, leg="cc")
    gate_result_leap=calc_four_gates(result, bb_veto_mode=bb_veto_mode, soft_penalty=soft_penalty, leg="leap")

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
            "csp_timing_label":csp_lbl,"csp_timing_score":csp_tsc,"csp_timing_reasons":csp_treasons}

# ── Signal engines ─────────────────────────────────────────────────────────────
def leap_signal(hvr,rsi_val,above_50ma,above_200ma,vix_lvl):
    score=0; reasons=[]
    if hvr is not None:
        if hvr<25:   score+=3; reasons.append("✅ HV Rank low (<25) — cheap premium")
        elif hvr<40: score+=2; reasons.append("🟡 HV Rank moderate (25-40)")
        else:        score-=1; reasons.append("❌ HV Rank elevated — expensive entry")
    if rsi_val is not None:
        if 33<=rsi_val<=52:   score+=2; reasons.append("✅ RSI ideal recovery zone (33-52)")
        elif 52<rsi_val<=65:  score+=1; reasons.append("🟡 RSI extended, not overbought")
        elif rsi_val<30:      score+=1; reasons.append("🟡 RSI oversold — wait for turn")
        else:                 score-=1; reasons.append("❌ RSI overbought — avoid chasing")
    if above_200ma: score+=2; reasons.append("✅ Above 200MA — long term trend intact")
    else:           score-=1; reasons.append("❌ Below 200MA — trend broken")
    if above_50ma:  score+=1; reasons.append("✅ Above 50MA — medium term OK")
    if vix_lvl is not None:
        if vix_lvl<18:   score+=1; reasons.append("✅ VIX low — cheap index premium")
        elif vix_lvl>28: score-=1; reasons.append("⚠️ VIX elevated — vol expansion risk")
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

def cc_signal(hvr,pctb_s,rsi_s,df):
    score,reasons,_=_mean_reversion_score(pctb_s,rsi_s,df,"cc")
    if hvr is not None:
        if hvr>55:   score+=2; reasons.append("✅ HV Rank high — premium rich")
        elif hvr>35: score+=1; reasons.append("🟡 HV Rank moderate")
        else:        reasons.append("❌ HV Rank low — thin premium even if setup fires")
    label=("🟢 FULL SETUP — write now" if score>=10 else "🟡 PARTIAL SETUP" if score>=6
           else "🟠 EARLY / WATCH" if score>=3 else "🔴 NO SETUP")
    return label, score, reasons

def csp_signal(hvr,pctb_s,rsi_s,df,walking=False):
    score,reasons,_=_mean_reversion_score(pctb_s,rsi_s,df,"csp")
    if walking:
        score=max(0,score-4); reasons.append("❌ Still walking the lower band — breakdown, not a bounce (veto)")
    if hvr is not None:
        if hvr>55:   score+=2; reasons.append("✅ High HV Rank — CSP premium rich")
        elif hvr>35: score+=1; reasons.append("🟡 Moderate HV Rank")
    label=("🟢 FULL SETUP — sell put" if score>=10 else "🟡 PARTIAL SETUP" if score>=6
           else "🟠 EARLY / WATCH" if score>=3 else "🔴 NO SETUP")
    return label, score, reasons

def analyse(ticker, period, vix_current):
    df=fetch_prices(ticker,period)
    if df is None: return None
    cl=df["Close"].squeeze()
    curr=float(cl.iloc[-1]); prev=float(cl.iloc[-2]); pct_chg=(curr/prev-1)*100
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
            calls_df,puts_df,dte,chain_err=fetch_chain_cached(ticker,exp)
            if calls_df is not None:
                chain=type("_C",(),{"calls":calls_df,"puts":puts_df})()
                c_iv,p_iv=calc_atm_iv(chain,curr); pcr_val=calc_pcr(chain)
    leap_lbl,leap_sc,leap_r=leap_signal(hvr,rsi_cur,ab50,ab200,vix_current)
    cc_lbl,cc_sc,cc_r=cc_signal(hvr,pctb_s,rsi_s,df)
    csp_lbl,csp_sc,csp_r=csp_signal(hvr,pctb_s,rsi_s,df,walking_lower)
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
    return f"{v:{fs}}{su}" if v is not None else "—"

def greek_source_label(gs):
    return {"yahoo":"📡 Yahoo","bs_strike":"📐 BS (strike IV)",
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

def vix_term_chart(term_data):
    if not term_data or len(term_data)<2: return None
    labels=list(term_data.keys()); vals=list(term_data.values())
    colors=["#60a5fa"]+["#22c55e" if vals[i]>=vals[i-1] else "#ef4444" for i in range(1,len(vals))]
    fig=go.Figure(go.Bar(x=labels,y=vals,marker_color=colors,
        text=[f"{v:.1f}" for v in vals],textposition="outside"))
    fig.update_layout(height=210,template="plotly_dark",
        title={"text":"<b>VIX Term Structure</b>","font":{"size":12}},
        yaxis_range=[0,max(vals)*1.3],margin=dict(l=10,r=10,t=50,b=10),showlegend=False)
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
    .jay-tbl td{{padding:6px 10px;border-bottom:1px solid #21262d;white-space:nowrap;}}
    .jay-tbl tbody tr:nth-child(even) td{{background:#11151c;}}
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

    new_ticker=st.text_input("Add ticker",placeholder="e.g. AMZN",key="new_ticker_input")
    if new_ticker:
        t=new_ticker.upper().strip()
        if t and t not in st.session_state.watchlist:
            st.session_state.watchlist.append(t)
            _save_watchlist_to_params(st.session_state.watchlist)
            st.rerun()

    if st.session_state.watchlist:
        remove=st.selectbox("Remove ticker",["— select —"]+st.session_state.watchlist)
        if remove!="— select —":
            st.session_state.watchlist.remove(remove)
            _save_watchlist_to_params(st.session_state.watchlist)
            st.rerun()

    st.caption(f"**Watchlist:** {', '.join(st.session_state.watchlist)}")
    st.caption("💾 Saved in URL — bookmark the page to restore your list.")

    period=st.selectbox("Price History",["6mo","1y","2y"],index=1)

    st.divider()
    auto_refresh=st.toggle("🔄 Auto-refresh (60s)",value=False)
    if auto_refresh and HAS_AUTOREFRESH:
        st_autorefresh(interval=60_000,key="pulse_refresh")
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

tab_dash,tab_dive,tab_chain,tab_vix,tab_screener=st.tabs(
    ["Overview","Deep Dive","Options Chain","🌪️ Market Volatility","⚡ Screener"])

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
st.markdown('<div class="jay-nav-tt-row">'+"".join(
    f'<span class="jay-nav-tt">ⓘ {label}<span class="jay-nav-tt-text">{text}</span></span>'
    for label, text in _TAB_LEGEND
)+'</div>', unsafe_allow_html=True)

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
    st.caption(f"Updated: {datetime.utcnow().strftime('%H:%M:%S UTC')}  ·  ~15 min delayed  ·  Toggle 60s refresh in sidebar")

    pulse_data={ticker:fetch_quote(ticker) for ticker,*_ in PULSE_TICKERS}

    def render_pulse_col(col,ticker,label,prefix,is_yield):
        q=pulse_data.get(ticker)
        if q:
            p=q["price"]; pct=q["pct"]
            if is_yield:    disp=f"{p:.2f}%"
            elif p>10000:   disp=f"${p:,.0f}"
            elif p>100:     disp=f"{prefix}{p:,.2f}"
            else:           disp=f"{prefix}{p:.2f}"
            col.metric(label,disp,f"{pct:+.2f}%")
        else:
            col.metric(label,"—","—")

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
        if term_data:
            tc = vix_term_chart(term_data)
            if tc: st.plotly_chart(tc, use_container_width=True)

    with gcol4:
        st.markdown("**📊 Macro Signals**")
        tnx_q=pulse_data.get("^TNX"); irx_q=pulse_data.get("^IRX")
        if tnx_q and irx_q:
            spread=tnx_q["price"]-irx_q["price"]
            curve=("🟢 Normal" if spread>0.5 else "🟡 Flat" if spread>-0.3 else "🔴 Inverted")
            st.markdown(f"**Yield Curve (10Y−3M):** {spread:+.2f}%  {curve}")
        else:
            st.markdown("**Yield Curve:** —")
        if skew_val is not None:
            sk=("🔴 Elevated tail risk" if skew_val>145 else "🟡 Moderate" if skew_val>130 else "🟢 Low tail risk")
            st.markdown(f"**SKEW:** {skew_val:.1f}  {sk}")
        else:
            st.markdown("**SKEW:** —")
        if len(term_data)>=2:
            vals=list(term_data.values())
            st.markdown(f"**VIX Shape:** {'🟢 Contango' if vals[-1]>vals[0] else '🔴 Backwardation (stress)'}")
        if stock_fg_score is not None:
            color = fg_color(stock_fg_score)
            st.markdown(f"**Stocks F&G:** <span style='color:{color}'>{stock_fg_score:.0f} — {stock_fg_rating}</span>",
                        unsafe_allow_html=True)
            st.caption("Source: CNN")
        if crypto_fg_score is not None:
            color = fg_color(crypto_fg_score)
            st.markdown(f"**Crypto F&G:** <span style='color:{color}'>{crypto_fg_score:.0f} — {crypto_fg_rating}</span>",
                        unsafe_allow_html=True)
            st.caption("Source: Alternative.me")

    st.divider()

    # ── SECTOR HEATMAP ─────────────────────────────────────────────────────────
    st.subheader("🟩 Sector Heatmap")
    st.caption("SPDR sector ETFs + Bitcoin as Digital Assets · colour intensity = move strength · data ~15 min delayed")

    sector_quotes = []
    for ticker, label, short in SECTOR_TICKERS:
        q = fetch_quote(ticker)
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
                     "HV Rank":fmt(r["hvr"],".0f"),"HV%ile":fmt(r["hvpct"],".0f"),
                     "HV20":fmt(r["hv20"],".1f","%"),
                     "ATM IV C/P":f"{r['c_iv']:.0f}/{r['p_iv']:.0f}%" if r["c_iv"] else "—",
                     "RSI":fmt(r["rsi"],".0f"),
                     "200MA":"✅" if r["ab200"] else "❌",
                     "PCR":fmt(r["pcr"],".2f"),
                     "LEAP":r["leap"][0],"CC":r["cc"][0],"CSP":r["csp"][0]})
    if rows:
        tbl_height=38+len(rows)*35+4      # fits all rows exactly — no scrollbar
        # 22 June — same in-header hover-tooltip table as the Screener tab (st.dataframe's
        # header is a canvas-rendered grid and can't carry a real tooltip — see _html_table).
        _WATCH_LEGEND = [
            ("Ticker","Stock symbol"),
            ("Price","Current stock price"),
            ("Chg %","Today's percent change"),
            ("HV Rank","Historical volatility rank 0–100 vs its own 1-year range "
                       "(low = cheap premium, good for buying; high = rich premium, good for selling)"),
            ("HV%ile","Historical volatility percentile vs its own 1-year range"),
            ("HV20","20-day historical (realized) volatility, annualized"),
            ("ATM IV C/P","At-the-money implied volatility — call / put"),
            ("RSI","Relative Strength Index (14) — momentum; <30 oversold, >70 overbought"),
            ("200MA","Price above (✅) or below (❌) its 200-day moving average — long-term trend"),
            ("PCR","Put/call volume ratio — elevated readings skew bearish"),
            ("LEAP","LEAP-buy timing signal (low-IV, oversold-leaning setup)"),
            ("CC","Covered-call timing signal (overbought-leaning setup to write calls)"),
            ("CSP","Cash-secured-put timing signal (oversold-bounce setup to sell puts)"),
        ]
        _html_table(rows, _WATCH_LEGEND, tbl_height)

    hvr_data={t:r["hvr"] for t,r in results.items() if r["hvr"] is not None}
    if hvr_data:
        st.subheader("HV Rank — Entry Zones")
        colors=["#22c55e" if v<25 else "#eab308" if v<45 else "#f97316" if v<65 else "#ef4444"
                for v in hvr_data.values()]
        fig_hvr=go.Figure(go.Bar(x=list(hvr_data.keys()),y=list(hvr_data.values()),
            marker_color=colors,text=[f"{v:.0f}" for v in hvr_data.values()],textposition="outside"))
        fig_hvr.add_hline(y=25,line_dash="dash",line_color="#22c55e",annotation_text="25 = LEAP Zone")
        fig_hvr.add_hline(y=65,line_dash="dash",line_color="#ef4444",annotation_text="65 = CC Zone")
        fig_hvr.update_layout(height=320,template="plotly_dark",yaxis_title="HV Rank",
                               yaxis_range=[0,115],margin=dict(l=0,r=0,t=20,b=0))
        st.plotly_chart(fig_hvr,use_container_width=True)

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
        c1,c2,c3,c4,c5,c6=st.columns(6)
        c1.metric("Price",f"${r['price']:.2f}",f"{r['pct']:+.1f}%")
        c2.metric("HV Rank",fmt(r["hvr"],".0f"))
        c3.metric("HV Pctile",fmt(r["hvpct"],".0f","%"))
        c4.metric("RSI (14)",fmt(r["rsi"],".1f"))
        c5.metric("ATM Call IV",fmt(r["c_iv"],".1f","%"))
        c6.metric("PCR",fmt(r["pcr"],".2f"))
        sc1,sc2,sc3=st.columns(3)
        for col,key,lbl in [(sc1,"leap","LEAP"),(sc2,"cc","CC"),(sc3,"csp","CSP")]:
            with col:
                lb2,_,reasons=r[key]
                st.markdown(f"#### {lbl}: {lb2}")
                with st.expander("Breakdown"):
                    for reason in reasons: st.write(reason)
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
        if r["price"]:
            st.subheader("Position Sizing Guide")
            dte_ref=r["dte"] if r.get("dte") else 30
            iv_ref=(r["c_iv"] or r["p_iv"]) if (r.get("c_iv") or r.get("p_iv")) else None
            if iv_ref:
                exp_move=r["price"]*(iv_ref/100.0)*math.sqrt(dte_ref/365.0)
                st.markdown(f"""
| Metric | Value |
|---|---|
| IV expected move (1 SD, ~{dte_ref}d, ATM IV {iv_ref:.0f}%) | ±${exp_move:.2f} ({exp_move/r['price']*100:.1f}% of price) |
| Suggested CC strike (1 SD above) | ~${r['price']+exp_move:.2f} |
| Suggested CSP strike (1 SD below) | ~${r['price']-exp_move:.2f} |
                """)
                st.caption("Uses the option market's own forward-looking volatility (ATM IV), scaled to this expiry's actual DTE via √(DTE/365) — a proper 1-standard-deviation expected move. The old ATR-based row is retired: ATR is backward-looking and not time-scaled, so it's fully superseded by this.")
            else:
                st.caption("No ATM IV available for this name (no live option chain) — can't compute an expected-move sizing guide.")

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
            today_p=datetime.utcnow()
            st.markdown(f"**Expiry — {cur_tkr} (${price:.2f})**")
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
                st.markdown(f"Loaded: {sel_c} — {selected_exp}")
                calls_df,puts_df,dte,chain_err=fetch_chain_cached(sel_c,selected_exp)
                if calls_df is not None:
                    chain=type("_C",(),{"calls":calls_df,"puts":puts_df})()
                    st.markdown(f"<span style='font-size:1.9rem;font-weight:700;'>${price:.2f}</span>"
                                f"&nbsp;&nbsp;·&nbsp;&nbsp;"
                                f"<span style='font-size:1.9rem;font-weight:700;'>{dte} DTE</span>",
                                unsafe_allow_html=True)
                    def fmt_chain(df_raw,side):
                        df_raw=df_raw.copy()
                        df_raw["IV %"]=(df_raw["impliedVolatility"]*100).round(1)
                        df_raw["Moneyness"]=df_raw["strike"].apply(
                            lambda s:"ATM" if abs(s-price)/price<0.02
                            else("ITM" if((s<price and side=="call")or(s>price and side=="put"))else "OTM"))
                        cols=["strike","Moneyness","lastPrice","bid","ask","volume","openInterest","IV %","delta"]
                        available=[c for c in cols if c in df_raw.columns]
                        return(df_raw[available]
                               .rename(columns={"lastPrice":"Last","openInterest":"OI","strike":"Strike","volume":"Volume"})
                               .sort_values("Strike").reset_index(drop=True))
                    col_c,col_p=st.columns(2)
                    with col_c:
                        st.subheader("Calls"); st.dataframe(fmt_chain(chain.calls,"call"),use_container_width=True,hide_index=True)
                    with col_p:
                        st.subheader("Puts");  st.dataframe(fmt_chain(chain.puts,"put"),use_container_width=True,hide_index=True)
                    st.subheader("IV Smile")
                    fig_smile=go.Figure()
                    fig_smile.add_trace(go.Scatter(x=chain.calls["strike"],y=chain.calls["impliedVolatility"]*100,
                        name="Calls IV",mode="lines+markers",line=dict(color="#26a69a",width=2),marker=dict(size=5)))
                    fig_smile.add_trace(go.Scatter(x=chain.puts["strike"], y=chain.puts["impliedVolatility"]*100,
                        name="Puts IV", mode="lines+markers",line=dict(color="#ef5350",width=2),marker=dict(size=5)))
                    fig_smile.add_vline(x=price,line_dash="dash",line_color="white",annotation_text=f"${price:.2f}")
                    fig_smile.update_layout(height=350,template="plotly_dark",xaxis_title="Strike",
                                            yaxis_title="IV (%)",margin=dict(l=0,r=0,t=20,b=0))
                    st.plotly_chart(fig_smile,use_container_width=True)
                    st.subheader("Open Interest by Strike")
                    fig_oi=go.Figure()
                    fig_oi.add_trace(go.Bar(x=chain.calls["strike"],y=chain.calls["openInterest"],name="Call OI",marker_color="#26a69a",opacity=0.75))
                    fig_oi.add_trace(go.Bar(x=chain.puts["strike"], y=chain.puts["openInterest"],name="Put OI", marker_color="#ef5350",opacity=0.75))
                    fig_oi.add_vline(x=price,line_dash="dash",line_color="white")
                    fig_oi.update_layout(barmode="overlay",height=320,template="plotly_dark",
                                         xaxis_title="Strike",yaxis_title="OI",margin=dict(l=0,r=0,t=20,b=0))
                    st.plotly_chart(fig_oi,use_container_width=True)
                    with st.expander("📖 How to read IV Smile & Open Interest"):
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
                               + " Try again in a moment — this is usually a transient "
                                 "Yahoo Finance fetch issue, not cached, so a retry can help.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — MARKET VOLATILITY
# ══════════════════════════════════════════════════════════════════════════════
with tab_vix:
    st.caption("VIX is implied volatility, not historical — it's priced off S&P 500 options "
               "and represents what the market expects annualized volatility to be over the "
               "**next 30 days specifically**. That's why it jumps before an event (Fed, "
               "earnings) even before anything's happened — the next 30 days' uncertainty is "
               "already baked into the option prices it's built from.")
    if vix_df is not None and not vix_df.empty:
        vix_cl=vix_df["Close"].squeeze()
        c1,c2,c3,c4=st.columns(4)
        c1.metric("Current VIX",f"{vix_now:.1f}",f"{vix_chg:+.2f}")
        c2.metric("52wk High",f"{vix_cl.max():.1f}")
        c3.metric("52wk Low",f"{vix_cl.min():.1f}")
        c4.metric("52wk Avg",f"{vix_cl.mean():.1f}")
        fig_vix=make_subplots(rows=2,cols=1,shared_xaxes=True,row_heights=[0.7,0.3],
            subplot_titles=["VIX Level","30-day Rolling Avg"])
        fig_vix.add_trace(go.Scatter(x=vix_df.index,y=vix_cl,name="VIX",fill="tozeroy",
            line=dict(color="#f87171",width=1.5),fillcolor="rgba(248,113,113,0.12)"),row=1,col=1)
        for lo,hi,color,label in VIX_ZONES:
            fig_vix.add_hrect(y0=lo,y1=min(hi,50),fillcolor=color,opacity=0.05,row=1,col=1)
            fig_vix.add_hline(y=lo,line_dash="dot",line_color=color,opacity=0.4,row=1,col=1)
        fig_vix.add_trace(go.Scatter(x=vix_df.index,y=vix_cl.rolling(30).mean(),
            name="30d MA",line=dict(color="#fbbf24",width=1.5)),row=2,col=1)
        fig_vix.update_layout(height=520,template="plotly_dark",margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(fig_vix,use_container_width=True)
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
                st.metric(label,f"{now_v:.1f}",f"{now_v-prev_v:+.2f}")
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

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — SCREENER
# ══════════════════════════════════════════════════════════════════════════════
with tab_screener:
    st.subheader("⚡ Options Suitability Screener")
    st.markdown("<hr style='margin:0.2rem 0 0.6rem 0;'>", unsafe_allow_html=True)
    st.markdown("""<div style="font-size:1.15rem;font-weight:700;font-style:italic;line-height:1.7;margin-bottom:0.3rem;">
    CSP default settings: Δ30 | 30 DTE<br>
    CC default settings: Δ30 | 30 DTE<br>
    LEAP settings: Δ80 | 542 DTE (18 months)
    </div>""", unsafe_allow_html=True)
    st.caption("All greeks via Black-Scholes (strike IV → chain median → HV20 → 30% default)")

    # 26 June — manual target Δ/DTE overrides (Jay: keep Δ30/30DTE as the default, but let a
    # trader dial it manually off-default per chart/support-resistance read, rather than the
    # algorithm silently picking whatever's "closest" with no visibility into the target).
    # Wrapped in st.form (26 June fix) — without a form, every +/- click on a number_input
    # fires an immediate full-script rerun, which re-touches the cached per-ticker data for
    # the whole watchlist and visibly "recalculates" before you've landed on the number you
    # actually wanted. Inside a form, nothing reruns until "Apply targets" is clicked.
    if "applied_targets" not in st.session_state:
        st.session_state["applied_targets"]={"csp_d":30,"csp_dte":30,"cc_d":30,"leap_d":80,"leap_dte":542}

    # 26 June — make the confirm button impossible to miss (neon green, big bold text).
    # button[kind*="FormSubmit"] matches both primaryFormSubmit/secondaryFormSubmit across
    # Streamlit versions; this is the only st.form in the app so the selector is safe — it
    # won't touch the separate "Run Screener" button (a plain st.button, different kind).
    st.markdown("""<style>
        button[kind*="FormSubmit"] {
            background-color: #39FF14 !important;
            color: #000000 !important;
            font-size: 2rem !important;
            font-weight: 900 !important;
            letter-spacing: 0.05em !important;
            padding: 0.9em 3em !important;
            border: none !important;
            border-radius: 8px !important;
            width: 100% !important;
        }
        button[kind*="FormSubmit"]:hover { background-color: #2ee60e !important; }
        /* Scoped to this section's container (st-key-* class, Streamlit >=1.37) so it only
           hits this one expander's header bar, not every expander in the app. */
        .st-key-strike_targeting_section [data-testid="stExpander"] summary p {
            background-color: #04D9FF !important;
            display: inline-block !important;
            color: #000000 !important;
            font-size: 1rem !important;
            font-weight: 800 !important;
            letter-spacing: 0.02em !important;
            padding: 0.5rem 1rem !important;
            border-radius: 8px !important;
        }
        </style>""", unsafe_allow_html=True)

    with st.container(key="strike_targeting_section"):
        _hover_tip("ℹ️ Manual Strike Selection",
                   "Enter your option Strike price and DTE details, then click CONFIRM "
                   "to update the screener tables below.")
        with st.expander("🎯 Manual Strike Selection",expanded=False):
            with st.form("target_form"):
                tcol1,tcol2,tcol3=st.columns(3)
                at=st.session_state["applied_targets"]
                with tcol1:
                    in_delta_csp=st.number_input("CSP target Δ",min_value=5,max_value=50,value=at["csp_d"],step=1)
                    in_dte_csp=st.number_input("CSP target DTE",min_value=21,max_value=45,value=at["csp_dte"],step=1)
                with tcol2:
                    in_delta_cc=st.number_input("CC target Δ",min_value=5,max_value=50,value=at["cc_d"],step=1)
                with tcol3:
                    in_delta_leap=st.number_input("LEAP target Δ",min_value=50,max_value=95,value=at["leap_d"],step=1)
                    in_dte_leap=st.number_input("LEAP target DTE",min_value=180,max_value=900,value=at["leap_dte"],step=1)
                st.caption("Defaults match the locked Δ30/30DTE CSP target (§ trade criteria doc). "
                           "CSP/LEAP DTE targets are clamped to the 21–45 / 180–900 day windows that "
                           "define what counts as a CSP-ish / LEAP-ish expiry at all. Nothing recalculates "
                           "until you click Apply.")
                if st.form_submit_button("CONFIRM CHOICES"):
                    st.session_state["applied_targets"]={"csp_d":in_delta_csp,"csp_dte":in_dte_csp,
                        "cc_d":in_delta_cc,"leap_d":in_delta_leap,"leap_dte":in_dte_leap}
                    st.success("Targets applied — click Run Screener to use them.")

    _at=st.session_state["applied_targets"]
    target_delta_csp,target_dte_csp=_at["csp_d"],_at["csp_dte"]
    target_delta_cc=_at["cc_d"]
    target_delta_leap,target_dte_leap=_at["leap_d"],_at["leap_dte"]

    # §9.3 BB veto Hard/Soft/Off toggle (24 June) — Hard is the original behavior (G3 fail
    # blocks Status). Soft applies a points penalty to each leg's score instead of blocking.
    # Off makes G3 purely informational (shown in the gate reason, never gates or penalizes).
    bbcol1,bbcol2=st.columns([2,2])
    with bbcol1:
        bb_veto_mode=st.radio("BB Veto mode",["Hard","Soft","Off"],index=0,horizontal=True,
            help="Hard: walking the lower band 2+ sessions blocks Status (original behavior). "
                 "Soft: same condition costs each leg's score points instead of blocking. "
                 "Off: informational only — shown in Gates detail, never blocks or penalizes.")
    soft_penalty=10
    if bb_veto_mode=="Soft":
        with bbcol2:
            soft_penalty=st.number_input("Soft penalty (points off each leg's score)",
                min_value=1,max_value=50,value=10,step=1)

    _hover_tip("ℹ️ Run Screener",
               "Scans your whole watchlist for CSP, covered-call, and LEAP candidates using "
               "the target settings above.")
    col_run,col_note=st.columns([1,4])
    with col_run:
        run_btn=st.button("🔍 Run Screener",type="primary",use_container_width=True)
    with col_note:
        st.info("First run ~30–60 s. Cached 30 min.")

    show_debug=st.checkbox("🔧 Diagnostic output",value=False)

    if run_btn:
        if not results:
            st.warning("No market data loaded.")
        else:
            screener_rows=[]; prog=st.progress(0); debug_log=[]; n=len(results)
            for i,(ticker,result) in enumerate(results.items()):
                prog.progress((i+1)/n,text=f"Analysing {ticker} ({i+1}/{n})")
                row=get_screener_row(ticker,result,bb_veto_mode=bb_veto_mode,soft_penalty=soft_penalty,
                                      target_delta_csp=float(target_delta_csp),target_dte_csp=target_dte_csp,
                                      target_delta_cc=float(target_delta_cc),
                                      target_delta_leap=float(target_delta_leap),target_dte_leap=target_dte_leap)
                if row:
                    screener_rows.append(row)
                    debug_log.append(f"✅ {ticker} — CSP {row['csp_score']} ({row['greek_source']}, "
                                     f"inspected {row.get('_inspected')}, scored {row.get('_scored')})")
                else:
                    price=result.get("price"); all_exps=result.get("all_exps",[])
                    fetch_err=result.get("fetch_error")
                    if not price or price<=0: reason="no price"
                    # 25 June fix — distinguish a real fetch failure (rate limit/transient
                    # block, surfaced now instead of swallowed) from "no expiries" with no
                    # further info.
                    elif not all_exps:
                        reason="no expiries"+(f" — {fetch_err}" if fetch_err else "")
                    else:
                        today=datetime.utcnow()
                        ve=next((e for e in all_exps if 21<=(datetime.strptime(e,"%Y-%m-%d")-today).days<=45),None)
                        if not ve: reason=f"no 21–45 DTE expiry (have:{all_exps[:3]})"
                        else:
                            _,pdf,_,chain_err=fetch_chain_cached(ticker,ve)
                            if pdf is None or pdf.empty:
                                reason=f"chain failed for {ve}"+(f" — {chain_err}" if chain_err else "")
                            else:
                                oi_c=pd.to_numeric(pdf.get("openInterest",pd.Series(dtype=float)),errors="coerce").fillna(0)
                                iv_c=pd.to_numeric(pdf.get("impliedVolatility",pd.Series(dtype=float)),errors="coerce").fillna(0)
                                reason=(f"chain OK ({len(pdf)} rows, OI≥1:{int((oi_c>=1).sum())}, "
                                        f"IV>0:{int((iv_c>0).sum())}) — find_target_strike returned None")
                    debug_log.append(f"❌ {ticker} — skipped: {reason}")
            prog.empty()
            st.session_state["screener_results"]=screener_rows
            st.session_state["screener_debug"]=debug_log
            st.session_state["screener_schema_version"]=_SCREENER_SCHEMA_VERSION

    if show_debug and st.session_state.get("screener_debug"):
        with st.expander("🔧 Log",expanded=True):
            for line in st.session_state["screener_debug"]: st.text(line)

    # Discard a stale cache from before a row-schema change (see _SCREENER_SCHEMA_VERSION
    # above) instead of crashing on a missing key — just falls back to "click Run Screener".
    if st.session_state.get("screener_schema_version")==_SCREENER_SCHEMA_VERSION:
        screener_rows=st.session_state.get("screener_results",[])
    else:
        screener_rows=[]

    if screener_rows:
        screener_rows_sorted=sorted(screener_rows,key=lambda x:x["csp_score"],reverse=True)
        low_prec=[r["ticker"] for r in screener_rows_sorted if r.get("greek_source") in ("bs_hv20","bs_default")]
        if low_prec:
            st.warning(f"⚠️ Low-precision greeks for: **{', '.join(low_prec)}**")

        # §9.4/§10.5 step 5 — split into three per-leg tables (24 June). Every ticker appears
        # in every table (revised §10.4.1 — the mean-reversion signal is a column, not a
        # filter), so "qualifies for the CC table" is no longer a thing. Gates/Status are now
        # per-leg (22 June) — CSP and CC each get their own gate_result with an opposite-
        # direction G4 Median check; LEAP still shares the original 3-gate logic.
        def _gate_cols(r, leg="leap"):
            # 22 June — Gates are now per-leg: CSP and CC each carry their own gate_result
            # (with a 4th Median gate, opposite pass conditions per leg); LEAP still uses the
            # original 3-gate result. leg picks which one this table's row should read.
            gr=r[f"gate_result_{leg}"]; gates=gr["gates"]
            icons="".join("✅" if gates[k]["pass"] else "❌" for k in sorted(gates.keys()))
            return icons, ("🟢 TRADE" if gr["all_pass"] else "🔴 WAIT")

        # 22 June — column_config.Column(help=...) on st.dataframe (added below) gives a
        # native hover icon on the column header, but it silently doesn't render on the
        # deployed Streamlit Cloud runtime — same root cause as the st.expander(help=...)
        # crash fixed earlier this session (deployed version is older than what the column
        # header tooltip feature needs). st.caption(help=...) IS confirmed working on this
        # deployment (that's how the Manual Strike Selection info icon was fixed), so this
        # glossary strip uses that as the reliable hover mechanism — keep column_config's
        # help= too since it's free and will start working on its own once the Streamlit
        # Cloud app is rebooted onto a current version.
        def _col_legend(items, per_row=7):
            for i in range(0, len(items), per_row):
                chunk = items[i:i+per_row]
                cols = st.columns(len(chunk))
                for c, (label, text) in zip(cols, chunk):
                    with c:
                        st.caption(label, help=text)

        # 22 June — prototype: pure-CSS hover tooltip, no click required (Jay: "cursor just
        # touches the column header and automatically pops up... HTML-like"). This is plain
        # HTML/CSS (:hover + a positioned span), not a Streamlit feature, so it doesn't depend
        # on the deployed Streamlit Cloud version at all.
        # Important constraint: this floats in a strip just above the table, not literally
        # fused into st.dataframe's own header row — Streamlit renders dataframe headers via
        # a canvas-based grid component (glide-data-grid), which doesn't accept arbitrary HTML
        # in its cells, so true in-grid hover isn't reachable without a custom component.
        # Trying this on CSP only first — once Jay confirms it actually pops on hover (no
        # click) on the deployed app, roll the same helper out to CC_LEGEND/LEAP_LEGEND too.
        def _col_legend_hover(items):
            style = """<style>
            .jay-tt-row{display:flex;flex-wrap:wrap;gap:0.7rem 1.5rem;margin:0.2rem 0 0.7rem 0;}
            .jay-tt-wrap{position:relative;display:inline-block;cursor:help;
                font-size:0.85rem;color:#9ca3af;border-bottom:1px dotted #6b7280;}
            .jay-tt-wrap .jay-tt-text{
                visibility:hidden;opacity:0;transition:opacity 0.15s ease;
                position:absolute;bottom:135%;left:50%;transform:translateX(-50%);
                background:#1f2937;color:#f9fafb;text-align:center;border-radius:6px;
                padding:6px 10px;font-size:0.78rem;font-weight:400;line-height:1.35;
                white-space:normal;width:max-content;max-width:220px;
                box-shadow:0 4px 14px rgba(0,0,0,0.4);z-index:9999;pointer-events:none;
            }
            .jay-tt-wrap .jay-tt-text::after{
                content:"";position:absolute;top:100%;left:50%;margin-left:-5px;
                border-width:5px;border-style:solid;
                border-color:#1f2937 transparent transparent transparent;
            }
            .jay-tt-wrap:hover .jay-tt-text{visibility:visible;opacity:1;}
            </style>"""
            spans = "".join(
                f'<span class="jay-tt-wrap">{label}<span class="jay-tt-text">{text}</span></span>'
                for label, text in items
            )
            st.markdown(style + f'<div class="jay-tt-row">{spans}</div>', unsafe_allow_html=True)

        # 22 June (step 2) — Jay confirmed the hover-strip prototype above works and asked for
        # the popup to live on the ACTUAL column header instead of a separate row. st.dataframe
        # can't do that (its header is a canvas-rendered grid, not real HTML — see comment
        # above), so this renders the table itself as a plain HTML <table>, with each <th>
        # wrapping the same hover-tooltip span used above. Tradeoff vs st.dataframe: we lose
        # the built-in click-to-sort columns, fullscreen/search icon, and copy-to-clipboard —
        # gained true in-header hover. Scrollable via a max-height wrapper with a sticky thead
        # so it still behaves like the old fixed-height table. _html_table itself now lives
        # at module scope (above the Sidebar section) so the Watchlist Overview table in the
        # Overview tab can reuse it too — see comment there.
        _CSP_LEGEND = [
            ("Ticker","Stock symbol"),("Price","Current stock price"),
            ("Expiry","Option expiration date"),("DTE","Days to Expiry"),
            ("Strike","Option strike price"),("Δ","Delta — sensitivity per $1 stock move"),
            ("θ/day","Theta — premium collected per day (you're the seller)"),
            ("Put IV %","Implied volatility of this put"),
            ("OI","Open interest — contracts outstanding"),
            ("Vol","Volume — contracts traded today"),
            ("Spread %","Bid-ask spread, % of mid"),("Mid","Midpoint of bid/ask"),
            ("POP %","Probability of profit (Black-Scholes)"),
            ("Liquidity","Liquidity score 0–100 (OI + volume)"),
            ("NIS","Normalised Income Score — premium per $ risk & time"),
            ("Score","Composite suitability score (NIS + DTE fit + Δ fit)"),
            ("Timing","Mean-reversion timing signal (flag, not a filter)"),
            ("Gates","G1 Trend · G2 Session · G3 BB Veto · G4 Median — pass/fail "
                     "(G4 fails if price is above the median band — catch the bounce early, "
                     "before the up-move's runway is used up)"),
            ("Status","Trade/Wait — all four gates must pass"),
        ]
        _CC_LEGEND = [(l, t) if l!="Put IV %" else ("Call IV %","Implied volatility of this call") for l,t in _CSP_LEGEND]
        _cc_gates_i = next(i for i,(l,_) in enumerate(_CC_LEGEND) if l=="Gates")
        _CC_LEGEND[_cc_gates_i] = ("Gates","G1 Trend · G2 Session · G3 BB Veto · G4 Median — pass/fail "
                                           "(G4 fails if price is below the median band — opposite of CSP)")
        _LEAP_LEGEND = [(l,t) for l,t in _CSP_LEGEND if l not in ("Put IV %","Timing")]
        _LEAP_LEGEND.insert(7, ("IV %","Implied volatility of this option"))
        _leap_gates_i = next(i for i,(l,_) in enumerate(_LEAP_LEGEND) if l=="Gates")
        _LEAP_LEGEND[_leap_gates_i] = ("Gates","G1 Trend · G2 Session · G3 BB Veto — pass/fail "
                                                "(no median gate for LEAP)")
        _leap_status_i = next(i for i,(l,_) in enumerate(_LEAP_LEGEND) if l=="Status")
        _LEAP_LEGEND[_leap_status_i] = ("Status","Trade/Wait — all three gates must pass")
        _leap_nis_i = next(i for i,(l,_) in enumerate(_LEAP_LEGEND) if l=="NIS")
        _LEAP_LEGEND[_leap_nis_i] = ("NIS","Normalised Income Score, inverted — lower means cheaper to buy")
        _leap_theta_i = next(i for i,(l,_) in enumerate(_LEAP_LEGEND) if l=="θ/day")
        _LEAP_LEGEND[_leap_theta_i] = ("θ/day","Theta — premium you pay away per day (you're the buyer)")
        _leap_intr_i = next(i for i,(l,_) in enumerate(_LEAP_LEGEND) if l=="Mid") + 1
        _LEAP_LEGEND[_leap_intr_i:_leap_intr_i] = [
            ("Intrinsic","In-the-money value"),
            ("Extrinsic $","Time value paid above intrinsic"),
            ("Extrinsic $/day","Time value cost per day"),
        ]

        # 26 June — unified column set/order across CSP/CC/LEAP (Jay's request): Greeks column
        # dropped (superfluous — greek source is already surfaced via the low-precision
        # warning above and the per-row tooltip), Ann Return %/Breakeven/BE % dropped from
        # CSP as clutter, and CC/LEAP now mirror CSP's column set/order as closely as possible.
        # LEAP has no mean-reversion Timing signal (only computed for CSP/CC), so that column
        # is the one unavoidable omission there.
        st.subheader("CSP Targets")
        csp_rows=[]
        for r in screener_rows_sorted:
            icons,status=_gate_cols(r,"csp")
            csp_rows.append({"Ticker":r["ticker"],"Price":f"${r['price']:.2f}",
                "Expiry":r["expiry"],"DTE":r["dte"],
                "Strike":f"${r['csp_strike']:.1f}","Δ":r["csp_delta"],"θ/day":f"${r['csp_theta']:.3f}",
                "Put IV %":r["csp_iv"] if r["csp_iv"] else "—",
                "OI":r["csp_oi"],"Vol":r.get("csp_volume","—"),
                "Spread %":r["csp_spread"] if r["csp_spread"] else "—",
                "Mid":r.get("csp_mid","—"),
                "POP %":r["csp_pop"] if r.get("csp_pop") is not None else "—",
                "Liquidity":r.get("csp_liquidity","—"),
                "NIS":r["nis"],"Score":r["csp_score"],"Timing":r.get("csp_timing_label","—"),
                "Gates":icons,"Status":status})
        csp_h=38+len(csp_rows)*35+12  # no cap — full table height, no internal scrollbar
        _html_table(csp_rows, _CSP_LEGEND, csp_h)

        st.subheader("CC Targets")
        cc_sorted=sorted(screener_rows_sorted,key=lambda x:x["cc_score"],reverse=True)
        cc_rows=[]
        for r in cc_sorted:
            icons,status=_gate_cols(r,"cc")
            has_cc=r.get("cc_strike") is not None
            cc_rows.append({"Ticker":r["ticker"],"Price":f"${r['price']:.2f}",
                "Expiry":r["expiry"],"DTE":r["dte"],
                "Strike":f"${r['cc_strike']:.1f}" if has_cc else "—","Δ":r.get("cc_delta","—"),
                "θ/day":f"${r['cc_theta']:.3f}" if r.get("cc_theta") is not None else "—",
                "Call IV %":r["cc_iv"] if r.get("cc_iv") else "—",
                "OI":r.get("cc_oi","—"),"Vol":r.get("cc_volume","—"),
                "Spread %":r["cc_spread"] if r.get("cc_spread") else "—",
                "Mid":r.get("cc_mid","—"),
                "POP %":r["cc_pop"] if r.get("cc_pop") is not None else "—",
                "Liquidity":r.get("cc_liquidity","—"),
                "NIS":r.get("cc_nis","—"),"Score":r["cc_score"],"Timing":r.get("cc_timing_label","—"),
                "Gates":icons,"Status":status})
        cc_h=38+len(cc_rows)*35+12  # no cap — full table height, no internal scrollbar
        _html_table(cc_rows, _CC_LEGEND, cc_h)

        st.subheader("LEAP Targets")
        leap_sorted=sorted(screener_rows_sorted,key=lambda x:(x["leap_score"] if x.get("leap_score") is not None else -1),reverse=True)
        leap_rows=[]
        for r in leap_sorted:
            icons,status=_gate_cols(r,"leap")
            has_leap=r.get("leap_strike") is not None
            leap_rows.append({"Ticker":r["ticker"],"Price":f"${r['price']:.2f}",
                "Expiry":r.get("leap_expiry","—") if has_leap else "—",
                "DTE":r.get("leap_dte","—") if has_leap else "—",
                "Strike":f"${r['leap_strike']:.1f}" if has_leap else "—","Δ":r.get("leap_delta","—"),
                "θ/day":f"${r['leap_theta']:.3f}" if r.get("leap_theta") is not None else "—",
                "IV %":r["leap_iv"] if r.get("leap_iv") else "—",
                "OI":r.get("leap_oi","—"),"Vol":r.get("leap_volume","—"),
                "Spread %":r["leap_spread"] if r.get("leap_spread") else "—",
                "Mid":r.get("leap_mid","—"),
                "Intrinsic":f"${r['leap_intrinsic']:.2f}" if r.get("leap_intrinsic") is not None else "—",
                "Extrinsic $":f"${r['leap_extrinsic']:.2f}" if r.get("leap_extrinsic") is not None else "—",
                "Extrinsic $/day":f"${r['leap_extrinsic_per_day']:.3f}" if r.get("leap_extrinsic_per_day") is not None else "—",
                "POP %":r["leap_pop"] if r.get("leap_pop") is not None else "—",
                "Liquidity":r.get("leap_liquidity","—"),
                "NIS":r.get("leap_nis","—"),
                "Score":r["leap_score"] if r.get("leap_score") is not None else "—",
                "Gates":icons,"Status":status})
        leap_h=38+len(leap_rows)*35+12  # no cap — full table height, no internal scrollbar
        _html_table(leap_rows, _LEAP_LEGEND, leap_h)
        st.caption("LEAP now scores a real ~80Δ contract in the 180–900 DTE window (closest to "
                   "542 DTE) — fixed 24 June, was previously reusing the CSP's ~30Δ/30DTE "
                   "numbers. Shows — if no expiry in that window exists for the ticker.")

        tickers=[r["ticker"] for r in screener_rows_sorted]
        csp_scores=[r["csp_score"] for r in screener_rows_sorted]

        st.subheader("CSP Suitability Ranking")
        fig_sc=go.Figure(go.Bar(x=tickers,y=csp_scores,marker_color=[score_color(s) for s in csp_scores],
            text=[f"{s:.0f}" for s in csp_scores],textposition="outside"))
        for lvl,col,lbl in [(80,"#22c55e","80—Optimal"),(60,"#eab308","60—Acceptable"),(40,"#f97316","40—Marginal")]:
            fig_sc.add_hline(y=lvl,line_dash="dash",line_color=col,annotation_text=lbl)
        fig_sc.update_layout(height=340,template="plotly_dark",yaxis_title="CSP Score",
                              yaxis_range=[0,115],margin=dict(l=0,r=0,t=20,b=0))
        st.plotly_chart(fig_sc,use_container_width=True)

        st.subheader("Strategy Score Comparison")
        fig_cmp=go.Figure()
        fig_cmp.add_trace(go.Bar(name="CSP",   x=tickers,y=[r["csp_score"]  for r in screener_rows_sorted],marker_color="#22c55e",opacity=0.85))
        fig_cmp.add_trace(go.Bar(name="CC",    x=tickers,y=[r["cc_score"]   for r in screener_rows_sorted],marker_color="#60a5fa",opacity=0.85))
        fig_cmp.add_trace(go.Bar(name="LEAP*", x=tickers,y=[r["leap_score"] for r in screener_rows_sorted],marker_color="#a78bfa",opacity=0.85))
        fig_cmp.update_layout(barmode="group",height=340,template="plotly_dark",
                               yaxis_title="Score",yaxis_range=[0,115],
                               legend=dict(orientation="h",y=1.05,x=0),margin=dict(l=0,r=0,t=20,b=0))
        st.plotly_chart(fig_cmp,use_container_width=True)

        st.subheader("Four-Gate Filter Detail (CSP)")
        st.caption(f"G1=Trend  G2=Session  G3=BB Veto (mode: {bb_veto_mode}"
                   + (f", −{soft_penalty} pts" if bb_veto_mode=="Soft" else "")
                   + ")  G4=Median (CSP fails above median)")
        for r in screener_rows_sorted:
            gr=r["gate_result_csp"]; gates=gr["gates"]; icon="🟢" if gr["all_pass"] else "🔴"
            with st.expander(f"{icon} {r['ticker']}  CSP:{r['csp_score']}  Strike${r['csp_strike']:.1f}  Δ{r['csp_delta']}  θ${r['csp_theta']:.3f}/d"):
                gcols=st.columns(len(gates))
                for idx,(gk,gv) in enumerate(gates.items()):
                    gcols[idx].markdown(f"**{gv['label']}** {'✅' if gv['pass'] else '❌'}")
                    gcols[idx].caption(gv["reason"])
                st.divider()
                rc1,rc2=st.columns(2)
                csp_ar = r.get('csp_ann_return'); csp_pop = r.get('csp_pop'); csp_liq = r.get('csp_liquidity')
                rc1.markdown(f"""
**CSP Strike**
- Strike: **${r['csp_strike']:.1f}**
- Delta: **{r['csp_delta']}** ({greek_source_label(r.get('greek_source'))})
- Theta: **${r['csp_theta']:.3f}/day**
- IV: **{r['csp_iv'] or '—'}%**
- OI: **{r['csp_oi']}** · Volume: **{r.get('csp_volume','—')}** · Liquidity: **{csp_liq if csp_liq is not None else '—'}**
- Spread: **{r['csp_spread']}%** {'✅' if r['csp_spread'] and r['csp_spread']<20 else '⚠️ Wide' if r['csp_spread'] else '—'}
- Mid premium: **${r.get('csp_mid','—')}**
- Annualized return: **{f'{csp_ar:.1f}%' if csp_ar is not None else '—'}**
- Breakeven: **${r.get('csp_breakeven','—')}** ({f"{r.get('csp_breakeven_pct'):.1f}% below spot" if r.get('csp_breakeven_pct') is not None else '—'})
- POP (N(d2)): **{f'{csp_pop:.1f}%' if csp_pop is not None else '—'}**

**CSP Timing — {r.get('csp_timing_label','—')}** (score {r.get('csp_timing_score',0)})
                """)
                for reason in r.get("csp_timing_reasons",[]): rc1.caption(reason)
                if r["cc_strike"]:
                    cc_ar = r.get('cc_ann_return'); cc_pop = r.get('cc_pop'); cc_liq = r.get('cc_liquidity')
                    rc2.markdown(f"""
**CC Strike**
- Strike: **${r['cc_strike']:.1f}**
- Delta: **{r['cc_delta']}**
- CC Score: **{r['cc_score']}**
- OI / Volume / Liquidity: **{r.get('cc_oi','—') if r.get('cc_oi') is not None else '—'} / {r.get('cc_volume','—')} / {cc_liq if cc_liq is not None else '—'}**
- Mid premium: **${r.get('cc_mid','—')}**
- Annualized return: **{f'{cc_ar:.1f}%' if cc_ar is not None else '—'}**
- POP (N(-d2), call expires OTM): **{f'{cc_pop:.1f}%' if cc_pop is not None else '—'}**

**CC Timing — {r.get('cc_timing_label','—')}** (score {r.get('cc_timing_score',0)})
                    """)
                    for reason in r.get("cc_timing_reasons",[]): rc2.caption(reason)

                st.divider()
                if r.get("leap_strike"):
                    st.markdown(f"""
**LEAP Contract** — fixed 24 June, now a real ~80Δ long-dated call (was reusing CSP's numbers)
- Expiry: **{r.get('leap_expiry','—')}** ({r.get('leap_dte','—')} DTE)
- Strike: **${r['leap_strike']:.1f}** · Delta: **{r['leap_delta']}** ({greek_source_label(r.get('leap_greek_source'))})
- Theta: **${r['leap_theta']:.3f}/day** · IV: **{r['leap_iv'] or '—'}%** · OI: **{r.get('leap_oi','—')}**
- Mid premium: **${r.get('leap_mid','—')}** · LEAP NIS: **{r.get('leap_nis','—')}**
- Intrinsic: **${r.get('leap_intrinsic','—')}** · Extrinsic (time value): **${r.get('leap_extrinsic','—')}** · Avg cost to hold: **${r.get('leap_extrinsic_per_day','—')}/day**
- **LEAP Score: {r.get('leap_score','—')}**
                    """)
                else:
                    st.caption("LEAP — no expiry in the 180–900 DTE window for this ticker.")

    elif "screener_results" not in st.session_state:
        st.markdown("""<div style='text-align:center;padding:60px;color:#94a3b8;'>
        <h3>Click <strong>Run Screener</strong> to analyse your watchlist</h3>
        <p>Δ30 · 30 DTE · Black-Scholes greeks · Works any time of day</p>
        </div>""",unsafe_allow_html=True)
    else:
        st.warning("No screener data. Enable diagnostic output and re-run the screener.")
