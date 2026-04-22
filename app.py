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

# ── Auto-refresh (optional) ────────────────────────────────────────────────────
try:
    from streamlit_autorefresh import st_autorefresh
    HAS_AUTOREFRESH = True
except ImportError:
    HAS_AUTOREFRESH = False

DEFAULT_WATCHLIST = ["NVDA", "META", "TSLA", "IBIT", "GLD", "GDXJ", "BE", "VST", "CRWV"]

VIX_ZONES = [
    (0,  15, "#16a34a", "LOW — Ideal LEAP buying zone"),
    (15, 20, "#ca8a04", "NORMAL — Balanced regime"),
    (20, 30, "#ea580c", "ELEVATED — CC writing premium rich"),
    (30, 99, "#dc2626", "HIGH — Aggressive premium selling regime"),
]

# ── Market Pulse tickers ───────────────────────────────────────────────────────
# (yahoo_ticker, display_label, prefix, is_yield)
PULSE_TICKERS = [
    ("SPY",       "S&P 500",    "$",  False),
    ("QQQ",       "Nasdaq 100", "$",  False),
    ("DIA",       "Dow Jones",  "$",  False),
    ("IWM",       "R2000",      "$",  False),
    ("DX-Y.NYB",  "DXY",        "",   False),
    ("CL=F",      "Crude Oil",  "$",  False),
    ("GC=F",      "Gold",       "$",  False),
    ("BTC-USD",   "Bitcoin",    "$",  False),
    ("^TNX",      "10Y Yield",  "",   True),
    ("^IRX",      "3M Yield",   "",   True),
]

# VIX term structure tickers
VIX_TERM_TICKERS = [
    ("^VIX9D",  "9-Day VIX"),
    ("^VIX",    "30-Day VIX"),
    ("^VIX3M",  "3-Month VIX"),
    ("^VIX6M",  "6-Month VIX"),
]

# ── Screener constants ─────────────────────────────────────────────────────────
NIS_FLOOR = 0.00157
NIS_CEIL  = 0.01253

STRATEGY_PARAMS = {
    "CSP": {
        "delta_opt": 18, "delta_lo": 10, "delta_hi": 30,
        "dte_opt": 37,  "dte_lo": 21,  "dte_hi": 60,
        "w_iv": 0.50,   "w_dte": 0.30, "w_delta": 0.20,
        "iv_dir": 1,    "option_type": "put",
    },
    "CC": {
        "delta_opt": 35, "delta_lo": 20, "delta_hi": 50,
        "dte_opt": 37,  "dte_lo": 21,  "dte_hi": 60,
        "w_iv": 0.50,   "w_dte": 0.30, "w_delta": 0.20,
        "iv_dir": 1,    "option_type": "call",
    },
    "LEAP": {
        "delta_opt": 80, "delta_lo": 60, "delta_hi": 95,
        "dte_opt": 547, "dte_lo": 180, "dte_hi": 900,
        "w_iv": 0.30,   "w_dte": 0.40, "w_delta": 0.30,
        "iv_dir": -1,   "option_type": "call",
    },
}

RISK_FREE_RATE = 0.045

# ── Session state ──────────────────────────────────────────────────────────────
if "watchlist" not in st.session_state:
    st.session_state.watchlist = DEFAULT_WATCHLIST.copy()

# ══════════════════════════════════════════════════════════════════════════════
# MARKET PULSE FETCHERS  (short TTL — 60s)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=60, show_spinner=False)
def fetch_quote(ticker):
    """Fetch last price + 1-day % change for a single ticker."""
    try:
        df = yf.download(ticker, period="5d", auto_adjust=True, progress=False)
        if df is None or df.empty:
            return None
        cl = df["Close"].squeeze().dropna()
        if len(cl) < 2:
            return None
        curr = float(cl.iloc[-1])
        prev = float(cl.iloc[-2])
        return {"price": curr, "pct": (curr / prev - 1) * 100}
    except Exception:
        return None

@st.cache_data(ttl=60, show_spinner=False)
def fetch_fear_greed():
    """CNN Fear & Greed index — public endpoint, no auth required."""
    try:
        r = requests.get(
            "https://production.dataviz.cnn.io/index/fearandgreed/graphdata",
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=8,
        )
        d = r.json()["fear_and_greed"]
        return round(float(d["score"]), 1), str(d["rating"]).replace("_", " ").title()
    except Exception:
        return None, None

@st.cache_data(ttl=60, show_spinner=False)
def fetch_vix_term():
    """Fetch VIX9D, VIX30 (^VIX), VIX3M, VIX6M for term structure."""
    out = {}
    for ticker, label in VIX_TERM_TICKERS:
        q = fetch_quote(ticker)
        if q:
            out[label] = q["price"]
    return out

@st.cache_data(ttl=60, show_spinner=False)
def fetch_skew():
    """CBOE SKEW index — proxy for tail risk / black swan demand."""
    q = fetch_quote("^SKEW")
    return q["price"] if q else None

# ══════════════════════════════════════════════════════════════════════════════
# EXISTING DATA FETCHERS  (long TTL — 30 min)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=1800, show_spinner=False)
def fetch_prices(ticker, period="1y"):
    try:
        df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
        if df.empty or len(df) < 30:
            return None
        return df
    except Exception:
        return None

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_vix(period="1y"):
    return fetch_prices("^VIX", period)

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_all_expiries(ticker):
    try:
        tk = yf.Ticker(ticker)
        return list(tk.options) or []
    except Exception:
        return []

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_chain_cached(ticker, expiry):
    try:
        tk    = yf.Ticker(ticker)
        chain = tk.option_chain(expiry)
        dte   = (datetime.strptime(expiry, "%Y-%m-%d") - datetime.utcnow()).days
        return chain.calls, chain.puts, dte
    except Exception:
        return None, None, None

def fetch_chain(ticker, expiry):
    calls, puts, dte = fetch_chain_cached(ticker, expiry)
    if calls is None:
        return None, None
    class _Chain:
        pass
    c = _Chain()
    c.calls = calls
    c.puts  = puts
    return c, dte

# ══════════════════════════════════════════════════════════════════════════════
# BLACK-SCHOLES GREEKS
# ══════════════════════════════════════════════════════════════════════════════
def _bs_greeks(S, K, T, sigma, r=RISK_FREE_RATE, option_type="call"):
    try:
        if any(x is None for x in (S, K, T, sigma)):
            return None, None
        S, K, T, sigma = float(S), float(K), float(T), float(sigma)
        if (pd.isna(S) or pd.isna(K) or pd.isna(T) or pd.isna(sigma)
                or T <= 0 or sigma <= 0 or S <= 0 or K <= 0):
            return None, None
        sigma  = max(0.05, min(sigma, 3.0))
        T_yr   = T / 365.0
        sqrtT  = math.sqrt(T_yr)
        d1     = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T_yr) / (sigma * sqrtT)
        d2     = d1 - sigma * sqrtT
        delta  = norm.cdf(d1) if option_type == "call" else norm.cdf(d1) - 1.0
        pdf_d1 = norm.pdf(d1)
        theta_call = (-(S * pdf_d1 * sigma) / (2 * sqrtT)
                      - r * K * math.exp(-r * T_yr) * norm.cdf(d2))
        theta = (theta_call if option_type == "call"
                 else theta_call + r * K * math.exp(-r * T_yr)) / 365.0
        return round(abs(delta) * 100.0, 2), round(abs(theta), 4)
    except Exception:
        return None, None

# ── Technical indicators ───────────────────────────────────────────────────────
def calc_hv(close, window=20):
    log_ret = np.log(close / close.shift(1))
    return log_ret.rolling(window).std() * np.sqrt(252) * 100

def calc_iv_rank(hv_series, lookback=252):
    s = hv_series.dropna().tail(lookback)
    if len(s) < 30:
        return None
    lo, hi, cur = s.min(), s.max(), s.iloc[-1]
    return round((cur - lo) / (hi - lo) * 100, 1) if hi != lo else 50.0

def calc_iv_percentile(hv_series, lookback=252):
    s = hv_series.dropna().tail(lookback)
    if len(s) < 30:
        return None
    cur = s.iloc[-1]
    return round((s < cur).sum() / len(s) * 100, 1)

def calc_rsi(close, window=14):
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    ag    = gain.ewm(alpha=1/window, min_periods=window).mean()
    al    = loss.ewm(alpha=1/window, min_periods=window).mean()
    rs    = ag / al.replace(0, np.nan)
    return 100 - 100 / (1 + rs)

def calc_atr(df, window=14):
    hi = df["High"].squeeze()
    lo = df["Low"].squeeze()
    cl = df["Close"].squeeze()
    pc = cl.shift(1)
    tr = pd.concat([hi - lo, (hi - pc).abs(), (lo - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(window).mean()

def calc_bb_width(close, window=20):
    mid = close.rolling(window).mean()
    std = close.rolling(window).std()
    return ((mid + 2*std) - (mid - 2*std)) / mid * 100

def calc_bb_bands(close, window=20):
    mid   = close.rolling(window).mean()
    std   = close.rolling(window).std()
    upper = mid + 2 * std
    lower = mid - 2 * std
    return upper, mid, lower

def calc_stochastics(df, k_period=14, d_period=3):
    high  = df["High"].squeeze()
    low   = df["Low"].squeeze()
    close = df["Close"].squeeze()
    lo_min = low.rolling(k_period).min()
    hi_max = high.rolling(k_period).max()
    raw_k  = 100.0 * (close - lo_min) / (hi_max - lo_min + 1e-10)
    k      = raw_k.rolling(d_period).mean()
    d      = k.rolling(d_period).mean()
    return k, d

def calc_atm_iv(chain, price):
    try:
        c_atm = chain.calls.iloc[(chain.calls["strike"] - price).abs().argsort()[:1]]
        p_atm = chain.puts.iloc[(chain.puts["strike"]  - price).abs().argsort()[:1]]
        return (
            round(c_atm["impliedVolatility"].values[0] * 100, 1),
            round(p_atm["impliedVolatility"].values[0] * 100, 1),
        )
    except Exception:
        return None, None

def calc_pcr(chain):
    try:
        pv = chain.puts["volume"].sum()
        cv = chain.calls["volume"].sum()
        return round(pv / cv, 2) if cv > 0 else None
    except Exception:
        return None

# ── Screener helpers ───────────────────────────────────────────────────────────
def find_target_strike(chain_df, target_delta_abs, option_type, price, dte, hv_pct=None):
    if chain_df is None or chain_df.empty:
        return None
    if price is None or price <= 0 or dte is None or dte <= 0:
        return None

    iv_col_raw = chain_df.get("impliedVolatility", pd.Series(dtype=float))
    iv_clean   = pd.to_numeric(iv_col_raw, errors="coerce").dropna() if iv_col_raw is not None else pd.Series(dtype=float)
    valid_ivs  = iv_clean[(iv_clean > 0.01) & (iv_clean < 5.0)]
    median_iv  = float(valid_ivs.median()) if len(valid_ivs) >= 3 else None

    best      = None
    min_score = float("inf")
    inspected = 0
    scored    = 0

    for _, row in chain_df.iterrows():
        inspected += 1
        try:
            K      = float(row.get("strike", 0) or 0)
            oi_raw = row.get("openInterest", 0)
            oi     = 0.0 if pd.isna(oi_raw) else float(oi_raw or 0)
            iv_raw = row.get("impliedVolatility", 0)
            iv     = 0.0 if pd.isna(iv_raw) else float(iv_raw or 0)
            bid    = float(row.get("bid", 0) or 0)
            ask    = float(row.get("ask", 0) or 0)
        except (TypeError, ValueError):
            continue

        if K <= 0:
            continue

        delta_raw = row.get("delta", None)
        theta_raw = row.get("theta", None)
        yahoo_has_greeks = False
        if delta_raw is not None and theta_raw is not None:
            try:
                if not pd.isna(delta_raw) and not pd.isna(theta_raw):
                    yahoo_has_greeks = True
            except Exception:
                pass

        if yahoo_has_greeks:
            d_abs        = abs(float(delta_raw)) * 100.0
            theta        = abs(float(theta_raw))
            greek_source = "yahoo"
        else:
            if 0.01 < iv < 5.0:
                sigma     = iv;        iv_source = "strike"
            elif median_iv is not None:
                sigma     = median_iv; iv_source = "chain_median"
            elif hv_pct is not None and hv_pct > 0:
                sigma     = float(hv_pct); iv_source = "hv20"
            else:
                sigma     = 0.30;      iv_source = "default"

            d_abs, theta = _bs_greeks(price, K, dte, sigma, option_type=option_type)
            if d_abs is None or theta is None:
                continue
            greek_source = f"bs_{iv_source}"

        if pd.isna(d_abs) or pd.isna(theta):
            continue

        scored     += 1
        delta_diff  = abs(d_abs - target_delta_abs)
        if   oi >= 100: oi_penalty = 0
        elif oi >= 10:  oi_penalty = 0.5
        elif oi >= 1:   oi_penalty = 2
        else:           oi_penalty = 4
        score = delta_diff + oi_penalty

        if score < min_score:
            min_score = score
            mid        = (bid + ask) / 2.0
            eff_iv_pct = round(iv * 100.0, 1) if (greek_source == "yahoo" and iv > 0.01) else round(sigma * 100.0, 1)
            best = {
                "strike":       K,
                "delta":        round(d_abs, 1),
                "theta":        round(theta, 4),
                "iv":           eff_iv_pct,
                "oi":           int(oi),
                "bid":          bid,
                "ask":          ask,
                "spread_pct":   round((ask - bid) / mid * 100, 1) if mid > 0 else None,
                "greek_source": greek_source,
                "_inspected":   inspected,
                "_scored":      scored,
            }
    return best

def calc_nis(theta, dte, price):
    if theta <= 0 or dte <= 0 or price <= 0:
        return 0.0
    raw = theta * math.sqrt(dte) / price
    return min(100.0, max(0.0, (raw - NIS_FLOOR) / (NIS_CEIL - NIS_FLOOR) * 100.0))

def _tri_score(value, optimal, lo, hi):
    if value < lo or value > hi:
        return 0.0
    half = max(abs(optimal - lo), abs(hi - optimal))
    return max(0.0, 100.0 * (1.0 - abs(value - optimal) / half)) if half > 0 else 100.0

def calc_suitability(nis, dte, delta_abs, strategy):
    p = STRATEGY_PARAMS[strategy]
    a = nis if p["iv_dir"] == 1 else (100.0 - nis)
    b = _tri_score(dte,       p["dte_opt"],   p["dte_lo"],   p["dte_hi"])
    c = _tri_score(delta_abs, p["delta_opt"], p["delta_lo"], p["delta_hi"])
    return round(p["w_iv"] * a + p["w_dte"] * b + p["w_delta"] * c, 1)

def score_color(s):
    if s >= 80: return "#22c55e"
    if s >= 60: return "#eab308"
    if s >= 40: return "#f97316"
    return "#ef4444"

def calc_four_gates(r):
    df    = r.get("df")
    cl    = r.get("cl")
    price = r.get("price", 0)
    pct   = r.get("pct", 0)
    gates = {}

    if cl is not None and len(cl.dropna()) >= 50:
        ma20 = float(cl.rolling(20).mean().dropna().iloc[-1])
        ma50 = float(r.get("ma50", 0))
        g1   = (ma20 > ma50) or (price > ma20)
        gates["G1"] = {"pass": g1, "label": "Trend (MA)",
            "reason": f"20MA={ma20:.2f} {'>' if ma20>ma50 else '<'} 50MA={ma50:.2f}  |  Price {'>' if price>ma20 else '<'} 20MA"}
    else:
        gates["G1"] = {"pass": False, "label": "Trend (MA)", "reason": "Insufficient history"}

    if df is not None and len(df) >= 25:
        k_s, d_s = calc_stochastics(df)
        k_clean  = k_s.dropna()
        d_clean  = d_s.dropna()
        if len(k_clean) >= 6 and len(d_clean) >= 6:
            touched_sub20 = bool((k_clean.iloc[-5:] < 20).any())
            k_cur, k_prev = float(k_clean.iloc[-1]), float(k_clean.iloc[-2])
            d_cur, d_prev = float(d_clean.iloc[-1]), float(d_clean.iloc[-2])
            crossed_up    = (k_cur > d_cur) and (k_prev <= d_prev)
            g2            = touched_sub20 and crossed_up
            gates["G2"] = {"pass": g2, "label": "Stoch (14,3,3)",
                "reason": f"%K={k_cur:.1f}  %D={d_cur:.1f}  |  Sub-20: {'✓' if touched_sub20 else '✗'}  |  Cross ↑: {'✓' if crossed_up else '✗'}"}
        else:
            gates["G2"] = {"pass": False, "label": "Stoch (14,3,3)", "reason": "Insufficient data"}
    else:
        gates["G2"] = {"pass": False, "label": "Stoch (14,3,3)", "reason": "No OHLCV data"}

    g3 = pct > -2.5
    gates["G3"] = {"pass": g3, "label": "Session",
        "reason": f"Today: {pct:+.2f}%  ({'OK' if g3 else 'FAIL — down >2.5%'})"}

    if cl is not None and len(cl.dropna()) >= 22:
        _, _, lower = calc_bb_bands(cl)
        lower_clean = lower.dropna()
        cl_aligned  = cl.loc[lower_clean.index]
        if len(lower_clean) >= 2:
            below_today = float(cl_aligned.iloc[-1]) < float(lower_clean.iloc[-1])
            below_prev  = float(cl_aligned.iloc[-2]) < float(lower_clean.iloc[-2])
            walking     = below_today and below_prev
            g4          = not walking
            gates["G4"] = {"pass": g4, "label": "BB Veto",
                "reason": f"Lower band: {float(lower_clean.iloc[-1]):.2f}  |  {'Walking lower band ❌' if walking else 'Price inside bands ✓'}"}
        else:
            gates["G4"] = {"pass": True, "label": "BB Veto", "reason": "Sparse data — pass"}
    else:
        gates["G4"] = {"pass": True, "label": "BB Veto", "reason": "Insufficient data — pass"}

    return {"gates": gates, "all_pass": all(g["pass"] for g in gates.values())}

def get_screener_row(ticker, result):
    price      = result.get("price")
    all_exps   = result.get("all_exps", [])
    hv_pct_raw = result.get("hv20")

    if not all_exps or not price or price <= 0:
        return None

    today    = datetime.utcnow()
    exp_csp  = None
    dte_csp  = None
    min_diff = 999

    for exp in all_exps:
        try:
            dte  = (datetime.strptime(exp, "%Y-%m-%d") - today).days
            diff = abs(dte - 37)
            if 21 <= dte <= 60 and diff < min_diff:
                min_diff = diff; exp_csp = exp; dte_csp = dte
        except Exception:
            continue

    if exp_csp is None:
        return None

    calls_df, puts_df, _ = fetch_chain_cached(ticker, exp_csp)
    if puts_df is None or puts_df.empty:
        return None

    hv_sigma = (hv_pct_raw / 100.0) if (hv_pct_raw and hv_pct_raw > 0) else None
    csp = find_target_strike(puts_df, 18.0, "put", price, dte_csp, hv_sigma)
    if csp is None:
        return None

    cc = None
    if calls_df is not None and not calls_df.empty:
        cc = find_target_strike(calls_df, 35.0, "call", price, dte_csp, hv_sigma)

    nis       = calc_nis(csp["theta"], dte_csp, price)
    csp_score = calc_suitability(nis, dte_csp, csp["delta"], "CSP")
    cc_score  = 0.0
    if cc:
        cc_score = calc_suitability(calc_nis(cc["theta"], dte_csp, price), dte_csp, cc["delta"], "CC")
    leap_score   = calc_suitability(nis, dte_csp, csp["delta"], "LEAP")
    gate_result  = calc_four_gates(result)
    greek_source = csp.get("greek_source", "unknown")

    return {
        "ticker": ticker, "price": price, "expiry": exp_csp, "dte": dte_csp,
        "csp_strike": csp["strike"], "csp_delta": csp["delta"], "csp_theta": csp["theta"],
        "csp_iv": csp["iv"], "csp_oi": csp["oi"], "csp_spread": csp["spread_pct"],
        "cc_strike": cc["strike"] if cc else None, "cc_delta": cc["delta"] if cc else None,
        "nis": round(nis, 1), "csp_score": csp_score, "cc_score": cc_score,
        "leap_score": leap_score, "gate_result": gate_result, "greek_source": greek_source,
        "_inspected": csp.get("_inspected"), "_scored": csp.get("_scored"),
    }

# ── Signal engines ─────────────────────────────────────────────────────────────
def leap_signal(hvr, rsi_val, above_50ma, above_200ma, vix_lvl):
    score = 0; reasons = []
    if hvr is not None:
        if hvr < 25:   score += 3; reasons.append("✅ HV Rank low (<25) — cheap premium")
        elif hvr < 40: score += 2; reasons.append("🟡 HV Rank moderate (25-40)")
        else:          score -= 1; reasons.append("❌ HV Rank elevated — expensive entry")
    if rsi_val is not None:
        if 33 <= rsi_val <= 52:    score += 2; reasons.append("✅ RSI in ideal recovery zone (33-52)")
        elif 52 < rsi_val <= 65:   score += 1; reasons.append("🟡 RSI extended but not overbought")
        elif rsi_val < 30:         score += 1; reasons.append("🟡 RSI oversold — wait for turn upward")
        else:                      score -= 1; reasons.append("❌ RSI overbought — avoid chasing")
    if above_200ma: score += 2; reasons.append("✅ Above 200MA — long term trend intact")
    else:           score -= 1; reasons.append("❌ Below 200MA — long term trend broken")
    if above_50ma:  score += 1; reasons.append("✅ Above 50MA — medium term trend OK")
    if vix_lvl is not None:
        if vix_lvl < 18:  score += 1; reasons.append("✅ VIX low — index premium cheap")
        elif vix_lvl > 28: score -= 1; reasons.append("⚠️ VIX elevated — vol expansion risk")
    label = ("🟢 STRONG ENTRY" if score >= 7 else "🟡 DECENT ENTRY" if score >= 4
             else "🟠 MARGINAL" if score >= 2 else "🔴 AVOID")
    return label, score, reasons

def cc_signal(hvr, rsi_val, above_50ma, pcr_val):
    score = 0; reasons = []
    if hvr is not None:
        if hvr > 65:   score += 3; reasons.append("✅ HV Rank high (>65) — premium rich")
        elif hvr > 45: score += 2; reasons.append("🟡 HV Rank moderate (45-65)")
        else:          reasons.append("❌ HV Rank low — thin premium")
    if rsi_val is not None:
        if rsi_val > 65:   score += 2; reasons.append("✅ RSI overbought — capping upside is smart")
        elif rsi_val > 50: score += 1; reasons.append("🟡 RSI bullish but not extreme")
        elif rsi_val < 35: score -= 1; reasons.append("❌ RSI oversold — don't cap at the bottom")
    if above_50ma: score += 1; reasons.append("✅ Above 50MA — safe to sell calls")
    else:          score -= 1; reasons.append("⚠️ Below 50MA — trend weakening")
    if pcr_val is not None and pcr_val > 1.2:
        score += 1; reasons.append("✅ Elevated PCR — fear = good premium")
    label = ("🟢 WRITE NOW" if score >= 5 else "🟡 DECENT" if score >= 3
             else "🟠 MARGINAL" if score >= 1 else "🔴 WAIT")
    return label, score, reasons

def csp_signal(hvr, rsi_val, above_200ma):
    score = 0; reasons = []
    if hvr is not None:
        if hvr > 55:   score += 2; reasons.append("✅ High HV Rank — CSP premium rich")
        elif hvr > 35: score += 1; reasons.append("🟡 Moderate HV Rank")
    if rsi_val is not None:
        if 30 <= rsi_val <= 50: score += 2; reasons.append("✅ RSI in recovery — good put strike support")
        elif rsi_val < 30:      score += 1; reasons.append("⚠️ Deeply oversold — wait for stabilisation")
        elif rsi_val > 70:      score -= 1; reasons.append("❌ RSI overbought — don't sell puts at the top")
    if above_200ma: score += 2; reasons.append("✅ Above 200MA — reduces assignment risk")
    else:           score -= 1; reasons.append("❌ Below 200MA — assignment risk elevated")
    label = ("🟢 SELL PUT" if score >= 4 else "🟡 DECENT" if score >= 2
             else "🟠 MARGINAL" if score >= 1 else "🔴 AVOID")
    return label, score, reasons

def analyse(ticker, period, vix_current):
    df = fetch_prices(ticker, period)
    if df is None:
        return None
    cl      = df["Close"].squeeze()
    curr    = float(cl.iloc[-1])
    prev    = float(cl.iloc[-2])
    pct_chg = (curr / prev - 1) * 100
    hv20_s  = calc_hv(cl, 20);  hv60_s = calc_hv(cl, 60)
    hv_cur  = float(hv20_s.dropna().iloc[-1]) if not hv20_s.dropna().empty else None
    hvr     = calc_iv_rank(hv20_s); hvpct = calc_iv_percentile(hv20_s)
    rsi_s   = calc_rsi(cl)
    rsi_cur = float(rsi_s.dropna().iloc[-1]) if not rsi_s.dropna().empty else None
    atr_s   = calc_atr(df)
    atr_cur = float(atr_s.dropna().iloc[-1]) if not atr_s.dropna().empty else None
    bbw_s   = calc_bb_width(cl)
    bbw_cur = float(bbw_s.dropna().iloc[-1]) if not bbw_s.dropna().empty else None
    ma50    = float(cl.rolling(50).mean().iloc[-1])
    ma200   = float(cl.rolling(200).mean().iloc[-1])
    ab50    = curr > ma50; ab200 = curr > ma200
    all_exps = fetch_all_expiries(ticker)
    c_iv = p_iv = pcr_val = chain = exp = dte = None
    if all_exps:
        today = datetime.utcnow()
        valid = [e for e in all_exps if (datetime.strptime(e, "%Y-%m-%d") - today).days > 14]
        if valid:
            exp = valid[0]
            calls_df, puts_df, dte = fetch_chain_cached(ticker, exp)
            if calls_df is not None:
                chain = type("_C", (), {"calls": calls_df, "puts": puts_df})()
                c_iv, p_iv = calc_atm_iv(chain, curr)
                pcr_val    = calc_pcr(chain)
    leap_lbl, leap_sc, leap_r = leap_signal(hvr, rsi_cur, ab50, ab200, vix_current)
    cc_lbl,   cc_sc,   cc_r   = cc_signal(hvr, rsi_cur, ab50, pcr_val)
    csp_lbl,  csp_sc,  csp_r  = csp_signal(hvr, rsi_cur, ab200)
    return {
        "ticker": ticker, "price": curr, "pct": pct_chg,
        "hv20": hv_cur, "hvr": hvr, "hvpct": hvpct,
        "hv20_s": hv20_s, "hv60_s": hv60_s,
        "rsi": rsi_cur, "rsi_s": rsi_s,
        "atr": atr_cur, "bbw": bbw_cur, "bbw_s": bbw_s,
        "ma50": ma50, "ma200": ma200, "ab50": ab50, "ab200": ab200,
        "c_iv": c_iv, "p_iv": p_iv, "pcr": pcr_val,
        "exp": exp, "dte": dte, "all_exps": all_exps, "df": df, "cl": cl,
        "leap": (leap_lbl, leap_sc, leap_r),
        "cc":   (cc_lbl,   cc_sc,   cc_r),
        "csp":  (csp_lbl,  csp_sc,  csp_r),
    }

def vix_zone(v):
    for lo, hi, color, label in VIX_ZONES:
        if lo <= v < hi:
            return color, label
    return "#6b7280", "Unknown"

def fmt(v, fs=".1f", su=""):
    return f"{v:{fs}}{su}" if v is not None else "—"

def greek_source_label(gs):
    if gs == "yahoo":           return "📡 Yahoo"
    if gs == "bs_strike":       return "📐 BS (strike IV)"
    if gs == "bs_chain_median": return "📐 BS (chain med IV)"
    if gs == "bs_hv20":         return "📐 BS (HV20)"
    if gs == "bs_default":      return "📐 BS (30% def)"
    return gs or "—"

# ══════════════════════════════════════════════════════════════════════════════
# MARKET PULSE CHART HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def fg_color(score):
    if score is None:   return "#6b7280"
    if score < 25:      return "#dc2626"
    if score < 45:      return "#ea580c"
    if score < 55:      return "#ca8a04"
    if score < 75:      return "#16a34a"
    return "#15803d"

def fg_gauge(score, rating):
    color = fg_color(score)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"font": {"size": 36, "color": color}},
        title={"text": f"<b>Fear & Greed</b><br><span style='color:{color};font-size:0.85em'>{rating}</span>",
               "font": {"size": 13}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#6b7280",
                     "tickvals": [0, 25, 45, 55, 75, 100],
                     "ticktext": ["0", "Ex Fear", "Fear", "Greed", "Ex Greed", "100"]},
            "bar":  {"color": color, "thickness": 0.25},
            "steps": [
                {"range": [0,  25],  "color": "rgba(220,38,38,0.12)"},
                {"range": [25, 45],  "color": "rgba(234,88,12,0.12)"},
                {"range": [45, 55],  "color": "rgba(202,138,4,0.12)"},
                {"range": [55, 75],  "color": "rgba(22,163,74,0.12)"},
                {"range": [75, 100], "color": "rgba(21,128,61,0.12)"},
            ],
            "threshold": {"line": {"color": color, "width": 4}, "thickness": 0.75, "value": score},
        },
    ))
    fig.update_layout(height=200, template="plotly_dark",
                      margin=dict(l=10, r=10, t=50, b=10))
    return fig

def vix_gauge(vix_val):
    color, label = vix_zone(vix_val)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=vix_val,
        number={"font": {"size": 36, "color": color}, "suffix": ""},
        title={"text": f"<b>VIX</b><br><span style='color:{color};font-size:0.85em'>{label.split('—')[0].strip()}</span>",
               "font": {"size": 13}},
        gauge={
            "axis": {"range": [0, 50], "tickwidth": 1, "tickcolor": "#6b7280",
                     "tickvals": [0, 15, 20, 30, 50]},
            "bar":  {"color": color, "thickness": 0.25},
            "steps": [
                {"range": [0,  15], "color": "rgba(22,163,74,0.12)"},
                {"range": [15, 20], "color": "rgba(202,138,4,0.12)"},
                {"range": [20, 30], "color": "rgba(234,88,12,0.12)"},
                {"range": [30, 50], "color": "rgba(220,38,38,0.12)"},
            ],
            "threshold": {"line": {"color": color, "width": 4}, "thickness": 0.75, "value": vix_val},
        },
    ))
    fig.update_layout(height=200, template="plotly_dark",
                      margin=dict(l=10, r=10, t=50, b=10))
    return fig

def vix_term_chart(term_data):
    """Bar chart of VIX term structure (9D, 30D, 3M, 6M)."""
    if not term_data or len(term_data) < 2:
        return None
    labels = list(term_data.keys())
    vals   = list(term_data.values())
    # Contango = rising = green / Backwardation = falling = red
    bar_colors = []
    for i, v in enumerate(vals):
        if i == 0:
            bar_colors.append("#60a5fa")
        else:
            bar_colors.append("#22c55e" if v >= vals[i-1] else "#ef4444")
    fig = go.Figure(go.Bar(
        x=labels, y=vals,
        marker_color=bar_colors,
        text=[f"{v:.1f}" for v in vals],
        textposition="outside",
    ))
    fig.update_layout(
        height=200, template="plotly_dark",
        title={"text": "<b>VIX Term Structure</b>", "font": {"size": 13}},
        yaxis_title="VIX Level",
        yaxis_range=[0, max(vals) * 1.25],
        margin=dict(l=10, r=10, t=40, b=10),
        showlegend=False,
    )
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## Settings")
    new_ticker = st.text_input("Add a ticker", placeholder="e.g. AMZN", key="new_ticker_input")
    if new_ticker:
        t = new_ticker.upper().strip()
        if t and t not in st.session_state.watchlist:
            st.session_state.watchlist.append(t)
            st.rerun()
    if st.session_state.watchlist:
        remove = st.selectbox("Remove a ticker", ["— select to remove —"] + st.session_state.watchlist)
        if remove != "— select to remove —":
            st.session_state.watchlist.remove(remove)
            st.rerun()
    st.markdown(f"**Watchlist:** {', '.join(st.session_state.watchlist)}")
    period = st.selectbox("Price History", ["6mo", "1y", "2y"], index=1)

    st.divider()
    auto_refresh = st.toggle("🔄 Auto-refresh market data (60s)", value=False)
    if auto_refresh and HAS_AUTOREFRESH:
        st_autorefresh(interval=60_000, key="pulse_refresh")
    elif auto_refresh and not HAS_AUTOREFRESH:
        st.warning("Install `streamlit-autorefresh` to enable this.")

    if st.button("🧹 Clear all cached data", use_container_width=True):
        st.cache_data.clear()
        st.success("Cache cleared.")
        st.rerun()

    st.divider()
    vix_df  = fetch_vix("1y")
    vix_now = None; vix_chg = 0
    if vix_df is not None and not vix_df.empty:
        vix_cl_s = vix_df["Close"].squeeze()
        vix_now  = float(vix_cl_s.iloc[-1])
        vix_prev = float(vix_cl_s.iloc[-2])
        vix_chg  = vix_now - vix_prev
        vc, vl   = vix_zone(vix_now)
        st.markdown(f"### VIX: {vix_now:.1f} ({vix_chg:+.2f})")
        st.markdown(f"<span style='color:{vc};font-weight:700'>{vl}</span>", unsafe_allow_html=True)
        vix_52hi = float(vix_cl_s.max()); vix_52lo = float(vix_cl_s.min())
        vix_rank = (vix_now - vix_52lo) / (vix_52hi - vix_52lo) * 100 if vix_52hi != vix_52lo else 50
        st.progress(int(vix_rank), text=f"VIX 52wk Rank: {vix_rank:.0f}%")
    st.divider()
    st.caption("Pulse data refreshes every 60s if toggle enabled. Watchlist data cached 30 min.")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
watchlist = st.session_state.watchlist
st.title("Options Intelligence Dashboard")

tab_dash, tab_dive, tab_chain, tab_vix, tab_screener = st.tabs(
    ["Overview", "Deep Dive", "Options Chain", "VIX", "⚡ Screener"]
)

results = {}
with st.spinner("Loading market data..."):
    for t in watchlist:
        r = analyse(t, period, vix_now)
        if r:
            results[t] = r

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab_dash:

    # ── MARKET PULSE ────────────────────────────────────────────────────────
    st.subheader("🌍 Market Pulse")
    updated_at = datetime.utcnow().strftime("%H:%M:%S UTC")
    st.caption(f"Last updated: {updated_at}  ·  Data ~15 min delayed  ·  Toggle auto-refresh in sidebar for 60s updates")

    # Fetch all pulse quotes (60s TTL)
    pulse_data = {}
    for ticker, label, prefix, is_yield in PULSE_TICKERS:
        pulse_data[ticker] = fetch_quote(ticker)

    # Row 1: Equities + DXY
    pulse_row1 = [("SPY","S&P 500","$",False), ("QQQ","Nasdaq 100","$",False),
                  ("DIA","Dow Jones","$",False), ("IWM","R2000","$",False), ("DX-Y.NYB","DXY","",False)]
    cols_p1 = st.columns(5)
    for col, (ticker, label, prefix, is_yield) in zip(cols_p1, pulse_row1):
        q = pulse_data.get(ticker)
        if q:
            price = q["price"]
            pct   = q["pct"]
            # Format price
            if is_yield:
                disp = f"{price:.2f}%"
            elif price > 10000:
                disp = f"${price:,.0f}"
            elif price > 100:
                disp = f"{prefix}{price:,.2f}"
            else:
                disp = f"{prefix}{price:.2f}"
            delta_str = f"{pct:+.2f}%"
            col.metric(label, disp, delta_str)
        else:
            col.metric(label, "—", "—")

    # Row 2: Commodities + Crypto + Yields
    pulse_row2 = [("CL=F","Crude Oil","$",False), ("GC=F","Gold","$",False),
                  ("BTC-USD","Bitcoin","$",False), ("^TNX","10Y Yield","",True), ("^IRX","3M Yield","",True)]
    cols_p2 = st.columns(5)
    for col, (ticker, label, prefix, is_yield) in zip(cols_p2, pulse_row2):
        q = pulse_data.get(ticker)
        if q:
            price = q["price"]
            pct   = q["pct"]
            if is_yield:
                disp = f"{price:.2f}%"
            elif price > 10000:
                disp = f"${price:,.0f}"
            elif price > 100:
                disp = f"{prefix}{price:,.2f}"
            else:
                disp = f"{prefix}{price:.2f}"
            delta_str = f"{pct:+.2f}%"
            col.metric(label, disp, delta_str)
        else:
            col.metric(label, "—", "—")

    st.divider()

    # ── GAUGES ROW ────────────────────────────────────────────────────────────
    fg_score, fg_rating = fetch_fear_greed()
    term_data = fetch_vix_term()
    skew_val  = fetch_skew()

    gcol1, gcol2, gcol3, gcol4 = st.columns([1.2, 1.2, 1.4, 1.2])

    with gcol1:
        if fg_score is not None:
            st.plotly_chart(fg_gauge(fg_score, fg_rating), use_container_width=True)
        else:
            st.metric("Fear & Greed", "—", help="CNN data unavailable")

    with gcol2:
        if vix_now is not None:
            st.plotly_chart(vix_gauge(vix_now), use_container_width=True)
        else:
            st.metric("VIX", "—")

    with gcol3:
        if term_data:
            tc = vix_term_chart(term_data)
            if tc:
                st.plotly_chart(tc, use_container_width=True)
        else:
            st.metric("VIX Term Structure", "—", help="Could not load VIX term tickers")

    with gcol4:
        st.markdown("**📊 Macro Signals**")
        # Yield curve
        tnx_q = pulse_data.get("^TNX")
        irx_q = pulse_data.get("^IRX")
        if tnx_q and irx_q:
            spread = tnx_q["price"] - irx_q["price"]
            curve_status = (
                "🟢 Normal" if spread > 0.5 else
                "🟡 Flat"   if spread > -0.3 else
                "🔴 Inverted"
            )
            st.markdown(f"**Yield Curve (10Y−3M):** {spread:+.2f}%  {curve_status}")
        else:
            st.markdown("**Yield Curve:** —")
        # SKEW
        if skew_val is not None:
            skew_status = (
                "🔴 Elevated tail risk" if skew_val > 145 else
                "🟡 Moderate"          if skew_val > 130 else
                "🟢 Low tail risk"
            )
            st.markdown(f"**SKEW Index:** {skew_val:.1f}  {skew_status}")
        else:
            st.markdown("**SKEW Index:** —")
        # VIX term structure shape
        if len(term_data) >= 2:
            vals = list(term_data.values())
            shape = "🟢 Contango (normal)" if vals[-1] > vals[0] else "🔴 Backwardation (stress)"
            st.markdown(f"**VIX Shape:** {shape}")
        # F&G label
        if fg_score is not None:
            color = fg_color(fg_score)
            st.markdown(f"**CNN F&G:** <span style='color:{color}'>{fg_score:.0f} — {fg_rating}</span>",
                        unsafe_allow_html=True)

    st.divider()

    # ── WATCHLIST TABLE ───────────────────────────────────────────────────────
    st.subheader("📋 Watchlist Overview")
    rows = []
    for t, r in results.items():
        rows.append({
            "Ticker":     t,
            "Price":      f"${r['price']:.2f}",
            "Chg %":      f"{r['pct']:+.1f}%",
            "HV Rank":    fmt(r["hvr"], ".0f"),
            "HV%ile":     fmt(r["hvpct"], ".0f"),
            "HV20":       fmt(r["hv20"], ".1f", "%"),
            "ATM IV C/P": f"{r['c_iv']:.0f}/{r['p_iv']:.0f}%" if r["c_iv"] else "—",
            "RSI":        fmt(r["rsi"], ".0f"),
            "50MA":       "✅" if r["ab50"] else "❌",
            "200MA":      "✅" if r["ab200"] else "❌",
            "PCR":        fmt(r["pcr"], ".2f"),
            "ATR":        fmt(r["atr"], ".2f"),
            "BB Width":   fmt(r["bbw"], ".1f", "%"),
            "LEAP":       r["leap"][0],
            "CC":         r["cc"][0],
            "CSP":        r["csp"][0],
        })
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True, height=420)

    hvr_data = {t: r["hvr"] for t, r in results.items() if r["hvr"] is not None}
    if hvr_data:
        st.subheader("HV Rank — Entry Zones")
        colors = ["#22c55e" if v < 25 else "#eab308" if v < 45 else "#f97316" if v < 65 else "#ef4444"
                  for v in hvr_data.values()]
        fig_hvr = go.Figure(go.Bar(
            x=list(hvr_data.keys()), y=list(hvr_data.values()),
            marker_color=colors,
            text=[f"{v:.0f}" for v in hvr_data.values()], textposition="outside",
        ))
        fig_hvr.add_hline(y=25, line_dash="dash", line_color="#22c55e", annotation_text="25 = LEAP Buy Zone")
        fig_hvr.add_hline(y=65, line_dash="dash", line_color="#ef4444", annotation_text="65 = CC Writing Zone")
        fig_hvr.update_layout(height=320, template="plotly_dark", yaxis_title="HV Rank",
                               yaxis_range=[0, 115], margin=dict(l=0, r=0, t=20, b=0))
        st.plotly_chart(fig_hvr, use_container_width=True)

    rsi_data = {t: r["rsi"] for t, r in results.items() if r["rsi"] is not None}
    if rsi_data:
        st.subheader("RSI Snapshot")
        rsi_colors = ["#22c55e" if 33 <= v <= 52 else "#eab308" if 52 < v <= 65 else "#f97316" if v > 65 else "#94a3b8"
                      for v in rsi_data.values()]
        fig_rsi = go.Figure(go.Bar(
            x=list(rsi_data.keys()), y=list(rsi_data.values()),
            marker_color=rsi_colors,
            text=[f"{v:.0f}" for v in rsi_data.values()], textposition="outside",
        ))
        for level, color, label in [(30, "#22c55e", "30"), (50, "#94a3b8", "50"), (70, "#ef4444", "70")]:
            fig_rsi.add_hline(y=level, line_dash="dash", line_color=color, annotation_text=label)
        fig_rsi.update_layout(height=280, template="plotly_dark", yaxis_title="RSI (14)",
                               yaxis_range=[0, 115], margin=dict(l=0, r=0, t=20, b=0))
        st.plotly_chart(fig_rsi, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — DEEP DIVE
# ══════════════════════════════════════════════════════════════════════════════
with tab_dive:
    sel = st.selectbox("Select Ticker", list(results.keys()), key="dd_sel")
    if sel and sel in results:
        r = results[sel]; df = r["df"]; cl = r["cl"]
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Price",       f"${r['price']:.2f}", f"{r['pct']:+.1f}%")
        c2.metric("HV Rank",     fmt(r["hvr"], ".0f"))
        c3.metric("HV Pctile",   fmt(r["hvpct"], ".0f", "%"))
        c4.metric("RSI (14)",    fmt(r["rsi"], ".1f"))
        c5.metric("ATM Call IV", fmt(r["c_iv"], ".1f", "%"))
        c6.metric("PCR",         fmt(r["pcr"], ".2f"))
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            lbl, sc, reasons = r["leap"]
            st.markdown(f"#### LEAP: {lbl}")
            with st.expander("Breakdown"):
                for reason in reasons: st.write(reason)
        with sc2:
            lbl, sc, reasons = r["cc"]
            st.markdown(f"#### CC: {lbl}")
            with st.expander("Breakdown"):
                for reason in reasons: st.write(reason)
        with sc3:
            lbl, sc, reasons = r["csp"]
            st.markdown(f"#### CSP: {lbl}")
            with st.expander("Breakdown"):
                for reason in reasons: st.write(reason)
        st.divider()
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
            row_heights=[0.48, 0.18, 0.17, 0.17],
            subplot_titles=[f"{sel} — Price & MAs", "HV20 (Volatility Proxy)", "RSI (14)", "BB Width"],
            vertical_spacing=0.04)
        fig.add_trace(go.Candlestick(x=df.index,
            open=df["Open"].squeeze(), high=df["High"].squeeze(),
            low=df["Low"].squeeze(), close=cl, name="Price",
            increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
            increasing_fillcolor="#26a69a", decreasing_fillcolor="#ef5350"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=cl.rolling(50).mean(), name="50MA",
            line=dict(color="#f97316", width=1.4)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=cl.rolling(200).mean(), name="200MA",
            line=dict(color="#60a5fa", width=1.6)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=r["hv20_s"], name="HV20",
            fill="tozeroy", line=dict(color="#a78bfa", width=1.5),
            fillcolor="rgba(167,139,250,0.12)"), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=r["hv60_s"], name="HV60",
            line=dict(color="#7c3aed", width=1, dash="dot")), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=r["rsi_s"], name="RSI",
            line=dict(color="#fbbf24", width=1.5)), row=3, col=1)
        for lvl, col in [(70, "#ef4444"), (50, "#94a3b8"), (30, "#22c55e")]:
            fig.add_hline(y=lvl, line_dash="dash", line_color=col, row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=r["bbw_s"], name="BB Width",
            fill="tozeroy", line=dict(color="#34d399", width=1.4),
            fillcolor="rgba(52,211,153,0.10)"), row=4, col=1)
        fig.update_layout(height=820, template="plotly_dark",
                          xaxis_rangeslider_visible=False,
                          legend=dict(orientation="h", y=1.01, x=0),
                          margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)
        if r["atr"] and r["price"]:
            atr_pct = r["atr"] / r["price"] * 100
            st.subheader("Position Sizing Guide")
            st.markdown(f"""
| Metric | Value |
|---|---|
| ATR 14-day | ${r['atr']:.2f} ({atr_pct:.1f}% of price) |
| Suggested CC strike (1.5× ATR above) | ~${r['price'] + r['atr']*1.5:.2f} |
| Suggested CSP strike (1.5× ATR below) | ~${r['price'] - r['atr']*1.5:.2f} |
            """)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — OPTIONS CHAIN
# ══════════════════════════════════════════════════════════════════════════════
with tab_chain:
    sel_c = st.selectbox("Select Ticker", list(results.keys()), key="chain_sel")
    if sel_c and sel_c in results:
        r = results[sel_c]; price = r["price"]; all_exps = r.get("all_exps", [])
        if all_exps:
            selected_exp = st.selectbox("Select Expiry", all_exps, key=f"exp_sel_{sel_c}")
            calls_df, puts_df, dte = fetch_chain_cached(sel_c, selected_exp)
            if calls_df is not None:
                chain = type("_C", (), {"calls": calls_df, "puts": puts_df})()
                st.caption(f"Expiry: {selected_exp} ({dte} DTE) | Current Price: ${price:.2f}")
                def fmt_chain(df_raw, side):
                    df_raw = df_raw.copy()
                    df_raw["IV %"] = (df_raw["impliedVolatility"] * 100).round(1)
                    df_raw["Moneyness"] = df_raw["strike"].apply(
                        lambda s: "ATM" if abs(s - price) / price < 0.02
                        else ("ITM" if ((s < price and side=="call") or (s > price and side=="put")) else "OTM"))
                    cols = ["strike","Moneyness","lastPrice","bid","ask","volume","openInterest","IV %","delta"]
                    available = [c for c in cols if c in df_raw.columns]
                    return (df_raw[available]
                            .rename(columns={"lastPrice":"Last","openInterest":"OI","strike":"Strike","volume":"Volume"})
                            .sort_values("Strike").reset_index(drop=True))
                col_c, col_p = st.columns(2)
                with col_c:
                    st.subheader("Calls")
                    st.dataframe(fmt_chain(chain.calls, "call"), use_container_width=True, hide_index=True)
                with col_p:
                    st.subheader("Puts")
                    st.dataframe(fmt_chain(chain.puts, "put"), use_container_width=True, hide_index=True)
                st.subheader("IV Smile")
                fig_smile = go.Figure()
                fig_smile.add_trace(go.Scatter(x=chain.calls["strike"], y=chain.calls["impliedVolatility"]*100,
                    name="Calls IV", mode="lines+markers", line=dict(color="#26a69a", width=2), marker=dict(size=5)))
                fig_smile.add_trace(go.Scatter(x=chain.puts["strike"],  y=chain.puts["impliedVolatility"]*100,
                    name="Puts IV",  mode="lines+markers", line=dict(color="#ef5350", width=2), marker=dict(size=5)))
                fig_smile.add_vline(x=price, line_dash="dash", line_color="white", annotation_text=f"${price:.2f}")
                fig_smile.update_layout(height=350, template="plotly_dark", xaxis_title="Strike",
                                        yaxis_title="IV (%)", margin=dict(l=0,r=0,t=20,b=0))
                st.plotly_chart(fig_smile, use_container_width=True)
                st.subheader("Open Interest by Strike")
                fig_oi = go.Figure()
                fig_oi.add_trace(go.Bar(x=chain.calls["strike"], y=chain.calls["openInterest"],
                    name="Call OI", marker_color="#26a69a", opacity=0.75))
                fig_oi.add_trace(go.Bar(x=chain.puts["strike"],  y=chain.puts["openInterest"],
                    name="Put OI",  marker_color="#ef5350", opacity=0.75))
                fig_oi.add_vline(x=price, line_dash="dash", line_color="white")
                fig_oi.update_layout(barmode="overlay", height=320, template="plotly_dark",
                                     xaxis_title="Strike", yaxis_title="Open Interest",
                                     margin=dict(l=0,r=0,t=20,b=0))
                st.plotly_chart(fig_oi, use_container_width=True)
            else:
                st.warning("Could not load chain for this expiry.")
        else:
            st.warning(f"No options data for {sel_c}.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — VIX
# ══════════════════════════════════════════════════════════════════════════════
with tab_vix:
    if vix_df is not None and not vix_df.empty:
        vix_cl = vix_df["Close"].squeeze()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Current VIX", f"{vix_now:.1f}", f"{vix_chg:+.2f}")
        c2.metric("52wk High",   f"{vix_cl.max():.1f}")
        c3.metric("52wk Low",    f"{vix_cl.min():.1f}")
        c4.metric("52wk Avg",    f"{vix_cl.mean():.1f}")
        fig_vix = make_subplots(rows=2, cols=1, shared_xaxes=True,
            row_heights=[0.7, 0.3], subplot_titles=["VIX Level", "30-day Rolling Average"])
        fig_vix.add_trace(go.Scatter(x=vix_df.index, y=vix_cl, name="VIX",
            fill="tozeroy", line=dict(color="#f87171", width=1.5),
            fillcolor="rgba(248,113,113,0.12)"), row=1, col=1)
        for lo, hi, color, label in VIX_ZONES:
            fig_vix.add_hrect(y0=lo, y1=min(hi, 50), fillcolor=color, opacity=0.05, row=1, col=1)
            fig_vix.add_hline(y=lo, line_dash="dot", line_color=color, opacity=0.4, row=1, col=1)
        fig_vix.add_trace(go.Scatter(x=vix_df.index, y=vix_cl.rolling(30).mean(),
            name="30d MA", line=dict(color="#fbbf24", width=1.5)), row=2, col=1)
        fig_vix.update_layout(height=520, template="plotly_dark", margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(fig_vix, use_container_width=True)
        st.markdown("""
| VIX | Regime | LEAP Buying | CC Writing | CSP Writing |
|---|---|---|---|---|
| Below 15 | Low | Best — cheapest premium | Thin premium | Good if trend up |
| 15 to 20 | Normal | Decent | Moderate premium | OK |
| 20 to 30 | Elevated | Expensive — be selective | Rich premium | Rich premium |
| Above 30 | Fear | Very expensive — wait | Maximum premium | High premium, high risk |
        """)
    else:
        st.error("Could not load VIX data.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — SCREENER
# ══════════════════════════════════════════════════════════════════════════════
with tab_screener:
    st.subheader("⚡ Options Suitability Screener")
    st.caption(
        "Ranks each watchlist stock for CSP / CC / LEAP suitability using the θ × √DTE / S formula. "
        "yfinance does NOT return greeks — all delta/theta computed via Black-Scholes "
        "using live strike IV (preferred), chain-median IV, or HV20 as progressive fallbacks."
    )

    with st.expander("How the score is calculated", expanded=False):
        st.markdown("""
**Core formula**

For the nearest 30–45 DTE expiry:
- Put strike closest to **Δ18** → CSP target
- Call strike closest to **Δ35** → CC target

**Normalised IV Score (NIS):**
> Raw = θ × √DTE / Price → normalised 0–100

**Score weights**

| Component | CSP | CC | LEAP |
|---|---|---|---|
| A — NIS | 50% | 50% | 30% |
| B — DTE fit | 30% | 30% | 40% |
| C — Delta fit | 20% | 20% | 30% |

**Four-gate filter:**
G1 Trend · G2 Stochastics · G3 Session · G4 BB Veto

**Greek sourcing:** Strike IV → Chain median IV → HV20 → 30% default
        """)

    with st.expander("🔬 Chain Inspector", expanded=False):
        insp_ticker = st.selectbox("Ticker", list(results.keys()), key="insp_ticker")
        if insp_ticker and insp_ticker in results:
            insp_exps = results[insp_ticker].get("all_exps", [])
            today_i   = datetime.utcnow()
            valid_i   = [e for e in insp_exps if 21 <= (datetime.strptime(e,"%Y-%m-%d")-today_i).days <= 60]
            if valid_i:
                calls_i, puts_i, dte_i = fetch_chain_cached(insp_ticker, valid_i[0])
                if puts_i is not None and not puts_i.empty:
                    st.write(f"**Columns:** `{list(puts_i.columns)}`")
                    st.write(f"**Rows:** {len(puts_i)}  |  **DTE:** {dte_i}")
                    price_i    = results[insp_ticker]["price"]
                    puts_sorted = puts_i.iloc[(puts_i["strike"]-price_i).abs().argsort()].head(12)
                    disp_cols  = [c for c in ["strike","bid","ask","impliedVolatility","volume","openInterest"]
                                  if c in puts_sorted.columns]
                    st.dataframe(puts_sorted[disp_cols].reset_index(drop=True), use_container_width=True)
                    oi_c = pd.to_numeric(puts_i.get("openInterest", pd.Series(dtype=float)), errors="coerce").fillna(0)
                    iv_c = pd.to_numeric(puts_i.get("impliedVolatility", pd.Series(dtype=float)), errors="coerce").fillna(0)
                    st.write(f"OI≥1: **{int((oi_c>=1).sum())}** / {len(puts_i)}  ·  IV>0: **{int((iv_c>0).sum())}** / {len(puts_i)}")
                else:
                    st.warning("Chain returned empty.")
            else:
                st.warning(f"No 21–60 DTE expiry found for {insp_ticker}.")

    col_run, col_note = st.columns([1, 4])
    with col_run:
        run_btn = st.button("🔍 Run Screener", type="primary", use_container_width=True)
    with col_note:
        st.info("First run ~30–60 s. Results cached 30 min.")

    show_debug = st.checkbox("🔧 Show diagnostic output", value=False)

    if run_btn:
        if not results:
            st.warning("No market data loaded.")
        else:
            screener_rows = []
            prog      = st.progress(0, text="Initialising...")
            n         = len(results)
            debug_log = []
            for i, (ticker, result) in enumerate(results.items()):
                prog.progress((i+1)/n, text=f"Analysing {ticker}  ({i+1}/{n})")
                row = get_screener_row(ticker, result)
                if row:
                    screener_rows.append(row)
                    debug_log.append(
                        f"✅ {ticker} — CSP {row['csp_score']} ({row['greek_source']}, "
                        f"inspected {row.get('_inspected')}, scored {row.get('_scored')})"
                    )
                else:
                    price    = result.get("price")
                    all_exps = result.get("all_exps", [])
                    reason   = "unknown"
                    if not price or price <= 0:
                        reason = "no price"
                    elif not all_exps:
                        reason = "no expiries"
                    else:
                        today = datetime.utcnow()
                        valid_exp = next((e for e in all_exps
                                         if 21 <= (datetime.strptime(e,"%Y-%m-%d")-today).days <= 60), None)
                        if not valid_exp:
                            reason = f"no 21–60 DTE expiry (have: {all_exps[:3]})"
                        else:
                            _, puts_df, _ = fetch_chain_cached(ticker, valid_exp)
                            if puts_df is None or puts_df.empty:
                                reason = f"chain fetch failed for {valid_exp}"
                            else:
                                oi_c = pd.to_numeric(puts_df.get("openInterest", pd.Series(dtype=float)), errors="coerce").fillna(0)
                                iv_c = pd.to_numeric(puts_df.get("impliedVolatility", pd.Series(dtype=float)), errors="coerce").fillna(0)
                                reason = (f"chain OK ({len(puts_df)} rows, OI≥1:{int((oi_c>=1).sum())}, "
                                          f"IV>0:{int((iv_c>0).sum())}) — find_target_strike returned None")
                    debug_log.append(f"❌ {ticker} — skipped: {reason}")
            prog.empty()
            st.session_state["screener_results"] = screener_rows
            st.session_state["screener_debug"]   = debug_log

    if show_debug and st.session_state.get("screener_debug"):
        with st.expander("🔧 Diagnostic log", expanded=True):
            for line in st.session_state["screener_debug"]:
                st.text(line)

    screener_rows = st.session_state.get("screener_results", [])

    if screener_rows:
        screener_rows_sorted = sorted(screener_rows, key=lambda x: x["csp_score"], reverse=True)

        src_counts   = {}
        for r in screener_rows_sorted:
            k = r.get("greek_source","unknown")
            src_counts[k] = src_counts.get(k,0) + 1
        st.info("**Greeks:** " + "  ·  ".join(f"{greek_source_label(k)}: {v}" for k,v in src_counts.items()))

        low_prec = [r["ticker"] for r in screener_rows_sorted if r.get("greek_source") in ("bs_hv20","bs_default")]
        if low_prec:
            st.warning(f"⚠️ Low-precision greeks for: **{', '.join(low_prec)}** — HV20 or 30% default used.")

        table_rows = []
        for r in screener_rows_sorted:
            gr    = r["gate_result"]; gates = gr["gates"]
            gate_icons = "".join(["✅" if gates[f"G{i}"]["pass"] else "❌" for i in range(1,5)])
            table_rows.append({
                "Ticker":      r["ticker"],
                "Price":       f"${r['price']:.2f}",
                "Expiry":      r["expiry"],
                "DTE":         r["dte"],
                "Greeks":      greek_source_label(r.get("greek_source")),
                "CSP Strike":  f"${r['csp_strike']:.1f}",
                "CSP Δ":       r["csp_delta"],
                "θ/day":       f"${r['csp_theta']:.3f}",
                "Put IV %":    r["csp_iv"] if r["csp_iv"] else "—",
                "OI":          r["csp_oi"],
                "Spread %":    r["csp_spread"] if r["csp_spread"] else "—",
                "NIS":         r["nis"],
                "CSP Score":   r["csp_score"],
                "CC Score":    r["cc_score"],
                "LEAP Score*": r["leap_score"],
                "Gates":       gate_icons,
                "Status":      "🟢 TRADE" if gr["all_pass"] else "🔴 WAIT",
            })
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True, height=480)
        st.caption("\\* LEAP Score at short DTE — use as relative IV indicator only.")

        tickers    = [r["ticker"]    for r in screener_rows_sorted]
        csp_scores = [r["csp_score"] for r in screener_rows_sorted]
        st.subheader("CSP Suitability Ranking")
        fig_sc = go.Figure(go.Bar(x=tickers, y=csp_scores,
            marker_color=[score_color(s) for s in csp_scores],
            text=[f"{s:.0f}" for s in csp_scores], textposition="outside"))
        for lvl, col, lbl in [(80,"#22c55e","80—Optimal"),(60,"#eab308","60—Acceptable"),(40,"#f97316","40—Marginal")]:
            fig_sc.add_hline(y=lvl, line_dash="dash", line_color=col, annotation_text=lbl)
        fig_sc.update_layout(height=340, template="plotly_dark", yaxis_title="CSP Score",
                              yaxis_range=[0,115], margin=dict(l=0,r=0,t=20,b=0))
        st.plotly_chart(fig_sc, use_container_width=True)

        st.subheader("Strategy Score Comparison")
        fig_cmp = go.Figure()
        fig_cmp.add_trace(go.Bar(name="CSP",   x=tickers, y=[r["csp_score"]  for r in screener_rows_sorted], marker_color="#22c55e", opacity=0.85))
        fig_cmp.add_trace(go.Bar(name="CC",    x=tickers, y=[r["cc_score"]   for r in screener_rows_sorted], marker_color="#60a5fa", opacity=0.85))
        fig_cmp.add_trace(go.Bar(name="LEAP*", x=tickers, y=[r["leap_score"] for r in screener_rows_sorted], marker_color="#a78bfa", opacity=0.85))
        fig_cmp.update_layout(barmode="group", height=340, template="plotly_dark",
                               yaxis_title="Score", yaxis_range=[0,115],
                               legend=dict(orientation="h", y=1.05, x=0),
                               margin=dict(l=0,r=0,t=20,b=0))
        st.plotly_chart(fig_cmp, use_container_width=True)

        st.subheader("Four-Gate Filter Detail")
        st.caption("G1=Trend  G2=Stochastics  G3=Session  G4=BB Veto")
        for r in screener_rows_sorted:
            gr = r["gate_result"]; gates = gr["gates"]
            icon = "🟢" if gr["all_pass"] else "🔴"
            with st.expander(f"{icon} {r['ticker']}  CSP:{r['csp_score']}  Strike${r['csp_strike']:.1f}  Δ{r['csp_delta']}  θ${r['csp_theta']:.3f}/d"):
                gcols = st.columns(4)
                for idx, (gk, gv) in enumerate(gates.items()):
                    gcols[idx].markdown(f"**{gv['label']}** {'✅' if gv['pass'] else '❌'}")
                    gcols[idx].caption(gv["reason"])
                st.divider()
                rc1, rc2 = st.columns(2)
                rc1.markdown(f"""
**CSP Strike**
- Strike: **${r['csp_strike']:.1f}**
- Delta: **{r['csp_delta']}** ({greek_source_label(r.get('greek_source'))})
- Theta: **${r['csp_theta']:.3f}/day**
- IV: **{r['csp_iv'] or '—'}%**
- OI: **{r['csp_oi']}**
- Spread: **{r['csp_spread']}%** {'✅' if r['csp_spread'] and r['csp_spread']<20 else '⚠️ Wide' if r['csp_spread'] else '—'}
                """)
                if r["cc_strike"]:
                    rc2.markdown(f"""
**CC Strike**
- Strike: **${r['cc_strike']:.1f}**
- Delta: **{r['cc_delta']}**
- CC Score: **{r['cc_score']}**
                    """)
    elif "screener_results" not in st.session_state:
        st.markdown("""
        <div style='text-align:center;padding:60px;color:#94a3b8;'>
        <h3>Click <strong>Run Screener</strong> to analyse your watchlist</h3>
        <p>All greeks via Black-Scholes · Works market hours and after-hours</p>
        </div>""", unsafe_allow_html=True)
    else:
        st.warning("No screener data. Check Chain Inspector above and enable diagnostic output.")
