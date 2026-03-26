import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import math
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Options Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

DEFAULT_WATCHLIST = ["NVDA", "META", "TSLA", "IBIT", "GLD", "GDXJ", "BE", "VST", "CRWV"]

VIX_ZONES = [
    (0,  15, "#16a34a", "LOW — Ideal LEAP buying zone"),
    (15, 20, "#ca8a04", "NORMAL — Balanced regime"),
    (20, 30, "#ea580c", "ELEVATED — CC writing premium rich"),
    (30, 99, "#dc2626", "HIGH — Aggressive premium selling regime"),
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

SCORE_RATINGS = [
    (80, "🟢 Optimal"),
    (60, "🟡 Acceptable"),
    (40, "🟠 Marginal"),
    (0,  "🔴 Unsuitable"),
]

# ── Session state ──────────────────────────────────────────────────────────────
if "watchlist" not in st.session_state:
    st.session_state.watchlist = DEFAULT_WATCHLIST.copy()

# ── Data fetchers ──────────────────────────────────────────────────────────────
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

def fetch_chain(ticker, expiry):
    try:
        tk = yf.Ticker(ticker)
        chain = tk.option_chain(expiry)
        dte = (datetime.strptime(expiry, "%Y-%m-%d") - datetime.utcnow()).days
        return chain, dte
    except Exception:
        return None, None

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_chain_cached(ticker, expiry):
    try:
        tk = yf.Ticker(ticker)
        chain = tk.option_chain(expiry)
        dte = (datetime.strptime(expiry, "%Y-%m-%d") - datetime.utcnow()).days
        return chain.calls, chain.puts, dte
    except Exception:
        return None, None, None

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
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    ag = gain.ewm(alpha=1/window, min_periods=window).mean()
    al = loss.ewm(alpha=1/window, min_periods=window).mean()
    rs = ag / al.replace(0, np.nan)
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
        calls = chain.calls.copy()
        puts  = chain.puts.copy()
        c_atm = calls.iloc[(calls["strike"] - price).abs().argsort()[:1]]
        p_atm = puts.iloc[(puts["strike"]  - price).abs().argsort()[:1]]
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
def find_target_strike(chain_df, target_delta_abs, option_type="put"):
    """
    Finds the strike whose delta (absolute value, 0-100 scale) is closest
    to target_delta_abs. Uses Yahoo's native delta and theta columns directly.
    Skips strikes with OI < 50, zero/missing IV, or missing greeks.
    """
    best     = None
    min_diff = 999.0

    for _, row in chain_df.iterrows():
        try:
            K     = float(row.get("strike",            0) or 0)
            oi    = float(row.get("openInterest",       0) or 0)
            iv    = float(row.get("impliedVolatility",  0) or 0)
            bid   = float(row.get("bid",                0) or 0)
            ask   = float(row.get("ask",                0) or 0)
            delta_raw = row.get("delta", None)
            theta_raw = row.get("theta", None)

            if delta_raw is None or theta_raw is None:
                continue
            delta = float(delta_raw)
            theta = float(theta_raw)
        except (TypeError, ValueError):
            continue

        if (K <= 0 or iv <= 0 or np.isnan(iv)
                or np.isnan(oi) or oi < 50
                or np.isnan(delta) or np.isnan(theta)):
            continue

        d_abs = abs(delta) * 100.0
        diff  = abs(d_abs - target_delta_abs)

        if diff < min_diff:
            min_diff = diff
            mid = (bid + ask) / 2.0
            best = {
                "strike":     K,
                "delta":      round(d_abs, 1),
                "theta":      round(abs(theta), 4),
                "iv":         round(iv * 100.0, 1),
                "oi":         int(oi),
                "bid":        bid,
                "ask":        ask,
                "spread_pct": round((ask - bid) / mid * 100, 1) if mid > 0 else None,
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
    b = _tri_score(dte,       p["dte_opt"],  p["dte_lo"],  p["dte_hi"])
    c = _tri_score(delta_abs, p["delta_opt"], p["delta_lo"], p["delta_hi"])
    return round(p["w_iv"] * a + p["w_dte"] * b + p["w_delta"] * c, 1)

def score_color(s):
    if s >= 80: return "#22c55e"
    if s >= 60: return "#eab308"
    if s >= 40: return "#f97316"
    return "#ef4444"

# ── Four-gate filter ───────────────────────────────────────────────────────────
def calc_four_gates(r):
    df    = r.get("df")
    cl    = r.get("cl")
    price = r.get("price", 0)
    pct   = r.get("pct", 0)
    gates = {}

    # G1: Trend
    if cl is not None and len(cl.dropna()) >= 50:
        ma20 = float(cl.rolling(20).mean().dropna().iloc[-1])
        ma50 = float(r.get("ma50", 0))
        g1   = (ma20 > ma50) or (price > ma20)
        gates["G1"] = {
            "pass": g1, "label": "Trend (MA)",
            "reason": (
                f"20MA={ma20:.2f} {'>' if ma20 > ma50 else '<'} 50MA={ma50:.2f}  |  "
                f"Price {'>' if price > ma20 else '<'} 20MA"
            ),
        }
    else:
        gates["G1"] = {"pass": False, "label": "Trend (MA)", "reason": "Insufficient history"}

    # G2: Stochastics
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
            gates["G2"] = {
                "pass": g2, "label": "Stoch (14,3,3)",
                "reason": (
                    f"%K={k_cur:.1f}  %D={d_cur:.1f}  |  "
                    f"Sub-20 touch: {'✓' if touched_sub20 else '✗'}  |  "
                    f"%K cross ↑ %D: {'✓' if crossed_up else '✗'}"
                ),
            }
        else:
            gates["G2"] = {"pass": False, "label": "Stoch (14,3,3)", "reason": "Insufficient data"}
    else:
        gates["G2"] = {"pass": False, "label": "Stoch (14,3,3)", "reason": "No OHLCV data"}

    # G3: Session
    g3 = pct > -2.5
    gates["G3"] = {
        "pass": g3, "label": "Session",
        "reason": f"Today: {pct:+.2f}%  ({'OK' if g3 else 'FAIL — down >2.5%'})",
    }

    # G4: BB Veto
    if cl is not None and len(cl.dropna()) >= 22:
        _, _, lower = calc_bb_bands(cl)
        lower_clean = lower.dropna()
        cl_aligned  = cl.loc[lower_clean.index]
        if len(lower_clean) >= 2:
            below_today = float(cl_aligned.iloc[-1]) < float(lower_clean.iloc[-1])
            below_prev  = float(cl_aligned.iloc[-2]) < float(lower_clean.iloc[-2])
            walking     = below_today and below_prev
            g4          = not walking
            gates["G4"] = {
                "pass": g4, "label": "BB Veto",
                "reason": (
                    f"Lower band: {float(lower_clean.iloc[-1]):.2f}  |  "
                    f"{'Walking lower band ❌' if walking else 'Price inside bands ✓'}"
                ),
            }
        else:
            gates["G4"] = {"pass": True, "label": "BB Veto", "reason": "Sparse data — pass"}
    else:
        gates["G4"] = {"pass": True, "label": "BB Veto", "reason": "Insufficient data — pass"}

    return {"gates": gates, "all_pass": all(g["pass"] for g in gates.values())}

# ── Screener: per-stock assembly ───────────────────────────────────────────────
def get_screener_row(ticker, result):
    price    = result.get("price")
    all_exps = result.get("all_exps", [])
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
                min_diff = diff
                exp_csp  = exp
                dte_csp  = dte
        except Exception:
            continue

    if exp_csp is None:
        return None

    calls_df, puts_df, _ = fetch_chain_cached(ticker, exp_csp)
    if puts_df is None or puts_df.empty:
        return None

    csp = find_target_strike(puts_df, 18.0, "put")
    if csp is None:
        return None

    cc = None
    if calls_df is not None and not calls_df.empty:
        cc = find_target_strike(calls_df, 35.0, "call")

    nis        = calc_nis(csp["theta"], dte_csp, price)
    csp_score  = calc_suitability(nis, dte_csp, csp["delta"], "CSP")
    cc_score   = 0.0
    if cc:
        cc_nis   = calc_nis(cc["theta"], dte_csp, price)
        cc_score = calc_suitability(cc_nis, dte_csp, cc["delta"], "CC")
    leap_score = calc_suitability(nis, dte_csp, csp["delta"], "LEAP")
    gate_result = calc_four_gates(result)

    return {
        "ticker":      ticker,
        "price":       price,
        "expiry":      exp_csp,
        "dte":         dte_csp,
        "csp_strike":  csp["strike"],
        "csp_delta":   csp["delta"],
        "csp_theta":   csp["theta"],
        "csp_iv":      csp["iv"],
        "csp_oi":      csp["oi"],
        "csp_spread":  csp["spread_pct"],
        "cc_strike":   cc["strike"] if cc else None,
        "cc_delta":    cc["delta"]  if cc else None,
        "nis":         round(nis, 1),
        "csp_score":   csp_score,
        "cc_score":    cc_score,
        "leap_score":  leap_score,
        "gate_result": gate_result,
    }

# ── Signal engines ─────────────────────────────────────────────────────────────
def leap_signal(hvr, rsi_val, above_50ma, above_200ma, vix_lvl):
    score = 0
    reasons = []
    if hvr is not None:
        if hvr < 25:
            score += 3; reasons.append("✅ HV Rank low (<25) — cheap premium")
        elif hvr < 40:
            score += 2; reasons.append("🟡 HV Rank moderate (25-40)")
        else:
            score -= 1; reasons.append("❌ HV Rank elevated — expensive entry")
    if rsi_val is not None:
        if 33 <= rsi_val <= 52:
            score += 2; reasons.append("✅ RSI in ideal recovery zone (33-52)")
        elif 52 < rsi_val <= 65:
            score += 1; reasons.append("🟡 RSI extended but not overbought")
        elif rsi_val < 30:
            score += 1; reasons.append("🟡 RSI oversold — wait for turn upward")
        else:
            score -= 1; reasons.append("❌ RSI overbought — avoid chasing")
    if above_200ma:
        score += 2; reasons.append("✅ Above 200MA — long term trend intact")
    else:
        score -= 1; reasons.append("❌ Below 200MA — long term trend broken")
    if above_50ma:
        score += 1; reasons.append("✅ Above 50MA — medium term trend OK")
    if vix_lvl is not None:
        if vix_lvl < 18:
            score += 1; reasons.append("✅ VIX low — index premium cheap")
        elif vix_lvl > 28:
            score -= 1; reasons.append("⚠️ VIX elevated — vol expansion risk")
    label = (
        "🟢 STRONG ENTRY" if score >= 7 else
        "🟡 DECENT ENTRY" if score >= 4 else
        "🟠 MARGINAL"     if score >= 2 else
        "🔴 AVOID"
    )
    return label, score, reasons

def cc_signal(hvr, rsi_val, above_50ma, pcr_val):
    score = 0
    reasons = []
    if hvr is not None:
        if hvr > 65:
            score += 3; reasons.append("✅ HV Rank high (>65) — premium rich")
        elif hvr > 45:
            score += 2; reasons.append("🟡 HV Rank moderate (45-65)")
        else:
            reasons.append("❌ HV Rank low — thin premium")
    if rsi_val is not None:
        if rsi_val > 65:
            score += 2; reasons.append("✅ RSI overbought — capping upside is smart")
        elif rsi_val > 50:
            score += 1; reasons.append("🟡 RSI bullish but not extreme")
        elif rsi_val < 35:
            score -= 1; reasons.append("❌ RSI oversold — don't cap at the bottom")
    if above_50ma:
        score += 1; reasons.append("✅ Above 50MA — safe to sell calls")
    else:
        score -= 1; reasons.append("⚠️ Below 50MA — trend weakening")
    if pcr_val is not None and pcr_val > 1.2:
        score += 1; reasons.append("✅ Elevated PCR — fear = good premium")
    label = (
        "🟢 WRITE NOW" if score >= 5 else
        "🟡 DECENT"    if score >= 3 else
        "🟠 MARGINAL"  if score >= 1 else
        "🔴 WAIT"
    )
    return label, score, reasons

def csp_signal(hvr, rsi_val, above_200ma):
    score = 0
    reasons = []
    if hvr is not None:
        if hvr > 55:
            score += 2; reasons.append("✅ High HV Rank — CSP premium rich")
        elif hvr > 35:
            score += 1; reasons.append("🟡 Moderate HV Rank")
    if rsi_val is not None:
        if 30 <= rsi_val <= 50:
            score += 2; reasons.append("✅ RSI in recovery — good put strike support")
        elif rsi_val < 30:
            score += 1; reasons.append("⚠️ Deeply oversold — wait for stabilisation")
        elif rsi_val > 70:
            score -= 1; reasons.append("❌ RSI overbought — don't sell puts at the top")
    if above_200ma:
        score += 2; reasons.append("✅ Above 200MA — reduces assignment risk")
    else:
        score -= 1; reasons.append("❌ Below 200MA — assignment risk elevated")
    label = (
        "🟢 SELL PUT" if score >= 4 else
        "🟡 DECENT"   if score >= 2 else
        "🟠 MARGINAL" if score >= 1 else
        "🔴 AVOID"
    )
    return label, score, reasons

# ── Main analyse ───────────────────────────────────────────────────────────────
def analyse(ticker, period, vix_current):
    df = fetch_prices(ticker, period)
    if df is None:
        return None
    cl      = df["Close"].squeeze()
    curr    = float(cl.iloc[-1])
    prev    = float(cl.iloc[-2])
    pct_chg = (curr / prev - 1) * 100
    hv20_s  = calc_hv(cl, 20)
    hv60_s  = calc_hv(cl, 60)
    hv_cur  = float(hv20_s.dropna().iloc[-1]) if not hv20_s.dropna().empty else None
    hvr     = calc_iv_rank(hv20_s)
    hvpct   = calc_iv_percentile(hv20_s)
    rsi_s   = calc_rsi(cl)
    rsi_cur = float(rsi_s.dropna().iloc[-1]) if not rsi_s.dropna().empty else None
    atr_s   = calc_atr(df)
    atr_cur = float(atr_s.dropna().iloc[-1]) if not atr_s.dropna().empty else None
    bbw_s   = calc_bb_width(cl)
    bbw_cur = float(bbw_s.dropna().iloc[-1]) if not bbw_s.dropna().empty else None
    ma50    = float(cl.rolling(50).mean().iloc[-1])
    ma200   = float(cl.rolling(200).mean().iloc[-1])
    ab50    = curr > ma50
    ab200   = curr > ma200
    all_exps = fetch_all_expiries(ticker)
    c_iv, p_iv, pcr_val, chain, exp, dte = None, None, None, None, None, None
    if all_exps:
        today = datetime.utcnow()
        valid = [e for e in all_exps if (datetime.strptime(e, "%Y-%m-%d") - today).days > 14]
        if valid:
            exp = valid[0]
            chain, dte = fetch_chain(ticker, exp)
            if chain:
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
        "exp": exp, "dte": dte, "all_exps": all_exps,
        "df": df, "cl": cl,
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

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## Settings")
    new_ticker = st.text_input("Add a ticker (press Enter)", placeholder="e.g. AMZN", key="new_ticker_input")
    if new_ticker:
        t = new_ticker.upper().strip()
        if t and t not in st.session_state.watchlist:
            st.session_state.watchlist.append(t)
            st.rerun()
    if st.session_state.watchlist:
        remove = st.selectbox("Remove a ticker", ["— select to remove —"] + st.session_state.watchlist, key="remove_ticker")
        if remove != "— select to remove —":
            st.session_state.watchlist.remove(remove)
            st.rerun()
    st.markdown(f"**Active watchlist:** {', '.join(st.session_state.watchlist)}")
    period = st.selectbox("Price History", ["6mo", "1y", "2y"], index=1)
    st.divider()
    vix_df  = fetch_vix("1y")
    vix_now = None
    vix_chg = 0
    if vix_df is not None and not vix_df.empty:
        vix_cl_s = vix_df["Close"].squeeze()
        vix_now  = float(vix_cl_s.iloc[-1])
        vix_prev = float(vix_cl_s.iloc[-2])
        vix_chg  = vix_now - vix_prev
        vc, vl   = vix_zone(vix_now)
        st.markdown(f"### VIX: {vix_now:.1f} ({vix_chg:+.2f})")
        st.markdown(f"<span style='color:{vc};font-weight:700'>{vl}</span>", unsafe_allow_html=True)
        vix_52hi = float(vix_cl_s.max())
        vix_52lo = float(vix_cl_s.min())
        vix_rank = (vix_now - vix_52lo) / (vix_52hi - vix_52lo) * 100 if vix_52hi != vix_52lo else 50
        st.progress(int(vix_rank), text=f"VIX 52wk Rank: {vix_rank:.0f}%")
    st.divider()
    st.caption("HV Rank is a proxy for IV Rank. ATM IV is live from Yahoo options chain.")

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
            "50MA":       "YES" if r["ab50"] else "NO",
            "200MA":      "YES" if r["ab200"] else "NO",
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
        colors = [
            "#22c55e" if v < 25 else
            "#eab308" if v < 45 else
            "#f97316" if v < 65 else
            "#ef4444"
            for v in hvr_data.values()
        ]
        fig_hvr = go.Figure(go.Bar(
            x=list(hvr_data.keys()), y=list(hvr_data.values()),
            marker_color=colors,
            text=[f"{v:.0f}" for v in hvr_data.values()],
            textposition="outside",
        ))
        fig_hvr.add_hline(y=25, line_dash="dash", line_color="#22c55e", annotation_text="25 = LEAP Buy Zone")
        fig_hvr.add_hline(y=65, line_dash="dash", line_color="#ef4444", annotation_text="65 = CC Writing Zone")
        fig_hvr.update_layout(height=320, template="plotly_dark", yaxis_title="HV Rank",
                               yaxis_range=[0, 115], margin=dict(l=0, r=0, t=20, b=0))
        st.plotly_chart(fig_hvr, use_container_width=True)

    rsi_data = {t: r["rsi"] for t, r in results.items() if r["rsi"] is not None}
    if rsi_data:
        st.subheader("RSI Snapshot")
        rsi_colors = [
            "#22c55e" if 33 <= v <= 52 else
            "#eab308" if 52 < v <= 65 else
            "#f97316" if v > 65 else
            "#94a3b8"
            for v in rsi_data.values()
        ]
        fig_rsi = go.Figure(go.Bar(
            x=list(rsi_data.keys()), y=list(rsi_data.values()),
            marker_color=rsi_colors,
            text=[f"{v:.0f}" for v in rsi_data.values()],
            textposition="outside",
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
        r  = results[sel]
        df = r["df"]
        cl = r["cl"]
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
        fig = make_subplots(
            rows=4, cols=1, shared_xaxes=True,
            row_heights=[0.48, 0.18, 0.17, 0.17],
            subplot_titles=[
                f"{sel} — Price & Moving Averages",
                "HV20 (Volatility Proxy)", "RSI (14)", "Bollinger Band Width",
            ],
            vertical_spacing=0.04,
        )
        fig.add_trace(go.Candlestick(
            x=df.index, open=df["Open"].squeeze(), high=df["High"].squeeze(),
            low=df["Low"].squeeze(), close=cl, name="Price",
            increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
            increasing_fillcolor="#26a69a",  decreasing_fillcolor="#ef5350",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=cl.rolling(50).mean(),
            name="50MA", line=dict(color="#f97316", width=1.4)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=cl.rolling(200).mean(),
            name="200MA", line=dict(color="#60a5fa", width=1.6)), row=1, col=1)
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
| Suggested CC strike (1.5× ATR above price) | ~${r['price'] + r['atr']*1.5:.2f} |
| Suggested CSP strike (1.5× ATR below price) | ~${r['price'] - r['atr']*1.5:.2f} |
            """)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — OPTIONS CHAIN
# ══════════════════════════════════════════════════════════════════════════════
with tab_chain:
    sel_c = st.selectbox("Select Ticker", list(results.keys()), key="chain_sel")
    if sel_c and sel_c in results:
        r        = results[sel_c]
        price    = r["price"]
        all_exps = r.get("all_exps", [])
        if all_exps:
            selected_exp = st.selectbox("Select Expiry", all_exps, index=0, key=f"exp_sel_{sel_c}")
            chain, dte   = fetch_chain(sel_c, selected_exp)
            if chain:
                st.caption(f"Expiry: {selected_exp} ({dte} DTE) | Current Price: ${price:.2f}")
                def fmt_chain(df_raw, side):
                    df_raw = df_raw.copy()
                    df_raw["IV %"] = (df_raw["impliedVolatility"] * 100).round(1)
                    df_raw["Moneyness"] = df_raw["strike"].apply(
                        lambda s: "ATM" if abs(s - price) / price < 0.02
                        else ("ITM" if ((s < price and side == "call") or (s > price and side == "put")) else "OTM")
                    )
                    cols      = ["strike", "Moneyness", "lastPrice", "bid", "ask", "volume", "openInterest", "IV %", "delta"]
                    available = [c for c in cols if c in df_raw.columns]
                    return (df_raw[available]
                            .rename(columns={"lastPrice": "Last", "openInterest": "OI",
                                             "strike": "Strike", "volume": "Volume"})
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
                fig_smile.add_trace(go.Scatter(
                    x=chain.calls["strike"], y=chain.calls["impliedVolatility"]*100,
                    name="Calls IV", mode="lines+markers",
                    line=dict(color="#26a69a", width=2), marker=dict(size=5)))
                fig_smile.add_trace(go.Scatter(
                    x=chain.puts["strike"],  y=chain.puts["impliedVolatility"]*100,
                    name="Puts IV",  mode="lines+markers",
                    line=dict(color="#ef5350", width=2), marker=dict(size=5)))
                fig_smile.add_vline(x=price, line_dash="dash", line_color="white",
                                    annotation_text=f"${price:.2f}")
                fig_smile.update_layout(height=350, template="plotly_dark",
                                        xaxis_title="Strike", yaxis_title="IV (%)",
                                        margin=dict(l=0, r=0, t=20, b=0))
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
                                     margin=dict(l=0, r=0, t=20, b=0))
                st.plotly_chart(fig_oi, use_container_width=True)
            else:
                st.warning("Could not load chain for this expiry.")
        else:
            st.warning(f"No options data available for {sel_c}.")

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
            row_heights=[0.7, 0.3],
            subplot_titles=["VIX Level", "30-day Rolling Average"])
        fig_vix.add_trace(go.Scatter(x=vix_df.index, y=vix_cl, name="VIX",
            fill="tozeroy", line=dict(color="#f87171", width=1.5),
            fillcolor="rgba(248,113,113,0.12)"), row=1, col=1)
        for lo, hi, color, label in VIX_ZONES:
            fig_vix.add_hrect(y0=lo, y1=min(hi, 50), fillcolor=color, opacity=0.05, row=1, col=1)
            fig_vix.add_hline(y=lo, line_dash="dot", line_color=color, opacity=0.4, row=1, col=1)
        fig_vix.add_trace(go.Scatter(x=vix_df.index, y=vix_cl.rolling(30).mean(),
            name="30d MA", line=dict(color="#fbbf24", width=1.5)), row=2, col=1)
        fig_vix.update_layout(height=520, template="plotly_dark",
                               margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig_vix, use_container_width=True)
        st.markdown("""
| VIX | Regime | LEAP Buying | CC Writing | CSP Writing |
|---|---|---|---|---|
| Below 15 | Low | Best time — cheapest premium | Thin premium | Good if trend up |
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
        "Greeks sourced directly from Yahoo Finance. Requires live market hours for valid delta/theta data."
    )

    with st.expander("How the score is calculated", expanded=False):
        st.markdown("""
**Core formula**

For the nearest 30–45 DTE expiry, the screener finds:
- The put strike closest to **Δ18** → CSP target
- The call strike closest to **Δ35** → CC target

Then calculates the **Normalised IV Score (NIS)**:

> **Raw = θ × √DTE / Price** → normalised 0–100 (0 = IV≈15%, 100 = IV≈120%)

**Composite score weights**

| Component | CSP | CC | LEAP |
|---|---|---|---|
| A — NIS (inverted for LEAP) | 50% | 50% | 30% |
| B — DTE fit (37d / 37d / 547d optimal) | 30% | 30% | 40% |
| C — Delta fit (18 / 35 / 80 optimal) | 20% | 20% | 30% |

**Four-gate filter** — all must pass to trade:
- **G1** 20MA > 50MA or price > 20MA
- **G2** Stoch %K sub-20 touch + %K cross above %D
- **G3** Not down >2.5% today
- **G4** Not walking lower Bollinger Band 2+ sessions

**Ratings:** ≥80 Optimal · ≥60 Acceptable · ≥40 Marginal · <40 Unsuitable

> ⚠️ Delta/theta only populate during US market hours — Dubai: 18:30 to 01:00
        """)

    col_run, col_note = st.columns([1, 4])
    with col_run:
        run_btn = st.button("🔍 Run Screener", type="primary", use_container_width=True)
    with col_note:
        st.info("First run ~30–60 s (fetches live chains). Results cached 30 min.")

    if run_btn:
        if not results:
            st.warning("No market data loaded. Check watchlist.")
        else:
            screener_rows = []
            prog = st.progress(0, text="Initialising...")
            n    = len(results)
            for i, (ticker, result) in enumerate(results.items()):
                prog.progress((i + 1) / n, text=f"Analysing {ticker}  ({i+1}/{n})")
                row = get_screener_row(ticker, result)
                if row:
                    screener_rows.append(row)
            prog.empty()
            st.session_state["screener_results"] = screener_rows

    screener_rows = st.session_state.get("screener_results", [])

    if screener_rows:
        screener_rows_sorted = sorted(screener_rows, key=lambda x: x["csp_score"], reverse=True)

        table_rows = []
        for r in screener_rows_sorted:
            gr         = r["gate_result"]
            gates      = gr["gates"]
            gate_icons = "".join(["✅" if gates[f"G{i}"]["pass"] else "❌" for i in range(1, 5)])
            status     = "🟢 TRADE" if gr["all_pass"] else "🔴 WAIT"
            table_rows.append({
                "Ticker":      r["ticker"],
                "Price":       f"${r['price']:.2f}",
                "Expiry":      r["expiry"],
                "DTE":         r["dte"],
                "CSP Strike":  f"${r['csp_strike']:.1f}",
                "CSP Δ":       r["csp_delta"],
                "θ/day":       f"${r['csp_theta']:.3f}",
                "Put IV %":    r["csp_iv"],
                "OI":          r["csp_oi"],
                "Spread %":    r["csp_spread"] if r["csp_spread"] else "—",
                "NIS":         r["nis"],
                "CSP Score":   r["csp_score"],
                "CC Score":    r["cc_score"],
                "LEAP Score*": r["leap_score"],
                "Gates":       gate_icons,
                "Status":      status,
            })

        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True, height=480)
        st.caption("\\* LEAP Score at short DTE — DTE component near zero. Use as relative IV indicator only.")

        st.subheader("CSP Suitability Ranking")
        tickers    = [r["ticker"]    for r in screener_rows_sorted]
        csp_scores = [r["csp_score"] for r in screener_rows_sorted]
        fig_sc = go.Figure(go.Bar(
            x=tickers, y=csp_scores,
            marker_color=[score_color(s) for s in csp_scores],
            text=[f"{s:.0f}" for s in csp_scores],
            textposition="outside",
        ))
        for lvl, col, lbl in [
            (80, "#22c55e", "80 — Optimal"),
            (60, "#eab308", "60 — Acceptable"),
            (40, "#f97316", "40 — Marginal"),
        ]:
            fig_sc.add_hline(y=lvl, line_dash="dash", line_color=col, annotation_text=lbl)
        fig_sc.update_layout(height=340, template="plotly_dark",
                              yaxis_title="CSP Score (0–100)", yaxis_range=[0, 115],
                              margin=dict(l=0, r=0, t=20, b=0))
        st.plotly_chart(fig_sc, use_container_width=True)

        st.subheader("Strategy Score Comparison")
        fig_cmp = go.Figure()
        fig_cmp.add_trace(go.Bar(name="CSP",    x=tickers, y=[r["csp_score"]  for r in screener_rows_sorted], marker_color="#22c55e", opacity=0.85))
        fig_cmp.add_trace(go.Bar(name="CC",     x=tickers, y=[r["cc_score"]   for r in screener_rows_sorted], marker_color="#60a5fa", opacity=0.85))
        fig_cmp.add_trace(go.Bar(name="LEAP*",  x=tickers, y=[r["leap_score"] for r in screener_rows_sorted], marker_color="#a78bfa", opacity=0.85))
        fig_cmp.update_layout(barmode="group", height=340, template="plotly_dark",
                               yaxis_title="Score (0–100)", yaxis_range=[0, 115],
                               legend=dict(orientation="h", y=1.05, x=0),
                               margin=dict(l=0, r=0, t=20, b=0))
        st.plotly_chart(fig_cmp, use_container_width=True)

        st.subheader("Four-Gate Filter Detail")
        st.caption("G1=Trend  G2=Stochastics  G3=Session  G4=BB Veto — all four ✅ required to trade")
        for r in screener_rows_sorted:
            gr    = r["gate_result"]
            gates = gr["gates"]
            icon  = "🟢" if gr["all_pass"] else "🔴"
            with st.expander(
                f"{icon} {r['ticker']}  —  CSP Score: {r['csp_score']}  |  "
                f"Strike ${r['csp_strike']:.1f}  Δ{r['csp_delta']}  "
                f"θ${r['csp_theta']:.3f}/day  IV {r['csp_iv']}%"
            ):
                gcols = st.columns(4)
                for idx, (gk, gv) in enumerate(gates.items()):
                    gcols[idx].markdown(f"**{gv['label']}**  {'✅' if gv['pass'] else '❌'}")
                    gcols[idx].caption(gv["reason"])
                st.divider()
                rc1, rc2 = st.columns(2)
                rc1.markdown(f"""
**CSP Recommended Strike**
- Strike: **${r['csp_strike']:.1f}**
- Delta: **{r['csp_delta']}**
- Theta: **${r['csp_theta']:.3f}/day**
- IV: **{r['csp_iv']}%**
- Open Interest: **{r['csp_oi']}**
- Spread: **{r['csp_spread']}%** {'✅' if r['csp_spread'] and r['csp_spread'] < 20 else '⚠️ Wide' if r['csp_spread'] else '—'}
                """)
                if r["cc_strike"]:
                    rc2.markdown(f"""
**CC Recommended Strike**
- Strike: **${r['cc_strike']:.1f}**
- Delta: **{r['cc_delta']}**
- CC Score: **{r['cc_score']}**
                    """)

    elif "screener_results" not in st.session_state:
        st.markdown("""
        <div style='text-align:center; padding:60px; color:#94a3b8;'>
            <h3>Click <strong>Run Screener</strong> to analyse your watchlist</h3>
            <p>Live delta and theta required — markets open 18:30 Dubai time.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning(
            "No screener data returned. Yahoo Finance may not have greeks available right now. "
            "Try again after 18:30 Dubai time when US markets open."
        )
