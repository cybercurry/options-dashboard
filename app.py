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

DEFAULT_WATCHLIST = ["NVDA", "META", "TSLA", "IBIT", "GLD", "GDXJ", "BE", "VST", "CRWV"]

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
        "delta_opt": 35, "delta_lo": 20, "delta_hi": 50,
        "dte_opt":   30, "dte_lo":   21, "dte_hi":   45,
        "w_iv": 0.50,    "w_dte": 0.30,  "w_delta": 0.20,
        "iv_dir": 1,     "option_type": "call",
    },
    "LEAP": {
        "delta_opt": 80, "delta_lo": 60, "delta_hi": 95,
        "dte_opt":  547, "dte_lo":  180, "dte_hi":  900,
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

# ══════════════════════════════════════════════════════════════════════════════
# BLACK-SCHOLES GREEKS
# ══════════════════════════════════════════════════════════════════════════════
def _bs_greeks(S, K, T, sigma, r=RISK_FREE_RATE, option_type="call"):
    try:
        if any(x is None for x in (S, K, T, sigma)): return None, None
        S, K, T, sigma = float(S), float(K), float(T), float(sigma)
        if any(pd.isna(x) for x in (S,K,T,sigma)): return None, None
        if T<=0 or sigma<=0 or S<=0 or K<=0:         return None, None
        sigma  = max(0.05, min(sigma, 3.0))
        T_yr   = T/365.0; sqrtT = math.sqrt(T_yr)
        d1     = (math.log(S/K) + (r + 0.5*sigma**2)*T_yr)/(sigma*sqrtT)
        d2     = d1 - sigma*sqrtT
        delta  = norm.cdf(d1) if option_type=="call" else norm.cdf(d1)-1.0
        pdf_d1 = norm.pdf(d1)
        tc     = -(S*pdf_d1*sigma)/(2*sqrtT) - r*K*math.exp(-r*T_yr)*norm.cdf(d2)
        theta  = (tc if option_type=="call" else tc + r*K*math.exp(-r*T_yr))/365.0
        return round(abs(delta)*100.0, 2), round(abs(theta), 4)
    except Exception:
        return None, None

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

def calc_stochastics(df, k_period=14, d_period=3):
    high=df["High"].squeeze(); low=df["Low"].squeeze(); close=df["Close"].squeeze()
    raw_k = 100.0*(close-low.rolling(k_period).min())/(high.rolling(k_period).max()-low.rolling(k_period).min()+1e-10)
    k = raw_k.rolling(d_period).mean()
    return k, k.rolling(d_period).mean()

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

        if yahoo_ok:
            d_abs=abs(float(d_raw))*100.0; theta=abs(float(t_raw)); gs="yahoo"
        else:
            if 0.01<iv<5.0:              sigma=iv;          src="strike"
            elif median_iv is not None:  sigma=median_iv;   src="chain_median"
            elif hv_pct and hv_pct>0:   sigma=float(hv_pct); src="hv20"
            else:                        sigma=0.30;         src="default"
            d_abs, theta = _bs_greeks(price, K, dte, sigma, option_type=option_type)
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
            best={"strike":K,"delta":round(d_abs,1),"theta":round(theta,4),
                  "iv":eff_iv,"oi":int(oi),"bid":bid,"ask":ask,
                  "spread_pct":round((ask-bid)/mid*100,1) if mid>0 else None,
                  "greek_source":gs,"_inspected":inspected,"_scored":scored}
    return best

def calc_nis(theta, dte, price):
    if theta<=0 or dte<=0 or price<=0: return 0.0
    raw = theta*math.sqrt(dte)/price
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

def calc_four_gates(r):
    df=r.get("df"); cl=r.get("cl"); price=r.get("price",0); pct=r.get("pct",0)
    gates={}

    if cl is not None and len(cl.dropna())>=50:
        ma20=float(cl.rolling(20).mean().dropna().iloc[-1]); ma50=float(r.get("ma50",0))
        g1=(ma20>ma50) or (price>ma20)
        gates["G1"]={"pass":g1,"label":"Trend (MA)",
            "reason":f"20MA={ma20:.2f} {'>' if ma20>ma50 else '<'} 50MA={ma50:.2f}  |  Price {'>' if price>ma20 else '<'} 20MA"}
    else:
        gates["G1"]={"pass":False,"label":"Trend (MA)","reason":"Insufficient history"}

    if df is not None and len(df)>=25:
        k_s,d_s=calc_stochastics(df); k_c=k_s.dropna(); d_c=d_s.dropna()
        if len(k_c)>=6 and len(d_c)>=6:
            touched=bool((k_c.iloc[-5:]<20).any())
            k_cur,k_prev=float(k_c.iloc[-1]),float(k_c.iloc[-2])
            d_cur,d_prev=float(d_c.iloc[-1]),float(d_c.iloc[-2])
            crossed=(k_cur>d_cur) and (k_prev<=d_prev)
            gates["G2"]={"pass":touched and crossed,"label":"Stoch (14,3,3)",
                "reason":f"%K={k_cur:.1f} %D={d_cur:.1f}  Sub-20:{'✓' if touched else '✗'}  Cross↑:{'✓' if crossed else '✗'}"}
        else:
            gates["G2"]={"pass":False,"label":"Stoch (14,3,3)","reason":"Insufficient data"}
    else:
        gates["G2"]={"pass":False,"label":"Stoch (14,3,3)","reason":"No OHLCV data"}

    g3=pct>-2.5
    gates["G3"]={"pass":g3,"label":"Session","reason":f"Today:{pct:+.2f}% ({'OK' if g3 else 'FAIL — down >2.5%'})"}

    if cl is not None and len(cl.dropna())>=22:
        _,_,lower=calc_bb_bands(cl); lo_c=lower.dropna(); cl_a=cl.loc[lo_c.index]
        if len(lo_c)>=2:
            walking=(float(cl_a.iloc[-1])<float(lo_c.iloc[-1])) and (float(cl_a.iloc[-2])<float(lo_c.iloc[-2]))
            gates["G4"]={"pass":not walking,"label":"BB Veto",
                "reason":f"Lower band:{float(lo_c.iloc[-1]):.2f}  |  {'Walking lower ❌' if walking else 'Inside bands ✓'}"}
        else:
            gates["G4"]={"pass":True,"label":"BB Veto","reason":"Sparse — pass"}
    else:
        gates["G4"]={"pass":True,"label":"BB Veto","reason":"Insufficient — pass"}

    return {"gates":gates,"all_pass":all(g["pass"] for g in gates.values())}

def get_screener_row(ticker, result):
    price=result.get("price"); all_exps=result.get("all_exps",[]); hv_raw=result.get("hv20")
    if not all_exps or not price or price<=0: return None

    today=datetime.utcnow(); exp_csp=None; dte_csp=None; min_diff=999
    for exp in all_exps:
        try:
            dte=(datetime.strptime(exp,"%Y-%m-%d")-today).days
            diff=abs(dte-30)
            if 21<=dte<=45 and diff<min_diff:
                min_diff=diff; exp_csp=exp; dte_csp=dte
        except Exception:
            continue
    if exp_csp is None: return None

    calls_df,puts_df,_=fetch_chain_cached(ticker,exp_csp)
    if puts_df is None or puts_df.empty: return None

    hv_sigma=(hv_raw/100.0) if (hv_raw and hv_raw>0) else None
    csp=find_target_strike(puts_df, 30.0,"put", price,dte_csp,hv_sigma)
    if csp is None: return None

    cc=None
    if calls_df is not None and not calls_df.empty:
        cc=find_target_strike(calls_df,35.0,"call",price,dte_csp,hv_sigma)

    nis=calc_nis(csp["theta"],dte_csp,price)
    csp_score=calc_suitability(nis,dte_csp,csp["delta"],"CSP")
    cc_score =(calc_suitability(calc_nis(cc["theta"],dte_csp,price),dte_csp,cc["delta"],"CC") if cc else 0.0)
    leap_score=calc_suitability(nis,dte_csp,csp["delta"],"LEAP")
    gate_result=calc_four_gates(result)

    return {"ticker":ticker,"price":price,"expiry":exp_csp,"dte":dte_csp,
            "csp_strike":csp["strike"],"csp_delta":csp["delta"],"csp_theta":csp["theta"],
            "csp_iv":csp["iv"],"csp_oi":csp["oi"],"csp_spread":csp["spread_pct"],
            "cc_strike":cc["strike"] if cc else None,"cc_delta":cc["delta"] if cc else None,
            "nis":round(nis,1),"csp_score":csp_score,"cc_score":cc_score,
            "leap_score":leap_score,"gate_result":gate_result,
            "greek_source":csp.get("greek_source","unknown"),
            "_inspected":csp.get("_inspected"),"_scored":csp.get("_scored")}

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

def cc_signal(hvr,rsi_val,above_50ma,pcr_val):
    score=0; reasons=[]
    if hvr is not None:
        if hvr>65:   score+=3; reasons.append("✅ HV Rank high — premium rich")
        elif hvr>45: score+=2; reasons.append("🟡 HV Rank moderate")
        else:        reasons.append("❌ HV Rank low — thin premium")
    if rsi_val is not None:
        if rsi_val>65:   score+=2; reasons.append("✅ RSI overbought — smart to cap")
        elif rsi_val>50: score+=1; reasons.append("🟡 RSI bullish, not extreme")
        elif rsi_val<35: score-=1; reasons.append("❌ RSI oversold — don't cap the bottom")
    if above_50ma: score+=1; reasons.append("✅ Above 50MA — safe to sell calls")
    else:          score-=1; reasons.append("⚠️ Below 50MA — trend weakening")
    if pcr_val is not None and pcr_val>1.2:
        score+=1; reasons.append("✅ Elevated PCR — fear = premium")
    label=("🟢 WRITE NOW" if score>=5 else "🟡 DECENT" if score>=3
           else "🟠 MARGINAL" if score>=1 else "🔴 WAIT")
    return label, score, reasons

def csp_signal(hvr,rsi_val,above_200ma):
    score=0; reasons=[]
    if hvr is not None:
        if hvr>55:   score+=2; reasons.append("✅ High HV Rank — CSP premium rich")
        elif hvr>35: score+=1; reasons.append("🟡 Moderate HV Rank")
    if rsi_val is not None:
        if 30<=rsi_val<=50: score+=2; reasons.append("✅ RSI in recovery — put support good")
        elif rsi_val<30:    score+=1; reasons.append("⚠️ Deep oversold — wait to stabilise")
        elif rsi_val>70:    score-=1; reasons.append("❌ RSI overbought — don't sell puts here")
    if above_200ma: score+=2; reasons.append("✅ Above 200MA — reduces assignment risk")
    else:           score-=1; reasons.append("❌ Below 200MA — assignment risk elevated")
    label=("🟢 SELL PUT" if score>=4 else "🟡 DECENT" if score>=2
           else "🟠 MARGINAL" if score>=1 else "🔴 AVOID")
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
    all_exps=fetch_all_expiries(ticker)
    c_iv=p_iv=pcr_val=chain=exp=dte=None
    if all_exps:
        today=datetime.utcnow()
        valid=[e for e in all_exps if (datetime.strptime(e,"%Y-%m-%d")-today).days>14]
        if valid:
            exp=valid[0]
            calls_df,puts_df,dte=fetch_chain_cached(ticker,exp)
            if calls_df is not None:
                chain=type("_C",(),{"calls":calls_df,"puts":puts_df})()
                c_iv,p_iv=calc_atm_iv(chain,curr); pcr_val=calc_pcr(chain)
    leap_lbl,leap_sc,leap_r=leap_signal(hvr,rsi_cur,ab50,ab200,vix_current)
    cc_lbl,cc_sc,cc_r=cc_signal(hvr,rsi_cur,ab50,pcr_val)
    csp_lbl,csp_sc,csp_r=csp_signal(hvr,rsi_cur,ab200)
    return {"ticker":ticker,"price":curr,"pct":pct_chg,
            "hv20":hv_cur,"hvr":hvr,"hvpct":hvpct,"hv20_s":hv20_s,"hv60_s":hv60_s,
            "rsi":rsi_cur,"rsi_s":rsi_s,"atr":atr_cur,"bbw":bbw_cur,"bbw_s":bbw_s,
            "ma50":ma50,"ma200":ma200,"ab50":ab50,"ab200":ab200,
            "c_iv":c_iv,"p_iv":p_iv,"pcr":pcr_val,"exp":exp,"dte":dte,
            "all_exps":all_exps,"df":df,"cl":cl,
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

# ── Gauge helpers ──────────────────────────────────────────────────────────────
def fg_color(score):
    if score is None: return "#6b7280"
    if score<25:  return "#dc2626"
    if score<45:  return "#ea580c"
    if score<55:  return "#ca8a04"
    if score<75:  return "#16a34a"
    return "#15803d"

# Sector zone config (shared by both stock and crypto gauges)
_FG_ZONES = [
    (0,  25, "#ef4444", "EXTREME\nFEAR"),
    (25, 45, "#f97316", "FEAR"),
    (45, 55, "#eab308", "NEUTRAL"),
    (55, 75, "#22c55e", "GREED"),
    (75,100, "#15803d", "EXTREME\nGREED"),
]

def semicircle_gauge(score, title, rating, source_label=""):
    """
    CNN-style semicircle gauge with coloured zones and a needle pointer.
    score: 0–100
    """
    # Which zone is active?
    def active_zone(s):
        for i,(lo,hi,_,__) in enumerate(_FG_ZONES):
            if s < hi: return i
        return 4

    active = active_zone(score if score is not None else 50)

    # Build pie slices — top semicircle only (bottom half is invisible)
    # Each zone is proportional to its width out of 100
    sizes  = [hi-lo for lo,hi,_,__ in _FG_ZONES]   # [25,20,10,20,25]
    bright = [c  for _,__,c,___ in _FG_ZONES]
    dim    = ["rgba(239,68,68,0.18)","rgba(249,115,22,0.18)",
              "rgba(234,179,8,0.18)", "rgba(34,197,94,0.18)","rgba(21,128,61,0.18)"]
    z_colors = [bright[i] if i==active else dim[i] for i in range(5)]

    fig = go.Figure()

    # Semicircle via Pie: top half = gauge, bottom half = transparent spacer
    fig.add_trace(go.Pie(
        values=sizes + [sum(sizes)],          # spacer = 100
        labels=[z[3] for z in _FG_ZONES] + [""],
        marker=dict(
            colors=z_colors + ["rgba(0,0,0,0)"],
            line=dict(color="#0f172a", width=3),
        ),
        hole=0.44,
        rotation=90,      # 0 → left, 100 → right
        sort=False,
        direction="clockwise",
        textinfo="none",
        hoverinfo="skip",
        showlegend=False,
    ))

    # Zone labels inside each slice
    label_positions = [
        (12,  0.62, 0.82),   # Extreme Fear
        (35,  0.30, 0.82),   # Fear
        (50,  0.50, 0.95),   # Neutral
        (65,  0.70, 0.82),   # Greed
        (88,  0.88, 0.82),   # Extreme Greed
    ]
    zone_names = ["EXTREME\nFEAR", "FEAR", "NEUTRAL", "GREED", "EXTREME\nGREED"]
    for i, (mid_score, px, py) in enumerate(label_positions):
        angle = math.pi * (1 - mid_score/100)
        r_lbl = 0.36
        lx = 0.5 + r_lbl * math.cos(angle)
        ly = 0.47 + r_lbl * math.sin(angle)
        fig.add_annotation(
            x=lx, y=ly, xref="paper", yref="paper",
            text=zone_names[i].replace("\n","<br>"),
            showarrow=False,
            font=dict(size=7.5, color="white" if i==active else "rgba(255,255,255,0.45)"),
            align="center", xanchor="center", yanchor="middle",
        )

    # Needle
    if score is not None:
        angle = math.pi * (1 - score/100)
        needle_r = 0.35
        cx, cy = 0.5, 0.47
        nx = cx + needle_r * math.cos(angle)
        ny = cy + needle_r * math.sin(angle)

        # Needle line
        fig.add_shape(type="line",
            x0=cx, y0=cy, x1=nx, y1=ny,
            xref="paper", yref="paper",
            line=dict(color="white", width=3))

        # Hub dot
        fig.add_shape(type="circle",
            x0=cx-0.022, y0=cy-0.028, x1=cx+0.022, y1=cy+0.028,
            xref="paper", yref="paper",
            fillcolor="white", line_color="white")

        # Scale ticks: 0, 25, 50, 75, 100
        for tick_val in [0, 25, 50, 75, 100]:
            ta = math.pi * (1 - tick_val/100)
            r_in, r_out = 0.41, 0.44
            fig.add_shape(type="line",
                x0=cx + r_in * math.cos(ta), y0=cy + r_in * math.sin(ta),
                x1=cx + r_out* math.cos(ta), y1=cy + r_out* math.sin(ta),
                xref="paper", yref="paper",
                line=dict(color="rgba(255,255,255,0.5)", width=1.5))
            fig.add_annotation(
                x=cx + 0.50*math.cos(ta), y=cy + 0.50*math.sin(ta),
                xref="paper", yref="paper",
                text=str(tick_val),
                showarrow=False,
                font=dict(size=8, color="rgba(255,255,255,0.5)"),
                xanchor="center", yanchor="middle")

    # Score + rating text
    score_txt  = f"{score:.0f}" if score is not None else "—"
    rating_color = bright[active] if score is not None else "#6b7280"
    fig.add_annotation(x=0.5, y=0.18, xref="paper", yref="paper",
        text=f"<b>{score_txt}</b>",
        font=dict(size=38, color="white"), showarrow=False,
        xanchor="center", yanchor="middle")
    fig.add_annotation(x=0.5, y=0.06, xref="paper", yref="paper",
        text=f"<b>{rating}</b>" if rating else "",
        font=dict(size=13, color=rating_color), showarrow=False,
        xanchor="center", yanchor="middle")

    # Title + source
    title_full = f"<b>{title}</b>"
    if source_label:
        title_full += f"<br><span style='font-size:10px;color:rgba(255,255,255,0.45)'>{source_label}</span>"

    fig.update_layout(
        title=dict(text=title_full, font=dict(size=14, color="white"), x=0.5, xanchor="center", y=0.98),
        height=300,
        margin=dict(l=10, r=10, t=40, b=0),
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
        st.cache_data.clear(); st.success("Cache cleared."); st.rerun()

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
    ["Overview","Deep Dive","Options Chain","VIX","⚡ Screener"])

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
                     "50MA":"✅" if r["ab50"] else "❌","200MA":"✅" if r["ab200"] else "❌",
                     "PCR":fmt(r["pcr"],".2f"),"ATR":fmt(r["atr"],".2f"),
                     "BB Width":fmt(r["bbw"],".1f","%"),
                     "LEAP":r["leap"][0],"CC":r["cc"][0],"CSP":r["csp"][0]})
    if rows:
        tbl_height=38+len(rows)*35+4      # fits all rows exactly — no scrollbar
        st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True,height=tbl_height)

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
        fig=make_subplots(rows=4,cols=1,shared_xaxes=True,row_heights=[0.48,0.18,0.17,0.17],
            subplot_titles=[f"{sel} — Price & MAs","HV20","RSI (14)","BB Width"],vertical_spacing=0.04)
        fig.add_trace(go.Candlestick(x=df.index,open=df["Open"].squeeze(),high=df["High"].squeeze(),
            low=df["Low"].squeeze(),close=cl,name="Price",
            increasing_line_color="#26a69a",decreasing_line_color="#ef5350",
            increasing_fillcolor="#26a69a",decreasing_fillcolor="#ef5350"),row=1,col=1)
        fig.add_trace(go.Scatter(x=df.index,y=cl.rolling(50).mean(),name="50MA",
            line=dict(color="#f97316",width=1.4)),row=1,col=1)
        fig.add_trace(go.Scatter(x=df.index,y=cl.rolling(200).mean(),name="200MA",
            line=dict(color="#60a5fa",width=1.6)),row=1,col=1)
        fig.add_trace(go.Scatter(x=df.index,y=r["hv20_s"],name="HV20",fill="tozeroy",
            line=dict(color="#a78bfa",width=1.5),fillcolor="rgba(167,139,250,0.12)"),row=2,col=1)
        fig.add_trace(go.Scatter(x=df.index,y=r["hv60_s"],name="HV60",
            line=dict(color="#7c3aed",width=1,dash="dot")),row=2,col=1)
        fig.add_trace(go.Scatter(x=df.index,y=r["rsi_s"],name="RSI",
            line=dict(color="#fbbf24",width=1.5)),row=3,col=1)
        for lvl,col in [(70,"#ef4444"),(50,"#94a3b8"),(30,"#22c55e")]:
            fig.add_hline(y=lvl,line_dash="dash",line_color=col,row=3,col=1)
        fig.add_trace(go.Scatter(x=df.index,y=r["bbw_s"],name="BB Width",fill="tozeroy",
            line=dict(color="#34d399",width=1.4),fillcolor="rgba(52,211,153,0.10)"),row=4,col=1)
        fig.update_layout(height=820,template="plotly_dark",xaxis_rangeslider_visible=False,
                          legend=dict(orientation="h",y=1.01,x=0),margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(fig,use_container_width=True)
        if r["atr"] and r["price"]:
            atr_pct=r["atr"]/r["price"]*100
            st.subheader("Position Sizing Guide")
            st.markdown(f"""
| Metric | Value |
|---|---|
| ATR 14-day | ${r['atr']:.2f} ({atr_pct:.1f}% of price) |
| Suggested CC strike (1.5× ATR above) | ~${r['price']+r['atr']*1.5:.2f} |
| Suggested CSP strike (1.5× ATR below) | ~${r['price']-r['atr']*1.5:.2f} |
            """)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — OPTIONS CHAIN
# ══════════════════════════════════════════════════════════════════════════════
with tab_chain:
    sel_c=st.selectbox("Select Ticker",list(results.keys()),key="chain_sel")
    if sel_c and sel_c in results:
        r=results[sel_c]; price=r["price"]; all_exps=r.get("all_exps",[])
        if all_exps:
            selected_exp=st.selectbox("Select Expiry",all_exps,key=f"exp_sel_{sel_c}")
            calls_df,puts_df,dte=fetch_chain_cached(sel_c,selected_exp)
            if calls_df is not None:
                chain=type("_C",(),{"calls":calls_df,"puts":puts_df})()
                st.caption(f"Expiry: {selected_exp} ({dte} DTE) | Price: ${price:.2f}")
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
            else:
                st.warning("Could not load chain for this expiry.")
        else:
            st.warning(f"No options data for {sel_c}.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — VIX
# ══════════════════════════════════════════════════════════════════════════════
with tab_vix:
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

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — SCREENER
# ══════════════════════════════════════════════════════════════════════════════
with tab_screener:
    st.subheader("⚡ Options Suitability Screener")
    st.caption("CSP target: **Δ30 · 30 DTE** · All greeks via Black-Scholes (strike IV → chain median → HV20 → 30% default)")

    with st.expander("How the score is calculated", expanded=False):
        st.markdown("""
**Screener targets**

| Param | CSP | CC | LEAP |
|---|---|---|---|
| Delta target | **Δ30** | Δ35 | Δ80 |
| DTE target | **30d** | 30d | 547d |
| DTE window | 21–45d | 21–45d | 180–900d |

---

**What is NIS? (Normalised Income Score)**

NIS measures how much income a strike generates relative to the stock price and time remaining,
so you can compare premium quality across tickers of wildly different prices.

> **Raw = θ × √DTE ÷ Price**
>
> - **θ** = theta (dollars of decay per day per share)
> - **√DTE** = square root of days to expiry — accounts for the fact that theta accelerates as expiry approaches; √DTE normalises it to a time-consistent basis
> - **÷ Price** = scales by stock price so a $10 theta on a $1,000 stock is not the same as on a $100 stock

The raw number is then normalised to a 0–100 scale:
- **0** ≈ IV around 15% (very cheap premium — thin income)
- **100** ≈ IV around 120% (very expensive premium — rich income)

For **CSP and CC**: higher NIS is better — more premium per dollar of risk.
For **LEAP buying**: NIS is *inverted* — lower NIS means cheaper options to buy.

---

**Composite score weights**

| Component | CSP | CC | LEAP |
|---|---|---|---|
| A — NIS (inverted for LEAP) | 50% | 50% | 30% |
| B — DTE fit (30d / 30d / 547d optimal) | 30% | 30% | 40% |
| C — Delta fit (Δ30 / Δ35 / Δ80 optimal) | 20% | 20% | 30% |

Scores ≥80 = Optimal · ≥60 = Acceptable · ≥40 = Marginal · <40 = Unsuitable

---

**Four-gate filter** — all four must pass before trading:
- **G1 Trend** — 20MA > 50MA, or price > 20MA
- **G2 Stochastics** — %K touched sub-20 in last 5 sessions + %K crossed above %D
- **G3 Session** — not down more than 2.5% today
- **G4 BB Veto** — price not walking the lower Bollinger Band 2+ sessions in a row

**Greek sourcing (priority order):**
Strike's own IV → Chain-wide median IV → HV20 of underlying → 30% default
        """)

    with st.expander("🔬 Chain Inspector",expanded=False):
        insp_ticker=st.selectbox("Ticker",list(results.keys()),key="insp_ticker")
        if insp_ticker and insp_ticker in results:
            insp_exps=results[insp_ticker].get("all_exps",[])
            today_i=datetime.utcnow()
            valid_i=[e for e in insp_exps if 21<=(datetime.strptime(e,"%Y-%m-%d")-today_i).days<=45]
            if valid_i:
                calls_i,puts_i,dte_i=fetch_chain_cached(insp_ticker,valid_i[0])
                if puts_i is not None and not puts_i.empty:
                    st.write(f"**Columns:** `{list(puts_i.columns)}`  |  **Rows:** {len(puts_i)}  |  **DTE:** {dte_i}")
                    price_i=results[insp_ticker]["price"]
                    disp_cols=[c for c in ["strike","bid","ask","impliedVolatility","volume","openInterest"] if c in puts_i.columns]
                    st.dataframe(puts_i.iloc[(puts_i["strike"]-price_i).abs().argsort()].head(12)[disp_cols].reset_index(drop=True),use_container_width=True)
                    oi_c=pd.to_numeric(puts_i.get("openInterest",pd.Series(dtype=float)),errors="coerce").fillna(0)
                    iv_c=pd.to_numeric(puts_i.get("impliedVolatility",pd.Series(dtype=float)),errors="coerce").fillna(0)
                    st.write(f"OI≥1: **{int((oi_c>=1).sum())}** / {len(puts_i)}  ·  IV>0: **{int((iv_c>0).sum())}** / {len(puts_i)}")
                else:
                    st.warning("Chain returned empty.")
            else:
                st.warning(f"No 21–45 DTE expiry for {insp_ticker}.")

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
                row=get_screener_row(ticker,result)
                if row:
                    screener_rows.append(row)
                    debug_log.append(f"✅ {ticker} — CSP {row['csp_score']} ({row['greek_source']}, "
                                     f"inspected {row.get('_inspected')}, scored {row.get('_scored')})")
                else:
                    price=result.get("price"); all_exps=result.get("all_exps",[])
                    if not price or price<=0: reason="no price"
                    elif not all_exps:        reason="no expiries"
                    else:
                        today=datetime.utcnow()
                        ve=next((e for e in all_exps if 21<=(datetime.strptime(e,"%Y-%m-%d")-today).days<=45),None)
                        if not ve: reason=f"no 21–45 DTE expiry (have:{all_exps[:3]})"
                        else:
                            _,pdf,_=fetch_chain_cached(ticker,ve)
                            if pdf is None or pdf.empty: reason=f"chain failed for {ve}"
                            else:
                                oi_c=pd.to_numeric(pdf.get("openInterest",pd.Series(dtype=float)),errors="coerce").fillna(0)
                                iv_c=pd.to_numeric(pdf.get("impliedVolatility",pd.Series(dtype=float)),errors="coerce").fillna(0)
                                reason=(f"chain OK ({len(pdf)} rows, OI≥1:{int((oi_c>=1).sum())}, "
                                        f"IV>0:{int((iv_c>0).sum())}) — find_target_strike returned None")
                    debug_log.append(f"❌ {ticker} — skipped: {reason}")
            prog.empty()
            st.session_state["screener_results"]=screener_rows
            st.session_state["screener_debug"]=debug_log

    if show_debug and st.session_state.get("screener_debug"):
        with st.expander("🔧 Log",expanded=True):
            for line in st.session_state["screener_debug"]: st.text(line)

    screener_rows=st.session_state.get("screener_results",[])

    if screener_rows:
        screener_rows_sorted=sorted(screener_rows,key=lambda x:x["csp_score"],reverse=True)
        src_counts={}
        for r in screener_rows_sorted:
            k=r.get("greek_source","?"); src_counts[k]=src_counts.get(k,0)+1
        st.info("**Greeks:** "+"  ·  ".join(f"{greek_source_label(k)}: {v}" for k,v in src_counts.items()))

        low_prec=[r["ticker"] for r in screener_rows_sorted if r.get("greek_source") in ("bs_hv20","bs_default")]
        if low_prec:
            st.warning(f"⚠️ Low-precision greeks for: **{', '.join(low_prec)}**")

        table_rows=[]
        for r in screener_rows_sorted:
            gr=r["gate_result"]; gates=gr["gates"]
            gate_icons="".join(["✅" if gates[f"G{i}"]["pass"] else "❌" for i in range(1,5)])
            table_rows.append({"Ticker":r["ticker"],"Price":f"${r['price']:.2f}",
                "Expiry":r["expiry"],"DTE":r["dte"],
                "Greeks":greek_source_label(r.get("greek_source")),
                "CSP Strike":f"${r['csp_strike']:.1f}","CSP Δ":r["csp_delta"],
                "θ/day":f"${r['csp_theta']:.3f}",
                "Put IV %":r["csp_iv"] if r["csp_iv"] else "—",
                "OI":r["csp_oi"],"Spread %":r["csp_spread"] if r["csp_spread"] else "—",
                "NIS":r["nis"],"CSP Score":r["csp_score"],"CC Score":r["cc_score"],
                "LEAP Score*":r["leap_score"],"Gates":gate_icons,
                "Status":"🟢 TRADE" if gr["all_pass"] else "🔴 WAIT"})

        scr_height=min(38+len(table_rows)*35+4, 600)
        st.dataframe(pd.DataFrame(table_rows),use_container_width=True,hide_index=True,height=scr_height)
        st.caption("\\* LEAP Score at short DTE — relative IV indicator only.")

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

        st.subheader("Four-Gate Filter Detail")
        st.caption("G1=Trend  G2=Stochastics  G3=Session  G4=BB Veto")
        for r in screener_rows_sorted:
            gr=r["gate_result"]; gates=gr["gates"]; icon="🟢" if gr["all_pass"] else "🔴"
            with st.expander(f"{icon} {r['ticker']}  CSP:{r['csp_score']}  Strike${r['csp_strike']:.1f}  Δ{r['csp_delta']}  θ${r['csp_theta']:.3f}/d"):
                gcols=st.columns(4)
                for idx,(gk,gv) in enumerate(gates.items()):
                    gcols[idx].markdown(f"**{gv['label']}** {'✅' if gv['pass'] else '❌'}")
                    gcols[idx].caption(gv["reason"])
                st.divider()
                rc1,rc2=st.columns(2)
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
        st.markdown("""<div style='text-align:center;padding:60px;color:#94a3b8;'>
        <h3>Click <strong>Run Screener</strong> to analyse your watchlist</h3>
        <p>Δ30 · 30 DTE · Black-Scholes greeks · Works any time of day</p>
        </div>""",unsafe_allow_html=True)
    else:
        st.warning("No screener data. Check Chain Inspector and enable diagnostic output.")
