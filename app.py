import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
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

# ── Session state — persists across reruns ─────────────────────────────────────
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

def calc_atm_iv(chain, price):
    try:
        calls = chain.calls.copy()
        puts = chain.puts.copy()
        c_atm = calls.iloc[(calls["strike"] - price).abs().argsort()[:1]]
        p_atm = puts.iloc[(puts["strike"] - price).abs().argsort()[:1]]
        return round(c_atm["impliedVolatility"].values[0] * 100, 1), round(p_atm["impliedVolatility"].values[0] * 100, 1)
    except Exception:
        return None, None

def calc_pcr(chain):
    try:
        pv = chain.puts["volume"].sum()
        cv = chain.calls["volume"].sum()
        return round(pv / cv, 2) if cv > 0 else None
    except Exception:
        return None

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
    label = ("🟢 STRONG ENTRY" if score >= 7 else "🟡 DECENT ENTRY" if score >= 4 else "🟠 MARGINAL" if score >= 2 else "🔴 AVOID")
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
    label = ("🟢 WRITE NOW" if score >= 5 else "🟡 DECENT" if score >= 3 else "🟠 MARGINAL" if score >= 1 else "🔴 WAIT")
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
    label = ("🟢 SELL PUT" if score >= 4 else "🟡 DECENT" if score >= 2 else "🟠 MARGINAL" if score >= 1 else "🔴 AVOID")
    return label, score, reasons

def analyse(ticker, period, vix_current):
    df = fetch_prices(ticker, period)
    if df is None:
        return None
    cl = df["Close"].squeeze()
    curr = float(cl.iloc[-1])
    prev = float(cl.iloc[-2])
    pct_chg = (curr / prev - 1) * 100
    hv20_s = calc_hv(cl, 20)
    hv60_s = calc_hv(cl, 60)
    hv_cur = float(hv20_s.dropna().iloc[-1]) if not hv20_s.dropna().empty else None
    hvr = calc_iv_rank(hv20_s)
    hvpct = calc_iv_percentile(hv20_s)
    rsi_s = calc_rsi(cl)
    rsi_cur = float(rsi_s.dropna().iloc[-1]) if not rsi_s.dropna().empty else None
    atr_s = calc_atr(df)
    atr_cur = float(atr_s.dropna().iloc[-1]) if not atr_s.dropna().empty else None
    bbw_s = calc_bb_width(cl)
    bbw_cur = float(bbw_s.dropna().iloc[-1]) if not bbw_s.dropna().empty else None
    ma50 = float(cl.rolling(50).mean().iloc[-1])
    ma200 = float(cl.rolling(200).mean().iloc[-1])
    ab50 = curr > ma50
    ab200 = curr > ma200
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
                pcr_val = calc_pcr(chain)
    leap_lbl, leap_sc, leap_r = leap_signal(hvr, rsi_cur, ab50, ab200, vix_current)
    cc_lbl, cc_sc, cc_r = cc_signal(hvr, rsi_cur, ab50, pcr_val)
    csp_lbl, csp_sc, csp_r = csp_signal(hvr, rsi_cur, ab200)
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
        "cc": (cc_lbl, cc_sc, cc_r),
        "csp": (csp_lbl, csp_sc, csp_r),
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

    # Add ticker
    new_ticker = st.text_input("Add a ticker (press Enter)", placeholder="e.g. AMZN", key="new_ticker_input")
    if new_ticker:
        t = new_ticker.upper().strip()
        if t and t not in st.session_state.watchlist:
            st.session_state.watchlist.append(t)
            st.rerun()

    # Remove ticker
    if st.session_state.watchlist:
        remove = st.selectbox("Remove a ticker", ["— select to remove —"] + st.session_state.watchlist, key="remove_ticker")
        if remove != "— select to remove —":
            st.session_state.watchlist.remove(remove)
            st.rerun()

    st.markdown(f"**Active watchlist:** {', '.join(st.session_state.watchlist)}")

    period = st.selectbox("Price History", ["6mo", "1y", "2y"], index=1)
    st.divider()

    vix_df = fetch_vix("1y")
    vix_now = None
    vix_chg = 0
    if vix_df is not None and not vix_df.empty:
        vix_cl_s = vix_df["Close"].squeeze()
        vix_now = float(vix_cl_s.iloc[-1])
        vix_prev = float(vix_cl_s.iloc[-2])
        vix_chg = vix_now - vix_prev
        vc, vl = vix_zone(vix_now)
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
tab_dash, tab_dive, tab_chain, tab_vix = st.tabs(["Overview", "Deep Dive", "Options Chain", "VIX"])

results = {}
with st.spinner("Loading market data..."):
    for t in watchlist:
        r = analyse(t, period, vix_now)
        if r:
            results[t] = r

# TAB 1 — OVERVIEW
with tab_dash:
    rows = []
    for t, r in results.items():
        rows.append({
            "Ticker": t, "Price": f"${r['price']:.2f}", "Chg %": f"{r['pct']:+.1f}%",
            "HV Rank": fmt(r["hvr"], ".0f"), "HV%ile": fmt(r["hvpct"], ".0f"),
            "HV20": fmt(r["hv20"], ".1f", "%"),
            "ATM IV C/P": f"{r['c_iv']:.0f}/{r['p_iv']:.0f}%" if r["c_iv"] else "—",
            "RSI": fmt(r["rsi"], ".0f"),
            "50MA": "YES" if r["ab50"] else "NO",
            "200MA": "YES" if r["ab200"] else "NO",
            "PCR": fmt(r["pcr"], ".2f"), "ATR": fmt(r["atr"], ".2f"),
            "BB Width": fmt(r["bbw"], ".1f", "%"),
            "LEAP": r["leap"][0], "CC": r["cc"][0], "CSP": r["csp"][0],
        })
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True, height=420)

    hvr_data = {t: r["hvr"] for t, r in results.items() if r["hvr"] is not None}
    if hvr_data:
        st.subheader("HV Rank — Entry Zones")
        colors = ["#22c55e" if v < 25 else "#eab308" if v < 45 else "#f97316" if v < 65 else "#ef4444" for v in hvr_data.values()]
        fig_hvr = go.Figure(go.Bar(x=list(hvr_data.keys()), y=list(hvr_data.values()), marker_color=colors, text=[f"{v:.0f}" for v in hvr_data.values()], textposition="outside"))
        fig_hvr.add_hline(y=25, line_dash="dash", line_color="#22c55e", annotation_text="25 = LEAP Buy Zone")
        fig_hvr.add_hline(y=65, line_dash="dash", line_color="#ef4444", annotation_text="65 = CC Writing Zone")
        fig_hvr.update_layout(height=320, template="plotly_dark", yaxis_title="HV Rank", yaxis_range=[0, 115], margin=dict(l=0, r=0, t=20, b=0))
        st.plotly_chart(fig_hvr, use_container_width=True)

    rsi_data = {t: r["rsi"] for t, r in results.items() if r["rsi"] is not None}
    if rsi_data:
        st.subheader("RSI Snapshot")
        rsi_colors = ["#22c55e" if 33 <= v <= 52 else "#eab308" if 52 < v <= 65 else "#f97316" if v > 65 else "#94a3b8" for v in rsi_data.values()]
        fig_rsi = go.Figure(go.Bar(x=list(rsi_data.keys()), y=list(rsi_data.values()), marker_color=rsi_colors, text=[f"{v:.0f}" for v in rsi_data.values()], textposition="outside"))
        for level, color, label in [(30, "#22c55e", "30"), (50, "#94a3b8", "50"), (70, "#ef4444", "70")]:
            fig_rsi.add_hline(y=level, line_dash="dash", line_color=color, annotation_text=label)
        fig_rsi.update_layout(height=280, template="plotly_dark", yaxis_title="RSI (14)", yaxis_range=[0, 115], margin=dict(l=0, r=0, t=20, b=0))
        st.plotly_chart(fig_rsi, use_container_width=True)

# TAB 2 — DEEP DIVE
with tab_dive:
    sel = st.selectbox("Select Ticker", list(results.keys()), key="dd_sel")
    if sel and sel in results:
        r = results[sel]
        df = r["df"]
        cl = r["cl"]
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Price", f"${r['price']:.2f}", f"{r['pct']:+.1f}%")
        c2.metric("HV Rank", fmt(r["hvr"], ".0f"))
        c3.metric("HV Pctile", fmt(r["hvpct"], ".0f", "%"))
        c4.metric("RSI (14)", fmt(r["rsi"], ".1f"))
        c5.metric("ATM Call IV", fmt(r["c_iv"], ".1f", "%"))
        c6.metric("PCR", fmt(r["pcr"], ".2f"))
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            lbl, sc, reasons = r["leap"]
            st.markdown(f"#### LEAP: {lbl}")
            with st.expander("Breakdown"):
                for reason in reasons:
                    st.write(reason)
        with sc2:
            lbl, sc, reasons = r["cc"]
            st.markdown(f"#### CC: {lbl}")
            with st.expander("Breakdown"):
                for reason in reasons:
                    st.write(reason)
        with sc3:
            lbl, sc, reasons = r["csp"]
            st.markdown(f"#### CSP: {lbl}")
            with st.expander("Breakdown"):
                for reason in reasons:
                    st.write(reason)
        st.divider()
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True, row_heights=[0.48, 0.18, 0.17, 0.17],
            subplot_titles=[f"{sel} — Price & Moving Averages", "HV20 (Volatility Proxy)", "RSI (14)", "Bollinger Band Width"],
            vertical_spacing=0.04)
        fig.add_trace(go.Candlestick(x=df.index, open=df["Open"].squeeze(), high=df["High"].squeeze(), low=df["Low"].squeeze(), close=cl, name="Price", increasing_line_color="#26a69a", decreasing_line_color="#ef5350", increasing_fillcolor="#26a69a", decreasing_fillcolor="#ef5350"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=cl.rolling(50).mean(), name="50MA", line=dict(color="#f97316", width=1.4)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=cl.rolling(200).mean(), name="200MA", line=dict(color="#60a5fa", width=1.6)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=r["hv20_s"], name="HV20", fill="tozeroy", line=dict(color="#a78bfa", width=1.5), fillcolor="rgba(167,139,250,0.12)"), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=r["hv60_s"], name="HV60", line=dict(color="#7c3aed", width=1, dash="dot")), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=r["rsi_s"], name="RSI", line=dict(color="#fbbf24", width=1.5)), row=3, col=1)
        for lvl, col in [(70, "#ef4444"), (50, "#94a3b8"), (30, "#22c55e")]:
            fig.add_hline(y=lvl, line_dash="dash", line_color=col, row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=r["bbw_s"], name="BB Width", fill="tozeroy", line=dict(color="#34d399", width=1.4), fillcolor="rgba(52,211,153,0.10)"), row=4, col=1)
        fig.update_layout(height=820, template="plotly_dark", xaxis_rangeslider_visible=False, legend=dict(orientation="h", y=1.01, x=0), margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)
        if r["atr"] and r["price"]:
            atr_pct = r["atr"] / r["price"] * 100
            st.subheader("Position Sizing Guide")
            st.markdown(f"""
| Metric | Value |
|---|---|
| ATR 14-day | ${r['atr']:.2f} ({atr_pct:.1f}% of price) |
| Suggested CC strike (1.5x ATR above price) | ~${r['price'] + r['atr']*1.5:.2f} |
| Suggested CSP strike (1.5x ATR below price) | ~${r['price'] - r['atr']*1.5:.2f} |
            """)

# TAB 3 — OPTIONS CHAIN
with tab_chain:
    sel_c = st.selectbox("Select Ticker", list(results.keys()), key="chain_sel")
    if sel_c and sel_c in results:
        r = results[sel_c]
        price = r["price"]
        all_exps = r.get("all_exps", [])
        if all_exps:
            selected_exp = st.selectbox("Select Expiry", all_exps, index=0, key=f"exp_sel_{sel_c}")
            chain, dte = fetch_chain(sel_c, selected_exp)
            if chain:
                st.caption(f"Expiry: {selected_exp} ({dte} DTE) | Current Price: ${price:.2f}")
                def fmt_chain(df_raw, side):
                    df_raw = df_raw.copy()
                    df_raw["IV %"] = (df_raw["impliedVolatility"] * 100).round(1)
                    df_raw["Moneyness"] = df_raw["strike"].apply(lambda s: "ATM" if abs(s - price) / price < 0.02 else ("ITM" if ((s < price and side == "call") or (s > price and side == "put")) else "OTM"))
                    cols = ["strike", "Moneyness", "lastPrice", "bid", "ask", "volume", "openInterest", "IV %", "delta"]
                    available = [c for c in cols if c in df_raw.columns]
                    return df_raw[available].rename(columns={"lastPrice": "Last", "openInterest": "OI", "strike": "Strike", "volume": "Volume"}).sort_values("Strike").reset_index(drop=True)
                col_c, col_p = st.columns(2)
                with col_c:
                    st.subheader("Calls")
                    st.dataframe(fmt_chain(chain.calls, "call"), use_container_width=True, hide_index=True)
                with col_p:
                    st.subheader("Puts")
                    st.dataframe(fmt_chain(chain.puts, "put"), use_container_width=True, hide_index=True)
                st.subheader("IV Smile")
                fig_smile = go.Figure()
                fig_smile.add_trace(go.Scatter(x=chain.calls["strike"], y=chain.calls["impliedVolatility"]*100, name="Calls IV", mode="lines+markers", line=dict(color="#26a69a", width=2), marker=dict(size=5)))
                fig_smile.add_trace(go.Scatter(x=chain.puts["strike"], y=chain.puts["impliedVolatility"]*100, name="Puts IV", mode="lines+markers", line=dict(color="#ef5350", width=2), marker=dict(size=5)))
                fig_smile.add_vline(x=price, line_dash="dash", line_color="white", annotation_text=f"${price:.2f}")
                fig_smile.update_layout(height=350, template="plotly_dark", xaxis_title="Strike", yaxis_title="IV (%)", margin=dict(l=0, r=0, t=20, b=0))
                st.plotly_chart(fig_smile, use_container_width=True)
                st.subheader("Open Interest by Strike")
                fig_oi = go.Figure()
                fig_oi.add_trace(go.Bar(x=chain.calls["strike"], y=chain.calls["openInterest"], name="Call OI", marker_color="#26a69a", opacity=0.75))
                fig_oi.add_trace(go.Bar(x=chain.puts["strike"], y=chain.puts["openInterest"], name="Put OI", marker_color="#ef5350", opacity=0.75))
                fig_oi.add_vline(x=price, line_dash="dash", line_color="white")
                fig_oi.update_layout(barmode="overlay", height=320, template="plotly_dark", xaxis_title="Strike", yaxis_title="Open Interest", margin=dict(l=0, r=0, t=20, b=0))
                st.plotly_chart(fig_oi, use_container_width=True)
            else:
                st.warning("Could not load chain for this expiry.")
        else:
            st.warning(f"No options data available for {sel_c}.")

# TAB 4 — VIX
with tab_vix:
    if vix_df is not None and not vix_df.empty:
        vix_cl = vix_df["Close"].squeeze()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Current VIX", f"{vix_now:.1f}", f"{vix_chg:+.2f}")
        c2.metric("52wk High", f"{vix_cl.max():.1f}")
        c3.metric("52wk Low", f"{vix_cl.min():.1f}")
        c4.metric("52wk Avg", f"{vix_cl.mean():.1f}")
        fig_vix = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], subplot_titles=["VIX Level", "30-day Rolling Average"])
        fig_vix.add_trace(go.Scatter(x=vix_df.index, y=vix_cl, name="VIX", fill="tozeroy", line=dict(color="#f87171", width=1.5), fillcolor="rgba(248,113,113,0.12)"), row=1, col=1)
        for lo, hi, color, label in VIX_ZONES:
            fig_vix.add_hrect(y0=lo, y1=min(hi, 50), fillcolor=color, opacity=0.05, row=1, col=1)
            fig_vix.add_hline(y=lo, line_dash="dot", line_color=color, opacity=0.4, row=1, col=1)
        fig_vix.add_trace(go.Scatter(x=vix_df.index, y=vix_cl.rolling(30).mean(), name="30d MA", line=dict(color="#fbbf24", width=1.5)), row=2, col=1)
        fig_vix.update_layout(height=520, template="plotly_dark", margin=dict(l=0, r=0, t=40, b=0))
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