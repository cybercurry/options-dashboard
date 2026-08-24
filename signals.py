"""
Signals engine — a headless, Streamlit-free scan that finds wheel-strategy premium-selling
opportunities across the watchlist and ranks them. Used by BOTH:
  • scripts/scan_signals.py  (the GitHub Actions cron → writes data/signals.json), and
  • the Signals tab in app.py (a live "Scan now" button).

Design (per Jay, watchlist-only mode — no IBKR portfolio):
  • Universe  = watchlist.json (the established names of interest).
  • Core      = cash-secured PUTs (CSP) for cashflow; covered CALLs (CC) as an "if you hold it"
                secondary list. LEAP is a long buy, not premium, so it's out of scope here.
  • Real data = Tradier chains with vendor greeks (delta/IV) + Tradier daily history for the
                median gate. yfinance only for sector + earnings date (context, not the trade).
  • Per-trade premium %% = CSP: put-mid / strike ; CC: call-mid / spot. Floor 1.2%, strong ≥1.5%.
  • Gates (the SAME rules as the rest of the app): Δ≈0.30, 21-45 DTE, median rule (CSP below the
    20-day median, CC above), liquidity (OI + tight spread), earnings blackout before expiry.
  • Capital-independent: this module ranks opportunities. Position sizing to real capital happens
    in the UI, so the cron never needs to know account balances.

Everything degrades quietly: one bad ticker never aborts the scan.
"""

import datetime
import math
import statistics

import tradier

# ── tunable parameters (one place) ───────────────────────────────────────────────
P = {
    # CSP / CC (the wheel)
    "dte_lo": 21, "dte_hi": 45, "dte_opt": 30,
    "delta_opt": 0.30, "delta_lo": 0.20, "delta_hi": 0.45,   # matches Screener CSP window
    "min_premium": 0.012,        # 1.2% per-trade floor — premium ÷ SPOT (Jay's definition)
    "strong_premium": 0.015,     # 1.5%+ = flagged strong
    "min_oi": 100,
    "max_spread_pct": 0.15,      # (ask-bid)/mid
    "earnings_blackout": True,   # no new position within `earnings_blackout_days` of earnings
    "earnings_blackout_days": 7, # Jay's rule: skip only in the 7 days BEFORE an earnings call
    "iv_anchor": 0.30, "iv_high": 0.60,   # vol buckets: <30 anchor · 30-60 med · ≥60 high
    "max_per_sector": 2,         # shortlist diversification
    "shortlist_n": 10,
    # mean-reversion timing (criteria doc §10) — lookback windows
    "mr_bb_lookback": 3,         # BB %B peak/trough within the last 3 sessions
    "mr_rsi_lookback": 5,        # RSI >70 / <30 rollover within the last 5 sessions
    "mr_candle_lookback": 3,     # candle-reversal search window (Jay: "3 days trading")
    # LEAP (growth + PMCC basis) — long-dated deep-ITM call, Δ75-80 (criteria doc §2/§10.5)
    "leap_dte_lo": 180, "leap_dte_hi": 900, "leap_dte_opt": 542,
    "leap_delta_opt": 0.78, "leap_delta_lo": 0.70, "leap_delta_hi": 0.85,
    "leap_max_ext_share": 0.40,  # Gate A: extrinsic ≤40% of premium (pay for value, not rented time)
    "leap_max_carry": 0.08,      # Gate C: annualised carry (extrinsic/spot × 365/DTE) ≤ 8%/yr
    "leap_trend_lookback": 5,    # 5-session trend confirmation (Jay: longer-term, clear trend)
}


# ── small helpers ────────────────────────────────────────────────────────────────
def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _mid(bid, ask):
    b, a = _f(bid), _f(ask)
    if b is None or a is None or a <= 0:
        return None
    if b <= 0:
        return a / 2.0        # no bid → half the ask as a conservative mark
    return (a + b) / 2.0


def _spread_pct(bid, ask):
    b, a = _f(bid), _f(ask)
    m = _mid(bid, ask)
    if b is None or a is None or not m:
        return None
    return (a - b) / m if m else None


def pick_expiry(exps, today):
    """Nearest expiry inside the DTE window; else the closest to dte_opt overall."""
    dated = []
    for e in exps:
        try:
            d = (datetime.date.fromisoformat(e) - today).days
        except Exception:
            continue
        dated.append((d, e))
    if not dated:
        return None, None
    inwin = [(d, e) for d, e in dated if P["dte_lo"] <= d <= P["dte_hi"]]
    pool = inwin or [(d, e) for d, e in dated if d > 0]
    if not pool:
        return None, None
    d, e = min(pool, key=lambda t: abs(t[0] - P["dte_opt"]))
    return e, d


def vol_bucket(iv):
    if iv is None:
        return "—"
    if iv < P["iv_anchor"]:
        return "anchor"
    if iv <= P["iv_high"]:
        return "med"
    return "high"


def _sma20_pctb(closes):
    """20-day SMA (the 'median' midline the whole app gates on) and Bollinger %B for context.
    Returns (sma20, pct_b) or (None, None)."""
    c = [x for x in closes if x is not None]
    if len(c) < 20:
        return None, None
    window = c[-20:]
    sma = sum(window) / 20.0
    sd = statistics.pstdev(window)
    if sd == 0:
        return sma, 0.5
    lower = sma - 2 * sd
    pctb = (c[-1] - lower) / (4 * sd)
    return sma, pctb


# ── technicals for the §10 mean-reversion timing signal ──────────────────────────
# Plain-Python ports of app.py's calc_rsi / calc_bb_pctb / _candle_reversal /
# _mean_reversion_score, so the SAME rules drive the Signals tab, Overview, Screener
# AND the static site — one engine, one source of truth (criteria doc §10).
_IND_CACHE = {}   # ticker -> indicator bundle, cleared at the start of every scan()


def _sma(vals, n):
    v = [x for x in vals if x is not None]
    return sum(v[-n:]) / n if len(v) >= n else None


def _rsi_series(closes, period=14):
    """Wilder's RSI, aligned to `closes` (None during warm-up)."""
    n = len(closes)
    out = [None] * n
    if n < period + 1:
        return out
    gains = [max(closes[i] - closes[i - 1], 0.0) for i in range(1, n)]
    losses = [max(closes[i - 1] - closes[i], 0.0) for i in range(1, n)]
    avg_g = sum(gains[:period]) / period
    avg_l = sum(losses[:period]) / period

    def _rsi(g, l):
        if l == 0:
            return 100.0
        rs = g / l
        return 100.0 - 100.0 / (1.0 + rs)

    out[period] = _rsi(avg_g, avg_l)
    for i in range(period + 1, n):
        avg_g = (avg_g * (period - 1) + gains[i - 1]) / period
        avg_l = (avg_l * (period - 1) + losses[i - 1]) / period
        out[i] = _rsi(avg_g, avg_l)
    return out


def _pctb_series(closes, window=20):
    """Bollinger %B per day (None during warm-up). 0.5 = the 20-day midline (the 'median')."""
    n = len(closes)
    out = [None] * n
    for i in range(window - 1, n):
        w = closes[i - window + 1:i + 1]
        sma = sum(w) / window
        sd = statistics.pstdev(w)
        if sd == 0:
            out[i] = 0.5
            continue
        lower = sma - 2 * sd
        out[i] = (closes[i] - lower) / (4 * sd)
    return out


def _hv20(closes):
    """Annualised 20-day realised (historical) volatility, as a decimal."""
    c = [x for x in closes if x]
    if len(c) < 21:
        return None
    rets = [math.log(c[i] / c[i - 1]) for i in range(len(c) - 20, len(c)) if c[i - 1] > 0]
    if len(rets) < 2:
        return None
    return statistics.pstdev(rets) * math.sqrt(252)


def _hv_percentile(closes, window=20, lookback=252):
    """Percentile rank (0-100) of today's 20-day HV within the last ~year of daily 20-day HVs.
    Mirrors app.py's HV%ile column."""
    c = [x for x in closes if x]
    if len(c) < window + 3:
        return None
    hvs = []
    for i in range(window, len(c)):
        seg = c[i - window:i + 1]
        rets = [math.log(seg[j] / seg[j - 1]) for j in range(1, len(seg)) if seg[j - 1] > 0]
        if len(rets) >= 2:
            hvs.append(statistics.pstdev(rets) * math.sqrt(252))
    if len(hvs) < 3:
        return None
    tail = hvs[-lookback:]
    cur = tail[-1]
    return sum(1 for v in tail if v <= cur) / len(tail) * 100.0


# Chain cache — keeps Tradier calls sane when scan_ticker, scan_leap_ticker AND the Overview
# row all want the same expiry's chain. Cleared at the start of every scan().
_CHAIN_CACHE = {}


def _get_chain(ticker, expiry):
    key = (ticker, expiry)
    if key not in _CHAIN_CACHE:
        try:
            _CHAIN_CACHE[key] = tradier.get_option_chain(ticker, expiry, greeks=True) or []
        except Exception:
            _CHAIN_CACHE[key] = []
    return _CHAIN_CACHE[key]


def _candle_reversal(ohlc, direction, lookback=3):
    """Port of app.py _candle_reversal — any one bullish/bearish 2-3 day pattern fires (OR logic)."""
    n = len(ohlc)
    if n < 4:
        return False, None
    for off in range(lookback):
        t = n - 1 - off; t1 = t - 1; t2 = t - 2
        if t1 < 0:
            continue
        o_t, h_t, l_t, c_t = ohlc[t]["o"], ohlc[t]["h"], ohlc[t]["l"], ohlc[t]["c"]
        o1, h1, l1, c1 = ohlc[t1]["o"], ohlc[t1]["h"], ohlc[t1]["l"], ohlc[t1]["c"]
        body_t = abs(c_t - o_t); body1 = abs(c1 - o1); rng1 = max(h1 - l1, 1e-9)
        uw1 = h1 - max(o1, c1); lw1 = min(o1, c1) - l1
        if direction == "bearish":
            if c1 > o1 and c_t < o_t and o_t >= c1 and c_t <= o1 and body_t > body1 * 0.9:
                return True, "Bearish engulfing"
            mid1 = (o1 + c1) / 2
            if c1 > o1 and o_t > c1 and c_t < o_t and o1 < c_t < mid1:
                return True, "Dark cloud cover"
            if abs(h_t - h1) / rng1 < 0.015 and c_t < o_t:
                return True, "Tweezer top"
            if uw1 >= 2 * body1 and lw1 <= body1 * 0.3 and c_t < o_t and c_t < c1:
                return True, "Shooting star + confirmation"
            if t2 >= 0:
                o2, c2 = ohlc[t2]["o"], ohlc[t2]["c"]
                body2 = abs(c2 - o2); rngt = max(h_t - l_t, 1e-9)
                if c2 > o2 and body2 > rngt * 0.5 and body1 < rng1 * 0.3 and c_t < o_t and c_t < (o2 + c2) / 2:
                    return True, "Evening star"
        else:
            if c1 < o1 and c_t > o_t and o_t <= c1 and c_t >= o1 and body_t > body1 * 0.9:
                return True, "Bullish engulfing"
            mid1 = (o1 + c1) / 2
            if c1 < o1 and o_t < c1 and c_t > o_t and mid1 < c_t < o1:
                return True, "Piercing line"
            if abs(l_t - l1) / rng1 < 0.015 and c_t > o_t:
                return True, "Tweezer bottom"
            if lw1 >= 2 * body1 and uw1 <= body1 * 0.3 and c_t > o_t and c_t > c1:
                return True, "Hammer + confirmation"
            if t2 >= 0:
                o2, c2 = ohlc[t2]["o"], ohlc[t2]["c"]
                body2 = abs(c2 - o2); rngt = max(h_t - l_t, 1e-9)
                if c2 < o2 and body2 > rngt * 0.5 and body1 < rng1 * 0.3 and c_t > o_t and c_t > (o2 + c2) / 2:
                    return True, "Morning star"
    return False, None


def _indicators(ticker):
    """One daily-history fetch per ticker → every technical the signal rules need. Cached per
    scan() run so scan_ticker (CSP/CC) and scan_leap_ticker (LEAP) share identical numbers."""
    if ticker in _IND_CACHE:
        return _IND_CACHE[ticker]
    bundle = {"ok": False}
    try:
        today = datetime.date.today()
        start = (today - datetime.timedelta(days=420)).isoformat()   # ~290 trading days → 200-MA
        hist = tradier.get_history(ticker, interval="daily", start=start, end=today.isoformat())
        rows = []
        for d in (hist or []):
            c = _f(d.get("close")); o = _f(d.get("open")); h = _f(d.get("high")); l = _f(d.get("low"))
            if c is None:
                continue
            rows.append({"o": o if o is not None else c, "h": h if h is not None else c,
                         "l": l if l is not None else c, "c": c})
        closes = [r["c"] for r in rows]
        if len(closes) < 20:
            _IND_CACHE[ticker] = bundle
            return bundle
        pctb = _pctb_series(closes, 20)
        rsi = _rsi_series(closes, 14)
        pctb_c = [x for x in pctb if x is not None]
        rsi_c = [x for x in rsi if x is not None]
        spot = closes[-1]
        ma50 = _sma(closes, 50); ma200 = _sma(closes, 200)
        bundle = {
            "ok": True, "ohlc": rows, "closes": closes,
            "pctb": pctb_c[-1] if pctb_c else None,
            "pctb_prev": pctb_c[-2] if len(pctb_c) >= 2 else None,
            "pctb_3": pctb_c[-3:] if len(pctb_c) >= 3 else pctb_c,
            "rsi": rsi_c[-1] if rsi_c else None,
            "rsi_prev": rsi_c[-2] if len(rsi_c) >= 2 else None,
            "rsi_5": rsi_c[-5:] if len(rsi_c) >= 5 else rsi_c,
            "rsi_5ago": rsi_c[-6] if len(rsi_c) >= 6 else (rsi_c[0] if rsi_c else None),
            "sma20": _sma(closes, 20), "ma50": ma50, "ma200": ma200, "hv20": _hv20(closes),
            "hvpct": _hv_percentile(closes),
            "pct_chg": ((closes[-1] / closes[-2]) - 1) * 100 if (len(closes) >= 2 and closes[-2]) else None,
            "walking_lower": bool(len(pctb_c) >= 2 and pctb_c[-1] <= 0.2 and pctb_c[-2] <= 0.2),
            "above_50": bool(ma50 is not None and spot > ma50),
            "above_200": bool(ma200 is not None and spot > ma200),
        }
    except Exception:
        bundle = {"ok": False}
    _IND_CACHE[ticker] = bundle
    return bundle


def _mr_score(ind, direction):
    """§10.1/10.2 mean-reversion score — fade a bottom (csp) or a top (cc). (score, reasons, pattern)."""
    pctb_today = ind.get("pctb"); pctb_prev = ind.get("pctb_prev")
    pctb_3 = ind.get("pctb_3") or []
    rsi_today = ind.get("rsi"); rsi_prev = ind.get("rsi_prev")
    rsi_5 = ind.get("rsi_5") or []
    if (pctb_today is None or pctb_prev is None or len(pctb_3) < 3
            or rsi_today is None or rsi_prev is None or len(rsi_5) < 5):
        return 0, ["Insufficient history for mean-reversion read"], None
    score = 0; reasons = []
    lb = P["mr_candle_lookback"]
    if direction == "cc":
        fired, pattern = _candle_reversal(ind["ohlc"], "bearish", lb)
        if pctb_today >= 0.85: score += 2; reasons.append("Near/touching upper BB (%.2f)" % pctb_today)
        if max(pctb_3) >= 0.95 and pctb_today < pctb_prev: score += 3; reasons.append("Spiked then rolled over")
        if pctb_today > 0.5: score += 1; reasons.append("Above midline")
        if max(rsi_5) > 70 and rsi_today < rsi_prev: score += 3; reasons.append("RSI exceeded 70, turning down (%.0f)" % rsi_today)
        if fired: score += 3; reasons.append(pattern)
    else:
        fired, pattern = _candle_reversal(ind["ohlc"], "bullish", lb)
        if pctb_today <= 0.15: score += 2; reasons.append("Near/touching lower BB (%.2f)" % pctb_today)
        if min(pctb_3) <= 0.05 and pctb_today > pctb_prev: score += 3; reasons.append("Dropped then bounced")
        if pctb_today < 0.5: score += 1; reasons.append("Below midline")
        if min(rsi_5) < 30 and rsi_today > rsi_prev: score += 3; reasons.append("RSI dropped below 30, turning up (%.0f)" % rsi_today)
        if fired: score += 3; reasons.append(pattern)
    return score, reasons, (pattern if fired else None)


def _setup_label(score, kind, blocked=False):
    tier = "full" if score >= 10 else "partial" if score >= 6 else "early" if score >= 3 else "none"
    if blocked and tier in ("full", "partial"):
        return "🟡 Timing ok — wrong side of median"
    verb = "write now" if kind == "cc" else "sell put"
    return {"full": "🟢 FULL SETUP — " + verb, "partial": "🟡 PARTIAL SETUP",
            "early": "🟠 BUILDING UP / WAIT", "none": "🔴 NO SETUP"}[tier]


def _timing(ind, iv_ratio, direction):
    """Wrap _mr_score with the IV-richness add-on + the G4 median block (§10.3/§10.5)."""
    score, reasons, _ = _mr_score(ind, direction)
    if iv_ratio is not None:            # premium seller wants RICH premium (IV > realized)
        if iv_ratio >= 1.25: score += 2; reasons.append("IV rich vs realized — premium fat")
        elif iv_ratio >= 1.0: score += 1; reasons.append("IV fair vs realized")
        else: reasons.append("IV below realized — thin premium")
    pctb = ind.get("pctb")
    if direction == "csp":
        if ind.get("walking_lower"):
            score = max(0, score - 4); reasons.append("Still walking the lower band — breakdown, veto")
        blocked = pctb is not None and pctb > 0.5      # CSP needs BELOW median
        if blocked: reasons.append("⛔ Above median (%%B %.2f) — CSP needs below" % pctb)
        return _setup_label(score, "csp", blocked), score, reasons
    blocked = pctb is not None and pctb < 0.5          # CC needs ABOVE median
    if blocked: reasons.append("⛔ Below median (%%B %.2f) — CC needs above" % pctb)
    return _setup_label(score, "cc", blocked), score, reasons


def _sector(ticker):
    try:
        import yfinance as yf
        return (yf.Ticker(ticker).info or {}).get("sector") or "—"
    except Exception:
        return "—"


def _earnings_date(ticker):
    """Next earnings date (a date) or None. Best-effort via Yahoo's earnings calendar (yfinance);
    None on any failure or when Yahoo has no confirmed date (fail-open — we don't block or warn on
    data we don't have)."""
    try:
        import yfinance as yf
        cal = yf.Ticker(ticker).calendar
        dt = None
        if isinstance(cal, dict):
            ed = cal.get("Earnings Date")
            if isinstance(ed, (list, tuple)) and ed:
                dt = ed[0]
            else:
                dt = ed
        if dt is None:
            return None
        if hasattr(dt, "date"):
            dt = dt.date()
        return dt
    except Exception:
        return None


def _nearest_delta(options, opt_type, target=0.30):
    """Pick the contract whose |delta| is closest to target, within [delta_lo, delta_hi]."""
    best, best_gap = None, 1e9
    for o in options:
        if o.get("option_type") != opt_type:
            continue
        g = (o.get("greeks") or {})
        dl = _f(g.get("delta"))
        if dl is None:
            continue
        ad = abs(dl)
        if not (P["delta_lo"] <= ad <= P["delta_hi"]):
            continue
        gap = abs(ad - target)
        if gap < best_gap:
            best, best_gap = o, gap
    return best


def _build(ticker, strat, o, spot, expiry, dte, sector, iv_atm, earn_soon, earn_window):
    """Assemble one signal dict from a chosen option contract, or None if it fails filters.
    `earn_soon` (earnings ≤7d out) EXCLUDES the trade; `earn_window` (earnings before expiry but
    further out) only sets a warning flag — the trade still shows."""
    strike = _f(o.get("strike"))
    g = o.get("greeks") or {}
    delta = _f(g.get("delta"))
    iv = _f(g.get("mid_iv")) or _f(g.get("smv_vol"))
    mid = _mid(o.get("bid"), o.get("ask"))
    oi = o.get("open_interest") or 0
    spr = _spread_pct(o.get("bid"), o.get("ask"))
    if strike is None or mid is None or not strike or not spot:
        return None
    prem_pct = mid / spot          # Jay's definition: premium ÷ stock price (both CSP and CC)
    pop = (1 - abs(delta)) if delta is not None else None    # ≈ prob of expiring OTM
    # filters
    if prem_pct < P["min_premium"]:
        return None
    if oi < P["min_oi"]:
        return None
    if spr is not None and spr > P["max_spread_pct"]:
        return None
    if P["earnings_blackout"] and earn_soon:
        return None
    return {
        "ticker": ticker, "strategy": strat, "spot": round(spot, 2),
        "expiry": expiry, "dte": dte, "strike": strike,
        "delta": round(delta, 3) if delta is not None else None,
        "mid": round(mid, 2), "premium_pct": round(prem_pct * 100, 2),
        "pop": round(pop * 100, 1) if pop is not None else None,
        "iv": round(iv * 100, 1) if iv is not None else None,
        "iv_atm": round(iv_atm * 100, 1) if iv_atm is not None else None,
        "sector": sector, "vol_bucket": vol_bucket(iv_atm),
        "oi": int(oi), "spread_pct": round(spr * 100, 1) if spr is not None else None,
        "earnings_soon": bool(earn_soon),
        "earnings_in_window": bool(earn_window),
        "strong": prem_pct >= P["strong_premium"],
    }


def _leg_gates(s, iv_ratio):
    """Full pass/fail scorecard for a CSP/CC leg — one node per gate, so the TA-tile web can
    show every parameter (passed or failed), not just the timing reasons. 's': ok/no/warn."""
    d = abs(s.get("delta") or 0)
    dte = s.get("dte") or 0
    prem = s.get("premium_pct")           # already ×100 (percent)
    oi = s.get("oi") or 0
    spr = s.get("spread_pct")             # already ×100 (percent)
    mo = s.get("median_ok")
    ts = s.get("timing_score")
    g = [
        {"l": "Δ %.2f" % d,        "s": "ok" if P["delta_lo"] <= d <= P["delta_hi"] else "no"},
        {"l": "DTE %d" % dte,           "s": "ok" if P["dte_lo"] <= dte <= P["dte_hi"] else "no"},
        {"l": "prem %.1f%%" % (prem or 0), "s": "ok" if (prem or 0) >= P["min_premium"] * 100 else "no"},
        {"l": "OI %d" % oi,             "s": "ok" if oi >= P["min_oi"] else "no"},
        {"l": "spr %.0f%%" % (spr or 0), "s": "ok" if (spr is not None and spr <= P["max_spread_pct"] * 100) else "no"},
        {"l": "earnings",               "s": "warn" if s.get("earnings_in_window") else "ok"},
        {"l": "median",                 "s": "ok" if mo else ("no" if mo is not None else "warn")},
    ]
    if iv_ratio is not None:
        g.append({"l": "IV rich" if iv_ratio >= 1.0 else "IV thin", "s": "ok" if iv_ratio >= 1.0 else "no"})
    else:
        g.append({"l": "IV vs real", "s": "warn"})
    g.append({"l": "timing", "s": "ok" if (ts or 0) > 0 else ("no" if ts is not None else "warn")})
    return g


def scan_ticker(ticker):
    """Best qualifying CSP and CC for one ticker. Returns a list (0-2 signals)."""
    out = []
    try:
        today = datetime.date.today()
        quotes = tradier.get_quotes(ticker)
        spot = _f(quotes[0].get("last")) if quotes else None
        if not spot:
            return out
        exps = tradier.get_expirations(ticker)
        expiry, dte = pick_expiry(exps, today)
        if not expiry:
            return out
        chain = _get_chain(ticker, expiry)
        if not chain:
            return out
        # ATM IV (nearest strike, avg call/put mid_iv) → vol bucket
        atm = min(chain, key=lambda o: abs((_f(o.get("strike")) or 1e9) - spot), default=None)
        atm_strike = _f(atm.get("strike")) if atm else None
        atm_ivs = [_f((o.get("greeks") or {}).get("mid_iv")) for o in chain
                   if _f(o.get("strike")) == atm_strike]
        atm_ivs = [v for v in atm_ivs if v]
        iv_atm = (sum(atm_ivs) / len(atm_ivs)) if atm_ivs else None

        # Technicals: median gate + §10 mean-reversion timing, from one shared bundle so the
        # Signals tab, Overview, Screener and the static site all read identical numbers.
        ind = _indicators(ticker)
        pctb = ind.get("pctb")
        below_median = (pctb < 0.5) if pctb is not None else None   # CSP wants below, CC above
        hv20 = ind.get("hv20")
        iv_ratio = (iv_atm / hv20) if (iv_atm and hv20) else None   # ATM IV vs realized (richness)

        sector = _sector(ticker)
        # Earnings: within 7 days → exclude (entry blackout); before expiry but further out →
        # warn only. One calendar lookup, two flags.
        edate = _earnings_date(ticker)
        exp_date = datetime.date.fromisoformat(expiry)
        earn_soon = bool(edate and today <= edate <= today + datetime.timedelta(days=P["earnings_blackout_days"]))
        earn_window = bool(edate and today <= edate <= exp_date)

        # CSP — needs price BELOW the median to go on the shortlist (else flagged)
        put = _nearest_delta(chain, "put", P["delta_opt"])
        if put:
            s = _build(ticker, "CSP", put, spot, expiry, dte, sector, iv_atm, earn_soon, earn_window)
            if s:
                s["median_ok"] = bool(below_median) if below_median is not None else None
                s["pct_b"] = round(pctb, 2) if pctb is not None else None
                if ind.get("ok"):
                    lbl, sc, rs = _timing(ind, iv_ratio, "csp")
                    s["timing_label"], s["timing_score"], s["timing_reasons"] = lbl, sc, rs
                s["gates"] = _leg_gates(s, iv_ratio)
                out.append(s)
        # CC — needs price ABOVE the median
        call = _nearest_delta(chain, "call", P["delta_opt"])
        if call:
            s = _build(ticker, "CC", call, spot, expiry, dte, sector, iv_atm, earn_soon, earn_window)
            if s:
                above_median = (not below_median) if below_median is not None else None
                s["median_ok"] = bool(above_median) if above_median is not None else None
                s["pct_b"] = round(pctb, 2) if pctb is not None else None
                if ind.get("ok"):
                    lbl, sc, rs = _timing(ind, iv_ratio, "cc")
                    s["timing_label"], s["timing_score"], s["timing_reasons"] = lbl, sc, rs
                s["gates"] = _leg_gates(s, iv_ratio)
                out.append(s)
    except Exception:
        return out
    return out


def _shortlist(signals):
    """Diversified shortlist: qualified (median_ok) CSPs, ranked by premium %, spread across
    sectors (≤ max_per_sector) and one per ticker."""
    cands = [s for s in signals if s["strategy"] == "CSP" and s.get("median_ok")]
    cands.sort(key=lambda s: (s["premium_pct"], s.get("pop") or 0), reverse=True)
    picked, per_sector, seen = [], {}, set()
    for s in cands:
        sec = s.get("sector") or "—"
        if s["ticker"] in seen:
            continue
        if per_sector.get(sec, 0) >= P["max_per_sector"]:
            continue
        picked.append(s); seen.add(s["ticker"]); per_sector[sec] = per_sector.get(sec, 0) + 1
        if len(picked) >= P["shortlist_n"]:
            break
    keys = {(s["ticker"], s["strategy"]) for s in picked}
    for s in signals:
        s["shortlist"] = (s["ticker"], s["strategy"]) in keys
    return picked


def scan_leap_ticker(ticker):
    """Best long-dated deep-ITM call = a LEAP for growth AND a PMCC basis. A BUY, not premium —
    ranked by low extrinsic (time value) so it's an efficient PMCC base. Returns 0-1 dict."""
    try:
        today = datetime.date.today()
        quotes = tradier.get_quotes(ticker)
        spot = _f(quotes[0].get("last")) if quotes else None
        if not spot:
            return None
        exps = tradier.get_expirations(ticker)
        dated = []
        for e in exps:
            try:
                d = (datetime.date.fromisoformat(e) - today).days
            except Exception:
                continue
            if P["leap_dte_lo"] <= d <= P["leap_dte_hi"]:
                dated.append((d, e))
        if not dated:
            return None
        dte, expiry = min(dated, key=lambda t: abs(t[0] - P["leap_dte_opt"]))
        chain = _get_chain(ticker, expiry)
        if not chain:
            return None
        call = None; best = 1e9
        for o in chain:
            if o.get("option_type") != "call":
                continue
            dl = _f((o.get("greeks") or {}).get("delta"))
            if dl is None or not (P["leap_delta_lo"] <= dl <= P["leap_delta_hi"]):
                continue
            gap = abs(dl - P["leap_delta_opt"])
            if gap < best:
                call, best = o, gap
        if not call:
            return None
        strike = _f(call.get("strike"))
        mid = _mid(call.get("bid"), call.get("ask"))
        dl = _f((call.get("greeks") or {}).get("delta"))
        iv = _f((call.get("greeks") or {}).get("mid_iv"))
        if strike is None or mid is None:
            return None
        intrinsic = max(0.0, spot - strike)
        extrinsic = max(0.0, mid - intrinsic)
        ext_share = (extrinsic / mid) if mid else None                          # Gate A basis
        carry = (extrinsic / spot) * (365.0 / dte) if (spot and dte) else None  # Gate C: annualised carry
        gate_a = bool(ext_share is not None and ext_share <= P["leap_max_ext_share"])
        gate_c = bool(carry is not None and carry <= P["leap_max_carry"])
        # CLEAR-TREND confirmation (revised 23 Aug — Jay: "we want a clear trend", after AAPL
        # slipped through a too-weak gate: it was below the 50-MA with a bearish candle, yet the
        # old `above_200 + RSI-rising` gate passed it). A LEAP buy now needs an ESTABLISHED
        # uptrend, not a dip: above BOTH 50 & 200 MA, 50≥200 (aligned), RSI≥50 & rising over
        # 5 sessions, and NO bearish reversal candle in the last 5 sessions.
        ind = _indicators(ticker)
        above_200 = bool(ind.get("above_200")); above_50 = bool(ind.get("above_50"))
        ma50 = ind.get("ma50"); ma200 = ind.get("ma200")
        aligned = bool(ma50 is not None and ma200 is not None and ma50 >= ma200)
        rsi_now = ind.get("rsi"); rsi_5ago = ind.get("rsi_5ago")
        rsi_rising = bool(rsi_now is not None and rsi_5ago is not None and rsi_now > rsi_5ago)
        rsi_bullish = bool(rsi_now is not None and rsi_now >= 50 and rsi_rising)
        bearish_recent = False
        if ind.get("ok"):
            bearish_recent, _bpat = _candle_reversal(ind["ohlc"], "bearish", P["leap_trend_lookback"])
        trend_ok = bool(above_50 and above_200 and aligned and rsi_bullish and not bearish_recent)
        hv20 = ind.get("hv20")
        iv_ratio = (iv / hv20) if (iv and hv20) else None       # IV cheap-vs-realized = timing, NOT cost
        qualifies = bool(gate_a and gate_c and trend_ok)
        # entry-timing score (green dot among qualifiers): cheap IV + healthy-bullish RSI + aligned trend
        tscore = 0
        if iv_ratio is not None:
            tscore += 3 if iv_ratio < 1.0 else 2 if iv_ratio < 1.25 else -1
        if rsi_now is not None:
            tscore += 2 if 50 <= rsi_now <= 65 else 1 if 65 < rsi_now <= 72 else -1 if rsi_now > 72 else 0
        tscore += 2 if above_200 else -1
        if above_50: tscore += 1
        if aligned: tscore += 1
        tlabel = ("🟢 STRONG ENTRY" if tscore >= 7 else "🟡 DECENT ENTRY" if tscore >= 4
                  else "🟠 MARGINAL" if tscore >= 2 else "🔴 AVOID")
        return {
            "ticker": ticker, "strategy": "LEAP", "spot": round(spot, 2),
            "expiry": expiry, "dte": dte, "strike": strike,
            "delta": round(dl, 3) if dl is not None else None,
            "mid": round(mid, 2), "cost": round(mid * 100, 0),
            "intrinsic": round(intrinsic, 2), "extrinsic": round(extrinsic, 2),
            "extrinsic_pct": round(ext_share * 100, 1) if ext_share is not None else None,
            "carry_pct": round(carry * 100, 2) if carry is not None else None,
            "iv": round(iv * 100, 1) if iv is not None else None,
            "iv_ratio": round(iv_ratio, 2) if iv_ratio is not None else None,
            "above_200ma": bool(above_200), "above_50ma": bool(above_50), "ma_aligned": aligned,
            "rsi": round(rsi_now, 1) if rsi_now is not None else None,
            "rsi_rising": rsi_rising,
            "gate_a": gate_a, "gate_c": gate_c, "trend_ok": trend_ok,
            "qualifies": qualifies,
            "timing_label": tlabel, "timing_score": tscore,
            "sector": _sector(ticker),
            "good_pmcc": qualifies,   # back-compat alias for any UI still reading good_pmcc
        }
    except Exception:
        return None


_OVERVIEW = {}   # ticker -> Watchlist-Overview row, rebuilt each scan()


def _iv_richness(c_iv, p_iv, hv):
    """app.py's 'IV vs HV' column: is the premium rich/fair/cheap vs realized vol?"""
    ivs = [v for v in (c_iv, p_iv) if v]
    if not ivs or not hv or hv <= 0:
        return "—"
    ratio = (sum(ivs) / len(ivs)) / hv
    return "rich" if ratio >= 1.25 else "fair" if ratio >= 1.0 else "cheap"


def overview_row(ticker):
    """One Watchlist-Overview row per ticker (the app's tab_dash table), computed headless:
    price/chg, HV%ile, HV20, ATM IV C/P, IV-vs-HV, RSI, 200-MA, PCR, median (%B), CC/CSP timing."""
    try:
        today = datetime.date.today()
        ind = _indicators(ticker)
        quotes = tradier.get_quotes(ticker)
        spot = _f(quotes[0].get("last")) if quotes else (ind.get("closes")[-1] if ind.get("ok") else None)
        c_iv = p_iv = pcr = None
        try:
            expiry, _dte = pick_expiry(tradier.get_expirations(ticker), today)
            chain = _get_chain(ticker, expiry) if expiry else []
            if chain and spot:
                ats = _f(min(chain, key=lambda o: abs((_f(o.get("strike")) or 1e9) - spot)).get("strike"))
                c_iv = next((_f((o.get("greeks") or {}).get("mid_iv")) for o in chain
                             if _f(o.get("strike")) == ats and o.get("option_type") == "call"
                             and _f((o.get("greeks") or {}).get("mid_iv"))), None)
                p_iv = next((_f((o.get("greeks") or {}).get("mid_iv")) for o in chain
                             if _f(o.get("strike")) == ats and o.get("option_type") == "put"
                             and _f((o.get("greeks") or {}).get("mid_iv"))), None)
                cvol = sum((o.get("volume") or 0) for o in chain if o.get("option_type") == "call")
                pvol = sum((o.get("volume") or 0) for o in chain if o.get("option_type") == "put")
                pcr = (pvol / cvol) if cvol else None
        except Exception:
            pass
        hv = ind.get("hv20")
        iv_ratio = (((c_iv + p_iv) / 2) / hv) if (c_iv and p_iv and hv) else None
        csp_tl = cc_tl = None
        if ind.get("ok"):
            csp_tl = _timing(ind, iv_ratio, "csp")[0]
            cc_tl = _timing(ind, iv_ratio, "cc")[0]
        pctb = ind.get("pctb")
        _OVERVIEW[ticker] = {
            "ticker": ticker, "sector": _sector(ticker),
            "price": round(spot, 2) if spot else None,
            "pct": round(ind.get("pct_chg"), 1) if ind.get("pct_chg") is not None else None,
            "hv20": round(hv * 100, 1) if hv else None,
            "hvpct": round(ind["hvpct"]) if ind.get("hvpct") is not None else None,
            "c_iv": round(c_iv * 100) if c_iv else None, "p_iv": round(p_iv * 100) if p_iv else None,
            "iv_vs_hv": _iv_richness(c_iv, p_iv, hv),
            "rsi": round(ind["rsi"]) if ind.get("rsi") is not None else None,
            "above_200": bool(ind.get("above_200")),
            "pcr": round(pcr, 2) if pcr is not None else None,
            "pctb": round(pctb, 2) if pctb is not None else None,
            "csp_timing": csp_tl, "cc_timing": cc_tl,
        }
    except Exception:
        _OVERVIEW[ticker] = {"ticker": ticker}
    return _OVERVIEW[ticker]


def scan(universe):
    """Full scan → dict ready to serialise. Never raises.

    `universe` is a dict {"wheel": [...], "growth": [...]}. Wheel names → CSP/CC premium
    signals; wheel ∪ growth → LEAP (growth + PMCC basis) ideas. A plain list is treated as the
    wheel list with no growth names (back-compat)."""
    if isinstance(universe, dict):
        wheel = [str(t).strip().upper() for t in (universe.get("wheel") or []) if str(t).strip()]
        growth = [str(t).strip().upper() for t in (universe.get("growth") or []) if str(t).strip()]
    else:
        wheel = [str(t).strip().upper() for t in universe if str(t).strip()]
        growth = []

    _IND_CACHE.clear(); _CHAIN_CACHE.clear(); _OVERVIEW.clear()   # fresh per scan
    all_names = list(dict.fromkeys(wheel + growth))

    all_sigs = []
    for t in all_names:                 # CSP/CC for every name — CSPs also enter growth names
        all_sigs.extend(scan_ticker(t))
    _shortlist(all_sigs)
    all_sigs.sort(key=lambda s: (s["strategy"] != "CSP", -(s["premium_pct"] or 0)))

    leaps = []
    for t in list(dict.fromkeys(growth + wheel)):     # growth first, deduped
        lp = scan_leap_ticker(t)
        if lp:
            leaps.append(lp)
    leaps.sort(key=lambda s: (not s.get("qualifies"),
                              s.get("carry_pct") if s.get("carry_pct") is not None else 999))

    # Watchlist-Overview rows for EVERY name (the app analyses all of them, not just the wheel).
    for t in all_names:
        overview_row(t)
    leap_tl = {lp["ticker"]: lp.get("timing_label") for lp in leaps}
    overview = []
    for t in all_names:
        row = _OVERVIEW.get(t)
        if row:
            row["leap_timing"] = leap_tl.get(t)
            row["is_wheel"] = t in wheel
            overview.append(row)

    return {
        "signals": all_sigs,
        "leaps": leaps,
        "overview": overview,
        "params": {"min_premium_pct": P["min_premium"] * 100,
                   "strong_premium_pct": P["strong_premium"] * 100,
                   "dte": [P["dte_lo"], P["dte_hi"]], "delta_opt": P["delta_opt"],
                   "premium_basis": "premium ÷ spot",
                   "iv_buckets": [P["iv_anchor"] * 100, P["iv_high"] * 100],
                   "earnings_blackout": P["earnings_blackout"],
                   "earnings_blackout_days": P["earnings_blackout_days"]},
        "count": len(all_sigs), "leap_count": len(leaps),
    }
