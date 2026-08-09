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
import statistics

import tradier

# ── tunable parameters (one place) ───────────────────────────────────────────────
P = {
    # CSP / CC (the wheel)
    "dte_lo": 21, "dte_hi": 45, "dte_opt": 30,
    "delta_opt": 0.30, "delta_lo": 0.18, "delta_hi": 0.40,
    "min_premium": 0.012,        # 1.2% per-trade floor — premium ÷ SPOT (Jay's definition)
    "strong_premium": 0.015,     # 1.5%+ = flagged strong
    "min_oi": 100,
    "max_spread_pct": 0.15,      # (ask-bid)/mid
    "earnings_blackout": True,   # no new position if earnings falls before expiry
    "iv_anchor": 0.30, "iv_high": 0.60,   # vol buckets: <30 anchor · 30-60 med · ≥60 high
    "max_per_sector": 2,         # shortlist diversification
    "shortlist_n": 10,
    # LEAP (growth + PMCC basis) — long-dated deep-ITM call
    "leap_dte_lo": 300, "leap_dte_hi": 540, "leap_dte_opt": 420,
    "leap_delta_opt": 0.75, "leap_delta_lo": 0.65, "leap_delta_hi": 0.85,
    "leap_max_extrinsic": 0.12,  # ≤12% of the premium is time value → good PMCC basis
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


def _sector(ticker):
    try:
        import yfinance as yf
        return (yf.Ticker(ticker).info or {}).get("sector") or "—"
    except Exception:
        return "—"


def _earnings_before(ticker, expiry):
    """True if an earnings date falls between today and the expiry (blackout). Best-effort via
    yfinance; on any failure returns False (don't block a trade just because we couldn't check)."""
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
            return False
        if hasattr(dt, "date"):
            dt = dt.date()
        exp = datetime.date.fromisoformat(expiry)
        return datetime.date.today() <= dt <= exp
    except Exception:
        return False


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


def _build(ticker, strat, o, spot, expiry, dte, sector, iv_atm, earn):
    """Assemble one signal dict from a chosen option contract, or None if it fails filters."""
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
    if P["earnings_blackout"] and earn:
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
        "earnings_before_expiry": bool(earn),
        "strong": prem_pct >= P["strong_premium"],
    }


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
        chain = tradier.get_option_chain(ticker, expiry, greeks=True)
        if not chain:
            return out
        # ATM IV (nearest strike, avg call/put mid_iv) → vol bucket
        atm = min(chain, key=lambda o: abs((_f(o.get("strike")) or 1e9) - spot), default=None)
        atm_strike = _f(atm.get("strike")) if atm else None
        atm_ivs = [_f((o.get("greeks") or {}).get("mid_iv")) for o in chain
                   if _f(o.get("strike")) == atm_strike]
        atm_ivs = [v for v in atm_ivs if v]
        iv_atm = (sum(atm_ivs) / len(atm_ivs)) if atm_ivs else None

        # median gate from Tradier daily history
        start = (today - datetime.timedelta(days=60)).isoformat()
        hist = tradier.get_history(ticker, interval="daily", start=start, end=today.isoformat())
        closes = [_f(d.get("close")) for d in hist] if hist else []
        sma20, pctb = _sma20_pctb(closes)
        below_median = (spot < sma20) if sma20 else None      # CSP wants below, CC wants above

        sector = _sector(ticker)
        earn = _earnings_before(ticker, expiry)

        # CSP — needs price BELOW the median to go on the shortlist (else flagged)
        put = _nearest_delta(chain, "put", P["delta_opt"])
        if put:
            s = _build(ticker, "CSP", put, spot, expiry, dte, sector, iv_atm, earn)
            if s:
                s["median_ok"] = bool(below_median) if below_median is not None else None
                s["pct_b"] = round(pctb, 2) if pctb is not None else None
                out.append(s)
        # CC — needs price ABOVE the median
        call = _nearest_delta(chain, "call", P["delta_opt"])
        if call:
            s = _build(ticker, "CC", call, spot, expiry, dte, sector, iv_atm, earn)
            if s:
                above_median = (not below_median) if below_median is not None else None
                s["median_ok"] = bool(above_median) if above_median is not None else None
                s["pct_b"] = round(pctb, 2) if pctb is not None else None
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
        chain = tradier.get_option_chain(ticker, expiry, greeks=True)
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
        extrinsic = mid - intrinsic
        ext_pct = (extrinsic / mid) if mid else None
        return {
            "ticker": ticker, "strategy": "LEAP", "spot": round(spot, 2),
            "expiry": expiry, "dte": dte, "strike": strike,
            "delta": round(dl, 3) if dl is not None else None,
            "mid": round(mid, 2), "cost": round(mid * 100, 0),
            "intrinsic": round(intrinsic, 2), "extrinsic": round(extrinsic, 2),
            "extrinsic_pct": round(ext_pct * 100, 1) if ext_pct is not None else None,
            "iv": round(iv * 100, 1) if iv is not None else None,
            "sector": _sector(ticker),
            "good_pmcc": bool(ext_pct is not None and ext_pct <= P["leap_max_extrinsic"]),
        }
    except Exception:
        return None


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

    all_sigs = []
    for t in wheel:
        all_sigs.extend(scan_ticker(t))
    _shortlist(all_sigs)
    all_sigs.sort(key=lambda s: (s["strategy"] != "CSP", -(s["premium_pct"] or 0)))

    leaps = []
    for t in list(dict.fromkeys(growth + wheel)):     # growth first, deduped
        lp = scan_leap_ticker(t)
        if lp:
            leaps.append(lp)
    leaps.sort(key=lambda s: (not s.get("good_pmcc"), s.get("extrinsic_pct") or 99))

    return {
        "signals": all_sigs,
        "leaps": leaps,
        "params": {"min_premium_pct": P["min_premium"] * 100,
                   "strong_premium_pct": P["strong_premium"] * 100,
                   "dte": [P["dte_lo"], P["dte_hi"]], "delta_opt": P["delta_opt"],
                   "premium_basis": "premium ÷ spot",
                   "iv_buckets": [P["iv_anchor"] * 100, P["iv_high"] * 100],
                   "earnings_blackout": P["earnings_blackout"]},
        "count": len(all_sigs), "leap_count": len(leaps),
    }
