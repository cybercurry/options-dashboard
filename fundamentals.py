"""
Fundamentals door — REAL company fundamentals straight from SEC filings (10-K/10-Q XBRL),
not a vendor's derived numbers. One ticker in, a structured read out: valuation, quality,
balance-sheet health, and a red-flag list.

Source: SEC EDGAR's free XBRL API (no key, no login). Two endpoints:
  • https://www.sec.gov/files/company_tickers.json      → ticker → CIK map
  • https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json → every reported concept

SEC requires a descriptive User-Agent on every request (else 403). Everything here is REST and
works with the market closed — filings don't move intraday. Price (for P/E, FCF yield, market
cap) is passed in by the app from its live Tradier feed; this module never fetches a quote.

Design (per Jay): real data over calculated, extreme simplicity. The app renders three verdict
chips (Valuation / Quality / Health) + a red-flag list; every raw number rides in the hover.
"""

import os
import requests
from datetime import date, timedelta

try:                       # streamlit is present in the app; keep this importable standalone.
    import streamlit as _st
    _HAS_ST_SECRETS = True
except Exception:
    _HAS_ST_SECRETS = False

# SEC's "fair access" policy REQUIRES a User-Agent that names the app AND a contact email —
# a UA without an email is rejected with HTTP 403. This is a declaration header, not auth: no
# secret, no personal PII (a neutral project mailbox, not Jay's address). Accept-Encoding is
# also expected. https://www.sec.gov/os/webmaster-faq#developers
_HEADERS = {
    "User-Agent": "options-dashboard/1.0 (contact: options-dashboard@proton.me)",
    "Accept-Encoding": "gzip, deflate",
    "Accept": "application/json, text/plain, */*",
}
_TIMEOUT = 15

# ── thresholds (one place to tune) ───────────────────────────────────────────────
T = {
    "pe_rich": 40,   "pe_high": 60,
    "fcfy_thin": 0.02,                        # 2% free-cash-flow yield
    "peg_high": 3.0,
    "netmargin_thin": 0.05,                   # 5%
    "roe_thin": 0.08,  "roe_good": 0.15,      # 8% / 15%
    "de_high": 1.0,  "de_danger": 2.0,        # debt / equity
    "cr_thin": 1.5,  "cr_danger": 1.0,        # current ratio
    "cover_thin": 6,  "cover_danger": 3,      # interest coverage (x)
    "dilution": 0.05,                         # +5% shares YoY
}

# XBRL concept names vary by filer — try each in order until one has data.
_C = {
    "revenue":  ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues",
                 "SalesRevenueNet", "RevenueFromContractWithCustomerIncludingAssessedTax"],
    "net_income": ["NetIncomeLoss"],
    "gross_profit": ["GrossProfit"],
    "op_income": ["OperatingIncomeLoss"],
    "assets": ["Assets"],
    "liabilities": ["Liabilities"],
    "equity": ["StockholdersEquity",
               "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "cur_assets": ["AssetsCurrent"],
    "cur_liab": ["LiabilitiesCurrent"],
    "lt_debt": ["LongTermDebtNoncurrent", "LongTermDebt"],
    "debt_cur": ["LongTermDebtCurrent", "DebtCurrent"],
    "cfo": ["NetCashProvidedByUsedInOperatingActivities",
            "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets"],
    "rnd": ["ResearchAndDevelopmentExpense",
            "ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost"],
    "interest": ["InterestExpense", "InterestExpenseDebt"],
    "eps": ["EarningsPerShareDiluted", "EarningsPerShareBasic"],
    "sh_diluted": ["WeightedAverageNumberOfDilutedSharesOutstanding"],
    "shares_out": ["CommonStockSharesOutstanding", "EntityCommonStockSharesOutstanding"],
}


# ── SEC fetch ────────────────────────────────────────────────────────────────────
def _get(url):
    r = requests.get(url, headers=_HEADERS, timeout=_TIMEOUT)
    r.raise_for_status()
    return r.json()


_CIK_MAP_CACHE = None


def ticker_to_cik(ticker):
    """Map a ticker to its zero-padded 10-digit CIK via SEC's public map. The ~1 MB map is fetched
    once per process and reused (a concurrent batch would otherwise pull it once per name — needless
    load on SEC). A thread race just re-fetches the same bytes, so no lock is needed."""
    global _CIK_MAP_CACHE
    if _CIK_MAP_CACHE is None:
        _CIK_MAP_CACHE = _get("https://www.sec.gov/files/company_tickers.json")
    data = _CIK_MAP_CACHE
    tk = ticker.strip().upper()
    for row in data.values():
        if str(row.get("ticker", "")).upper() == tk:
            return str(row["cik_str"]).zfill(10), row.get("title", tk)
    return None, None


def company_facts(cik):
    return _get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json")


# ── concept extraction ───────────────────────────────────────────────────────────
def _rows(facts, concepts):
    """Return the raw fact rows for the first matching concept (us-gaap or dei)."""
    gaap = facts.get("facts", {}).get("us-gaap", {})
    dei  = facts.get("facts", {}).get("dei", {})
    for c in concepts:
        node = gaap.get(c) or dei.get(c)
        if not node:
            continue
        units = node.get("units", {})
        rows = None
        for uk in ("USD", "shares", "USD/shares"):
            if uk in units:
                rows = units[uk]; break
        if rows is None:
            rows = next(iter(units.values()), [])
        if rows:
            return rows
    return []


def _annual_last_two(facts, concepts):
    """Latest two ANNUAL values (from 10-K, full-year) as (latest, prior). Dedupes restatements
    by keeping the latest-filed value per period end."""
    by_end = {}
    for r in _rows(facts, concepts):
        if r.get("form") != "10-K" or (r.get("fp") and r.get("fp") != "FY"):
            continue
        end, val, filed = r.get("end"), r.get("val"), r.get("filed", "")
        if end is None or val is None:
            continue
        if end not in by_end or filed > by_end[end][1]:
            by_end[end] = (float(val), filed)
    if not by_end:
        return None, None
    ends = sorted(by_end)
    latest = by_end[ends[-1]][0]
    prior = by_end[ends[-2]][0] if len(ends) >= 2 else None
    return latest, prior


def _pdate(s):
    try:
        y, mo, d = str(s).split("-"); return date(int(y), int(mo), int(d))
    except Exception:
        return None


def _ddays(s, e):
    ds, de = _pdate(s), _pdate(e)
    return (de - ds).days if (ds and de) else 0


def _ttm(facts, concepts):
    """Trailing-twelve-months for a FLOW item (revenue, net income, EPS, cash flow): latest full
    fiscal year + current-year YTD − the prior-year matching YTD. Dedupes restatements (latest-filed
    per period). Degrades to the latest annual value when no newer quarter is on file, so annual-only
    / slow filers come out exactly as before. Returns a single float (or None).

    This is what keeps P/E current: a fast grower that has filed three 2026 quarters is valued on its
    trailing 12 months, not on a stale FY-2025 that under-counts earnings and inflates P/E."""
    pdate, dur = _pdate, _ddays
    seen = {}                                     # dedupe duration facts by (start,end)
    for r in _rows(facts, concepts):
        s, e, v, f = r.get("start"), r.get("end"), r.get("val"), r.get("filed", "")
        if not (s and e and v is not None):
            continue
        k = (s, e)
        if k not in seen or f > seen[k][1]:
            seen[k] = (float(v), f)
    items = [(s, e, v) for (s, e), (v, _f) in seen.items()]
    if not items:
        return None

    annuals = sorted([(s, e, v) for (s, e, v) in items if 350 <= dur(s, e) <= 380],
                     key=lambda x: x[1])
    if not annuals:                               # no annual on file → sum last 4 discrete quarters
        q = sorted([(s, e, v) for (s, e, v) in items if 80 <= dur(s, e) <= 100],
                   key=lambda x: x[1])
        return sum(v for _, _, v in q[-4:]) if len(q) >= 4 else None

    fy_s, fy_e, fy_v = annuals[-1]
    fye = pdate(fy_e)
    # current-year YTD: a cumulative period ending AFTER the FY end, within ~1 year of it
    cur = [(s, e, v) for (s, e, v) in items
           if pdate(e) and fye and pdate(e) > fye and (pdate(e) - fye).days <= 370
           and 60 <= dur(s, e) < 350]
    if not cur:
        return fy_v                               # no newer quarter → latest annual stands
    c_s, c_e, c_v = max(cur, key=lambda x: dur(x[0], x[1]))   # longest YTD (e.g. 9-month)
    clen, cend = dur(c_s, c_e), pdate(c_e)
    # prior-year matching YTD: same length, ending ~1 year before the current YTD
    pri = [(s, e, v) for (s, e, v) in items
           if abs(dur(s, e) - clen) <= 20 and pdate(e) and 330 <= (cend - pdate(e)).days <= 400]
    if not pri:
        return fy_v                               # can't match prior YTD → annual (never a wrong TTM)
    p_v = max(pri, key=lambda x: pdate(x[1]))[2]
    return fy_v + c_v - p_v


# ── per-quarter series (the Fundamentals tab's 3-quarters-side-by-side table) ─────
def _dur_facts(facts, concepts):
    """Deduped duration facts (latest-filed per start/end) as (start, end, val, days)."""
    seen = {}
    for r in _rows(facts, concepts):
        s, e, v, f = r.get("start"), r.get("end"), r.get("val"), r.get("filed", "")
        if not (s and e and v is not None):
            continue
        k = (s, e)
        if k not in seen or f > seen[k][1]:
            seen[k] = (float(v), f)
    out = []
    for (s, e), (v, _f) in seen.items():
        d = _ddays(s, e)
        if d > 0:
            out.append((s, e, v, d))
    return out


def _discrete_quarters(facts, concepts):
    """Map each fiscal-quarter END date → that quarter's DISCRETE (3-month) flow value.

    SEC files flows two ways: some as explicit ~90-day quarters, most as year-to-date cumulatives
    (Q1=3mo, Q2=6mo, Q3=9mo, FY=12mo, all sharing the fiscal-year start). We take explicit quarters
    directly, then fill any gap by differencing the cumulative chain — each increment is accepted only
    when it spans ~90 days, so a standalone annual (no YTD chain) never masquerades as a quarter and
    Q4 falls out cleanly as FY − 9-month."""
    fs = _dur_facts(facts, concepts)
    disc = {}
    for (s, e, v, d) in fs:                        # explicit ~quarter facts win
        if 80 <= d <= 100:
            disc[e] = v
    bystart = {}
    for (s, e, v, d) in fs:
        if d <= 385:                               # within a fiscal year (YTD legs + annual)
            bystart.setdefault(s, []).append((e, v, d))
    for s, lst in bystart.items():
        lst.sort(key=lambda x: x[0])               # by end date
        prev_v, prev_d = 0.0, 0
        for (e, v, d) in lst:
            seg = d - prev_d                        # implied length of this increment
            if e not in disc and 80 <= seg <= 100:  # ~one quarter → a real discrete value
                disc[e] = v - prev_v
            prev_v, prev_d = v, d
    return disc


def _points_by_end(facts, concepts):
    """Balance-sheet (point-in-time) value at each reported period end (latest-filed per end)."""
    seen = {}
    for r in _rows(facts, concepts):
        e, v, f = r.get("end"), r.get("val"), r.get("filed", "")
        if e is None or v is None:
            continue
        if e not in seen or f > seen[e][1]:
            seen[e] = (float(v), f)
    return {e: v for e, (v, _f) in seen.items()}


_QLBL_MON = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _q_label(end):
    """Short column header for a quarter-END date: fiscal-quarter guess + the period end, e.g.
    'Q3 ·Sep ’26'. Calendar-based quarter is an approximation for off-calendar fiscal years, but the
    month+year is exact, so the header is never misleading."""
    d = _pdate(end)
    if not d:
        return str(end)
    q = (d.month - 1) // 3 + 1
    return "Q%d · %s ’%02d" % (q, _QLBL_MON[d.month], d.year % 100)


def _recent_quarter_ends(facts, want=3):
    """The most recent `want` fiscal-quarter end dates we can form a discrete value for."""
    rev = _discrete_quarters(facts, _C["revenue"])
    ni = _discrete_quarters(facts, _C["net_income"])
    ends = sorted(e for e in set(list(rev) + list(ni)) if e)
    return ends[-want:]


def _quarters(facts, prices=None, want=3, lookback=7):
    """Build the per-quarter rows for the table. `prices` maps quarter-END date → closing price
    (used for that quarter's P/E and market cap); omit for no valuation columns. Per-quarter P/E uses
    trailing-twelve-months net income AT that quarter (the same basis as the headline P/E), so the
    valuation trend is apples-to-apples across the three columns."""
    prices = prices or {}
    rev_d = _discrete_quarters(facts, _C["revenue"])
    ni_d = _discrete_quarters(facts, _C["net_income"])
    gp_d = _discrete_quarters(facts, _C["gross_profit"])
    opi_d = _discrete_quarters(facts, _C["op_income"])
    cfo_d = _discrete_quarters(facts, _C["cfo"])
    cap_d = _discrete_quarters(facts, _C["capex"])
    rnd_d = _discrete_quarters(facts, _C["rnd"])
    eps_d = _discrete_quarters(facts, _C["eps"])
    eq_p = _points_by_end(facts, _C["equity"])
    ca_p = _points_by_end(facts, _C["cur_assets"])
    cl_p = _points_by_end(facts, _C["cur_liab"])
    ltd_p = _points_by_end(facts, _C["lt_debt"])
    dc_p = _points_by_end(facts, _C["debt_cur"])
    sh_p = _points_by_end(facts, _C["shares_out"])

    ends = sorted(e for e in set(list(rev_d) + list(ni_d)) if e)
    recent = ends[-lookback:]

    def ttm_ni(e):                                 # rolling 4-quarter NI ending at e (for ROE/PE)
        idx = recent.index(e)
        legs = recent[max(0, idx - 3): idx + 1]
        vals = [ni_d.get(x) for x in legs]
        return sum(vals) if (len(legs) == 4 and all(v is not None for v in vals)) else None

    def at_or_before(pts, e):                       # latest balance-sheet point ≤ e
        cand = [d for d in pts if d <= e]
        return pts[max(cand)] if cand else None

    rows = []
    for e in recent[-want:]:
        rev, ni = rev_d.get(e), ni_d.get(e)
        equity = at_or_before(eq_p, e); shares = at_or_before(sh_p, e)
        ca = at_or_before(ca_p, e); cl = at_or_before(cl_p, e)
        ltd = at_or_before(ltd_p, e); dc = at_or_before(dc_p, e)
        total_debt = ((ltd or 0) + (dc or 0)) if (ltd is not None or dc is not None) else None
        cfo, cap = cfo_d.get(e), cap_d.get(e)
        fcf = (cfo - cap) if (cfo is not None and cap is not None) else None
        tn = ttm_ni(e)
        px = prices.get(e)
        mcap = (px * shares) if (px and shares) else None
        pe = (mcap / tn) if (mcap and tn and tn > 0) else None
        rows.append({
            "label": _q_label(e), "end": e,
            "revenue": rev, "net_income": ni, "eps": eps_d.get(e),
            "gross_margin": _div(gp_d.get(e), rev), "op_margin": _div(opi_d.get(e), rev),
            "net_margin": _div(ni, rev), "fcf": fcf, "fcf_margin": _div(fcf, rev),
            "capex": cap, "rnd": rnd_d.get(e),
            "debt_to_equity": _div(total_debt, equity), "current_ratio": _div(ca, cl),
            "market_cap": mcap, "pe": pe, "price": px,
        })
    return rows


def _quarter_end_prices(symbol, ends):
    """Closing price near each quarter-END date, via Tradier monthly history (one lightweight call).
    {end_date: close}. Empty on any failure — valuation columns then show n/a, never a wrong price."""
    out = {}
    if not ends:
        return out
    try:
        import tradier
        if not tradier.is_configured():
            return out
        start = (min(_pdate(e) for e in ends) - timedelta(days=40)).isoformat()
        hist = tradier.get_history(symbol, interval="monthly", start=start)
        by_ym = {}
        for bar in hist:
            d = str(bar.get("date", ""))[:7]        # YYYY-MM
            c = bar.get("close")
            if d and isinstance(c, (int, float)):
                by_ym[d] = float(c)
        for e in ends:
            out[e] = by_ym.get(str(e)[:7])
    except Exception:
        return {}
    return {e: v for e, v in out.items() if v is not None}


def _fwd_pe(ticker):
    """Forward P/E from yfinance (price ÷ consensus forward EPS). It's the one estimate-based figure
    in the tab — SEC filings have no forward number — so it degrades to None (shown n/a) rather than
    ever guessing. Runs server-side per refresh, so Yahoo rate limits aren't hit per viewer."""
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info or {}
        v = info.get("forwardPE")
        return float(v) if isinstance(v, (int, float)) and v > 0 else None
    except Exception:
        return None


def _latest_point(facts, concepts):
    """Most recently reported value for a balance-sheet (point-in-time) item, any form — so a
    fresh 10-Q updates the balance sheet between annual reports."""
    by_end = {}
    for r in _rows(facts, concepts):
        end, val, filed = r.get("end"), r.get("val"), r.get("filed", "")
        if end is None or val is None:
            continue
        if end not in by_end or filed > by_end[end][1]:
            by_end[end] = (float(val), filed)
    if not by_end:
        return None
    return by_end[sorted(by_end)[-1]][0]


def _div(a, b):
    return (a / b) if (a is not None and b not in (None, 0)) else None


# ── assessment: raw metrics → red flags + group verdicts (source-agnostic) ────────
def _assess(m):
    """Turn a metrics dict (whatever the source) into (flags, groups). Shared by the SEC and
    Yahoo paths so the verdict logic is identical no matter where the numbers came from."""
    flags = []
    def flag(sev, text): flags.append({"sev": sev, "text": text})

    if m["net_margin"] is not None and m["net_margin"] < 0:
        flag("🔴", "Unprofitable — negative net margin")
    if m["fcf"] is not None and m["fcf"] < 0:
        flag("🔴", "Burning cash — negative free cash flow")
    if m["debt_to_equity"] is not None and m["debt_to_equity"] > T["de_danger"]:
        flag("🔴", f"High leverage — debt/equity {m['debt_to_equity']:.1f}×")
    if m["current_ratio"] is not None and m["current_ratio"] < T["cr_danger"]:
        flag("🔴", f"Liquidity tight — current ratio {m['current_ratio']:.2f}")
    if m["interest_coverage"] is not None and m["interest_coverage"] < T["cover_danger"]:
        flag("🔴", f"Thin interest coverage — {m['interest_coverage']:.1f}×")
    if m["rev_growth"] is not None and m["rev_growth"] < 0:
        flag("🟡", f"Revenue shrinking — {m['rev_growth']*100:+.0f}% YoY")
    if m["roe"] is not None and m["roe"] < 0:
        flag("🟡", "Negative return on equity")
    if m["share_change"] is not None and m["share_change"] > T["dilution"]:
        flag("🟡", f"Dilution — shares {m['share_change']*100:+.0f}% YoY")
    if m["pe"] is not None and m["pe"] > T["pe_rich"]:
        flag("🟡", f"Rich valuation — P/E {m['pe']:.0f}")

    def verdict(reds, ambers):
        if any(reds):   return "🔴"
        if any(ambers): return "🟡"
        return "🟢"

    groups = {
        "Valuation": {"verdict": verdict(
            [m["pe"] is not None and m["pe"] > T["pe_high"],
             m["fcf_yield"] is not None and m["fcf_yield"] < 0],
            [m["pe"] is not None and m["pe"] > T["pe_rich"],
             m["fcf_yield"] is not None and m["fcf_yield"] < T["fcfy_thin"],
             m["peg"] is not None and m["peg"] > T["peg_high"]])},
        "Quality": {"verdict": verdict(
            [m["net_margin"] is not None and m["net_margin"] < 0,
             m["roe"] is not None and m["roe"] < 0],
            [m["net_margin"] is not None and m["net_margin"] < T["netmargin_thin"],
             m["roe"] is not None and m["roe"] < T["roe_thin"]])},
        "Health": {"verdict": verdict(
            [m["debt_to_equity"] is not None and m["debt_to_equity"] > T["de_danger"],
             m["current_ratio"] is not None and m["current_ratio"] < T["cr_danger"],
             m["interest_coverage"] is not None and m["interest_coverage"] < T["cover_danger"],
             m["fcf"] is not None and m["fcf"] < 0],
            [m["debt_to_equity"] is not None and m["debt_to_equity"] > T["de_high"],
             m["current_ratio"] is not None and m["current_ratio"] < T["cr_thin"],
             m["interest_coverage"] is not None and m["interest_coverage"] < T["cover_thin"]])},
    }
    # A group with no data at all shows a neutral dot, not a false green.
    if m["pe"] is None and m["fcf_yield"] is None and m["peg"] is None:
        groups["Valuation"]["verdict"] = "⚪"
    if m["net_margin"] is None and m["roe"] is None:
        groups["Quality"]["verdict"] = "⚪"
    if m["debt_to_equity"] is None and m["current_ratio"] is None and m["fcf"] is None:
        groups["Health"]["verdict"] = "⚪"
    return flags, groups


def _metrics_from_sec(facts, price):
    """Build the metrics dict from SEC companyfacts. Point metrics (P/E, EPS, margins, FCF) use
    TRAILING-TWELVE-MONTHS (latest full fiscal year + current-year YTD − prior-year YTD) so fast
    growers aren't valued on a stale annual EPS; year-over-year growth stays full-year-vs-full-year;
    balance-sheet items use the most recent reported period. The displayed net income is the same
    TTM figure the P/E is built from, so market-cap ÷ net income reconciles to the shown P/E."""
    # Annual latest/prior — drives clean full-year-vs-full-year growth
    rev_a, rev_ap = _annual_last_two(facts, _C["revenue"])
    ni_a,  ni_ap  = _annual_last_two(facts, _C["net_income"])
    shd,   shd_p  = _annual_last_two(facts, _C["sh_diluted"])

    # TTM for the point metrics; fall back to the annual figure where a TTM can't be formed
    rev   = _ttm(facts, _C["revenue"])      or rev_a
    ni    = _ttm(facts, _C["net_income"])   or ni_a
    gp    = _ttm(facts, _C["gross_profit"]) or _annual_last_two(facts, _C["gross_profit"])[0]
    opi   = _ttm(facts, _C["op_income"])    or _annual_last_two(facts, _C["op_income"])[0]
    cfo   = _ttm(facts, _C["cfo"])          or _annual_last_two(facts, _C["cfo"])[0]
    capex = _ttm(facts, _C["capex"])        or _annual_last_two(facts, _C["capex"])[0]
    rnd   = _ttm(facts, _C["rnd"])          or _annual_last_two(facts, _C["rnd"])[0]
    intr  = _ttm(facts, _C["interest"])     or _annual_last_two(facts, _C["interest"])[0]

    equity   = _latest_point(facts, _C["equity"])
    cur_a    = _latest_point(facts, _C["cur_assets"])
    cur_l    = _latest_point(facts, _C["cur_liab"])
    lt_debt  = _latest_point(facts, _C["lt_debt"])
    debt_cur = _latest_point(facts, _C["debt_cur"])
    shares   = _latest_point(facts, _C["shares_out"])

    total_debt = None
    if lt_debt is not None or debt_cur is not None:
        total_debt = (lt_debt or 0) + (debt_cur or 0)
    fcf = (cfo - capex) if (cfo is not None and capex is not None) else None
    # EPS from TTM net income ÷ current shares (robust against quarterly-EPS rounding drift)
    eps = _div(ni, shares) if (ni is not None and shares) else _ttm(facts, _C["eps"])

    m = {
        "revenue": rev, "revenue_prev": rev_a,            # headline revenue is TTM; prior = last full FY
        "net_income": ni, "net_income_prev": ni_a,
        "rev_growth": _div(rev_a - rev_ap, rev_ap) if (rev_a is not None and rev_ap) else None,
        "ni_growth":  _div(ni_a - ni_ap, abs(ni_ap)) if (ni_a is not None and ni_ap) else None,
        "gross_margin": _div(gp, rev),
        "op_margin":    _div(opi, rev),
        "net_margin":   _div(ni, rev),
        "roe":          _div(ni, equity),
        "fcf": fcf, "fcf_margin": _div(fcf, rev),
        "debt_to_equity": _div(total_debt, equity),
        "current_ratio":  _div(cur_a, cur_l),
        "interest_coverage": _div(opi, intr),
        "eps": eps, "shares": shares,
        "share_change": _div(shd - shd_p, shd_p) if (shd is not None and shd_p) else None,
        "capex": capex, "rnd": rnd,
    }
    m["market_cap"] = (price * shares) if (price and shares) else None
    m["pe"] = _div(m["market_cap"], ni) if (m["market_cap"] and ni and ni > 0) else None
    m["fcf_yield"] = _div(fcf, m["market_cap"])
    m["peg"] = (m["pe"] / (m["ni_growth"] * 100)) if (m["pe"] and m["ni_growth"] and m["ni_growth"] > 0) else None
    return m


def _metrics_from_yf(ticker, price):
    """Fallback source: Yahoo Finance fundamentals (yfinance). Yahoo aggregates the same SEC
    filings, so it's a credible backup when SEC's endpoint refuses us. Fewer trend fields than
    the raw filings (no YoY share count / interest coverage), so those come back None → their
    checks simply don't fire. Returns None if Yahoo has nothing usable."""
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info or {}
    except Exception:
        return None
    if not info:
        return None
    rev = info.get("totalRevenue")
    ni  = info.get("netIncomeToCommon")
    fcf = info.get("freeCashflow")
    mcap = info.get("marketCap")
    de = info.get("debtToEquity")
    de = (de / 100.0) if de is not None else None          # yfinance reports D/E as a percent
    eps = info.get("trailingEps")
    shares = info.get("sharesOutstanding")
    pe = info.get("trailingPE") or (_div(price, eps) if (price and eps and eps > 0) else None)
    m = {
        "revenue": rev, "revenue_prev": None,
        "net_income": ni, "net_income_prev": None,
        "rev_growth": info.get("revenueGrowth"),
        "ni_growth":  info.get("earningsGrowth"),
        "gross_margin": info.get("grossMargins"),
        "op_margin":    info.get("operatingMargins"),
        "net_margin":   info.get("profitMargins"),
        "roe":          info.get("returnOnEquity"),
        "fcf": fcf, "fcf_margin": _div(fcf, rev),
        "debt_to_equity": de,
        "current_ratio":  info.get("currentRatio"),
        "interest_coverage": None,
        "eps": eps, "shares": shares,
        "share_change": None,
        "capex": None, "rnd": None,
        "market_cap": mcap or ((price * shares) if (price and shares) else None),
        "pe": pe,
        "fcf_yield": _div(fcf, mcap),
        "peg": info.get("trailingPegRatio") or info.get("pegRatio"),
    }
    # Guard against an empty/placeholder info dict — require at least one substantive number.
    if all(m[k] is None for k in ("revenue", "net_income", "pe", "net_margin", "market_cap")):
        return None
    return m


# ── company profile: sourced facts only, NOTHING generated ────────────────────────
def _extractive_short(text, max_sentences=3, max_chars=460):
    """A no-AI 'summary': the first 1-2 sentences of the company's own filing text. Clean because
    it ends on real sentence boundaries — never a mid-thought truncation. Zero hallucination risk
    (it IS the source text), zero setup."""
    import re
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    out, total = [], 0
    for p in parts:
        if out and total + len(p) > max_chars:
            break
        out.append(p); total += len(p)
        if len(out) >= max_sentences:
            break
    return " ".join(out) if out else text[:max_chars]


def _anthropic_key():
    # Optional. st.secrets first (Streamlit Cloud), then env. Guarded so an unconfigured app
    # never crashes — no key just means we use the extractive summary instead.
    if _HAS_ST_SECRETS:
        try:
            if "ANTHROPIC_API_KEY" in _st.secrets:
                return _st.secrets["ANTHROPIC_API_KEY"]
        except Exception:
            pass
    return os.environ.get("ANTHROPIC_API_KEY")


def ai_summary(text):
    """Short plain-English 'what this company does'. If an Anthropic key is configured, an LLM
    CONDENSES THE SUPPLIED filing text (summarizing given text, not recalling facts from memory —
    so it can't invent). With no key, falls back to the extractive first-sentences summary.
    Returns (text, is_ai). Never fabricates: no source text → (None, False)."""
    if not text:
        return None, False
    key = _anthropic_key()
    if key:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=key)
            msg = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=240,
                system=("Summarise the company in 2-3 plain-English sentences: what it does, its "
                        "main products or services, and the markets/customers it serves. Use ONLY "
                        "the text provided — do not add any fact that isn't in it. No preamble, "
                        "no lists."),
                messages=[{"role": "user", "content": text[:4000]}],
            )
            out = "".join(getattr(b, "text", "") for b in msg.content).strip()
            if out:
                return out, True
        except Exception:
            pass   # any failure (no package, bad key, network) → extractive fallback
    return _extractive_short(text), False


def _sec_sic(cik):
    """SEC's own industry classification (SIC description) from the submissions endpoint."""
    try:
        d = _get(f"https://data.sec.gov/submissions/CIK{cik}.json")
        return d.get("sicDescription")
    except Exception:
        return None


def company_profile(ticker, cik=None):
    """Factual profile — never AI-written. The business summary is Yahoo's `longBusinessSummary`,
    which is the company's own 'Item 1. Business' description aggregated from its filings, shown
    verbatim. Sector/industry/country/employees are provider metadata; industry is cross-checked
    against SEC's SIC classification. Returns {} if nothing is available."""
    prof = {}
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info or {}
        prof = {
            "summary":   info.get("longBusinessSummary"),
            "sector":    info.get("sector"),
            "industry":  info.get("industry"),
            "country":   info.get("country"),
            "employees": info.get("fullTimeEmployees"),
            "website":   info.get("website"),
        }
    except Exception:
        pass
    if cik:
        sic = _sec_sic(cik)
        if sic:
            prof["sic"] = sic
            prof.setdefault("industry", sic)   # fall back to SEC's label if Yahoo had none
    # Short perma-visible summary (AI-condensed from the filing text if a key is set, else an
    # extractive excerpt). The long verbatim text stays in `summary` for the drill-down.
    if prof.get("summary"):
        s, is_ai = ai_summary(prof["summary"])
        if s:
            prof["summary_short"] = s
            prof["summary_ai"] = is_ai
    return {k: v for k, v in prof.items() if v not in (None, "")}


def company_news(ticker, limit=6):
    """Recent REAL headlines from Yahoo Finance (yfinance `.news`) — actual articles with working
    links, nothing generated. yfinance has shipped two schemas (flat, and nested under 'content'),
    so parse both defensively. Returns [{title, publisher, link, when}]; [] if unavailable."""
    try:
        import yfinance as yf
        raw = yf.Ticker(ticker).news or []
    except Exception:
        return []
    import datetime as _dt
    out = []
    for it in raw:
        if not isinstance(it, dict):
            continue
        c = it.get("content") if isinstance(it.get("content"), dict) else it
        title = c.get("title")
        if not title:
            continue
        link = c.get("link")
        for k in ("canonicalUrl", "clickThroughUrl"):
            u = c.get(k)
            if isinstance(u, dict) and u.get("url"):
                link = u["url"]; break
        publisher = c.get("publisher")
        prov = c.get("provider")
        if isinstance(prov, dict) and prov.get("displayName"):
            publisher = prov["displayName"]
        when = None
        ts = c.get("providerPublishTime")
        pd_ = c.get("pubDate") or c.get("displayTime")
        try:
            if ts:
                when = _dt.datetime.utcfromtimestamp(int(ts)).strftime("%b %d")
            elif pd_:
                when = _dt.datetime.fromisoformat(str(pd_).replace("Z", "+00:00")).strftime("%b %d")
        except Exception:
            when = None
        if link:
            out.append({"title": title, "publisher": publisher, "link": link, "when": when})
        if len(out) >= limit:
            break
    return out


# ── the analysis: SEC first (raw filings), Yahoo as automatic fallback ────────────
def analyze(ticker, price=None):
    """One ticker → structured fundamentals. `price` (live, from Tradier) powers valuation.
    Tries SEC EDGAR first (the authoritative filings); if SEC is unreachable/unknown, falls
    back to Yahoo. Returns a dict the app renders directly; never raises."""
    sec_err = None
    try:
        cik, name = ticker_to_cik(ticker)
        if cik:
            facts = company_facts(cik)
            m = _metrics_from_sec(facts, price)
            ends = _recent_quarter_ends(facts, want=3)
            m["quarters"] = _quarters(facts, _quarter_end_prices(ticker, ends))
            m["fwd_pe"] = _fwd_pe(ticker)
            flags, groups = _assess(m)
            return {"ok": True, "error": None, "ticker": ticker.upper(), "company": name,
                    "cik": cik, "source": "SEC EDGAR — 10-K / 10-Q filings",
                    "profile": company_profile(ticker, cik),
                    "metrics": m, "groups": groups, "flags": flags}
        sec_err = f"{ticker.upper()} not in SEC's ticker list (US-listed filers only)."
    except Exception as e:
        sec_err = f"{type(e).__name__}: {e}"

    # SEC failed — fall back to Yahoo (same underlying filings, vendor-derived).
    m = _metrics_from_yf(ticker, price)
    if m:
        flags, groups = _assess(m)
        return {"ok": True, "error": None, "ticker": ticker.upper(),
                "company": ticker.upper(), "cik": None,
                "source": "Yahoo Finance — SEC unreachable, vendor-derived",
                "profile": company_profile(ticker, None),
                "metrics": m, "groups": groups, "flags": flags}

    return {"ok": False, "error": f"No fundamentals available. SEC: {sec_err} "
            "Yahoo fallback also returned nothing (foreign/ADR names may lack coverage)."}
