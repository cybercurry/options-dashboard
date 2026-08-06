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

import requests

# SEC asks for a UA that identifies the app and a contact. No secret, no PII.
_HEADERS = {"User-Agent": "options-dashboard (github.com/cybercurry/options-dashboard)"}
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


def ticker_to_cik(ticker):
    """Map a ticker to its zero-padded 10-digit CIK via SEC's public map."""
    data = _get("https://www.sec.gov/files/company_tickers.json")
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


# ── the analysis ─────────────────────────────────────────────────────────────────
def analyze(ticker, price=None):
    """One ticker → structured fundamentals. `price` (live, from Tradier) powers valuation.
    Returns a dict the app renders directly; never raises — errors come back in `error`."""
    try:
        cik, name = ticker_to_cik(ticker)
        if not cik:
            return {"ok": False, "error": f"{ticker.upper()} not found in SEC's filer list "
                    "(US-listed filers only — foreign/ADR names may be absent)."}
        facts = company_facts(cik)
    except Exception as e:
        return {"ok": False, "error": f"SEC fetch failed — {type(e).__name__}: {e}"}

    rev,  rev_p  = _annual_last_two(facts, _C["revenue"])
    ni,   ni_p   = _annual_last_two(facts, _C["net_income"])
    gp,   _      = _annual_last_two(facts, _C["gross_profit"])
    opi,  _      = _annual_last_two(facts, _C["op_income"])
    cfo,  _      = _annual_last_two(facts, _C["cfo"])
    capex, _     = _annual_last_two(facts, _C["capex"])
    intr, _      = _annual_last_two(facts, _C["interest"])
    eps,  _      = _annual_last_two(facts, _C["eps"])
    shd,  shd_p  = _annual_last_two(facts, _C["sh_diluted"])

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

    m = {
        "revenue": rev, "revenue_prev": rev_p,
        "net_income": ni, "net_income_prev": ni_p,
        "rev_growth": _div(rev - rev_p, rev_p) if (rev is not None and rev_p) else None,
        "ni_growth":  _div(ni - ni_p, abs(ni_p)) if (ni is not None and ni_p) else None,
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
    }
    # valuation needs a live price
    m["market_cap"] = (price * shares) if (price and shares) else None
    m["pe"] = _div(price, eps) if (price and eps and eps > 0) else None
    m["fcf_yield"] = _div(fcf, m["market_cap"])
    m["peg"] = (m["pe"] / (m["ni_growth"] * 100)) if (m["pe"] and m["ni_growth"] and m["ni_growth"] > 0) else None

    flags = []
    def flag(sev, text): flags.append({"sev": sev, "text": text})

    # ── red-flag detection ──
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

    # ── group verdicts (worst signal in the group wins) ──
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
    # A group with no data at all shows a dash, not a false green.
    if m["pe"] is None and m["fcf_yield"] is None and m["peg"] is None:
        groups["Valuation"]["verdict"] = "⚪"
    if m["net_margin"] is None and m["roe"] is None:
        groups["Quality"]["verdict"] = "⚪"
    if m["debt_to_equity"] is None and m["current_ratio"] is None and m["fcf"] is None:
        groups["Health"]["verdict"] = "⚪"

    return {"ok": True, "error": None, "ticker": ticker.upper(), "company": name,
            "cik": cik, "metrics": m, "groups": groups, "flags": flags}
