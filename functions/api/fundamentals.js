// Cloudflare Pages Function — on-demand SEC fundamentals for ANY US-listed ticker.
// Mirrors fundamentals.py (SEC EDGAR XBRL, 10-K/10-Q) exactly, so the tab can research the
// off-index small/mid caps a baked list never covers. Returns the same JSON the app renders.
//
// SEC requires a descriptive User-Agent with a contact email (else 403). Edge-caches the ticker
// map (1 day) and companyfacts (1 h) so per-viewer lookups stay well within SEC fair-access.

const UA = "options-dashboard/1.0 (contact: options-dashboard@proton.me)";

const T = { pe_rich: 40, pe_high: 60, fcfy_thin: 0.02, peg_high: 3.0, netmargin_thin: 0.05,
  roe_thin: 0.08, de_high: 1.0, de_danger: 2.0, cr_thin: 1.5, cr_danger: 1.0,
  cover_thin: 6, cover_danger: 3, dilution: 0.05 };

const C = {
  revenue: ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "SalesRevenueNet",
            "RevenueFromContractWithCustomerIncludingAssessedTax"],
  net_income: ["NetIncomeLoss"], gross_profit: ["GrossProfit"], op_income: ["OperatingIncomeLoss"],
  equity: ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
  cur_assets: ["AssetsCurrent"], cur_liab: ["LiabilitiesCurrent"],
  lt_debt: ["LongTermDebtNoncurrent", "LongTermDebt"], debt_cur: ["LongTermDebtCurrent", "DebtCurrent"],
  cfo: ["NetCashProvidedByUsedInOperatingActivities", "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"],
  capex: ["PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets"],
  rnd: ["ResearchAndDevelopmentExpense", "ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost"],
  interest: ["InterestExpense", "InterestExpenseDebt"], eps: ["EarningsPerShareDiluted", "EarningsPerShareBasic"],
  sh_diluted: ["WeightedAverageNumberOfDilutedSharesOutstanding"],
  shares_out: ["CommonStockSharesOutstanding", "EntityCommonStockSharesOutstanding"],
};

async function secGet(url, ttl) {
  const r = await fetch(url, {
    headers: { "User-Agent": UA, "Accept-Encoding": "gzip, deflate", "Accept": "application/json, text/plain, */*" },
    cf: { cacheTtl: ttl, cacheEverything: true },
  });
  if (!r.ok) throw new Error("SEC " + r.status);
  return await r.json();
}

function rows(facts, concepts) {
  const gaap = (facts.facts && facts.facts["us-gaap"]) || {};
  const dei = (facts.facts && facts.facts.dei) || {};
  for (const c of concepts) {
    const node = gaap[c] || dei[c];
    if (!node) continue;
    const units = node.units || {};
    let rws = null;
    for (const uk of ["USD", "shares", "USD/shares"]) { if (units[uk]) { rws = units[uk]; break; } }
    if (rws === null) { const vals = Object.values(units); rws = vals.length ? vals[0] : []; }
    if (rws && rws.length) return rws;
  }
  return [];
}

function annualLastTwo(facts, concepts) {
  const byEnd = {};
  for (const r of rows(facts, concepts)) {
    if (r.form !== "10-K" || (r.fp && r.fp !== "FY")) continue;
    const end = r.end, val = r.val, filed = r.filed || "";
    if (end == null || val == null) continue;
    if (!(end in byEnd) || filed > byEnd[end][1]) byEnd[end] = [Number(val), filed];
  }
  const ends = Object.keys(byEnd).sort();
  if (!ends.length) return [null, null];
  const latest = byEnd[ends[ends.length - 1]][0];
  const prior = ends.length >= 2 ? byEnd[ends[ends.length - 2]][0] : null;
  return [latest, prior];
}

function latestPoint(facts, concepts) {
  const byEnd = {};
  for (const r of rows(facts, concepts)) {
    const end = r.end, val = r.val, filed = r.filed || "";
    if (end == null || val == null) continue;
    if (!(end in byEnd) || filed > byEnd[end][1]) byEnd[end] = [Number(val), filed];
  }
  const ends = Object.keys(byEnd).sort();
  return ends.length ? byEnd[ends[ends.length - 1]][0] : null;
}

const div = (a, b) => (a != null && b != null && b !== 0) ? a / b : null;

function metricsFromSec(facts, price) {
  const [rev, rev_p] = annualLastTwo(facts, C.revenue);
  const [ni, ni_p] = annualLastTwo(facts, C.net_income);
  const [gp] = annualLastTwo(facts, C.gross_profit);
  const [opi] = annualLastTwo(facts, C.op_income);
  const [cfo] = annualLastTwo(facts, C.cfo);
  const [capex] = annualLastTwo(facts, C.capex);
  const [rnd] = annualLastTwo(facts, C.rnd);
  const [intr] = annualLastTwo(facts, C.interest);
  const [eps] = annualLastTwo(facts, C.eps);
  const [shd, shd_p] = annualLastTwo(facts, C.sh_diluted);
  const equity = latestPoint(facts, C.equity), cur_a = latestPoint(facts, C.cur_assets), cur_l = latestPoint(facts, C.cur_liab);
  const lt_debt = latestPoint(facts, C.lt_debt), debt_cur = latestPoint(facts, C.debt_cur), shares = latestPoint(facts, C.shares_out);
  let total_debt = null;
  if (lt_debt != null || debt_cur != null) total_debt = (lt_debt || 0) + (debt_cur || 0);
  const fcf = (cfo != null && capex != null) ? (cfo - capex) : null;
  const m = {
    revenue: rev, revenue_prev: rev_p, net_income: ni, net_income_prev: ni_p,
    rev_growth: (rev != null && rev_p) ? div(rev - rev_p, rev_p) : null,
    ni_growth: (ni != null && ni_p) ? div(ni - ni_p, Math.abs(ni_p)) : null,
    gross_margin: div(gp, rev), op_margin: div(opi, rev), net_margin: div(ni, rev),
    roe: div(ni, equity), fcf: fcf, fcf_margin: div(fcf, rev),
    debt_to_equity: div(total_debt, equity), current_ratio: div(cur_a, cur_l),
    interest_coverage: div(opi, intr), eps: eps, shares: shares,
    share_change: (shd != null && shd_p) ? div(shd - shd_p, shd_p) : null, capex: capex, rnd: rnd,
  };
  m.market_cap = (price && shares) ? price * shares : null;
  m.pe = (price && eps && eps > 0) ? div(price, eps) : null;
  m.fcf_yield = div(fcf, m.market_cap);
  m.peg = (m.pe && m.ni_growth && m.ni_growth > 0) ? (m.pe / (m.ni_growth * 100)) : null;
  return m;
}

function assess(m) {
  const flags = [];
  const flag = (sev, text) => flags.push({ sev, text });
  const pct = v => (v * 100 >= 0 ? "+" : "") + (v * 100).toFixed(0);
  if (m.net_margin != null && m.net_margin < 0) flag("🔴", "Unprofitable — negative net margin");
  if (m.fcf != null && m.fcf < 0) flag("🔴", "Burning cash — negative free cash flow");
  if (m.debt_to_equity != null && m.debt_to_equity > T.de_danger) flag("🔴", `High leverage — debt/equity ${m.debt_to_equity.toFixed(1)}×`);
  if (m.current_ratio != null && m.current_ratio < T.cr_danger) flag("🔴", `Liquidity tight — current ratio ${m.current_ratio.toFixed(2)}`);
  if (m.interest_coverage != null && m.interest_coverage < T.cover_danger) flag("🔴", `Thin interest coverage — ${m.interest_coverage.toFixed(1)}×`);
  if (m.rev_growth != null && m.rev_growth < 0) flag("🟡", `Revenue shrinking — ${pct(m.rev_growth)}% YoY`);
  if (m.roe != null && m.roe < 0) flag("🟡", "Negative return on equity");
  if (m.share_change != null && m.share_change > T.dilution) flag("🟡", `Dilution — shares ${pct(m.share_change)}% YoY`);
  if (m.pe != null && m.pe > T.pe_rich) flag("🟡", `Rich valuation — P/E ${m.pe.toFixed(0)}`);
  const verdict = (reds, ambers) => reds.some(Boolean) ? "🔴" : (ambers.some(Boolean) ? "🟡" : "🟢");
  const groups = {
    Valuation: { verdict: verdict(
      [m.pe != null && m.pe > T.pe_high, m.fcf_yield != null && m.fcf_yield < 0],
      [m.pe != null && m.pe > T.pe_rich, m.fcf_yield != null && m.fcf_yield < T.fcfy_thin, m.peg != null && m.peg > T.peg_high]) },
    Quality: { verdict: verdict(
      [m.net_margin != null && m.net_margin < 0, m.roe != null && m.roe < 0],
      [m.net_margin != null && m.net_margin < T.netmargin_thin, m.roe != null && m.roe < T.roe_thin]) },
    Health: { verdict: verdict(
      [m.debt_to_equity != null && m.debt_to_equity > T.de_danger, m.current_ratio != null && m.current_ratio < T.cr_danger, m.interest_coverage != null && m.interest_coverage < T.cover_danger, m.fcf != null && m.fcf < 0],
      [m.debt_to_equity != null && m.debt_to_equity > T.de_high, m.current_ratio != null && m.current_ratio < T.cr_thin, m.interest_coverage != null && m.interest_coverage < T.cover_thin]) },
  };
  if (m.pe == null && m.fcf_yield == null && m.peg == null) groups.Valuation.verdict = "⚪";
  if (m.net_margin == null && m.roe == null) groups.Quality.verdict = "⚪";
  if (m.debt_to_equity == null && m.current_ratio == null && m.fcf == null) groups.Health.verdict = "⚪";
  return { flags, groups };
}

async function priceOf(ticker) {
  try {
    const r = await fetch(`https://query1.finance.yahoo.com/v8/finance/chart/${encodeURIComponent(ticker)}?interval=1d&range=1d`,
      { headers: { "User-Agent": UA }, cf: { cacheTtl: 600, cacheEverything: true } });
    if (!r.ok) return null;
    const j = await r.json();
    const p = j?.chart?.result?.[0]?.meta?.regularMarketPrice;
    return (typeof p === "number") ? p : null;
  } catch (e) { return null; }
}

const J = (obj, status = 200) => new Response(JSON.stringify(obj),
  { status, headers: { "content-type": "application/json", "access-control-allow-origin": "*", "cache-control": "public, max-age=1800" } });

export async function onRequestGet(context) {
  const tk = (new URL(context.request.url).searchParams.get("t") || "").trim().toUpperCase();
  if (!tk) return J({ ok: false, error: "no ticker" }, 400);
  try {
    const map = await secGet("https://www.sec.gov/files/company_tickers.json", 86400);
    let cik = null, company = tk;
    for (const k in map) {
      const row = map[k];
      if (String(row.ticker || "").toUpperCase() === tk) { cik = String(row.cik_str).padStart(10, "0"); company = row.title || tk; break; }
    }
    if (!cik) return J({ ok: false, ticker: tk, error: `${tk} not in SEC's ticker list (US-listed filers only).` });
    const facts = await secGet(`https://data.sec.gov/api/xbrl/companyfacts/CIK${cik}.json`, 3600);
    const price = await priceOf(tk);
    const m = metricsFromSec(facts, price);
    const { flags, groups } = assess(m);
    let sic = null;
    try { const sub = await secGet(`https://data.sec.gov/submissions/CIK${cik}.json`, 86400); sic = sub.sicDescription || null; } catch (e) {}
    return J({ ok: true, error: null, ticker: tk, company, cik, source: "SEC EDGAR — 10-K / 10-Q filings",
      profile: sic ? { sic, industry: sic } : {}, metrics: m, groups, flags, price });
  } catch (e) {
    return J({ ok: false, ticker: tk, error: "SEC lookup failed: " + (e && e.message || e) });
  }
}

export { rows, annualLastTwo, latestPoint, metricsFromSec, assess };
