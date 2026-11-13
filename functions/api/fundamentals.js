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

// Trailing-twelve-months for a FLOW item: latest full fiscal year + current-year YTD − prior-year
// matching YTD. Degrades to the latest annual when no newer quarter is on file (slow filers
// unchanged). Mirrors fundamentals.py _ttm exactly so the live path and the baked JSON agree.
function ttm(facts, concepts) {
  const pdate = s => { const p = String(s).split("-"); const d = Date.UTC(+p[0], +p[1] - 1, +p[2]); return isNaN(d) ? null : d; };
  const DAY = 86400000;
  const days = (s, e) => { const ds = pdate(s), de = pdate(e); return (ds != null && de != null) ? Math.round((de - ds) / DAY) : 0; };
  const seen = {};
  for (const r of rows(facts, concepts)) {
    const s = r.start, e = r.end, v = r.val, f = r.filed || "";
    if (!(s && e && v != null)) continue;
    const k = s + "|" + e;
    if (!(k in seen) || f > seen[k][1]) seen[k] = [Number(v), f];
  }
  const items = Object.keys(seen).map(k => { const [s, e] = k.split("|"); return [s, e, seen[k][0]]; });
  if (!items.length) return null;
  const annuals = items.filter(([s, e]) => days(s, e) >= 350 && days(s, e) <= 380).sort((a, b) => a[1] < b[1] ? -1 : 1);
  if (!annuals.length) {                       // no annual → sum last 4 discrete quarters
    const q = items.filter(([s, e]) => days(s, e) >= 80 && days(s, e) <= 100).sort((a, b) => a[1] < b[1] ? -1 : 1);
    return q.length >= 4 ? q.slice(-4).reduce((t, x) => t + x[2], 0) : null;
  }
  const [, fy_e, fy_v] = annuals[annuals.length - 1];
  const fye = pdate(fy_e);
  const cur = items.filter(([s, e]) => { const pe = pdate(e); return pe != null && fye != null && pe > fye && (pe - fye) / DAY <= 370 && days(s, e) >= 60 && days(s, e) < 350; });
  if (!cur.length) return fy_v;
  const c = cur.reduce((best, x) => days(x[0], x[1]) > days(best[0], best[1]) ? x : best);
  const clen = days(c[0], c[1]), cend = pdate(c[1]);
  const pri = items.filter(([s, e]) => { const pe = pdate(e); return pe != null && Math.abs(days(s, e) - clen) <= 20 && (cend - pe) / DAY >= 330 && (cend - pe) / DAY <= 400; });
  if (!pri.length) return fy_v;
  const p = pri.reduce((best, x) => pdate(x[1]) > pdate(best[1]) ? x : best);
  return fy_v + c[2] - p[2];
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

// ── per-quarter series (mirrors fundamentals.py _discrete_quarters / _quarters) ──
function discreteQuarters(facts, concepts) {
  const pd = s => { const p = String(s).split("-"); return Date.UTC(+p[0], +p[1] - 1, +p[2]); };
  const DAY = 86400000;
  const days = (s, e) => Math.round((pd(e) - pd(s)) / DAY);
  const seen = {};
  for (const r of rows(facts, concepts)) {
    const s = r.start, e = r.end, v = r.val, f = r.filed || "";
    if (!(s && e && v != null)) continue;
    const k = s + "|" + e;
    if (!(k in seen) || f > seen[k][1]) seen[k] = [Number(v), f];
  }
  const fs = Object.keys(seen).map(k => { const [s, e] = k.split("|"); return [s, e, seen[k][0], days(s, e)]; }).filter(x => x[3] > 0);
  const disc = {};
  for (const [, e, v, d] of fs) if (d >= 80 && d <= 100) disc[e] = v;   // explicit quarters win
  const bystart = {};
  for (const [s, e, v, d] of fs) if (d <= 385) (bystart[s] = bystart[s] || []).push([e, v, d]);
  for (const s in bystart) {
    const lst = bystart[s].sort((a, b) => a[0] < b[0] ? -1 : 1);
    let pv = 0, pdv = 0;
    for (const [e, v, d] of lst) { const seg = d - pdv; if (!(e in disc) && seg >= 80 && seg <= 100) disc[e] = v - pv; pv = v; pdv = d; }
  }
  return disc;
}
function pointsByEnd(facts, concepts) {
  const seen = {};
  for (const r of rows(facts, concepts)) { const e = r.end, v = r.val, f = r.filed || ""; if (e == null || v == null) continue; if (!(e in seen) || f > seen[e][1]) seen[e] = [Number(v), f]; }
  const out = {}; for (const e in seen) out[e] = seen[e][0]; return out;
}
const QMON = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
function qLabel(end) { const p = String(end).split("-"); if (p.length < 3) return String(end); const mo = +p[1], q = Math.floor((mo - 1) / 3) + 1; return `Q${q} · ${QMON[mo]} ’${String(+p[0] % 100).padStart(2, "0")}`; }
function recentQuarterEnds(facts, want) {
  const rev = discreteQuarters(facts, C.revenue), ni = discreteQuarters(facts, C.net_income);
  return [...new Set([...Object.keys(rev), ...Object.keys(ni)])].filter(Boolean).sort().slice(-want);
}
function quarters(facts, prices, want = 3, lookback = 7) {
  prices = prices || {};
  const rev_d = discreteQuarters(facts, C.revenue), ni_d = discreteQuarters(facts, C.net_income),
    gp_d = discreteQuarters(facts, C.gross_profit), opi_d = discreteQuarters(facts, C.op_income),
    cfo_d = discreteQuarters(facts, C.cfo), cap_d = discreteQuarters(facts, C.capex),
    rnd_d = discreteQuarters(facts, C.rnd), eps_d = discreteQuarters(facts, C.eps);
  const eq_p = pointsByEnd(facts, C.equity), ca_p = pointsByEnd(facts, C.cur_assets), cl_p = pointsByEnd(facts, C.cur_liab),
    ltd_p = pointsByEnd(facts, C.lt_debt), dc_p = pointsByEnd(facts, C.debt_cur), sh_p = pointsByEnd(facts, C.shares_out);
  const ends = [...new Set([...Object.keys(rev_d), ...Object.keys(ni_d)])].filter(Boolean).sort();
  const recent = ends.slice(-lookback);
  const ttmNi = e => { const i = recent.indexOf(e); const legs = recent.slice(Math.max(0, i - 3), i + 1); const vals = legs.map(x => ni_d[x]); return (legs.length === 4 && vals.every(v => v != null)) ? vals.reduce((a, b) => a + b, 0) : null; };
  const atOrBefore = (pts, e) => { const c = Object.keys(pts).filter(d => d <= e).sort(); return c.length ? pts[c[c.length - 1]] : null; };
  const out = [];
  for (const e of recent.slice(-want)) {
    const rev = rev_d[e] ?? null, ni = ni_d[e] ?? null;
    const equity = atOrBefore(eq_p, e), shares = atOrBefore(sh_p, e), ca = atOrBefore(ca_p, e), cl = atOrBefore(cl_p, e), ltd = atOrBefore(ltd_p, e), dc = atOrBefore(dc_p, e);
    const total_debt = (ltd != null || dc != null) ? ((ltd || 0) + (dc || 0)) : null;
    const cfo = cfo_d[e], cap = cap_d[e];
    const fcf = (cfo != null && cap != null) ? (cfo - cap) : null;
    const tn = ttmNi(e), px = prices[e] ?? null;
    const mcap = (px && shares) ? px * shares : null;
    const pe = (mcap && tn && tn > 0) ? mcap / tn : null;
    out.push({
      label: qLabel(e), end: e, revenue: rev, net_income: ni, eps: eps_d[e] ?? null,
      gross_margin: div(gp_d[e], rev), op_margin: div(opi_d[e], rev), net_margin: div(ni, rev),
      fcf: fcf, fcf_margin: div(fcf, rev), capex: cap ?? null, rnd: rnd_d[e] ?? null,
      debt_to_equity: div(total_debt, equity), current_ratio: div(ca, cl),
      market_cap: mcap, pe: pe, price: px,
    });
  }
  return out;
}
// Quarter-end closes from the open Yahoo v8 chart (monthly bars) — one call, no crumb.
async function quarterEndPrices(ticker, ends) {
  const out = {};
  if (!ends || !ends.length) return out;
  try {
    const r = await fetch(`https://query1.finance.yahoo.com/v8/finance/chart/${encodeURIComponent(ticker)}?interval=1mo&range=3y`,
      { headers: { "User-Agent": UA }, cf: { cacheTtl: 86400, cacheEverything: true } });
    if (!r.ok) return out;
    const j = await r.json();
    const res = j && j.chart && j.chart.result && j.chart.result[0];
    if (!res) return out;
    const ts = res.timestamp || [], closes = (res.indicators && res.indicators.quote && res.indicators.quote[0] && res.indicators.quote[0].close) || [];
    const byYm = {};
    for (let i = 0; i < ts.length; i++) { const d = new Date(ts[i] * 1000); const ym = d.getUTCFullYear() + "-" + String(d.getUTCMonth() + 1).padStart(2, "0"); if (closes[i] != null) byYm[ym] = closes[i]; }
    for (const e of ends) { const ym = String(e).slice(0, 7); if (byYm[ym] != null) out[e] = byYm[ym]; }
  } catch (e) { return {}; }
  return out;
}
// Forward P/E is best-effort on the live path (Yahoo's estimate endpoint is crumb-gated and flaky);
// the authoritative source is yfinance in the server-side build. Degrades to null (shown n/a).
async function fwdPe(ticker) {
  try {
    const r = await fetch(`https://query1.finance.yahoo.com/v7/finance/quote?symbols=${encodeURIComponent(ticker)}`,
      { headers: { "User-Agent": UA }, cf: { cacheTtl: 3600, cacheEverything: true } });
    if (!r.ok) return null;
    const j = await r.json();
    const q = j && j.quoteResponse && j.quoteResponse.result && j.quoteResponse.result[0];
    const v = q && q.forwardPE;
    return (typeof v === "number" && v > 0) ? v : null;
  } catch (e) { return null; }
}

function metricsFromSec(facts, price) {
  // Annual latest/prior — drives clean full-year-vs-full-year growth
  const [rev_a, rev_ap] = annualLastTwo(facts, C.revenue);
  const [ni_a, ni_ap] = annualLastTwo(facts, C.net_income);
  const [shd, shd_p] = annualLastTwo(facts, C.sh_diluted);
  // TTM for the point metrics (P/E, EPS, margins, FCF); fall back to the annual figure
  const rev = ttm(facts, C.revenue) ?? rev_a;
  const ni = ttm(facts, C.net_income) ?? ni_a;
  const gp = ttm(facts, C.gross_profit) ?? annualLastTwo(facts, C.gross_profit)[0];
  const opi = ttm(facts, C.op_income) ?? annualLastTwo(facts, C.op_income)[0];
  const cfo = ttm(facts, C.cfo) ?? annualLastTwo(facts, C.cfo)[0];
  const capex = ttm(facts, C.capex) ?? annualLastTwo(facts, C.capex)[0];
  const rnd = ttm(facts, C.rnd) ?? annualLastTwo(facts, C.rnd)[0];
  const intr = ttm(facts, C.interest) ?? annualLastTwo(facts, C.interest)[0];
  const equity = latestPoint(facts, C.equity), cur_a = latestPoint(facts, C.cur_assets), cur_l = latestPoint(facts, C.cur_liab);
  const lt_debt = latestPoint(facts, C.lt_debt), debt_cur = latestPoint(facts, C.debt_cur), shares = latestPoint(facts, C.shares_out);
  let total_debt = null;
  if (lt_debt != null || debt_cur != null) total_debt = (lt_debt || 0) + (debt_cur || 0);
  const fcf = (cfo != null && capex != null) ? (cfo - capex) : null;
  // EPS from TTM net income ÷ current shares (robust against quarterly-EPS rounding drift)
  const eps = (ni != null && shares) ? div(ni, shares) : ttm(facts, C.eps);
  const m = {
    revenue: rev, revenue_prev: rev_a, net_income: ni, net_income_prev: ni_a,
    rev_growth: (rev_a != null && rev_ap) ? div(rev_a - rev_ap, rev_ap) : null,
    ni_growth: (ni_a != null && ni_ap) ? div(ni_a - ni_ap, Math.abs(ni_ap)) : null,
    gross_margin: div(gp, rev), op_margin: div(opi, rev), net_margin: div(ni, rev),
    roe: div(ni, equity), fcf: fcf, fcf_margin: div(fcf, rev),
    debt_to_equity: div(total_debt, equity), current_ratio: div(cur_a, cur_l),
    interest_coverage: div(opi, intr), eps: eps, shares: shares,
    share_change: (shd != null && shd_p) ? div(shd - shd_p, shd_p) : null, capex: capex, rnd: rnd,
  };
  m.market_cap = (price && shares) ? price * shares : null;
  m.pe = (m.market_cap && ni && ni > 0) ? div(m.market_cap, ni) : null;
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

// One Yahoo v8 chart call (the only open Yahoo endpoint — v7/v10 now require a crumb) gives us the
// live price AND the instrument metadata we use to explain non-filers (ETFs/ETNs, foreign/OTC).
async function assetMeta(ticker) {
  try {
    const r = await fetch(`https://query1.finance.yahoo.com/v8/finance/chart/${encodeURIComponent(ticker)}?interval=1d&range=1d`,
      { headers: { "User-Agent": UA }, cf: { cacheTtl: 600, cacheEverything: true } });
    if (!r.ok) return null;
    const j = await r.json();
    const meta = j?.chart?.result?.[0]?.meta;
    if (!meta) return null;
    const p = meta.regularMarketPrice;
    return {
      price: (typeof p === "number") ? p : null,
      type: meta.instrumentType || null,               // EQUITY | ETF | MUTUALFUND | CRYPTOCURRENCY | ...
      name: meta.shortName || meta.longName || null,
      exchange: meta.fullExchangeName || meta.exchangeName || null,
    };
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
    if (!cik) {
      // Not an SEC company filer. Use the Yahoo chart metadata to explain WHY, instead of a blank
      // "not US-listed" error: an ETF/ETN has no company filings by nature; a foreign/OTC name may
      // trade here yet file no US company reports. Always hand the user a working details link.
      const meta = await assetMeta(tk);
      const detailUrl = `https://finance.yahoo.com/quote/${encodeURIComponent(tk)}`;
      if (meta) {
        const fundLike = /^(ETF|MUTUALFUND)$/i.test(meta.type || "")
          || /\b(ETF|ETN|ETP|Fund|Trust)\b/i.test(meta.name || "");
        return J({
          ok: false, kind: fundLike ? "fund" : "nonfiler", ticker: tk,
          name: meta.name || tk, exchange: meta.exchange, price: meta.price, detailUrl,
          error: fundLike ? "ETF/ETN — no company filings."
                          : "Listed, but files no SEC company reports (foreign listing or OTC).",
        });
      }
      return J({ ok: false, kind: "unknown", ticker: tk, detailUrl,
        error: `${tk} — no SEC filing and no market data found. Check the symbol.` });
    }
    const facts = await secGet(`https://data.sec.gov/api/xbrl/companyfacts/CIK${cik}.json`, 3600);
    const price = (await assetMeta(tk))?.price ?? null;
    const m = metricsFromSec(facts, price);
    const qEnds = recentQuarterEnds(facts, 3);
    const [qPrices, fpe] = await Promise.all([quarterEndPrices(tk, qEnds), fwdPe(tk)]);
    m.quarters = quarters(facts, qPrices);
    m.fwd_pe = fpe;
    const { flags, groups } = assess(m);
    let sic = null;
    try { const sub = await secGet(`https://data.sec.gov/submissions/CIK${cik}.json`, 86400); sic = sub.sicDescription || null; } catch (e) {}
    return J({ ok: true, error: null, ticker: tk, company, cik, source: "SEC EDGAR — 10-K / 10-Q filings",
      profile: sic ? { sic, industry: sic } : {}, metrics: m, groups, flags, price });
  } catch (e) {
    return J({ ok: false, ticker: tk, error: "SEC lookup failed: " + (e && e.message || e) });
  }
}

export { rows, annualLastTwo, ttm, latestPoint, metricsFromSec, assess, discreteQuarters, quarters, recentQuarterEnds };
