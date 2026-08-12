# CLAUDE.md — options-dashboard (OI Dashboard)

**READ THIS FIRST. This repo has a hard privacy boundary.**

---

## ⚠️ THIS IS THE PUBLIC APP

This is the **Options Intelligence (OI) Dashboard** — a **PUBLIC** Streamlit app,
deployed to **https://options-dashboard-jay.streamlit.app** (anyone with the link
can view it, and its published-sheet CSV is a public URL).

### The one rule that overrides everything

**The OI dashboard must NEVER contain, read, display, or hint at the owner's
IBKR account data** — account numbers, held positions, open trades, share/lot
counts, cost basis, P&L, NLV, free-to-write capacity, or anything derived from
the brokerage account. It is **anonymous** and must stay that way.

If a task would put any of that on this app, **STOP** — it belongs in the private
`~/ibkr-tracker` repo instead. When unsure, treat it as private and ask.

**Incident (2026-08-12):** a prior session added a "📌 Positions" tab that read an
open-positions block from the published sheet and displayed it. Nothing had leaked
(the sheet had no positions block yet), but it was removed as a security fix. **Do
NOT re-add any positions / holdings / trade-management / "import my IBKR" view
here.** That capability lives only in the private tracker.

---

## The two-repo boundary (do not blur these)

| | `~/options-dashboard` (THIS repo) | `~/ibkr-tracker` |
|---|---|---|
| Audience | **PUBLIC** — shareable, anonymous | **PRIVATE** — owner only |
| Purpose | Options research / education on a hand-picked watchlist | Plan & manage the owner's real IBKR portfolio |
| Data in | Tradier + yfinance + a **published watchlist** CSV (tickers only) | IBKR Flex (positions, trades, cash) → private Google Sheet |
| Contains IBKR account data? | **NEVER** | Yes, by design |
| Deploy | Streamlit Cloud (public) | Local cron + private Sheet + GitHub Action |

They may share **engine code** (rules + market data), but **never a live data link
and never account data**. The owner jumps between the two from the same machine —
keep them cleanly separated.

- Watchlist tickers are **hand-curated** (watchlist.json / sidebar / the published
  sheet's WHEEL+GROWTH columns). They must **never** auto-populate from IBKR
  holdings — that would make the public watchlist equal the portfolio.

---

## What this app is (quick orient)

Single-file `app.py` + `tradier.py` (data) + `signals.py` (wheel engine) +
`sheets.py` (published-CSV **watchlist** reader — tickers only, no positions) +
`Selection_Trade_Criteria_v1.md` (rules spec) + `wheel_universe.json` +
`watchlist.json`.

- **Data layer:** Tradier-first (real quotes/chains/IV/greeks via `tradier.py`);
  yfinance is the fallback (crypto, `^`-indices, misses). Black-Scholes
  (`_bs_greeks`) survives only as a last-resort greek fallback.
- **Tabs:** Overview · Deep Dive · Options Chain · 📊 Market Stats (yield curve +
  Fed rate via keyless FRED + this-week economic calendar) · 🎯 Signals · 🔬
  Fundamentals.

## Security TODO (still open)

The git remote embeds a **plaintext GitHub PAT** (`ghp_…` in `git remote -v`).
Rotate it on GitHub and switch the remote to SSH
(`git@github.com:cybercurry/options-dashboard.git`). Do not echo the token.
