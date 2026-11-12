# OptionIntel — Worklog & Architecture Notes

Public app: **https://optionintel.app** · repo: `cybercurry/options-dashboard`
This repo is **PUBLIC and anonymous**. The hard rule (never put IBKR account data here) lives in
`CLAUDE.md` — read it first.

---

## Where we stand (updated 2026-11-12)

- **The live product is the static site** `site/index.html` (deployed by **Cloudflare Pages** to
  **optionintel.app**). It is built by **`build_json.py`** (engine `signals.py` → `tradier.py`),
  which writes `site/data/signals.json`.
- **`app.py`** is the **old Streamlit app, KEPT as a manual backup** (deploy: Streamlit Cloud,
  `options-dashboard-jay.streamlit.app`). It is not the live product but must keep working — it
  still imports `sheets.py`, so **do not delete `sheets.py`** even though the static build no longer
  uses it (the Google-sheet universe source was retired 2026-11-03; `wheel_universe.json` has no
  `source_url`, so the committed hand-curated lists are always used).
- **Refresh is driven SOLELY by cron-job.org** (every 10 min, 24/7) → GitHub `workflow_dispatch` →
  `refresh-optionintel.yml`. **No GitHub `schedule:` anywhere in this workflow** (GitHub's free cron
  is unreliable — it fired once in ~4h overnight on 2026-11-12).
- **Two freshness clocks** in `signals.json`: `generated_at` = the last **full Tradier scan**
  (prices/signals/chains/fundamentals); `market_at` = the last **tiles refresh** (indices/metals/
  crypto/FX/sector/market-stats). The UI's amber stale-alarm watches `market_at`.
- **Active GitHub workflows:** `refresh-optionintel.yml` (main, cron-job.org-driven),
  `log-iv.yml` (daily ATM-IV log → `data/iv_history.csv`, still GitHub cron `0 21 * * 1-5`),
  `refresh-fundamentals.yml` (daily → `fundamentals.json`, still GitHub cron `0 16 * * *`),
  `selftest.yml` (manual). *Removed 2026-11-12:* `scan-signals.yml` (legacy) and the `scheduler/`
  Cloudflare Worker dir.

---

## Data refresh — how it works (as of 2026-11-12)

**Pipeline (100% cloud — nothing runs on a local machine):**
1. A scheduler sends a `workflow_dispatch` to GitHub.
2. GitHub Actions runs `.github/workflows/refresh-optionintel.yml` → `build_json.py`
   (engine `signals.py`; data from Tradier + yfinance + FRED).
3. The fresh `signals.json` is force-pushed as a parent-less commit to the **`data` branch**
   (with `[skip ci]`) — keeps `main` small and decoupled from the Cloudflare build cap.
4. The site (`site/index.html`) fetches
   `https://raw.githubusercontent.com/cybercurry/options-dashboard/data/signals.json`
   (cache-busted), falling back to the bundled `site/data/signals.json` if raw is briefly down.

**Scheduler — the 10-minute heartbeat (cron-job.org is the ONLY trigger):**
- **cron-job.org** job "Optionintel 10 min scheduler":
  - Every 10 min, **all hours, all days (24/7)**, timezone **America/New_York** (DST-aware).
  - Crontab: `*/10 * * * *`
  - `POST` to `…/actions/workflows/refresh-optionintel.yml/dispatches`, body `{"ref":"main"}`.
  - Auth header: `Authorization: Bearer <optionintel-cron PAT>` — a **fine-grained** token
    scoped to **this repo only**, **Actions: Read & write**. (Token value is NOT stored here.)
- The workflow has **no GitHub `schedule:`** — GitHub's free cron throttles/skips runs badly (it
  fired once in ~4h overnight), so it is not used. cron-job.org is the sole heartbeat. The DST-aware
  **ET gate** routes each fire: weekday **04:00–20:00 ET → full Tradier scan**; everything else
  (nights, weekends, pre/post-market) **→ light tiles-only refresh**. `concurrency: optionintel-data`
  prevents overlapping runs.

**Device-independent:** laptop off/asleep = refresh still runs.

**24/7 market tiles (light refresh — off-hours):**
- The indices / metals / crypto / FX pulse, the sector heatmap and the Market-tab stats come from
  **yfinance / CoinGecko / FRED — never Tradier** — so they can refresh around the clock.
- **Off-hours** (outside 04:00–20:00 ET weekdays), every workflow fire runs `build_json.py --light`:
  it seeds the base from the `data` branch (the last full scan), re-pulls ONLY those tiles, and
  **preserves `generated_at`** — so the Signals/Overview Tradier data is kept exactly as the last
  market-hours scan left it (the Signals tab still shows the last read to plan the open), while the
  tiles move. The light build stamps its own `market_at`; the front-end re-renders open tabs when
  either `generated_at` (full scan) **or** `market_at` (tiles) changes.
- Cadence is the **same 24/7 cron-job.org heartbeat** (`*/10 * * * *`) — no second job needed. The
  gate auto-routes each fire by time, so off-hours fires land as light refreshes and in-hours fires
  as full scans. (An explicit `{"inputs":{"light":"true"}}` dispatch forces light and is honored only
  off-hours; it's optional and unused now that the main job runs 24/7.)

**Weekend (calendar flip + crypto) — now covered by the 24/7 job:**
- Since the main cron-job.org job runs `*/10 * * * *` (24/7), **Saturday and Sunday get a light
  refresh every 10 min** — which re-pulls `fetch_market_stats` (the ForexFactory econ calendar) and
  crypto/metals tiles. So the weekend calendar + crypto stay current automatically.
- The ET gate still admits two **Sunday full-scan bands** — **02:00–02:29 ET** (just after
  ForexFactory's weekly feed rolls to the new trading week, ~00:00 ET Sun) and **18:00–18:29 ET**.
  (There is **no free `nextweek` feed** — `ff_calendar_nextweek.json`/`lastweek` return **404** —
  so the only way to show next week on the weekend is to refresh after the `thisweek` feed rolls.)
- **The old dedicated cron-job.org "Optionintel Sunday" job (`15 2,18 * * 0`, force=true) is now
  redundant** and can be deleted — the 24/7 job + the Sunday gate bands cover it. (Harmless if left.)

### Why this design (history — so we don't repeat mistakes)
- A **Cloudflare Worker** heartbeat was used before but **lost its runtime secrets on every
  deploy** (every push to `main` redeployed it via Workers Builds "Include paths = `*`"),
  silently killing the refresh. Retired.
- A **local Mac cron** was also in play → died whenever the laptop slept. Retired.
- **cron-job.org** is the bulletproof replacement: external, free, cloud, nothing to wipe.

### If the refresh ever stops
**Symptom:** the top-right freshness pill on optionintel.app turns **amber** with a ⚠ (fires when
`market_at` is >30 min old — the UI's built-in alarm).
1. Check GitHub Actions history for `refresh-optionintel.yml` — are there `workflow_dispatch`
   runs every ~10 min, 24/7? (If they stopped, cron-job.org isn't firing.)
2. If not, open the cron-job.org job → **Test run** (expect **HTTP 204**).
   `401/403` = token problem · `404` = URL/workflow name · `422` = missing/bad body (must be `{"ref":"main"}`).
   Also confirm the job's schedule is still `*/10 * * * *` (24/7) and the Bearer token + body survived
   the last edit.
3. Manual kick anytime:
   `gh api -X POST repos/cybercurry/options-dashboard/actions/workflows/refresh-optionintel.yml/dispatches -f ref=main`
   (add `-F 'inputs[light]=true'` to force a tiles-only refresh off-hours; `-F 'inputs[force]=true'`
   to force a full scan off-hours.)
4. If a run **succeeds but nothing publishes**, the **publish guard** refused a degenerate scan —
   look for a `::warning::Refusing to publish` annotation (Tradier likely returned empty). The last
   good data is intentionally kept; the next healthy scan republishes.

---

## Data sources & reliability (audited 2026-11-12)

Every external source has an explicit `requests` timeout (12–20s) and catches failures, returning
`None`/`[]`. The tile builder (`refresh_tiles`) **replaces** each key wholesale every run (never a
field-merge), so a failed fetch shows an honest blank "—" — **it can never silently show a stale
value**. In `--light` mode the Tradier block (`signals`/`overview`/`chains`/`fundamentals`/
`universe`) is deliberately preserved from the base with its original `generated_at`, so its age is
always visible.

| Source | Feeds | In light build? |
|---|---|---|
| **Tradier** (api/sandbox) | overview, signals, leaps, chains (prices/IV/greeks) | full scan only |
| **yfinance** | pulse (indices/metals/FX/BTC/ETH), sectors, VIX, HV/RSI history, earnings, fundamentals fallback | both (tiles 24/7) |
| **CoinGecko** | HYPE (`cg:hyperliquid`) | tiles 24/7 |
| **FRED** (keyless) | yields, Fed funds/target | tiles 24/7 |
| **CNN** (dataviz.cnn.io) | stocks Fear & Greed | tiles 24/7 |
| **alternative.me** | crypto Fear & Greed | tiles 24/7 |
| **ForexFactory** (faireconomy.media) | econ calendar (`thisweek`) | tiles 24/7 |
| **SEC EDGAR** | fundamentals | full scan + daily `build_fundamentals.py` |
| **historyofmarket.com → Finviz** | S&P fwd P/E (fallback chain) | tiles 24/7 |
| **Supabase** (`SUPABASE_SERVICE_KEY` env) | folds registered users' watchlist tickers into the scan | full scan only |
| **raw.githubusercontent** (`data` branch) | the light-build base + the site's live data fetch | — |

**Secrets:** `TRADIER_TOKEN`, `SUPABASE_SERVICE_KEY` (service_role — reads all users' watchlists,
unattributed), `ANTHROPIC_API_KEY` are env/Actions secrets, never committed. SEC User-Agent carries
a neutral contact email (required by SEC). *Open TODO (CLAUDE.md):* the git remote reportedly still
embeds a plaintext GitHub PAT — rotate + switch to SSH.

## Reliability hardening (2026-11-12)

Goal: the site is trustworthy **to its last displayed timestamp** — never fresh-looking-but-wrong.
1. **Publish guard** (`refresh-optionintel.yml`, "Guard" step) — a full scan that returns an **empty
   or >50%-collapsed** overview vs the last-good copy (e.g. a Tradier outage) is **refused**; the
   last good `data` branch is kept (with its honest older timestamp) and a `::warning::` is logged.
   Light refreshes always publish (Tradier data preserved). Prevents blanking the site at a fresh stamp.
2. **Amber stale-alarm** (`index.html`, `updateFreshness()`) — the freshness pill flips **amber + ⚠**
   when `market_at` is >30 min old (heartbeat expected every ~10 min). Re-checked every minute. Keyed
   on `market_at`, NOT `generated_at` (which is meant to sit old off-hours).
3. **yfinance timeout** (`build_json.py`) — `socket.setdefaulttimeout(30)`; yfinance takes no
   `timeout=`, so this bounds a Yahoo hang. The `requests` calls keep their own tighter timeouts.
4. **SEC guard** (`build_fundamentals._cik_map`) — returns `{}` on a SEC failure instead of raising,
   so the daily fundamentals script hits its `_MIN_OK` guard and leaves the good file untouched.

## Recent work log
- **2026-11-12** — **24/7 freshness + reliability pass.** (1) Added a `build_json.py --light` mode
  (tiles-only, preserves the Tradier scan) and routed off-hours fires to it via the ET gate.
  (2) **Removed the GitHub `schedule:`** from `refresh-optionintel.yml` — cron-job.org is now the
  sole trigger; widened its job to `*/10 * * * *` (24/7) so indices/metals/crypto refresh every
  10 min around the clock (GitHub's cron had frozen data ~1h40m overnight). (3) Hardening: publish
  guard, amber stale-alarm (`market_at`), yfinance socket timeout, `build_fundamentals` SEC guard.
  (4) Cleanup: deleted the retired `scheduler/` Cloudflare Worker and the legacy `scan-signals.yml`
  + `scripts/scan_signals.py` + stale root `data/signals.json`. (5) Audited all 11 external sources.
  Overview tables fitted to content (less wasted width); sector tiles show their ETF ticker; pulse
  regrouped (indices / commodities / crypto / forex) incl. Hang Seng + Silver.
- **2026-10-18** — Added a **Sunday refresh** (2× — 02:15 + 18:15 ET) so the economic calendar
  shows the new trading week over the weekend and crypto stays current. ForexFactory's `thisweek`
  feed rolls to the new week ~00:00 ET Sun; there is **no free `nextweek` feed** (confirmed 404),
  so we refresh just after the roll. Driven by a new **cron-job.org** Sunday job (forced dispatch,
  body `{"ref":"main","inputs":{"force":"true"}}`); the workflow gate also gained two Sunday ET
  bands (02:00–02:29, 18:00–18:29) as a GitHub-native fallback. (commit `1b44357`)
- **2026-10-07** — Refresh heartbeat moved to **cron-job.org** (10-min, device-independent,
  single engine). Retired the Cloudflare worker and an interim hourly Claude Routine. Deleted
  2 stale GitHub PATs; `optionintel-cron` is the live one.
- **2026-10-02** — Overview: every data column click-to-sort; new **IV Rank (IVR)** column from
  `data/iv_history.csv`; **next-earnings** column + ⚐ marker on chain expiries after earnings;
  cyan ticker-entry box. Options-chain ladder fitted to width (OI column visible, no sideways
  scroll). Fed tile → FRED target range (`DFEDTARL`/`DFEDTARU`). (commits `15342bd`, `1532cf2`)

---

## Related (separate, PRIVATE) project
The IBKR portfolio tracker lives in the **private** `cybercurry/ibkr-tracker` repo with its own
refresh (IBKR Flex → private Google Sheet). It is **never** connected to this public app and holds
no data here. Keep the two in **separate single-repo sessions**.
