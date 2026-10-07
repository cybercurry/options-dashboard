# OptionIntel — Worklog & Architecture Notes

Public app: **https://optionintel.app** · repo: `cybercurry/options-dashboard`
This repo is **PUBLIC and anonymous**. The hard rule (never put IBKR account data here) lives in
`CLAUDE.md` — read it first.

---

## Data refresh — how it works (as of 2026-10-07)

**Pipeline (100% cloud — nothing runs on a local machine):**
1. A scheduler sends a `workflow_dispatch` to GitHub.
2. GitHub Actions runs `.github/workflows/refresh-optionintel.yml` → `build_json.py`
   (engine `signals.py`; data from Tradier + yfinance + FRED).
3. The fresh `signals.json` is force-pushed as a parent-less commit to the **`data` branch**
   (with `[skip ci]`) — keeps `main` small and decoupled from the Cloudflare build cap.
4. The site (`site/index.html`) fetches
   `https://raw.githubusercontent.com/cybercurry/options-dashboard/data/signals.json`
   (cache-busted), falling back to the bundled `site/data/signals.json` if raw is briefly down.

**Scheduler — the 10-minute heartbeat (single source of truth):**
- **cron-job.org** job "Optionintel 10 min scheduler":
  - Every 10 min, hours **7–18**, **Mon–Fri**, timezone **America/New_York** (DST-aware).
  - Crontab: `*/10 7-18 * * 1-5`
  - `POST` to `…/actions/workflows/refresh-optionintel.yml/dispatches`, body `{"ref":"main"}`.
  - Auth header: `Authorization: Bearer <optionintel-cron PAT>` — a **fine-grained** token
    scoped to **this repo only**, **Actions: Read & write**. (Token value is NOT stored here.)
- The workflow also has a built-in `schedule: */10` (GitHub-native) as a sparse passive
  fallback, plus its own DST-aware **ET gate (04:00–20:00 ET)** that safely no-ops any
  off-window fire. `concurrency: optionintel-data` prevents overlapping runs.

**Device-independent:** laptop off/asleep = refresh still runs.

### Why this design (history — so we don't repeat mistakes)
- A **Cloudflare Worker** heartbeat was used before but **lost its runtime secrets on every
  deploy** (every push to `main` redeployed it via Workers Builds "Include paths = `*`"),
  silently killing the refresh. Retired.
- A **local Mac cron** was also in play → died whenever the laptop slept. Retired.
- **cron-job.org** is the bulletproof replacement: external, free, cloud, nothing to wipe.

### If the refresh ever stops
1. Check GitHub Actions history for `refresh-optionintel.yml` — are there `workflow_dispatch`
   runs every ~10 min during 7–18 ET weekdays?
2. If not, open the cron-job.org job → **Test run** (expect **HTTP 204**).
   `401/403` = token problem · `404` = URL/workflow name · `422` = missing/bad body (must be `{"ref":"main"}`).
3. Manual kick anytime:
   `gh api -X POST repos/cybercurry/options-dashboard/actions/workflows/refresh-optionintel.yml/dispatches -f ref=main`

---

## Recent work log
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
