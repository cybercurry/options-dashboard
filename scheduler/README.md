# Refresh heartbeat — Cloudflare Worker

Reliable 30-min heartbeat that dispatches the `refresh-optionintel.yml` GitHub workflow, replacing
GitHub's own cron (which throttles/skips `*/30`). Timing is enforced by the workflow's DST-aware ET
gate; this Worker also checks the window so it doesn't spend Actions minutes off-hours.

**Window:** every 30 min, **08:30–17:00 ET** (market open −1h … close +1h), **every day** (incl.
weekends — keeps 24/7 BTC/ETH quotes and the data-pull stamp current), EDT/EST-aware.

## One-time setup (dashboard, ~10 min)

### 1. GitHub token (fine-grained PAT)
GitHub → Settings → Developer settings → **Fine-grained tokens** → Generate:
- **Repository access:** Only select repositories → `cybercurry/options-dashboard`
- **Permissions:** Repository permissions → **Actions: Read and write** (nothing else)
- Generate and copy the `github_pat_…` value.

### 2. Create the Worker
Cloudflare dashboard → **Workers & Pages** → **Create** → **Worker** → name it
`optionintel-refresh-scheduler` → **Deploy** → **Edit code** → paste the contents of
[`refresh-scheduler.js`](./refresh-scheduler.js) → **Deploy**.

### 3. Secrets
Worker → **Settings** → **Variables and Secrets** → add two **Secrets**:
- `GH_TOKEN` = the `github_pat_…` from step 1
- `REFRESH_KEY` = any random string (guards the manual test URL)

### 4. Cron trigger
Worker → **Settings** → **Triggers** → **Cron Triggers** → **Add** → `0,30 12-22 * * *`.
(UTC. The band covers the ET window in both seasons; the code refines the exact 08:30/17:00 edges.)

### 5. Test
Visit `https://optionintel-refresh-scheduler.<your-subdomain>.workers.dev/dispatch?key=<REFRESH_KEY>`
→ expect `dispatched -> GitHub status 204`. A **Refresh optionintel data** run should appear in the
repo's Actions tab within a few seconds (the test forces a run regardless of the window).

## CLI alternative (if you use wrangler)
```
cd scheduler
wrangler deploy
wrangler secret put GH_TOKEN       # paste the PAT
wrangler secret put REFRESH_KEY    # paste a random string
```
The cron trigger comes from `wrangler.toml`.

## Notes
- No secrets live in this repo — the token is a Worker secret.
- The workflow re-checks the ET window (defense-in-depth) and also runs on any code `push` and on a
  manual dispatch with `force: true`.
- To rotate the token: regenerate the PAT and update the `GH_TOKEN` secret; nothing else changes.
