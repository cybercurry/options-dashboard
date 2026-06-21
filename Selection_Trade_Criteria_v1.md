# Selection & Trade Criteria — v1 (draft for review)

Synthesizes the Handover doc (strategy rules, §5) and the 21 June session snapshot into the
specific, codeable rules the live dashboard should enforce. This is a working document —
expect edits as we go. Nothing here gets built into `app.py` until you sign off.

---

## 1. Role tag (per ticker) — the foundation everything else hangs on

Every watchlist name gets exactly one tag. This replaces the implicit assumption that one
signal set ("wheelable") fits all names.

| Role | Definition | Options behaviour |
|---|---|---|
| **Core Compounder** | A name you never want capped — the upside is the point | CSPs only, to accumulate at a price you'd be thrilled to own. **No covered calls, ever**, on this sleeve. |
| **Income Name** | Good business, fair price, not a 5x — you're fine owning it but not betting the farm on upside | True wheel candidate. CSP to enter, CC once assigned/owned — capping costs little. |
| **Split** | You want both the upside *and* the income | Position divided into a **core sleeve** (never capped) + a **separate wheel sleeve** (CSP/CC as normal). Track as two sub-positions, not one. |

**Proposed first pass for your 9 tickers** (draft — please correct, this is my read of the
handover/snapshot, not yours):

| Ticker | Proposed role | Why |
|---|---|---|
| NVDA | Core Compounder | Flagged repeatedly as the cleanest LEAP candidate; AI-infra core thesis |
| META | Core Compounder | Mega-cap compounder character |
| TSLA | Split | High-conviction upside but volatile enough to fund an income sleeve |
| IBIT | Core Compounder | Structural BTC exposure, thesis is appreciation not income |
| GLD | Core Compounder | Macro hedge / dry-powder proxy — capping it defeats its job |
| GDXJ | Income Name | High-vol miner basket, not a compounding thesis — good IV harvesting |
| BE | Income Name | Explicitly named in snapshot as a high-beta income play |
| VST | Split | Power/infra bottleneck name — real upside thesis, also volatile enough to wheel |
| CRWV | Income Name | Explicitly named in snapshot as high-beta income play; size puts conservatively (snapshot's caveat 2) |

---

## 2. Accumulation-phase entry rules (puts only — this is where you are now)

**LEAP entry (all three required):**
1. HV Rank < 25
2. RSI rising through 33–52, preferred trigger = RSI crosses above 50 on a daily close
3. Price above 200-day MA

**LEAP structure:** expiry Jan 2027 or later · delta 0.70–0.80 · roll at 9 months DTE.

**CSP (Wheel entry leg) — applies to Income Name and Split-sleeve tickers:**
- Target ~30 delta, ~30 DTE (21–45 day window)
- Strike = a price you'd be "thrilled to own," not just a delta target
- Elevated HV Rank → read as "premium is rich, sell here," **not** as a buy signal

**Hard rule carried from snapshot:** the call leg does not exist yet for Core Compounders.
Only the put leg belongs in accumulation phase, full stop.

---

## 3. Distribution-phase gate (covered calls) — currently NOT enforced in app.py

Current `cc_signal()` scores "above 50MA" as *safe to sell calls* — that's backwards from the
snapshot's directional-wheeling rule: **never sell calls into an uptrend.** Before this gets
coded, need your call on:

- What counts as "uptrend" for veto purposes — price structure (e.g. above rising 50MA) vs.
  a momentum read (RSI), vs. both?
- Does the veto apply per-name only to Income Name / Split-sleeve tickers (Core Compounders
  have no call leg regardless), or could it ever apply more broadly?

Until that's defined, CC scoring stays as-is but should be flagged in the UI as "legacy logic,
under revision" rather than presented as current strategy.

---

## 4. Premium-selling trigger — true IV Rank (this is a new data field, not relabeling)

**Important technical finding from the code:** `calc_iv_rank()` / `calc_iv_percentile()` in
`app.py` (lines 258–268) already exist, but they compute the rank/percentile of **HV20**
(historical/realized vol) — that's the "HV Rank" / "HV%ile" columns already on the Overview
table. That is *not* the same thing the snapshot is asking for.

What the snapshot wants is **true IV Rank**: where today's *implied* vol (ATM IV) sits versus
its own 52-week range. That's the number that tells you whether the premium you'd collect is
rich relative to recent option pricing — the actual vol-risk-premium signal, separate from how
much the stock has realized-moved.

**Constraint to flag now:** yfinance only returns the *current* option chain — there's no
historical ATM-IV time series to pull. Real IV Rank needs either (a) a paid historical-IV
source (CBOE DataShop, ORATS, etc.), or (b) we start recording daily ATM IV ourselves from
today forward and the "rank" only becomes meaningful after enough days accumulate (a 52-week
rank needs ~a year before it's real; even a 60–90 day rolling rank is informative sooner).
Option (b) is free and buildable now — flagging so expectations on "how reliable is this on
day one" are set correctly.

---

## 5. Sizing, reserve, and concentration — the actual foundation per the snapshot

These aren't optional dashboard nice-to-haves; the snapshot calls this "the foundation,
selection is the floor tile":

- **Correlation/concentration view:** the 9-name watchlist is ~one AI-capex macro bet. Needs a
  first-class view — e.g., % of total notional in AI-capex-correlated names (NVDA/META/TSLA/
  VST/CRWV/BE) vs. macro-hedge/diversifying names (GLD/IBIT/GDXJ).
- **Reserve/tranche status:** cash buffer tracked against the 2027 income need, so a drawdown
  never forces a sale of the core.
- **Put-sizing discipline on the high-beta sleeve** (CRWV, BE per snapshot) — these gap hardest
  in a correction; covered calls only on a *slice*, never the full position.

These need your input on actual numbers (position sizes, reserve target, concentration cap %)
before they can be built — I can scaffold the UI/fields once you give me the numbers, or
propose reasonable defaults if you'd rather I draft and you edit.

---

## 6. Dashboard changes — mapped to actual current tabs

Current tabs in `app.py`: **Overview, Deep Dive, Options Chain, VIX, ⚡ Screener**.

| Agreed change | Maps to | Note |
|---|---|---|
| Remove stochastics | ⚠️ Not found in Overview — stochastics (`calc_stochastics`, "G2") only appears in the **Screener** tab's 4-Gates logic. Please confirm: remove from Screener's gates, or is there a different stochastics display you're thinking of? |
| Remove "numbers-in-a-box" | ⚠️ Ambiguous — could be the 10-ticker Market Pulse metric row (Overview) or the 6-metric row in Deep Dive (Price/HVR/HVpct/RSI/ATM IV/PCR). Which one(s)? |
| Drop 50-day MA | Two locations: the "50MA ✅/❌" column in the Overview watchlist table, and the 50MA line on the Deep Dive price chart. Drop both, or just one? |
| Per-stock detail: 200MA + standard BB + RSI | Deep Dive tab — replace current 4-panel chart (Price+50MA+200MA / HV20+HV60 / RSI / BB Width) with Price+200MA+BB-bands-overlay / RSI. Drops the HV20/HV60 sub-panel and the BB-Width sub-panel in favor of literal bands on the price chart. |
| IV Rank/percentile as first-class field next to RSI | New field — see §4 on data constraint |
| New "News & Macro" tab | New tab — per-stock news headlines + macro context |
| Same tab: Event Calendar | Earnings dates + FOMC/CPI dates, so you check before setting a trade |
| Role tag + correlation/concentration view | New — likely lives on Overview or its own tab |

The flagged items (⚠️) are genuine ambiguities I didn't want to guess on before touching a
1,441-line file you've been iterating on for months — wrong guesses here cost you a confusing
diff to review, not just my time.

---

## 7. Suggested build order (brick by brick, per your stated preference)

1. Lock §1 role tags (your edits to my draft table)
2. Resolve the three ⚠️ ambiguities in §6 + the CC veto question in §3
3. Ship the Overview declutter + Deep Dive chart rework as one complete `app.py` rewrite
4. Add true IV Rank tracking (starts accumulating data from day one, useful within weeks)
5. News & Macro / Event Calendar tab
6. Correlation/concentration + reserve view (needs your numbers first)

Nothing in §3 or §6 gets coded until you've answered the open questions — that's the
"rules first, then build" rule.

---

## Status as of 21 June 2026 session

Shipped to `app.py` and live (pushed to GitHub main, auto-deployed):
- DEFAULT_WATCHLIST expanded from 9 to 31 tickers (merged/deduped list)
- `calc_bb_pctb()` added (Bollinger %B)
- `cc_signal()` and `csp_signal()` updated to take `pctb_val` (and `walking` for CSP) as new
  scoring inputs — near-upper-band rewarded for calls, near-lower-band-holding rewarded for
  puts, walking-the-lower-band penalized for puts
- ATR and BB Width columns dropped from the Overview watchlist table

Still open / not yet built — see §§3, 4, 5, 6 above. Nothing in those sections has been
touched in code; all open questions remain open.
