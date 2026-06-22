# Selection & Trade Criteria — v1 (draft for review)

Synthesizes the Handover doc (strategy rules, §5) and the 21 June session snapshot into the
specific, codeable rules the live dashboard should enforce. This is a working document —
expect edits as we go. Nothing here gets built into `app.py` until you sign off.

---

## 1. Role tag (per ticker) — REMOVED (24 June session)

**Status: dropped, not just deprioritized.** You flagged that you don't recall asking for this
and that it confuses the issues — fair, and worth being straight about where it came from: an
earlier session read a "Handover doc" and a prior session snapshot and proposed this as its own
interpretation, which the table below even labeled "draft — please correct, this is my read of
the handover/snapshot, not yours." It was never confirmed by you, and this session carried it
forward as still-open instead of just dropping it when it kept not getting confirmed — that's
on the process, not on you for not having flagged it sooner. It's removed now: no role tag,
anywhere. Where this was going to gate the new CC table (§9.4/§9.8), that gate is replaced by
the technical mean-reversion trigger in §10. Sleeve/name-split decisions (core vs. wheel
portion of a position) are entirely manual, per your call.

Original table kept below for the record only — not in effect.

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

## 3. Distribution-phase gate (covered calls) — superseded by §10

**Status: resolved (24 June) — not by answering the open questions below, but by replacing the
premise.** §10 defines the actual CC trigger you want: spike up, price at/near the upper BB,
RSI rolled over after exceeding 70, candle reversal — a mean-reversion-from-overbought setup.
That's the opposite of `cc_signal()`'s current "above 50MA = safe to sell calls," which rewards
selling into strength rather than into an exhaustion signal. §10 replaces this section's logic
wholesale. Role-tag gating (the second bullet below) is moot — no role tags (§1).

Original open questions, kept for the record:
- What counts as "uptrend" for veto purposes — price structure (e.g. above rising 50MA) vs.
  a momentum read (RSI), vs. both?
- Does the veto apply per-name only to Income Name / Split-sleeve tickers (Core Compounders
  have no call leg regardless), or could it ever apply more broadly?

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

## 5. Sizing, reserve, and concentration — mostly out of scope (clarified 24 June)

**Status: scoped down, not removed.** You do tranche/reserve status and position sizing
manually — that's not the dashboard's job, ignore those parts of this section. This tab
(Screener/Options) is dedicated to options only; a separate tab is coming for stock
selection/fundamentals/news (see §6), and correlation/concentration-across-the-watchlist might
live there, on Overview, or get its own tab — genuinely TBD, not decided. Don't build the
correlation matrix yet (this reverses §9.5/§9.7's "build it now, it's free" call — location
isn't settled so building it on Overview now risks building it in the wrong place).

Original section kept for the record only:

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
| New "Stock Selection" tab (renamed/expanded 24 June, was "News & Macro") | New tab — AI-driven financial filings analysis, AI-condensed news from multiple sources, macro economic calendar relevant to your market. Fundamentals/selection, deliberately separate from the options-only tabs. |
| Same tab: Event Calendar | Earnings dates + FOMC/CPI dates, so you check before setting a trade |
| Role tag + correlation/concentration view | **Role tag removed (§1).** Correlation/concentration view location TBD — Overview, the new Stock Selection tab, or its own tab; not decided, don't build yet (§5). |

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
- Deep Dive chart rebuilt to 4 panels: Price+MAs+BB-bands-overlay (with BB Mid line) / Volume /
  HV20+HV60 / RSI. Position Sizing Guide now IV-expected-move only (ATR row retired).
- Options Chain tab: dropdowns replaced with two-stage tile UI (click ticker tile → click expiry
  tile). Price + DTE now bold/large; "Loaded: ticker — date" line demoted to normal text. Added
  a "How to read IV Smile & Open Interest" guide expander.
- VIX tab renamed "🌪️ Market Volatility": kept VIX as primary gauge, added a VIX/IV explainer
  caption, added OVX (oil vol) and GVZ (gold vol) gauges with reference-range tables. MOVE
  dropped per your call; HV-vs-IV contrast dropped (already covered on Overview).
- Commits this session (oldest→newest): `9d6772b`, `effbeeb`, `b15b15d`, `a3461ea`, `06633db`,
  `61d4d3d`, `703a2fc` — all live on `main`.

Still open / not yet built — see §§3, 4, 5, 6 above. Nothing in those sections has been
touched in code; all open questions remain open.

---

## 8. OPEN THREAD — Screener table rework (paused, resume here next session)

You asked to rework the conditions that merit a CSP, a CC, or a LEAP buy, starting with two
questions. First one is now answered and it surfaced a real structural gap — this is where we
left off:

**Finding — Gates/Status are NOT strategy-specific, at all.** `calc_four_gates()` takes only
the stock's price, trend (20MA/50MA), today's % move, and Bollinger-Band-walk state — nothing
about delta, strike, theta, or option type goes in. It runs **once per ticker** and the same
Gates/Status verdict gets applied uniformly across the CSP, CC, and LEAP columns in that row.
My first explanation implied a per-leg relevance ("G3 matters more for CSP/LEAP") — that's my
own interpretive read, not something the code does. The table's other columns (CSP Strike, CSP
Δ, NIS, CSP Score, CC Score, LEAP Score) genuinely are strategy-specific; Gates/Status break
that pattern by being one generic technical check on the underlying, full stop.

**Why this matters:** a single Status can't say "fine to sell a CSP here, bad time to write a
CC here" — which the directional-wheeling logic (§3's CC veto) says is often the *correct*
answer for the same ticker on the same day. Fixing this properly likely means splitting Status
into three per-leg verdicts, each gated by logic appropriate to that leg's directional bias.

**Second open item — NIS score review (your "fresh pair of eyes" ask):** found one concrete
bug, not yet fixed pending your sign-off:
- `get_screener_row()` never fetches an actual long-dated LEAP contract. It reuses the **CSP's**
  ~30-delta/~30-DTE values and runs them through `calc_suitability(..., "LEAP")`, whose param
  set expects ~80 delta/~547 DTE. Because `_tri_score()` hard-zeroes anything outside its
  lo/hi window, the DTE-fit and delta-fit terms (weights 0.40 + 0.30) are *always* zero for
  LEAP — so **LEAP Score is mathematically capped at ~30/100** for every ticker, regardless of
  actual LEAP suitability. Fix: fetch a real expiry in the 180–900 DTE range, find its ~80-delta
  call via `find_target_strike` (same pattern already used for CC), score that contract on its
  own theta/delta/DTE.
- Side note: `NIS_FLOOR`/`NIS_CEIL` are hand-calibrated to the CSP's ~30-delta/30-DTE shape;
  may need a LEAP-specific recalibration once it's scoring real LEAP contracts, since
  theta-to-IV scaling differs across delta/DTE shapes.

**Next step (your call, next session):** restructure the screener table to carry more
information — likely: per-leg Status (CSP/CC/LEAP each gated separately) instead of one shared
Status, plus the LEAP-scoring fix above. Nothing coded yet — paused per your "discuss first"
rule, picking up here.

---

## 9. External research review (overnight pass, 21 June) — NIS, missing CSP metrics, screener split

You brought outside research on NIS, missing CSP metrics, composite score, and a correlation
view. Cross-checked every claim against the actual code (`app.py`), not just against the
research's own logic. Verdict up front: the research is directionally right on almost every
point, and several items are free wins — buildable from data already on hand, no new feed
required. A few points need pushback or are blocked on the same data gap §4 already flagged.
Nothing coded yet — this is the review, build order is at the end for your sign-off.

### 9.1 NIS denominator (Price → Strike) — agree, but don't conflate the fix with a replacement

The research is correct: `calc_nis()` (line 376-379) divides by spot `price`, but the capital
at risk on a CSP is the strike (×100, cash-secured). That inflates NIS for deep-OTM puts and
understates it for higher-delta ones — a real bias, not a nitpick.

Pushback on the research's framing: it treats "fix the denominator" and "switch to Annualized
Return %" as competing options. They're not — they answer different questions. NIS's job is
cross-ticker premium-richness comparison (a $30 stock vs NVDA), normalized for IV regime via
the 15%→120% scaling. Annualized Return % (Premium÷Strike×365/DTE) answers "what's my yield on
capital" — directly comparable to a savings rate or another trade, but it says nothing about
whether that yield is rich or cheap for the current vol regime. Recommend keeping both:
- Fix `calc_nis()` to divide by `strike` instead of `price`.
- Add Annualized Return % as a new, separate column — don't merge the two into one number.

### 9.2 Missing CSP metrics — feasibility triage (this is the useful split the research skipped)

**Buildable now, zero new data** — everything below is already sitting in `find_target_strike()`'s
return dict (`bid`, `ask`, `delta`, `theta`, `oi`) or one line away from it:

| Metric | Formula | Note |
|---|---|---|
| Breakeven price | Strike − mid premium | mid = (bid+ask)/2, already computed |
| Breakeven % below spot | (Spot − Breakeven) ÷ Spot × 100 | |
| Annualized Return % | mid premium ÷ Strike × 365/DTE × 100 | the "industry-standard" metric the research names |
| POP / P(OTM) | Recommend `N(d2)` over the research's own "delta approximation" — `_bs_greeks()` (line 244) already computes `d1`/`d2` internally and just discards them. Returning `d2` and computing `norm.cdf(d2)` is more correct than `100 − |delta|` for near the same cost. | Caveat either way: this is risk-neutral POP, not a guarantee — worth a one-line tooltip caveat |
| Liquidity Score | OI (already shown) + Volume (in the chain columns, just not captured by `find_target_strike`) blended, e.g. 60/40 | Spread % is already shown separately — keep it distinct, don't fold it into the same score |

**Blocked on the same data gap as §4 — not free:** True IV Rank/Percentile needs a historical
ATM-IV time series; yfinance only returns the current chain. This is the exact constraint §4
already flagged, with the same two paths (paid feed, or start logging daily ATM IV from today
and let a rolling rank become meaningful in 60-90 days). Nothing about the new research changes
that math — it just independently arrived at wanting the same field. Recommend: start the daily
capture now regardless of what else gets built, since every day of delay pushes the payoff back
by a day. Interim: surface the already-computed HV Rank as a labeled proxy ("HV Rank — IV
proxy, not true IV Rank") rather than waiting on nothing.

### 9.3 Composite score additions / BB veto tunability

Adding IV Rank as a score component or gate has to wait on 9.2's data pipeline — gating on HV
Rank today and calling it an IV Rank gate would mislead, which is the exact confusion this
research is trying to fix elsewhere. Fine to add a clearly-labeled HV Rank gate/weight now as a
placeholder if you want it sooner; swap the data source later without changing the UI.

BB veto tunability: agree it's currently too blunt. `calc_four_gates()`'s G3 (line 414-423) hard
-zeros `all_pass` the moment the close has walked below the lower band two sessions running —
exactly the research's complaint, and it already applies identically across CSP/CC/LEAP (see
§8's "gates aren't strategy-specific" finding — same root cause, two symptoms). Recommend a
Hard / Soft(-N points) / Off toggle rather than a one-off "strong downtrend override" rule,
since a hand-written override is just a second hard-coded rule with the same brittleness.

### 9.4 Screener split into CSP / CC / LEAP tables — agree, and it's not just a data-overload fix

Splitting into three tables doesn't just solve "data overload" — it's the same fix §8 already
called for (per-leg Status instead of one shared Status), so this should be built as one
change, not two. But it inherits §8's open LEAP bug: `get_screener_row()` (line 427) never
fetches a real long-dated LEAP contract — it reuses the CSP's ~30-delta/30-DTE numbers, which
caps LEAP Score at ~30/100 for every ticker regardless of actual LEAP suitability (the
DTE-fit/delta-fit terms, 70% of LEAP's weight, hard-zero outside LEAP's 180-900 DTE / 60-95
delta window). A dedicated LEAP table built on that bug is more visibly broken than a buried
column was, not less. The bug fix is now a hard prerequisite for this item, not a nice-to-have.

**Resolved (24 June): no role tag.** CC table scope is now a pure technical filter — a ticker
qualifies for the CC table based on the mean-reversion-from-overbought trigger in §10, not on a
name classification. This is arguably the better outcome anyway: a repeatable, signal-based
filter instead of a static label that needed manual upkeep.

### 9.5 Overview correlation matrix — ON HOLD (location TBD, see §5)

Still true that it's a same-day build with zero new data — the full Close series (`cl`) is
already cached per ticker in the `results` dict from every Overview run (`analyse()`, line 567).
But per §5, where this view lives isn't decided (Overview vs. the new Stock Selection tab vs.
its own tab) — don't build it until that's settled, to avoid building it in the wrong place.

### 9.6 Tooltips + highlighting CSP Strike/Δ columns

No disagreement, low-risk UI polish, doesn't block on any decision above — can ship whenever.

### 9.7 Revised build order (superseded by §10.5 — see there for current order)

1. Start the daily ATM-IV capture pipeline (§9.2 / §4) — today, regardless of what else ships, because the clock only starts once this is running.
2. Fix the LEAP-contract bug (§8 / §9.4) — prerequisite for a LEAP table meaning anything.
3. Free-data wins: NIS denominator fix, + Annualized Return %, Breakeven, POP via `N(d2)`, Liquidity Score (§9.1, §9.2) — biggest decision-quality jump for the least effort, no data gap.
4. ~~Overview correlation matrix~~ — on hold, see §9.5.
5. Screener split into CSP/CC/LEAP tables (§9.4) — depends on #2, now gated by §10's signal instead of role tags.
6. BB veto Hard/Soft/Off toggle (§9.3).
7. Tooltips/highlighting (§9.6) — anytime, cosmetic.

Nothing above gets coded until you sign off — same rule as everywhere else in this doc.

### 9.8 Decisions locked (21 June session) — partially superseded 24 June, see §10

- **CC table scope:** ~~filter by §1 role tags~~ — role tags removed; now filtered by §10's
  mean-reversion trigger instead.
- **BB veto:** build the Hard / Soft(-N) / Off toggle (§9.3) — still stands.
- **Next build start:** still free wins first — NIS denominator fix, Annualized Return %,
  Breakeven, POP via `N(d2)`, Liquidity Score. Correlation matrix pulled from this batch (§9.5).
  IV-capture pipeline and the LEAP-bug-fix/table-split follow after.

---

## 10. CC/CSP signal redefinition — mean reversion (24 June session)

You redefined the actual trigger you want for both legs, replacing `cc_signal()`'s current
"above 50MA = safe to sell calls" logic, which the snapshot's directional-wheeling rule already
called backwards (§3). The new definition is mean-reversion-based and — per your instruction —
symmetric: CC fades a top, CSP fades a bottom, both gated on rich premium (HV Rank, pending true
IV Rank per §9.2) "all the same" in both directions.

### 10.1 CC trigger — spike up, fading from overbought

Your spec: price near/touching upper BB, above midline, RSI exceeded 70 and has turned down,
candle reversal signal. Proposed concrete parameters, built from indicators already computed
in `app.py` (`calc_bb_pctb`, `calc_rsi`) plus two new ones:

| Condition | Proposed rule | Why this threshold |
|---|---|---|
| Near/touching upper band | `pctb >= 0.85` (today) | Matches the existing `pctb_val>=0.8` convention already used in `cc_signal` (line 505), tightened slightly since "touching" should mean closer to the band than merely "extended" |
| Spike + rollover (not just sitting up there) | `pctb` peaked `>=0.95` within the last 3 sessions, and `pctb` today < `pctb` 1 session ago | The "spike then fade" part of your spec — without this, a stock just grinding along the upper band would also qualify, which isn't what you described |
| Above midline | `pctb > 0.5` | Mostly redundant once the above two hold, but cheap to check explicitly and catches the day price has pulled back toward mid but the RSI/candle conditions still apply |
| RSI exceeded 70 and turned down | RSI `>70` within the last 3-5 sessions, AND RSI today < RSI 1-2 sessions ago (or: RSI crossed back below 70 from above) | "Change to downwards" needs a reversal, not just an absolute level — this is the piece current `cc_signal` is missing entirely |
| Candle reversal signal | New code, not currently in `app.py` — needs Open/High/Low (already pulled via yfinance, just not used beyond ATR). Expanded set per your "more patterns, 2 days min" ask — see §10.3a | Fires if **any one** of §10.3a's bearish patterns is detected in the last 1-2 sessions (OR across patterns, not AND) |

### 10.2 CSP trigger — the mirror, per your "inverse CSP triggers" call

| Condition | Proposed rule |
|---|---|
| Near/touching lower band | `pctb <= 0.15` today |
| Drop + rollover | `pctb` troughed `<=0.05` within the last 3 sessions, and `pctb` today > `pctb` 1 session ago |
| Below midline | `pctb < 0.5` |
| RSI dropped below 30 and turned up | RSI `<30` within the last 3-5 sessions, AND RSI today > RSI 1-2 sessions ago |
| Candle reversal | Mirror set — see §10.3a |
| Not walking the band | Reuse the existing `walking_lower` flag (already computed, already in `csp_signal`) as a veto — a "rollover" that's actually still walking the lower band is a breakdown, not a bounce. This was already half-built for exactly this reason. |

### 10.3a Candle reversal patterns — expanded set (per "can we add more, 2 days min")

Single-candle patterns (shooting star, hammer) are technically 1-day; paired with a confirming
candle they satisfy your 2-day minimum and cut false positives, so kept that way below. All of
these are codable from Open/High/Low/Close with no new dependency — same complexity class as
the original two, just more of them. Pattern fires on an OR basis (any one is enough).

| CC (bearish reversal) | Span | CSP (bullish mirror) | Span |
|---|---|---|---|
| Bearish engulfing | 2-day | Bullish engulfing | 2-day |
| Dark cloud cover | 2-day | Piercing line | 2-day |
| Tweezer top | 2-day | Tweezer bottom | 2-day |
| Shooting star + red confirmation day | 2-day | Hammer + green confirmation day | 2-day |
| Evening star (optional, stronger signal) | 3-day | Morning star (optional, stronger signal) | 3-day |

Built with standard textbook thresholds first (body/wick ratios, overlap %) — these are
tunable once you see real fires against your watchlist, not claimed to be precisely calibrated
out of the gate.

### 10.3 IV/HV Rank gate — symmetric, per "with high IV all the same"

Current `cc_signal`/`csp_signal` use different breakpoints per side (CC: HVR>65/45, CSP:
HVR>55/35 — lines 493-494, 514-515), which is the asymmetry you're now saying to drop. Proposed:
identical breakpoints both directions, e.g. HVR>55 → strong, HVR>35 → moderate, same scoring
weight on each leg. Swap to true IV Rank once §9.2's pipeline has enough history; no UI change
needed when that happens, just the input source.

### 10.4 Decisions (locked 24 June, item 1 reversed 24 June — later in same session)

1. **Gate vs. score → Score, not a gate (REVISED).** Originally locked as a hard gate (below,
   kept for the record), then explicitly reversed: "Score please. The trader decides if he wants
   to take the trade or not. For both." Both CC and CSP triggers are now a score/label shown
   on **every** watchlist ticker — nothing gets removed from any table. You decide whether to
   act on the signal; the dashboard's job is to surface it, not to filter for you. This also
   makes CC and CSP symmetric again — item 5 below (the CC/CSP asymmetry) is now moot, since
   neither one gates.
   - *Original (superseded) text, for the record:* "Hard gate. A ticker only shows up in the new
     CC/CSP screener tables (and only carries a live signal card on Overview) when the trigger
     actually fires — not a ranked 'how close' score."
2. **Where this lives → Both.** One shared signal definition drives both `cc_signal()`/
   `csp_signal()` (Overview's per-ticker cards, lines 490-527) and the new CC/CSP Screener
   tables (§9.4) — not two separate logics showing two different things. Unaffected by the
   item-1 revision: same single source, now expressed as a score column in both places instead
   of a gate in one.
3. **Candle patterns → expanded set, see §10.3a.**
4. **Lookback window → keep as proposed.** 3-session BB peak/trough, 5-session RSI lookback for
   >70/<30. Ship as-is, revisit once it's firing on real tickers.
5. **CSP table filter → flag, not a gate.** ~~Unlike CC (hard gate, §10.4.1)~~ — now moot per the
   item-1 revision: CC and CSP are both flags, not gates. Every watchlist ticker keeps showing
   its Δ30/30DTE numbers regardless of the signal; the §10.1/10.2 trigger is a column, not a
   filter, on both legs.

**Status: implemented in `app.py` 24 June, pending your review/testing — not yet pushed to
GitHub main.** Added `_candle_reversal()` (§10.3a's 5 bearish + 5 bullish patterns, OR logic,
2-session lookback) and `_mean_reversion_score()` (BB peak/trough + RSI rollover, 3/5-session
lookback per §10.4.4). Rewrote `cc_signal()`/`csp_signal()` to call these plus the symmetric
HV Rank component (§10.3) — output is a score/label (revised §10.4.1), not a gate, on every
ticker. `get_screener_row()` now surfaces `cc_timing_label`/`csp_timing_label` (+ score +
reasons) per ticker; Screener table has new **CSP Timing** / **CC Timing** columns, and the
per-ticker detail expander shows the full reason breakdown. Overview's cards get this for free
since they call the same `cc_signal()`/`csp_signal()`. Ran a local smoke test against synthetic
OHLC data (not your live tickers) — compiles clean, candle/score logic fires sensibly in both
directions, no exceptions. Next: you review the diff and test against real tickers before this
goes live.

### 10.5 Build order (current — supersedes §9.7)

1. Resolve §10.4's open questions (gate-vs-score, which code path, candle patterns, lookback) — blocks everything else in this section from being coded correctly.
2. Build the §10.1/10.2/10.3 signal redefinition once #1 is answered.
3. Free-data wins from §9: NIS denominator fix, Annualized Return %, Breakeven, POP via `N(d2)`, Liquidity Score — independent of #1/#2, can run in parallel.
4. Fix the LEAP-contract bug (§8/§9.4) — prerequisite for a LEAP table.
5. Screener split into CSP/CC/LEAP tables (§9.4) — every ticker shown in each table, §10's
   signal/score is a column, not a filter (revised §10.4.1).
6. BB veto Hard/Soft/Off toggle (§9.3).
7. Daily ATM-IV capture pipeline (§9.2/§4) — **shelved, 24 June session.** Decision: rather than
   build a DIY daily-snapshot pipeline now (which needs its own persistence story since
   Streamlit Cloud's filesystem is ephemeral — see discussion this session), Jay will pursue
   Tradier (needs a cash account, signup takes time) or another data source that provides full
   historical data including historic IV, rather than rolling our own from scratch. Tradier
   integration itself stays paused — not urgent, data feed isn't critical to current work. The
   HV Rank proxy stays in place as the interim signal until a real historic-IV source is wired
   in. Revisit this line once a data-source decision lands.
8. Tooltips/highlighting (§9.6), correlation matrix (§9.5, once location is decided), Stock Selection tab (§6) — later, no urgency flagged yet.

**Status — item 3 (free-data-win metrics) implemented in `app.py` 24 June, pending your
review/testing — not yet pushed to GitHub main.** Changes:
- `_bs_greeks()` now also returns `d2` (was discarded before).
- `calc_nis()` denominator fixed: strike instead of price (§9.1). Caveat flagged in-code:
  `NIS_FLOOR`/`NIS_CEIL` were calibrated against the old price-denominated raw value: since
  strike sits close to but not exactly at price for ~30Δ contracts, the 0-100 scale shifts
  slightly. Worth eyeballing once tested against real chains — may need a floor/ceil
  re-calibration pass if the spread looks off.
- `find_target_strike()` now also returns `mid` (premium), `volume`, `pop` (POP via `N(d2)`
  for puts / `N(-d2)` for calls — only available on the Black-Scholes greek path, shows `None`
  if the raw chain supplies delta/theta directly), and `liquidity_score` (0-100, 60% OI / 40%
  volume blend, capped once OI≥100 or volume≥50/day — first-pass thresholds, untested on live
  data).
- `get_screener_row()` now also computes/returns Annualized Return % (mid premium ÷ strike,
  annualized off actual DTE) for both CSP and CC, and Breakeven + Breakeven % (CSP only — CC's
  "breakeven" needs a stock cost basis this dashboard doesn't track, so skipped rather than
  guessed at).
- All of the above surfaced in the per-ticker "Three-Gate Filter Detail" expander, **not** the
  main Screener table — table is already dense and is getting properly restructured in step 5
  (the CSP/CC/LEAP split), so new columns went into the detail view for now rather than adding
  to a table about to be rebuilt anyway.
- Smoke-tested against a synthetic options chain (not live tickers): target-delta strike
  selection, NIS, POP/d2, annualized return, and breakeven all produced sane numbers; compiles
  clean. Next: review against real chains before this goes live, then on to step 4 (LEAP bug
  fix).

**Status — item 4 (LEAP-contract bug, §8/§9.4) implemented in `app.py` 24 June, pending your
review/testing — not yet pushed to GitHub main.** `get_screener_row()` no longer reuses the
CSP's ~30Δ/30DTE numbers for LEAP. It now scans `all_exps` for the expiry closest to 547 DTE
within the 180–900 DTE window, fetches that expiry's call chain, and runs `find_target_strike`
targeting Δ80 — same pattern already used for CC — then scores that contract's own
theta/delta/DTE through `calc_suitability(..., "LEAP")`. If no expiry falls in the 180–900
window (or the chain fetch comes back empty), `leap_score` is `None` and the table/expander
show "—" rather than silently falling back to CSP's numbers. New fields returned: `leap_expiry`,
`leap_dte`, `leap_strike`, `leap_delta`, `leap_theta`, `leap_iv`, `leap_oi`, `leap_mid`,
`leap_nis`, `leap_score`, `leap_greek_source` — surfaced in the per-ticker detail expander
(main table still just shows LEAP Score, now correctly computed). Flagged in-code per §8's
side note: `NIS_FLOOR`/`NIS_CEIL` are calibrated to the CSP's ~30Δ/30DTE shape, so
`leap_nis`/`leap_score` may need their own floor/ceil once tested against real LEAP chains —
theta-to-IV scaling differs at Δ80/547DTE. Smoke-tested against a synthetic chain with
expiries out to 1000 DTE: correctly picked the 546-DTE expiry, found a Δ83.5 contract at
strike 80 (vs. a $100 underlying) and scored it 86.9 — versus the old logic's hard cap around
~30 — and correctly returned `None`/“—” when no expiry exists in the 180–900 window. Not yet
tested against your real watchlist. Next: review, then step 5 (CSP/CC/LEAP table split).

**Status — item 5 (CSP/CC/LEAP table split, §9.4) implemented in `app.py` 24 June, pending your
review/testing — not yet pushed to GitHub main.** The single combined Screener table is now
three separate tables — **CSP Targets**, **CC Targets**, **LEAP Targets** — each its own
`st.dataframe`, sorted by that leg's own score descending. Per the revised §10.4.1 (score, not
gate), **every ticker appears in every table** — there's no "qualifies for the CC table" filter
anymore, which makes §9.4's earlier "Resolved 24 June: no role tag... ticker qualifies for the
CC table based on the trigger firing" text stale — that was written before the later same-day
gate→score reversal in §10.4. Corrected behavior: a ticker with no CC contract or no LEAP-window
expiry just shows "—" in that table's row rather than being dropped. CSP/CC tables now also
carry the §9.2 metrics (Ann Return %, POP, Liquidity, and Breakeven/BE% for CSP) as real
columns instead of only living in the detail expander — this is the "table about to be rebuilt
anyway" redistribution flagged when those metrics first shipped. CC table also picked up
`cc_theta`/`cc_iv`/`cc_spread`, which `get_screener_row()` computed for scoring purposes but
hadn't been returned/displayed before this — without them the CC table had no θ/day or IV
column at all.

**Left as-is, not addressed by this change:** Gates/Status are still one shared per-ticker
computation (`calc_four_gates()` doesn't take delta/strike/option type as input), so the same
Gates/Status appears in all three tables for a given ticker. §8 floated true per-leg gating
("CSP/CC/LEAP each gated separately") as a likely next step but explicitly left it as "your
call, next session" — never locked as a decision — so it wasn't built here. Flag if you want
that as a separate follow-up item.

The two charts below the tables (CSP Suitability Ranking, Strategy Score Comparison) are
unchanged — still comparing all three legs' scores side by side, which seemed useful to keep
regardless of the table split.

Smoke-tested against two synthetic tickers (one with a 180-900 DTE expiry available, one
without): both tickers correctly appear in all three tables; the one without a LEAP-window
expiry shows "—" in the LEAP table instead of being dropped or showing reused CSP numbers.
Compiles clean. Not yet tested against your real watchlist — next: review/test, then step 6
(BB veto Hard/Soft/Off toggle).

**Status — item 6 (BB veto Hard/Soft/Off toggle, §9.3) implemented in `app.py` 24 June,
pending your review/testing — not yet pushed to GitHub main.** `calc_four_gates()` now takes
`bb_veto_mode` ("Hard"/"Soft"/"Off", default "Hard" — unchanged behavior unless you switch it)
and `soft_penalty` (default 10). **Hard** = original behavior, walking the lower band 2+
sessions fails G3 and blocks Status. **Soft** = G3 always passes (doesn't block Status), but
returns a `bb_penalty` that `get_screener_row()` now subtracts from CSP/CC/LEAP scores when the
walk condition is true. **Off** = G3 always passes, no penalty either — purely informational,
shown in the gate's reason text. A radio control (`Hard / Soft / Off`) sits above the Run
Screener button; selecting Soft reveals a number input for the penalty (1–50, default 10). The
per-ticker Gate Detail expander's caption now states which mode is active. This is the
"Hard / Soft(-N points) / Off toggle" §9.3 asked for, instead of a one-off override rule, since
a hand-written exception is just a second hard-coded rule with the same brittleness.

Smoke-tested all three modes against a synthetic downtrending close series engineered to walk
the lower band for 2 sessions: Hard correctly fails G3 and `all_pass`; Soft passes G3 but
returns `bb_penalty=10`; Off passes G3 with `bb_penalty=0`. Compiles clean. Not yet tested
against your real watchlist or with the UI rendering live.

**This completes all 6 build-order items from §10.5/§9.7** (daily ATM-IV capture, item 7, was
explicitly called out as independent/no-dependency and wasn't part of this run — still open
whenever you want to start it). Nothing here has been pushed to GitHub — it's all local,
pending your review across the board before any of it goes live.

Nothing coded until you sign off.

---

## 11. Data source for live greeks — Tradier (24 June session)

You raised whether the Black-Scholes-modeled greeks in the Screener are accurate enough, or
whether a real live feed is worth it. Resolved:

**BS error has two buckets, and they're not the same size.** Structural model error (BS
assumes European exercise, no skew, constant rate, no dividends) is genuinely small for this
app's actual strikes — Δ20-50, 21-45 DTE, mostly non-dividend megacaps — roughly low-single-digit
to ~10% on delta/theta when fed a real per-strike IV. Early-exercise premium on puts grows with
DTE/moneyness, so it matters more for the LEAP table than CSP/CC (flag for whenever §8's LEAP
fix gets built). The much bigger error bucket is **input-data error**, not model error: when
yfinance returns missing/zero IV and `find_target_strike` falls back to chain-median → HV20 →
flat 30% default, delta/theta can be off 20-50%+ — that's a yfinance reliability problem
surfacing as a "model accuracy" problem. Either Tradier or Polygon/Massive kills this entirely,
since both deliver already-computed exchange/vendor greeks — no BS formula involved at all.

**Decision: Tradier**, earmarked as the replacement. ORATS-sourced greeks/IV, bundled into a
$10/mo Pro brokerage subscription that doesn't require actually trading on it — used purely as
a data source. (Polygon/Massive was the alternative — pure data vendor, no brokerage tie-in, but
$79+/mo for the tier with Greeks, vs. Tradier's bundled $10/mo.)

**Market-hours behavior — why this matters for "is it worth switching":** US equity options
stop trading at session close; there's no continuous after-hours options market the way stocks
have extended-hours trading. So outside market hours, *every* vendor — Tradier included — is
just serving the last session's closing quote, same as yfinance does now. Switching doesn't
buy more "live-ness" overnight/weekends, because there isn't more live data to get from anyone
at that point. The real gap is **during market hours**: Tradier updates continuously and
reliably through the session; yfinance's intraday updates are the ones that sometimes
silently degrade to a stale/zero IV and trigger the proxy fallback chain above. Outside hours,
the difference is reliability of the frozen snapshot (Tradier's last-quote data is
consistently well-formed with a real timestamp; yfinance's is the one that sometimes just
breaks), not freshness.

**Decision (locked 24 June): Tradier only**, no yfinance fallback. Outside-hours data is
equally stale either way (frozen last-close snapshot), so a second source buys redundancy, not
better data quality — not worth the extra integration/schema-normalization surface.

**Rollout plan:**
1. Jay opens a free Tradier account and generates a **sandbox token** (delayed data) —
   this is on Jay, not buildable by me.
2. Build/test the Screener-tab integration against the sandbox token first — good enough to
   validate schema mapping, the new greek fields, and that the fallback-chain problem (§ above)
   actually goes away.
3. Switch to a **production token** later for real-time data during trading hours, once the
   sandbox-based integration is working end to end.

**POSTPONED (24 June session) — Tradier signup hit technical issues on their end.** Bypassing
this for now; Screener tab work proceeds on the current yfinance + Black-Scholes-modeled greeks
pipeline (§ above re: fallback chain still applies, including its failure modes). Revisit Tradier
once signup works. This does not block any of §9/§10's screener changes below — none of them
depend on the data source.
