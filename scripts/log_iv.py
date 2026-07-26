#!/usr/bin/env python3
"""Daily ATM-IV logger.

Appends one row (date, ticker, atm_iv) per watchlist ticker to data/iv_history.csv.
Run by .github/workflows/log-iv.yml on a weekday schedule — NOT by the Streamlit app
(the app's host has no persistent storage; a committed CSV is how we keep history).

Reads the Tradier token from the env (TRADIER_TOKEN), same door the app uses. After a few
weeks of rows this file is what powers a real IV Rank / percentile per ticker.
"""
import csv
import datetime
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))          # so `import tradier` (repo root) resolves
import tradier                          # noqa: E402

WATCHLIST = ROOT / "watchlist.json"
OUT       = ROOT / "data" / "iv_history.csv"


def atm_iv(ticker):
    """Average of the ATM call & put mid-IV (percent) for the nearest expiry ≥5 days out."""
    exps = tradier.get_expirations(ticker)
    if not exps:
        return None
    today = datetime.date.today()
    future = [e for e in exps if (datetime.date.fromisoformat(e) - today).days >= 5]
    expiry = (future or exps)[0]

    quotes = tradier.get_quotes(ticker)
    spot = quotes[0].get("last") if quotes else None
    if not spot:
        return None

    opts = tradier.get_option_chain(ticker, expiry, greeks=True)
    if not opts:
        return None

    # strike nearest spot
    strike = min((o.get("strike") for o in opts if o.get("strike") is not None),
                 key=lambda k: abs(k - spot), default=None)
    if strike is None:
        return None

    ivs = []
    for o in opts:
        if o.get("strike") == strike:
            g = o.get("greeks") or {}
            iv = g.get("mid_iv") or g.get("smv_vol")
            if iv:
                ivs.append(float(iv))
    if not ivs:
        return None
    return round(sum(ivs) / len(ivs) * 100.0, 2)   # store as percent


def main():
    tickers = json.loads(WATCHLIST.read_text())
    today = datetime.date.today().isoformat()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    write_header = not OUT.exists() or OUT.stat().st_size == 0

    rows = []
    for t in tickers:
        try:
            iv = atm_iv(t)
        except Exception as e:               # never let one ticker abort the batch
            iv = None
            print(f"{t}: ERROR {type(e).__name__}: {e}", file=sys.stderr)
        if iv is not None:
            rows.append([today, t, iv])
            print(f"{t}: {iv}")
        else:
            print(f"{t}: no IV", file=sys.stderr)

    if not rows:
        print("No IV rows collected — not writing.", file=sys.stderr)
        return

    with open(OUT, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["date", "ticker", "atm_iv"])
        w.writerows(rows)
    print(f"Appended {len(rows)} rows to {OUT}")


if __name__ == "__main__":
    main()
