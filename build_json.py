#!/usr/bin/env python3
"""build_json.py — headless engine runner for the optionintel.app static site.

Reuses the SAME engine as the Streamlit app (signals.py → tradier.py); no logic
rewrite. Meant to run on a schedule (GitHub Actions) with TRADIER_TOKEN in the
environment, or locally with the Streamlit secret. Writes site/data/signals.json,
which the static site fetches. Public/anonymous — watchlist only, never IBKR data.
"""
import json
import datetime
import pathlib

import signals
import sheets

HERE = pathlib.Path(__file__).parent
OUT = HERE / "site" / "data"


def load_universe():
    """Same source as the app: published-sheet CSV (tickers only), falling back to
    the committed wheel_universe.json so a Google hiccup never blanks the scan."""
    cfg = {}
    try:
        cfg = json.loads((HERE / "wheel_universe.json").read_text())
    except Exception:
        pass
    live = sheets.fetch_universe(cfg.get("source_url")) if cfg.get("source_url") else {}
    if live.get("wheel"):
        return {"wheel": live.get("wheel", []), "growth": live.get("growth", []), "_source": "sheet"}
    return {"wheel": cfg.get("wheel", []), "growth": cfg.get("growth", []), "_source": "fallback"}


def main():
    uni = load_universe()
    data = signals.scan(uni)                       # {"signals": [...], "leaps": [...], "params": {...}}
    data["generated_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    data["universe"] = {"wheel": len(uni.get("wheel", [])),
                        "growth": len(uni.get("growth", [])),
                        "source": uni.get("_source")}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "signals.json").write_text(json.dumps(data, indent=2, default=str))
    print("wrote {} — {} signals, {} leaps (universe: {})".format(
        OUT / "signals.json", len(data.get("signals", [])), len(data.get("leaps", [])), data["universe"]))


if __name__ == "__main__":
    main()
