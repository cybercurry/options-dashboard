#!/usr/bin/env python3
"""Wheel-signal scan → data/signals.json.

Run by .github/workflows/scan-signals.yml (3x/day on trading days + Sunday evening Dubai),
NOT by the Streamlit app (the app has no persistent storage; a committed JSON is how the
Signals tab keeps the latest shortlist between visits). The app can also run signals.scan()
live via its "Scan now" button — same engine, this just schedules it.

Reads the Tradier token from the env (TRADIER_TOKEN), same door the rest of the app uses.
"""
import datetime
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))          # so `import signals` (repo root) resolves
import signals                          # noqa: E402
import sheets                           # noqa: E402

UNIVERSE  = ROOT / "wheel_universe.json"
WATCHLIST = ROOT / "watchlist.json"
OUT       = ROOT / "data" / "signals.json"


def _load_universe():
    """Signal universe = wheel + growth. LIVE from the published Google-Sheet CSV each run;
    refreshes the committed wheel_universe.json cache so it stays a good fallback. Falls back to
    that cache (then the watchlist) if the sheet is unreachable."""
    cfg = {}
    try:
        cfg = json.loads(UNIVERSE.read_text())
    except Exception:
        pass
    live = sheets.fetch_universe(cfg.get("source_url")) if cfg.get("source_url") else {}
    if live.get("wheel"):
        merged = {**cfg, **live}          # keep note/source_url, refresh wheel/growth
        try:
            UNIVERSE.write_text(json.dumps(merged, indent=2) + "\n")
        except Exception:
            pass
        print(f"Universe from sheet: {len(live['wheel'])} wheel, {len(live.get('growth',[]))} growth")
        return merged
    if cfg.get("wheel"):
        print("Sheet unreachable — using committed wheel_universe.json cache")
        return cfg
    print("No universe file — using watchlist.json")
    return {"wheel": json.loads(WATCHLIST.read_text()), "growth": []}


def main():
    result = signals.scan(_load_universe())
    result["generated"] = datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2))
    print(f"Wrote {result['count']} signals ({sum(1 for s in result['signals'] if s.get('shortlist'))} "
          f"on the shortlist) to {OUT}")


if __name__ == "__main__":
    main()
