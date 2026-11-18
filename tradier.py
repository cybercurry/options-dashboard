"""
Tradier API access door — a single authenticated entry point for REAL market data
(quotes, option chains with vendor IV/Greeks, expirations, history, market clock).

Design goals (per Jay, 25 July):
  • ONE door, NO re-logons — a long-lived Personal Access Token (generated once in the
    Tradier dashboard) held in Streamlit secrets. No OAuth, no login flow, ever.
  • REAL data, not calculated — chains are pulled with greeks=true so IV and Greeks come
    straight from Tradier's ORATS feed instead of being Black-Scholes'd in-app.
  • Works with the market CLOSED — every function here is REST; off-hours you get the last
    session's quotes and the most recent ORATS Greeks snapshot. (Only live streaming ticks
    need the market open, and nothing here streams.)

Setup (done once, no token ever touches the repo):
  Local dev : export TRADIER_TOKEN=xxxxxxxx   (optionally TRADIER_ENV=production|sandbox)
  Streamlit : add to the app's Secrets:
                TRADIER_TOKEN = "xxxxxxxx"
                TRADIER_ENV   = "production"   # or "sandbox" for delayed test data
"""

import os
import time
import threading
import requests

try:                       # streamlit is present in the app, but keep this importable
    import streamlit as st # standalone (e.g. for offline checks) without it.
    _HAS_ST = True
except Exception:
    _HAS_ST = False

_BASES = {
    "production": "https://api.tradier.com",
    "sandbox":    "https://sandbox.tradier.com",
}
_TIMEOUT = 12

# Rate-limit safety. The engine scan now fetches tickers CONCURRENTLY (bounded pool), so _get must be
# safe under parallel callers: a shared requests.Session is fine (urllib3's pool is thread-safe), but
# we must never let a burst trip Tradier's per-minute cap and silently drop a ticker's signals. Tradier
# returns X-Ratelimit-Available / X-Ratelimit-Expiry on every market-data response; we track the last
# values and, when we're near the cap, pause until the window resets. A 429 is retried with backoff.
_RL_LOCK = threading.Lock()
_RL = {"available": None, "expiry": 0.0}   # last-seen remaining calls, and when the window resets (epoch s)


# ── credentials / config ────────────────────────────────────────────────────────
def _secret(name, default=None):
    # st.secrets first (Streamlit Cloud), then environment (local dev). Accessing
    # st.secrets with no secrets file raises, so guard it — an unconfigured app must
    # degrade quietly to "not connected", never crash.
    if _HAS_ST:
        try:
            if name in st.secrets:
                return st.secrets[name]
        except Exception:
            pass
    return os.environ.get(name, default)


def _token():
    tok = _secret("TRADIER_TOKEN")
    return tok.strip() if isinstance(tok, str) and tok.strip() else None


def _env():
    e = (_secret("TRADIER_ENV", "production") or "production").strip().lower()
    return e if e in _BASES else "production"


def _base():
    return _BASES[_env()]


def is_configured():
    """True once a token is present — used to show the door as available before we
    spend a network round-trip trying to open it."""
    return _token() is not None


# ── the door: one authenticated session, reused ─────────────────────────────────
def _build_session(token):
    s = requests.Session()
    s.headers.update({
        "Authorization": f"Bearer {token}",
        "Accept":        "application/json",
    })
    return s


if _HAS_ST:
    # Cache the Session as a resource, keyed by token so a rotated token rebuilds it.
    @st.cache_resource(show_spinner=False)
    def _session(token):
        return _build_session(token)
else:
    _SESS = {}
    def _session(token):
        if token not in _SESS:
            _SESS[token] = _build_session(token)
        return _SESS[token]


def _get(path, params=None, _tries=4):
    """Core GET through the door. Returns parsed JSON. Raises on missing token, HTTP error, or non-JSON
    body so callers can surface a precise reason. Thread-safe and rate-limit aware: when Tradier's
    headers say we're near the per-minute cap it pauses until the window resets, and a 429 is retried
    with backoff — so the concurrent scan can never silently drop a ticker by overrunning the limit."""
    token = _token()
    if not token:
        raise RuntimeError("No TRADIER_TOKEN configured (add it to Streamlit secrets).")
    sess, url = _session(token), _base() + path
    last_exc = None
    for attempt in range(_tries):
        # Proactive throttle: if the last response left ≤1 call in the window, wait for it to reset.
        with _RL_LOCK:
            avail, expiry = _RL["available"], _RL["expiry"]
        if avail is not None and avail <= 1 and expiry > time.time():
            time.sleep(min(3.0, max(0.0, expiry - time.time()) + 0.05))
        try:
            r = sess.get(url, params=params or {}, timeout=_TIMEOUT)
        except requests.RequestException as e:          # transient network blip → brief backoff + retry
            last_exc = e
            time.sleep(min(3.0, 0.5 * (attempt + 1)))
            continue
        # Record Tradier's rate-limit headers for the next caller (best-effort).
        try:
            with _RL_LOCK:
                if "X-Ratelimit-Available" in r.headers:
                    _RL["available"] = int(r.headers["X-Ratelimit-Available"])
                if "X-Ratelimit-Expiry" in r.headers:   # epoch MILLIS per Tradier
                    _RL["expiry"] = int(r.headers["X-Ratelimit-Expiry"]) / 1000.0
        except Exception:
            pass
        if r.status_code == 429:                        # over the cap → honor Retry-After / back off
            ra = r.headers.get("Retry-After")
            try:
                wait = float(ra) if ra else 1.0 * (attempt + 1)
            except ValueError:
                wait = 1.0 * (attempt + 1)
            time.sleep(min(5.0, max(0.5, wait)))
            continue
        if r.status_code == 401:
            raise RuntimeError("Tradier rejected the token (401). Check TRADIER_TOKEN / that it's "
                               "a production token for a funded account.")
        r.raise_for_status()
        try:
            return r.json()
        except ValueError:
            raise RuntimeError(f"Tradier returned non-JSON (HTTP {r.status_code}).")
    if last_exc:
        raise last_exc
    raise RuntimeError("Tradier rate-limited: still 429 after retries.")


# ── market data endpoints (all REST → all work with the market closed) ──────────
def get_clock():
    """Market clock — state is open/closed/premarket/postmarket. Works 24/7, so this
    doubles as the connection smoke-test that proves the door opens off-hours."""
    return _get("/v1/markets/clock").get("clock", {})


def get_profile():
    """Account profile — validates that the token carries account scope."""
    return _get("/v1/user/profile").get("profile", {})


def get_quotes(symbols):
    if not isinstance(symbols, str):
        symbols = ",".join(symbols)
    data = _get("/v1/markets/quotes", {"symbols": symbols, "greeks": "false"})
    q = (data.get("quotes") or {}).get("quote")
    if q is None:
        return []
    return q if isinstance(q, list) else [q]


def get_expirations(symbol):
    data = _get("/v1/markets/options/expirations",
                {"symbol": symbol, "includeAllRoots": "true", "strikes": "false"})
    exp = (data.get("expirations") or {}).get("date")
    if exp is None:
        return []
    exp = exp if isinstance(exp, list) else [exp]
    return list(dict.fromkeys(exp))   # dedupe (includeAllRoots can repeat a date per root)


def get_option_chain(symbol, expiration, greeks=True):
    """REAL option chain: every contract carries a `greeks` block (delta, gamma, theta,
    vega, rho, plus bid_iv/mid_iv/ask_iv/smv_vol) sourced from Tradier's ORATS feed."""
    data = _get("/v1/markets/options/chains",
                {"symbol": symbol, "expiration": expiration,
                 "greeks": "true" if greeks else "false"})
    opts = (data.get("options") or {}).get("option")
    if opts is None:
        return []
    return opts if isinstance(opts, list) else [opts]


def get_history(symbol, interval="daily", start=None, end=None):
    params = {"symbol": symbol, "interval": interval}
    if start: params["start"] = start
    if end:   params["end"]   = end
    data = _get("/v1/markets/history", params)
    days = (data.get("history") or {}).get("day")
    if days is None:
        return []
    return days if isinstance(days, list) else [days]


# ── one-call connection check for the UI ────────────────────────────────────────
def ping():
    """Open the door and report back, without throwing. Returns a dict the app can
    render directly:
        {ok, env, market_state, market_desc, sample, error}
    Designed to succeed with the market closed — it reads the clock (24/7) and one quote.
    """
    if not is_configured():
        return {"ok": False, "env": _env(), "error":
                "No token configured yet — add TRADIER_TOKEN to the app's Secrets."}
    try:
        clock  = get_clock()
        sample = None
        try:
            qs = get_quotes("SPY")
            if qs:
                qq = qs[0]
                sample = f"SPY {qq.get('last')}  (close {qq.get('prevclose')})"
        except Exception:
            pass  # quote is a nice-to-have; the clock already proves the door opened
        return {
            "ok": True,
            "env": _env(),
            "market_state": clock.get("state"),
            "market_desc":  clock.get("description"),
            "sample": sample,
            "error": None,
        }
    except Exception as e:
        return {"ok": False, "env": _env(), "error": f"{type(e).__name__}: {e}"}
