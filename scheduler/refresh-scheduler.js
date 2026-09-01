// optionintel.app — reliable refresh heartbeat (Cloudflare Worker).
//
// GitHub's own cron throttles/skips high-frequency schedules, so it can't be trusted for a
// dependable 30-min heartbeat. This Worker's Cron Trigger fires reliably and dispatches the
// `refresh-optionintel.yml` workflow via the GitHub API.
//
// Timing: the workflow itself is the source of truth for the ET market window (a DST-aware
// zoneinfo gate). This Worker ALSO checks the window before dispatching — purely to avoid spending
// GitHub Actions minutes on runs that would just gate out — using Intl with America/New_York, which
// tracks EDT/EST automatically. Window = 08:30–17:00 ET (open −1h … close +1h), Mon–Fri.
//
// Secrets (set in the Worker, never in this file):
//   GH_TOKEN     fine-grained GitHub PAT — repo cybercurry/options-dashboard, Actions: Read & write
//   REFRESH_KEY  any random string — guards the manual /dispatch test endpoint
//
// Cron trigger (UTC; the band covers the ET window in both seasons, the code refines the edges):
//   0,30 12-22 * * 1-5

const OWNER = "cybercurry";
const REPO = "options-dashboard";
const WORKFLOW = "refresh-optionintel.yml";

function inEtWindow(now = new Date()) {
  const parts = new Intl.DateTimeFormat("en-US", {
    timeZone: "America/New_York",
    weekday: "short", hour: "2-digit", minute: "2-digit", hour12: false,
  }).formatToParts(now);
  const val = (t) => parts.find((p) => p.type === t)?.value;
  const weekday = val("weekday");                 // "Mon", "Tue", …
  let hh = parseInt(val("hour"), 10);
  if (hh === 24) hh = 0;                           // some ICU builds emit "24" at midnight
  const mins = hh * 60 + parseInt(val("minute"), 10);
  const isWeekday = ["Mon", "Tue", "Wed", "Thu", "Fri"].includes(weekday);
  const inWindow = mins >= 8 * 60 + 30 && mins <= 17 * 60;   // 08:30 … 17:00 ET (inclusive)
  return isWeekday && inWindow;
}

async function dispatch(env, force = false) {
  const r = await fetch(
    `https://api.github.com/repos/${OWNER}/${REPO}/actions/workflows/${WORKFLOW}/dispatches`,
    {
      method: "POST",
      headers: {
        Authorization: "Bearer " + env.GH_TOKEN,
        Accept: "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "optionintel-refresh-scheduler",
      },
      body: JSON.stringify({ ref: "main", inputs: force ? { force: "true" } : {} }),
    }
  );
  return r.status; // 204 = accepted
}

export default {
  // Fires on the Cron Trigger. Dispatch only inside the ET window (the workflow re-checks too).
  async scheduled(event, env, ctx) {
    if (!inEtWindow()) return;
    ctx.waitUntil(dispatch(env, false));
  },

  // Manual test hook: GET /dispatch?key=REFRESH_KEY forces a run now (bypasses the window), so you
  // can verify the wiring without waiting for the next half-hour. Everything else returns a health OK.
  async fetch(req, env) {
    const url = new URL(req.url);
    if (url.pathname === "/dispatch") {
      if (url.searchParams.get("key") !== env.REFRESH_KEY) {
        return new Response("forbidden", { status: 403 });
      }
      const status = await dispatch(env, true);
      return new Response("dispatched -> GitHub status " + status + " (204 = accepted)", { status: 200 });
    }
    return new Response("optionintel refresh scheduler — ok. In ET window right now? " + inEtWindow(), { status: 200 });
  },
};
