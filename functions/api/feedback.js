// Cloudflare Pages Function — receive a visitor's feedback and email it to the owner via Resend.
// Works for anonymous or signed-in visitors. No secrets in the repo: reads RESEND_API_KEY and
// FEEDBACK_TO from the Pages project's environment variables (Settings → Environment variables).
//
// Anti-abuse: a hidden honeypot field, length caps, and a category whitelist. For heavier traffic
// add Cloudflare Turnstile later.

const J = (obj, status = 200) => new Response(JSON.stringify(obj),
  { status, headers: { "content-type": "application/json", "access-control-allow-origin": "*" } });

const esc = (s) => (s || "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

export async function onRequestPost(context) {
  const { request, env } = context;
  let b;
  try { b = await request.json(); } catch (e) { return J({ ok: false, error: "bad request" }, 400); }

  if (b.website) return J({ ok: true });                       // honeypot: bots fill this hidden field → drop silently

  const msg = String(b.message || "").trim();
  if (msg.length < 3)    return J({ ok: false, error: "Please write a little more." }, 400);
  if (msg.length > 4000) return J({ ok: false, error: "Message too long (4000 char max)." }, 400);

  const cats = ["Bug", "Idea", "Feedback", "Other"];
  const category = cats.includes(b.category) ? b.category : "Feedback";
  const email = String(b.email || "").trim().slice(0, 160);
  const page  = String(b.page  || "").trim().slice(0, 200);
  const validEmail = /^[^@\s]+@[^@\s]+\.[^@\s]+$/.test(email);

  const key = env.RESEND_API_KEY;
  if (!key) return J({ ok: false, error: "Feedback isn't configured yet." }, 500);
  // MUST be a mailbox you actually receive at (set FEEDBACK_TO in the Pages env). hello@ can send
  // but doesn't receive yet, so it's only a placeholder fallback.
  const to = env.FEEDBACK_TO || "hello@optionintel.app";

  const html = `<div style="font-family:Arial,Helvetica,sans-serif;max-width:600px;color:#111827">
    <h3 style="margin:0 0 8px">OptionIntel feedback · ${esc(category)}</h3>
    <p style="margin:2px 0;color:#6b7280"><b>From:</b> ${email ? esc(email) : "(anonymous)"}</p>
    <p style="margin:2px 0 12px;color:#6b7280"><b>Page:</b> ${esc(page) || "—"}</p>
    <pre style="white-space:pre-wrap;font-family:inherit;background:#f6f7f9;padding:12px;border-radius:8px;margin:0">${esc(msg)}</pre>
  </div>`;
  const payload = {
    from: "OptionIntel <hello@optionintel.app>",
    to,
    subject: `OptionIntel ${category.toLowerCase()}${email ? " from " + email : ""}`,
    html,
    text: `Category: ${category}\nFrom: ${email || "(anonymous)"}\nPage: ${page || "—"}\n\n${msg}`,
  };
  if (validEmail) payload.reply_to = email;                    // reply straight to the visitor

  try {
    const r = await fetch("https://api.resend.com/emails", {
      method: "POST",
      headers: { "Authorization": "Bearer " + key, "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!r.ok) return J({ ok: false, error: "Couldn't send right now — try again later." }, 502);
    return J({ ok: true });
  } catch (e) {
    return J({ ok: false, error: "Couldn't send right now — try again later." }, 502);
  }
}

// Simple GET probe so we can confirm the route deployed (no secret needed).
export async function onRequestGet() {
  return J({ ok: true, route: "feedback", method: "POST to submit" });
}
