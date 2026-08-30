-- optionintel.app — one-time WELCOME email for each new subscriber.
--
-- Sends a branded "how to use it" email the first time a new user is created (their first
-- magic-link sign-in), via Resend's HTTP API called straight from Postgres with pg_net. Fires
-- exactly once per subscriber (auth.users row is inserted once), never on later sign-ins, and
-- never blocks sign-up if the send fails.
--
-- PREREQUISITES (already done for SMTP): your domain is verified in Resend and you have a Resend
-- API key (re_…). You can reuse that key or make a new "welcome-email" one (Sending access,
-- domain optionintel.app).
--
-- INSTALL (Supabase dashboard):
--   1) Database → Extensions → enable  pg_net
--   2) SQL Editor → paste this whole file, then replace the two placeholders below:
--        RESEND_API_KEY_HERE   → your re_… key
--        REPLY_TO_EMAIL_HERE   → the inbox where replies should land (e.g. your real address),
--                                since hello@ can send but can't receive yet
--      → Run.
--   3) Test: sign in with a fresh alias (e.g. you+welcome@yourdomain) → the email arrives once.
--
-- SECURITY: the key lives only in this server-side function (SECURITY DEFINER); it is never sent
-- to the browser or the anon client. To keep it out of the function body, see the Vault note below.

create extension if not exists pg_net;

create or replace function public.send_welcome_email()
returns trigger
language plpgsql
security definer
set search_path = public
as $fn$
declare
  resend_key text := 'RESEND_API_KEY_HERE';   -- ← paste your Resend re_… key
  reply_to   text := 'REPLY_TO_EMAIL_HERE';   -- ← inbox that "Reply" should reach (hello@ can't receive yet)
begin
  perform net.http_post(
    url     := 'https://api.resend.com/emails',
    headers := jsonb_build_object(
                 'Authorization', 'Bearer ' || resend_key,
                 'Content-Type',  'application/json'),
    body    := jsonb_build_object(
      'from',     'OptionIntel <hello@optionintel.app>',
      'to',       NEW.email,
      'reply_to', reply_to,
      'subject',  'Welcome to OptionIntel — how to use it',
      'text',     'Welcome to OptionIntel (optionintel.app) — options intelligence for the wheel '
              || 'strategy; research, not advice. A clean way to work an idea, top to bottom: '
              || '1) Read the market (Market tab): macro, rates, volatility regime. '
              || '2) Set today''s goal: income (CSP/CC) or long exposure (LEAP). '
              || '3) Select the stock (Fundamentals): valuation & quality from SEC filings. '
              || '4) Confirm trend & timing (TA). '
              || '5) Scan the Signals for a match; note the timing score. '
              || '6) Structure the trade (Options Chain): hover a Bid to see the expected return at '
              || 'each Delta; pick your Delta (~30 preset) and DTE (30-35 preset), both editable. '
              || '7) Size it and plan the exit up front, for both outcomes. '
              || 'Above all — manage risk and position size. '
              || 'Feedback or a question? Just reply to this email — it comes straight to me. — Jay. '
              || 'Open https://optionintel.app',
      'html',     $html$
<div style="font-family:Arial,Helvetica,sans-serif;max-width:600px;margin:0 auto;color:#111827;background:#ffffff;padding:8px 4px">
  <h1 style="margin:0 0 4px;font-size:24px">Welcome to OptionIntel 👋</h1>
  <p style="color:#6b7280;margin:0 0 6px;font-size:14px">Options intelligence for the wheel strategy — research, not advice.</p>
  <p style="color:#5B8DEF;margin:0 0 22px;font-size:12px;font-weight:bold;letter-spacing:.06em;text-transform:uppercase">Context → Select → Confirm → Structure → Manage</p>
  <p style="line-height:1.6;font-size:14px;margin:0 0 22px">A clean way to work through an idea — top to bottom, using the tabs:</p>
  <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="border-collapse:collapse">
    <tr><td style="padding:0"><table role="presentation" width="100%" cellpadding="0" cellspacing="0"><tr>
      <td width="42" valign="top" style="padding:0 12px 0 0"><div style="width:30px;height:30px;border-radius:50%;background:#5B8DEF;color:#fff;font-weight:bold;font-size:14px;text-align:center;line-height:30px">1</div></td>
      <td valign="top" style="padding:2px 0"><div style="font-weight:bold;font-size:14px">Read the market <a href="https://optionintel.app/#market" style="color:#5B8DEF;text-decoration:none;font-weight:normal">· Market ↗</a></div>
      <div style="color:#4b5563;font-size:13px;line-height:1.5;margin-top:2px">Macro backdrop, rates, and the volatility regime. Trade with the tape, not against it.</div></td>
    </tr></table></td></tr>
    <tr><td style="padding:5px 0 5px 15px;color:#cbd2db;font-size:15px;line-height:1">↓</td></tr>
    <tr><td style="padding:0"><table role="presentation" width="100%" cellpadding="0" cellspacing="0"><tr>
      <td width="42" valign="top" style="padding:0 12px 0 0"><div style="width:30px;height:30px;border-radius:50%;background:#5B8DEF;color:#fff;font-weight:bold;font-size:14px;text-align:center;line-height:30px">2</div></td>
      <td valign="top" style="padding:2px 0"><div style="font-weight:bold;font-size:14px">Set today's goal</div>
      <div style="color:#4b5563;font-size:13px;line-height:1.5;margin-top:2px">Income (sell a CSP / CC) or long exposure (a LEAP)? Decide what you want <i>before</i> you look.</div></td>
    </tr></table></td></tr>
    <tr><td style="padding:5px 0 5px 15px;color:#cbd2db;font-size:15px;line-height:1">↓</td></tr>
    <tr><td style="padding:0"><table role="presentation" width="100%" cellpadding="0" cellspacing="0"><tr>
      <td width="42" valign="top" style="padding:0 12px 0 0"><div style="width:30px;height:30px;border-radius:50%;background:#5B8DEF;color:#fff;font-weight:bold;font-size:14px;text-align:center;line-height:30px">3</div></td>
      <td valign="top" style="padding:2px 0"><div style="font-weight:bold;font-size:14px">Select the stock <a href="https://optionintel.app/#fund" style="color:#5B8DEF;text-decoration:none;font-weight:normal">· Fundamentals ↗</a></div>
      <div style="color:#4b5563;font-size:13px;line-height:1.5;margin-top:2px">Valuation &amp; quality straight from SEC filings. Look for asymmetric upside, not just familiar names.</div></td>
    </tr></table></td></tr>
    <tr><td style="padding:5px 0 5px 15px;color:#cbd2db;font-size:15px;line-height:1">↓</td></tr>
    <tr><td style="padding:0"><table role="presentation" width="100%" cellpadding="0" cellspacing="0"><tr>
      <td width="42" valign="top" style="padding:0 12px 0 0"><div style="width:30px;height:30px;border-radius:50%;background:#5B8DEF;color:#fff;font-weight:bold;font-size:14px;text-align:center;line-height:30px">4</div></td>
      <td valign="top" style="padding:2px 0"><div style="font-weight:bold;font-size:14px">Confirm trend &amp; timing <a href="https://optionintel.app/#deep" style="color:#5B8DEF;text-decoration:none;font-weight:normal">· TA ↗</a></div>
      <div style="color:#4b5563;font-size:13px;line-height:1.5;margin-top:2px">Is the chart set up — and is <i>now</i> a sensible entry? Check the trend, then the timing read.</div></td>
    </tr></table></td></tr>
    <tr><td style="padding:5px 0 5px 15px;color:#cbd2db;font-size:15px;line-height:1">↓</td></tr>
    <tr><td style="padding:0"><table role="presentation" width="100%" cellpadding="0" cellspacing="0"><tr>
      <td width="42" valign="top" style="padding:0 12px 0 0"><div style="width:30px;height:30px;border-radius:50%;background:#5B8DEF;color:#fff;font-weight:bold;font-size:14px;text-align:center;line-height:30px">5</div></td>
      <td valign="top" style="padding:2px 0"><div style="font-weight:bold;font-size:14px">Scan the <a href="https://optionintel.app/#signals" style="color:#5B8DEF;text-decoration:none">Signals ↗</a></div>
      <div style="color:#4b5563;font-size:13px;line-height:1.5;margin-top:2px">Does anything match your criteria and interest? Note its timing score before acting.</div></td>
    </tr></table></td></tr>
    <tr><td style="padding:5px 0 5px 15px;color:#cbd2db;font-size:15px;line-height:1">↓</td></tr>
    <tr><td style="padding:0"><table role="presentation" width="100%" cellpadding="0" cellspacing="0"><tr>
      <td width="42" valign="top" style="padding:0 12px 0 0"><div style="width:30px;height:30px;border-radius:50%;background:#5B8DEF;color:#fff;font-weight:bold;font-size:14px;text-align:center;line-height:30px">6</div></td>
      <td valign="top" style="padding:2px 0"><div style="font-weight:bold;font-size:14px">Structure the trade <a href="https://optionintel.app/#chain" style="color:#5B8DEF;text-decoration:none;font-weight:normal">· Options Chain ↗</a></div>
      <div style="color:#4b5563;font-size:13px;line-height:1.5;margin-top:2px"><b>Hover a Bid</b> to see the expected return at each Delta. Pick your &#916; (&#8776;30 preset) and DTE (30&#8211;35 preset) — both editable.</div></td>
    </tr></table></td></tr>
    <tr><td style="padding:5px 0 5px 15px;color:#cbd2db;font-size:15px;line-height:1">↓</td></tr>
    <tr><td style="padding:0"><table role="presentation" width="100%" cellpadding="0" cellspacing="0"><tr>
      <td width="42" valign="top" style="padding:0 12px 0 0"><div style="width:30px;height:30px;border-radius:50%;background:#5B8DEF;color:#fff;font-weight:bold;font-size:14px;text-align:center;line-height:30px">7</div></td>
      <td valign="top" style="padding:2px 0"><div style="font-weight:bold;font-size:14px">Size it &amp; plan the exit</div>
      <div style="color:#4b5563;font-size:13px;line-height:1.5;margin-top:2px">Choose your position size, then decide your exit <i>up front</i> — for both cases: it works out, or it goes against you.</div></td>
    </tr></table></td></tr>
  </table>
  <div style="background:#FEF3C7;border:1px solid #FCD34D;border-radius:10px;padding:14px 16px;margin:24px 0 6px;color:#92400E;font-size:14px;line-height:1.55">
    <b>Above all — manage risk and position size.</b> No single trade should be able to hurt you. The plan matters more than the pick.
  </div>
  <p style="margin:24px 0 6px"><a href="https://optionintel.app" style="background:#5B8DEF;color:#ffffff;text-decoration:none;padding:12px 22px;border-radius:8px;display:inline-block;font-weight:bold;font-size:14px">Open OptionIntel →</a></p>
  <p style="font-size:14px;line-height:1.6;margin:22px 0 2px">Have feedback or a question? Just hit <b>Reply</b> — it comes straight to me.</p>
  <p style="font-size:14px;margin:0;color:#111827">— Jay</p>
  <p style="color:#9ca3af;font-size:12px;border-top:1px solid #e5e7eb;padding-top:14px;margin-top:20px;line-height:1.6">Research &amp; education only — not financial advice, an offer, or a recommendation. You're receiving this once because you signed in at optionintel.app.</p>
</div>
      $html$
    )
  );
  return NEW;
exception when others then
  return NEW;   -- never block sign-up if the email API call errors
end;
$fn$;

drop trigger if exists on_auth_user_created_welcome on auth.users;
create trigger on_auth_user_created_welcome
  after insert on auth.users
  for each row execute function public.send_welcome_email();

-- ── Optional hardening: keep the key in Supabase Vault instead of inline ──────────────────────
-- 1) In the dashboard: Project Settings → Vault → add a secret named  resend_api_key  = re_…
-- 2) In the function above, replace the resend_key line with:
--      resend_key text := (select decrypted_secret from vault.decrypted_secrets where name = 'resend_api_key');
--    Then the key never appears in the function definition.
