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
              || 'strategy; research, not advice. Registering lets you build your own watchlist and '
              || 'make it truly yours — your picks drive every tab, private and synced across '
              || 'devices; more personalization is on the way. A clean way to work an idea, top to bottom: '
              || '1) Read the market (Market tab): macro, rates, volatility regime. '
              || '2) Set the goal: income (a cash-secured put or covered call) or long exposure (a LEAP). '
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
<div style="font-family:'Helvetica Neue',Arial,sans-serif;max-width:600px;margin:0 auto;color:#111827;background:#ffffff;padding:8px 4px">
  <h1 style="margin:0 0 4px;font-size:24px">Welcome to OptionIntel</h1>
  <p style="color:#6b7280;margin:0 0 4px;font-size:14px">Options intelligence for the wheel strategy — research, not advice.</p>
  <p style="color:#5B8DEF;margin:0 0 22px;font-size:11.5px;font-weight:bold;letter-spacing:.09em;text-transform:uppercase">Context&nbsp;&rarr;&nbsp;Select&nbsp;&rarr;&nbsp;Confirm&nbsp;&rarr;&nbsp;Structure&nbsp;&rarr;&nbsp;Manage</p>
  <p style="line-height:1.65;font-size:14px;margin:0 0 16px">Thanks for registering. Signing in lets you <b>build your own watchlist</b> and make OptionIntel truly yours — your picks drive every tab, private to you and synced across your devices. This is just the start; more personalization is on the way.</p>
  <p style="line-height:1.65;font-size:14px;margin:0 0 24px">A clean way to work through an idea — top to bottom, using the tabs:</p>
  <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="border-collapse:collapse">
    <tr>
      <td width="38" valign="top" style="border-left:2px solid #D7E3F8;padding:0 0 26px 0"><div style="width:27px;height:27px;line-height:27px;border-radius:50%;background:#5B8DEF;color:#fff;font-size:13px;font-weight:bold;text-align:center;margin-left:-14px;box-shadow:0 0 0 4px #ffffff">1</div></td>
      <td valign="top" style="padding:0 0 26px 12px"><div style="font-size:14px;font-weight:bold">Read the market <a href="https://optionintel.app/#market" style="color:#5B8DEF;text-decoration:none;font-weight:normal">· Market &#8599;</a></div>
        <div style="color:#4b5563;font-size:13px;line-height:1.55;margin-top:3px">The macro backdrop, rates, and the volatility regime. Work with the tape, not against it.</div></td>
    </tr>
    <tr>
      <td width="38" valign="top" style="border-left:2px solid #D7E3F8;padding:0 0 26px 0"><div style="width:27px;height:27px;line-height:27px;border-radius:50%;background:#5B8DEF;color:#fff;font-size:13px;font-weight:bold;text-align:center;margin-left:-14px;box-shadow:0 0 0 4px #ffffff">2</div></td>
      <td valign="top" style="padding:0 0 26px 12px"><div style="font-size:14px;font-weight:bold">Set the goal</div>
        <div style="color:#4b5563;font-size:13px;line-height:1.55;margin-top:3px">Income (a cash-secured put or covered call) or long exposure (a LEAP)? Decide before you look.</div></td>
    </tr>
    <tr>
      <td width="38" valign="top" style="border-left:2px solid #D7E3F8;padding:0 0 26px 0"><div style="width:27px;height:27px;line-height:27px;border-radius:50%;background:#5B8DEF;color:#fff;font-size:13px;font-weight:bold;text-align:center;margin-left:-14px;box-shadow:0 0 0 4px #ffffff">3</div></td>
      <td valign="top" style="padding:0 0 26px 12px"><div style="font-size:14px;font-weight:bold">Select the stock <a href="https://optionintel.app/#fund" style="color:#5B8DEF;text-decoration:none;font-weight:normal">· Fundamentals &#8599;</a></div>
        <div style="color:#4b5563;font-size:13px;line-height:1.55;margin-top:3px">Valuation and quality straight from SEC filings — room to find asymmetric names, not just the index.</div></td>
    </tr>
    <tr>
      <td width="38" valign="top" style="border-left:2px solid #D7E3F8;padding:0 0 26px 0"><div style="width:27px;height:27px;line-height:27px;border-radius:50%;background:#5B8DEF;color:#fff;font-size:13px;font-weight:bold;text-align:center;margin-left:-14px;box-shadow:0 0 0 4px #ffffff">4</div></td>
      <td valign="top" style="padding:0 0 26px 12px"><div style="font-size:14px;font-weight:bold">Confirm trend &amp; timing <a href="https://optionintel.app/#deep" style="color:#5B8DEF;text-decoration:none;font-weight:normal">· TA &#8599;</a></div>
        <div style="color:#4b5563;font-size:13px;line-height:1.55;margin-top:3px">Is the chart set up — and is now a sensible entry? Trend first, then the timing read.</div></td>
    </tr>
    <tr>
      <td width="38" valign="top" style="border-left:2px solid #D7E3F8;padding:0 0 26px 0"><div style="width:27px;height:27px;line-height:27px;border-radius:50%;background:#5B8DEF;color:#fff;font-size:13px;font-weight:bold;text-align:center;margin-left:-14px;box-shadow:0 0 0 4px #ffffff">5</div></td>
      <td valign="top" style="padding:0 0 26px 12px"><div style="font-size:14px;font-weight:bold">Scan the <a href="https://optionintel.app/#signals" style="color:#5B8DEF;text-decoration:none">Signals &#8599;</a></div>
        <div style="color:#4b5563;font-size:13px;line-height:1.55;margin-top:3px">Does anything match your criteria and interest? Note its timing score before acting.</div></td>
    </tr>
    <tr>
      <td width="38" valign="top" style="border-left:2px solid #D7E3F8;padding:0 0 26px 0"><div style="width:27px;height:27px;line-height:27px;border-radius:50%;background:#5B8DEF;color:#fff;font-size:13px;font-weight:bold;text-align:center;margin-left:-14px;box-shadow:0 0 0 4px #ffffff">6</div></td>
      <td valign="top" style="padding:0 0 26px 12px"><div style="font-size:14px;font-weight:bold">Structure the trade <a href="https://optionintel.app/#chain" style="color:#5B8DEF;text-decoration:none;font-weight:normal">· Options Chain &#8599;</a></div>
        <div style="color:#4b5563;font-size:13px;line-height:1.55;margin-top:3px">Hover a Bid to see the expected return at each Delta. &#916;&#8776;30 and 30&#8211;35 DTE come preset — both editable.</div></td>
    </tr>
    <tr>
      <td width="38" valign="top" style="border-left:2px solid #D7E3F8;padding:0"><div style="width:27px;height:27px;line-height:27px;border-radius:50%;background:#5B8DEF;color:#fff;font-size:13px;font-weight:bold;text-align:center;margin-left:-14px;box-shadow:0 0 0 4px #ffffff">7</div></td>
      <td valign="top" style="padding:0 0 0 12px"><div style="font-size:14px;font-weight:bold">Size it &amp; plan the exit</div>
        <div style="color:#4b5563;font-size:13px;line-height:1.55;margin-top:3px">Choose the position size, then decide the exit before entering — for both cases: it works out, or it goes against you.</div></td>
    </tr>
  </table>
  <div style="background:#FEF3C7;border:1px solid #FCD34D;border-radius:12px;padding:15px 17px;margin:26px 0 6px;color:#92400E;font-size:14px;line-height:1.6">
    <b>Above all — manage risk and position size.</b> No single trade should be able to hurt you; decide the exit before entering, for both outcomes.
  </div>
  <p style="margin:24px 0 6px"><a href="https://optionintel.app" style="background:#5B8DEF;color:#ffffff;text-decoration:none;padding:13px 24px;border-radius:9px;display:inline-block;font-weight:bold;font-size:14px">Open OptionIntel &rarr;</a></p>
  <p style="font-size:14px;line-height:1.65;margin:22px 0 2px">Feedback or a question? Just hit <b>Reply</b> — it reaches me directly.</p>
  <p style="font-size:14px;margin:0;color:#111827">— Jay</p>
  <p style="color:#9ca3af;font-size:12px;border-top:1px solid #e5e7eb;padding-top:14px;margin-top:22px;line-height:1.6">Research and education only — not financial advice, an offer, or a recommendation. You're receiving this once because you signed in at optionintel.app.</p>
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
