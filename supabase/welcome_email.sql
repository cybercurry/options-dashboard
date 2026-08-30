-- optionintel.app — one-time WELCOME email for each new subscriber.
--
-- Sends a branded "what this is / how to use it" email the first time a new user is created
-- (i.e. their first magic-link sign-in), via Resend's HTTP API called straight from Postgres with
-- pg_net. Fires exactly once per subscriber (auth.users row is inserted once), never again on
-- later sign-ins, and never blocks sign-up if the send fails.
--
-- PREREQUISITES (already done for SMTP): your domain is verified in Resend and you have a Resend
-- API key (re_…). You can reuse that key or make a new "welcome-email" one (Sending access,
-- domain optionintel.app).
--
-- INSTALL (Supabase dashboard):
--   1) Database → Extensions → enable  pg_net
--   2) SQL Editor → paste this whole file, replace RESEND_API_KEY_HERE with your re_… key → Run
--   3) Test: sign in with a fresh address (e.g. you+test@yourdomain) → the welcome email arrives.
--
-- SECURITY: the key lives only in this server-side function (SECURITY DEFINER); it is never sent
-- to the browser or the anon client. To keep it out of the function body entirely, see the Vault
-- variant noted at the bottom.

create extension if not exists pg_net;

create or replace function public.send_welcome_email()
returns trigger
language plpgsql
security definer
set search_path = public
as $fn$
declare
  resend_key text := 'RESEND_API_KEY_HERE';   -- ← paste your Resend re_… key
begin
  perform net.http_post(
    url     := 'https://api.resend.com/emails',
    headers := jsonb_build_object(
                 'Authorization', 'Bearer ' || resend_key,
                 'Content-Type',  'application/json'),
    body    := jsonb_build_object(
      'from',    'OptionIntel <noreply@optionintel.app>',
      'to',      NEW.email,
      'subject', 'Welcome to OptionIntel — how to use it',
      'text',    'Welcome to OptionIntel (optionintel.app) — options intelligence for the wheel '
              || 'strategy, research only (not advice). It scans a curated watchlist and your own '
              || 'list for cash-secured puts, covered calls and LEAPs, using transparent rules you '
              || 'can inspect. Tabs: Overview (market pulse + watchlist), Market (rates, yield '
              || 'curve, volatility, calendar), Options Chain, Signals (suggested entries), TA '
              || '(per-ticker breakdown), Fundamentals (SEC filings, any US ticker), Rules (the '
              || 'exact logic). Add any ticker with + Add or the star, then switch to "My List" in '
              || 'the header — your list is private and synced across devices. Open '
              || 'https://optionintel.app',
      'html',    $html$
<div style="font-family:Arial,Helvetica,sans-serif;max-width:560px;margin:0 auto;color:#111827">
  <h2 style="margin:0 0 4px">Welcome to OptionIntel 👋</h2>
  <p style="color:#6b7280;margin:0 0 18px">Options intelligence for the wheel strategy — research, not advice.</p>
  <p style="line-height:1.6">Thanks for signing in. OptionIntel scans a curated watchlist — and now
     <b>your own list</b> — and shows where <b>cash-secured puts</b>, <b>covered calls</b> and
     <b>LEAPs</b> line up, using transparent rules you can inspect.</p>
  <p style="margin:18px 0 6px"><b>What each tab does</b></p>
  <ul style="padding-left:18px;color:#374151;line-height:1.7;margin:0">
    <li><b>Overview</b> — market pulse, sector heatmap, and the full watchlist read.</li>
    <li><b>Market</b> — rates, the yield curve, the volatility regime, and this week's calendar.</li>
    <li><b>Options Chain</b> — real calls &amp; puts with delta, IV and open interest.</li>
    <li><b>Signals</b> — the suggested CSP / CC entries and qualifying LEAPs, right now.</li>
    <li><b>TA</b> — one ticker, every leg, with the full reasoning behind each verdict.</li>
    <li><b>Fundamentals</b> — straight from SEC filings; look up any US-listed company.</li>
    <li><b>Rules</b> — exactly what the engine checks. No black box.</li>
  </ul>
  <p style="margin:18px 0 6px"><b>Make it yours</b></p>
  <p style="line-height:1.6;margin:0 0 18px">Add any ticker (the <b>+ Add</b> box or the ★ on any row),
     then flip the whole app to <b>My List</b> from the header switch. Your list is private to you
     and synced across your devices — the public default list never changes.</p>
  <p style="margin:22px 0">
    <a href="https://optionintel.app"
       style="background:#5B8DEF;color:#ffffff;text-decoration:none;padding:11px 20px;border-radius:8px;display:inline-block;font-weight:bold">Open OptionIntel →</a>
  </p>
  <p style="color:#9ca3af;font-size:12px;border-top:1px solid #e5e7eb;padding-top:12px;line-height:1.6">
     OptionIntel is for research and education only — it is not financial advice, an offer, or a
     recommendation. You're receiving this once because you signed in at optionintel.app.</p>
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
