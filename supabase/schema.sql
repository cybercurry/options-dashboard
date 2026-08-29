-- optionintel.app — Phase 1: per-user private watchlists.
-- Run once in the Supabase project's SQL editor (SQL Editor → New query → paste → Run).
--
-- Each registered user gets their own list of tickers, visible and editable ONLY by them
-- (row-level security). The public default watchlist is a separate static file and is NEVER
-- touched by this table — a user's private list can never bleed into what other visitors see.

create table if not exists public.watchlist (
  user_id    uuid        not null references auth.users(id) on delete cascade,
  ticker     text        not null check (ticker = upper(ticker) and char_length(ticker) between 1 and 8),
  created_at timestamptz not null default now(),
  primary key (user_id, ticker)
);

alter table public.watchlist enable row level security;

-- Own-rows-only policies.
create policy "watchlist select own" on public.watchlist
  for select using (auth.uid() = user_id);
create policy "watchlist insert own" on public.watchlist
  for insert with check (auth.uid() = user_id);
create policy "watchlist delete own" on public.watchlist
  for delete using (auth.uid() = user_id);

-- The registered-user email list lives in Supabase's managed `auth.users` table — view it in the
-- dashboard under Authentication → Users, or query it in the SQL editor:
--   select email, created_at, last_sign_in_at from auth.users order by created_at desc;
