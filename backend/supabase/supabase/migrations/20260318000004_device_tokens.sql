-- device_tokens: stores FCM tokens for push notifications.
-- Athletes manage their own tokens via RLS.
-- Applied: 2026-03-18

create table if not exists public.device_tokens (
  id uuid primary key default gen_random_uuid(),
  profile_id uuid not null references public.profiles(id) on delete cascade,
  token text not null,
  platform text not null default 'android',
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  unique(profile_id, token)
);

alter table public.device_tokens enable row level security;

create policy "Athletes manage their own tokens"
  on public.device_tokens
  for all
  using (
    profile_id in (
      select id from public.profiles
      where auth_user_id = auth.uid()
    )
  )
  with check (
    profile_id in (
      select id from public.profiles
      where auth_user_id = auth.uid()
    )
  );
