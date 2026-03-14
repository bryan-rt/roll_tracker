create table if not exists public.gym_checkins (
  id              uuid primary key default gen_random_uuid(),
  profile_id      uuid not null references public.profiles(id) on delete cascade,
  gym_id          uuid not null references public.gyms(id) on delete cascade,
  checked_in_at   timestamptz not null default now(),
  auto_expires_at timestamptz
                    generated always as (checked_in_at + interval '3 hours') stored,
  is_active       boolean not null default true
);

create index if not exists idx_gym_checkins_profile
  on public.gym_checkins(profile_id);

create index if not exists idx_gym_checkins_gym_time
  on public.gym_checkins(gym_id, checked_in_at);
