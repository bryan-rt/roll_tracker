-- Remove gym_members (replaced by home_gym_id on profiles)
-- gym_role enum is only used by gym_members so drop it too
drop table if exists public.gym_members;
drop type if exists gym_role;

-- Add home_gym_id to profiles
alter table public.profiles
  add column if not exists home_gym_id uuid
    references public.gyms(id) on delete set null;

create index if not exists idx_profiles_home_gym
  on public.profiles(home_gym_id);

-- Add coordinates to gyms for future map feature
alter table public.gyms
  add column if not exists latitude  numeric,
  add column if not exists longitude numeric;

-- Capture athlete demand signal for unenrolled gyms
-- Auth-required: profile_id is always the signed-in athlete
create table if not exists public.gym_interest_signals (
  id               uuid primary key default gen_random_uuid(),
  profile_id       uuid not null references public.profiles(id) on delete cascade,
  gym_name_entered text not null,
  submitted_at     timestamptz not null default now()
);

create index if not exists idx_gym_interest_signals_profile
  on public.gym_interest_signals(profile_id);
