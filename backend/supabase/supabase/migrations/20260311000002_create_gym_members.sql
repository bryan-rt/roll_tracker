create type gym_role as enum ('gym_owner', 'athlete');

create table if not exists public.gym_members (
  id         uuid primary key default gen_random_uuid(),
  profile_id uuid not null unique references public.profiles(id) on delete cascade,
  gym_id     uuid not null references public.gyms(id) on delete cascade,
  role       gym_role not null default 'athlete',
  joined_at  timestamptz default now()
);

create index if not exists idx_gym_members_gym
  on public.gym_members(gym_id);
