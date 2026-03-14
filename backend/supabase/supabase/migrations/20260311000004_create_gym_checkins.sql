create table if not exists public.gym_checkins (
  id              uuid primary key default gen_random_uuid(),
  profile_id      uuid not null references public.profiles(id) on delete cascade,
  gym_id          uuid not null references public.gyms(id) on delete cascade,
  checked_in_at   timestamptz not null default now(),
  auto_expires_at timestamptz not null default (now() + interval '3 hours'),
  is_active       boolean not null default true
);

-- Keep auto_expires_at in sync when checked_in_at is set or updated
create or replace function public.set_checkin_expiry()
returns trigger
language plpgsql
as $$
begin
  new.auto_expires_at = new.checked_in_at + interval '3 hours';
  return new;
end;
$$;

create trigger set_gym_checkins_expiry
before insert or update on public.gym_checkins
for each row
execute function public.set_checkin_expiry();

create index if not exists idx_gym_checkins_profile
  on public.gym_checkins(profile_id);

create index if not exists idx_gym_checkins_gym_time
  on public.gym_checkins(gym_id, checked_in_at);
