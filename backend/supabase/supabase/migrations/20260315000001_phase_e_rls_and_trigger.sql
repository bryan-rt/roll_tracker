-- Phase E: RLS policies, auth trigger, gym discovery function
-- Applied: 2026-03-15

begin;

-- ============================================================
-- 0. Schema adjustments required for onboarding flow
-- ============================================================

-- display_name must be nullable so the auth trigger can create
-- a bare profile row; onboarding collects the name afterward.
alter table public.profiles
  alter column display_name drop not null;

-- gym_interest_signals needs an owner_email column for the
-- "invite your gym" onboarding screen.
alter table public.gym_interest_signals
  add column owner_email text;

-- ============================================================
-- 1A. Auth trigger: auto-create profiles row on sign-up
-- ============================================================

create or replace function public.handle_new_user()
returns trigger as $$
begin
  insert into public.profiles (auth_user_id, email)
  values (new.id, new.email);
  return new;
end;
$$ language plpgsql security definer;

create or replace trigger on_auth_user_created
  after insert on auth.users
  for each row execute function public.handle_new_user();

-- ============================================================
-- 1B. Proximity function: gyms_near (Haversine, no PostGIS)
-- ============================================================

create or replace function public.gyms_near(
  lat float,
  lng float,
  radius_km float default 50
)
returns setof public.gyms as $$
  select * from public.gyms
  where latitude is not null and longitude is not null
    and (
      6371 * acos(
        cos(radians(lat)) * cos(radians(latitude)) *
        cos(radians(longitude) - radians(lng)) +
        sin(radians(lat)) * sin(radians(latitude))
      )
    ) < radius_km
  order by (
    6371 * acos(
      cos(radians(lat)) * cos(radians(latitude)) *
      cos(radians(longitude) - radians(lng)) +
      sin(radians(lat)) * sin(radians(latitude))
    )
  );
$$ language sql stable security definer;

-- ============================================================
-- 1C. Enable RLS on all public tables and apply policies
-- ============================================================

-- ---- profiles ----
alter table public.profiles enable row level security;

create policy "Athletes can read own profile"
  on public.profiles for select
  using (auth_user_id = auth.uid());

create policy "Gym owners can read checked-in athlete profiles"
  on public.profiles for select
  using (
    exists (
      select 1 from public.gym_checkins gc
      join public.gyms g on gc.gym_id = g.id
      join public.profiles op on g.owner_profile_id = op.id
      where gc.profile_id = profiles.id
        and gc.is_active = true
        and op.auth_user_id = auth.uid()
    )
  );

create policy "Athletes can update own profile"
  on public.profiles for update
  using (auth_user_id = auth.uid());

-- ---- gyms ----
alter table public.gyms enable row level security;

create policy "Authenticated users can read gyms"
  on public.gyms for select
  using (auth.role() = 'authenticated');

create policy "Gym owners can update their gym"
  on public.gyms for update
  using (
    owner_profile_id in (
      select id from public.profiles where auth_user_id = auth.uid()
    )
  );

-- ---- gym_checkins ----
alter table public.gym_checkins enable row level security;

create policy "Athletes can read own check-ins"
  on public.gym_checkins for select
  using (
    profile_id in (
      select id from public.profiles where auth_user_id = auth.uid()
    )
  );

create policy "Gym owners can read check-ins at their gym"
  on public.gym_checkins for select
  using (
    gym_id in (
      select id from public.gyms where owner_profile_id in (
        select id from public.profiles where auth_user_id = auth.uid()
      )
    )
  );

create policy "Athletes can insert own check-ins"
  on public.gym_checkins for insert
  with check (
    profile_id in (
      select id from public.profiles where auth_user_id = auth.uid()
    )
  );

-- ---- gym_subscriptions ----
alter table public.gym_subscriptions enable row level security;

create policy "Gym owners can manage their subscription"
  on public.gym_subscriptions for all
  using (
    gym_id in (
      select id from public.gyms where owner_profile_id in (
        select id from public.profiles where auth_user_id = auth.uid()
      )
    )
  )
  with check (
    gym_id in (
      select id from public.gyms where owner_profile_id in (
        select id from public.profiles where auth_user_id = auth.uid()
      )
    )
  );

-- ---- homography_configs ----
alter table public.homography_configs enable row level security;

create policy "Gym owners can manage homography configs for their gym"
  on public.homography_configs for all
  using (
    gym_id in (
      select id from public.gyms where owner_profile_id in (
        select id from public.profiles where auth_user_id = auth.uid()
      )
    )
  )
  with check (
    gym_id in (
      select id from public.gyms where owner_profile_id in (
        select id from public.profiles where auth_user_id = auth.uid()
      )
    )
  );

-- ---- videos ----
alter table public.videos enable row level security;

create policy "Gym owners can read their gym videos"
  on public.videos for select
  using (
    gym_id in (
      select id from public.gyms where owner_profile_id in (
        select id from public.profiles where auth_user_id = auth.uid()
      )
    )
  );

-- ---- clips ----
alter table public.clips enable row level security;

create policy "Athletes can read their own clips"
  on public.clips for select
  using (
    fighter_a_profile_id in (
      select id from public.profiles where auth_user_id = auth.uid()
    )
    or
    fighter_b_profile_id in (
      select id from public.profiles where auth_user_id = auth.uid()
    )
  );

create policy "Gym owners can read clips from their gym"
  on public.clips for select
  using (
    video_id in (
      select id from public.videos where gym_id in (
        select id from public.gyms where owner_profile_id in (
          select id from public.profiles where auth_user_id = auth.uid()
        )
      )
    )
  );

-- ---- log_events ----
alter table public.log_events enable row level security;
-- No user-facing policies — service role only

-- ---- gym_interest_signals ----
alter table public.gym_interest_signals enable row level security;

create policy "Authenticated users can submit gym interest"
  on public.gym_interest_signals for insert
  with check (auth.role() = 'authenticated');

commit;
