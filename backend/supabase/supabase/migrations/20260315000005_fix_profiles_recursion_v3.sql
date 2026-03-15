-- Fix: even SECURITY DEFINER plpgsql doesn't prevent recursion because
-- Postgres evaluates all SELECT policies on the table during UPDATE.
-- The gym owner policy must avoid touching profiles entirely.
-- Solution: store auth_user_id directly on gyms via a helper, or
-- restructure the policy to not need profile lookup.
--
-- We add owner_auth_user_id to gyms as a denormalized column so the
-- gym owner policy can check ownership without querying profiles.
-- This is maintained by the same code that sets owner_profile_id.

-- Step 1: Add denormalized column
alter table public.gyms add column if not exists owner_auth_user_id uuid;

-- Step 2: Backfill from current data
update public.gyms g
set owner_auth_user_id = p.auth_user_id
from public.profiles p
where g.owner_profile_id = p.id;

-- Step 3: Replace the recursive policy
drop policy if exists "Gym owners can read checked-in athlete profiles" on public.profiles;

create policy "Gym owners can read checked-in athlete profiles"
  on public.profiles for select
  using (
    exists (
      select 1 from public.gym_checkins gc
      join public.gyms g on gc.gym_id = g.id
      where gc.profile_id = profiles.id
        and gc.is_active = true
        and g.owner_auth_user_id = auth.uid()
    )
  );
