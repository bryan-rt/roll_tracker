-- Fix: profiles SELECT policy for gym owners causes infinite recursion (42P17)
-- because it queries profiles within a profiles policy.
-- Solution: use a SECURITY DEFINER helper function to look up the current
-- user's profile_id without triggering RLS on profiles.

create or replace function public.current_profile_id()
returns uuid as $$
  select id from public.profiles where auth_user_id = auth.uid() limit 1;
$$ language sql stable security definer;

drop policy if exists "Gym owners can read checked-in athlete profiles" on public.profiles;

create policy "Gym owners can read checked-in athlete profiles"
  on public.profiles for select
  using (
    exists (
      select 1 from public.gym_checkins gc
      join public.gyms g on gc.gym_id = g.id
      where gc.profile_id = profiles.id
        and gc.is_active = true
        and g.owner_profile_id = public.current_profile_id()
    )
  );
