-- Fix: profiles UPDATE policy needs WITH CHECK for PostgREST
-- Without WITH CHECK, updates are silently blocked.

drop policy if exists "Athletes can update own profile" on public.profiles;

create policy "Athletes can update own profile"
  on public.profiles for update
  using (auth_user_id = auth.uid())
  with check (auth_user_id = auth.uid());
