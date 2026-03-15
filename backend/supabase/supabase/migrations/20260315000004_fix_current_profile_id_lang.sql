-- Fix: current_profile_id() still causes recursion because SQL language
-- functions can be inlined by Postgres, bypassing SECURITY DEFINER.
-- Switching to plpgsql prevents inlining and ensures RLS bypass works.

create or replace function public.current_profile_id()
returns uuid as $$
declare
  pid uuid;
begin
  select id into pid from public.profiles where auth_user_id = auth.uid() limit 1;
  return pid;
end;
$$ language plpgsql stable security definer;
