-- Fix: the "Gym owners can read checked-in athlete profiles" policy
-- causes infinite recursion regardless of approach because:
-- profiles policy → EXISTS on gym_checkins → gym_checkins RLS
-- policies query profiles → recursion.
--
-- Dropping this policy for now. Gym owner profile reads will be
-- re-implemented via a SECURITY DEFINER RPC function that bypasses
-- RLS entirely, rather than as a row-level policy.

drop policy if exists "Gym owners can read checked-in athlete profiles" on public.profiles;
