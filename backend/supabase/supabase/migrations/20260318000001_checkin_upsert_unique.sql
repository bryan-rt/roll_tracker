-- Add unique constraint on (profile_id, gym_id) for gym_checkins
-- Enables upsert from Flutter CheckinService for sliding TTL.
-- The existing set_checkin_expiry trigger fires on INSERT OR UPDATE,
-- so each upsert with checked_in_at=now() slides auto_expires_at forward.
-- Applied: 2026-03-18

alter table public.gym_checkins
  add constraint gym_checkins_profile_gym_unique unique (profile_id, gym_id);
