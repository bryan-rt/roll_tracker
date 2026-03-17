-- Add app_version column to log_events
-- The Flutter AppLogger sends this field; without the column, inserts fail.
-- Applied: 2026-03-17

alter table public.log_events add column if not exists app_version text;
