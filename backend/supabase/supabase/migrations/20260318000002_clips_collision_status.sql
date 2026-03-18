-- Add CHECK constraint on clips.status to enumerate valid values.
-- Existing codebase values: 'created' (schema default), 'exported_local' (Stage F),
-- 'uploaded' (uploader). New value: 'collision_flagged' (uploader collision detection).
-- Applied: 2026-03-18

alter table public.clips
  add constraint clips_status_check
  check (status in ('created', 'exported_local', 'uploaded', 'collision_flagged'));
