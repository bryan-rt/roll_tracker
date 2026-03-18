-- Allow authenticated users to insert log events from the mobile app.
-- The app writes event_type, event_level, message, details, app_version.
-- clip_id and video_id are left null (set by uploader, not mobile app).
-- Applied: 2026-03-18

create policy "Authenticated users can insert log events"
  on public.log_events for insert
  with check (auth.role() = 'authenticated');
