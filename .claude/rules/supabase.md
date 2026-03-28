---
paths:
  - "backend/supabase/**"
---

# Supabase Schema & Conventions

## Tables (11 total, RLS on all, service role bypasses)
- `profiles` — auth_user_id FK, display_name (nullable), email, tag_id (0–586 cycling seq),
  home_gym_id FK→gyms
- `videos` — camera_id, source_path, recorded_at, status, metadata jsonb, gym_id FK→gyms
- `clips` — video_id FK, match_id, storage fields, fighter_a/b_tag_id, fighter_a/b_profile_id
  (nullable FK→profiles), global_person_id_a/b (text), source_video_ids text[].
  Status CHECK: created, exported_local, uploaded, collision_flagged.
- `log_events` — clip_id/video_id FK, event_type/level, message, details, app_version
- `gyms` — name, owner_profile_id, owner_auth_user_id (denormalized), address, wifi_ssid/bssid,
  lat/lng
- `gym_checkins` — profile_id, gym_id, checked_in_at, auto_expires_at (+3hr trigger),
  is_active, source (manual|wifi_auto). UNIQUE(profile_id, gym_id) for upsert.
- `gym_subscriptions` — gym_id, tier ENUM, started_at, ended_at, is_current
- `cameras` — gym_id FK, cam_id (last 6 of SDM path), device_path, display_name, is_active.
  UNIQUE(gym_id, cam_id). Auto-registered by nest_recorder.
- `homography_configs` — gym_id, camera_id, config_data JSONB
- `gym_interest_signals` — profile_id, gym_name_entered, owner_email, submitted_at
- `device_tokens` — profile_id FK, token (FCM), platform. UNIQUE(profile_id, token).

## Auth & RPCs
- `handle_new_user()` trigger on auth.users INSERT → auto-creates profile with cycling tag_id.
- `gyms_near(lat, lng, radius_km)` — Haversine, no PostGIS.
- `current_profile_id()` — SECURITY DEFINER, plpgsql (prevents inlining/recursion).
- `get_claimable_clips()` / `claim_clip()` — SECURITY DEFINER RPCs for athlete clip claiming.

## Storage
- Bucket: `match-clips` (private). RLS allows authenticated reads for signed URLs.

## Migration Conventions
- Naming: `YYYYMMDDHHMMSS_descriptive_name.sql`. Currently 23 applied.
- Never edit applied migrations. New changes = new migration file.
- Session pooler URL (port 5432, Supavisor) — not direct connection (IPv6-only fails in Docker).
- Remote uses `sb_publishable_`/`sb_secret_` key naming, but PostgREST needs classic JWT
  `eyJ...` keys from Dashboard > Settings > API Keys.

## Pending Schema (not yet migrated)
- notification_channel (drift alert delivery)
- Gym owner profile read policy (needs RPC to avoid RLS recursion)
- K + distortion coefficients per camera (extends homography_configs or new table)
- Drift scores per camera (daily score, baseline snapshot, alert status)
