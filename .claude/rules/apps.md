---
paths:
  - "app_mobile/**"
  - "app_web/**"
---

# Mobile & Web Apps

## Flutter Mobile (`app_mobile/`)
- **Auth:** supabase_flutter. Auth trigger auto-creates profile with tag_id on sign-up.
  Biometric login gated behind Settings toggle (default off).
- **Onboarding:** display name → gym select → invite gym (if not listed).
  AuthGate FutureBuilder with profile completeness check.
- **Clips:** Pull-to-refresh list. Tap to play via signed URL + video_player.
  RLS scopes to athlete's profile (fighter_a/b_profile_id match).
- **Check-in:** WiFi auto check-in (CheckinService) fires after auth + on WiFi changes.
  Upserts on `(profile_id, gym_id)` — sliding 3hr TTL via hourly periodic probe.
  Timer cancelled on WiFi disconnect. SSID-primary matching (BSSID optional).
  Source tracked as `wifi_auto` or `manual`.
- **Gym discovery:** Find a Gym screen with GPS proximity via `gyms_near` RPC.
- **Android:** `usesCleartextTraffic=true` for local HTTP Supabase.
  `ACCESS_FINE_LOCATION` for WiFi SSID + GPS.
- **Dev:** Flutter not on PATH — use `~/development/flutter/bin/flutter`.
  Local Supabase config commented out in `supabase_config.dart` (192.168.0.66:54321).
  Signed URLs rewrite `127.0.0.1` → configured host for phone access.

## Vite+React Web (`app_web/`)
- **Routes:** `/` mat blueprint editor (Konva canvas), `/admin/pricing` business model simulator.
- **Auth:** `AdminGate` component checks session email against `VITE_ADMIN_EMAIL` env var.
  Email+password sign-in via Supabase.
- **Dev:** `.env.example` provided. Set VITE_SUPABASE_URL, VITE_SUPABASE_ANON_KEY,
  VITE_ADMIN_EMAIL.

## Supabase Key Format
Remote uses `sb_publishable_`/`sb_secret_` naming, but PostgREST API requires classic
JWT keys (`eyJ...` format) from Dashboard > Settings > API Keys.
