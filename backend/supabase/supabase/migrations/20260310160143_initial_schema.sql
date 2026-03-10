begin;

create extension if not exists pgcrypto;

create or replace function public.set_updated_at()
returns trigger
language plpgsql
as $$
begin
  new.updated_at = now();
  return new;
end;
$$;

create table if not exists public.profiles (
  id uuid primary key default gen_random_uuid(),
  auth_user_id uuid unique,
  display_name text not null,
  email text unique,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table if not exists public.videos (
  id uuid primary key default gen_random_uuid(),
  camera_id text,
  source_path text,
  source_type text,
  recorded_at timestamptz,
  duration_seconds numeric,
  status text not null default 'new',
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table if not exists public.clips (
  id uuid primary key default gen_random_uuid(),
  video_id uuid references public.videos(id) on delete set null,
  match_id text,
  clip_type text,
  file_path text not null,
  storage_bucket text,
  storage_object_path text,
  start_seconds numeric,
  end_seconds numeric,
  duration_seconds numeric,
  camera_id text,
  status text not null default 'created',
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table if not exists public.log_events (
  id bigserial primary key,
  clip_id uuid references public.clips(id) on delete cascade,
  video_id uuid references public.videos(id) on delete cascade,
  event_type text not null,
  event_level text not null default 'info',
  message text,
  details jsonb not null default '{}'::jsonb,
  event_time timestamptz not null default now(),
  created_at timestamptz not null default now()
);

create index if not exists idx_profiles_auth_user_id
  on public.profiles(auth_user_id);

create index if not exists idx_videos_camera_id
  on public.videos(camera_id);

create index if not exists idx_videos_recorded_at
  on public.videos(recorded_at);

create index if not exists idx_clips_video_id
  on public.clips(video_id);

create index if not exists idx_clips_match_id
  on public.clips(match_id);

create index if not exists idx_log_events_clip_id
  on public.log_events(clip_id);

create index if not exists idx_log_events_video_id
  on public.log_events(video_id);

create index if not exists idx_log_events_event_type
  on public.log_events(event_type);

create trigger set_profiles_updated_at
before update on public.profiles
for each row
execute function public.set_updated_at();

create trigger set_videos_updated_at
before update on public.videos
for each row
execute function public.set_updated_at();

create trigger set_clips_updated_at
before update on public.clips
for each row
execute function public.set_updated_at();

commit;