-- profiles: tag identity + merchandise tracking
alter table public.profiles
  add column if not exists tag_id             integer,
  add column if not exists tag_assigned_at    timestamptz,
  add column if not exists starter_pack_sent_at timestamptz;

create index if not exists idx_profiles_tag_id
  on public.profiles(tag_id);
-- Note: no UNIQUE constraint on tag_id by design.
-- Uniqueness is enforced within (tag_id + gym_id + active time window).
-- See CV Design Constraints in CLAUDE.md.

-- videos: tie footage to the gym that owns it
alter table public.videos
  add column if not exists gym_id uuid references public.gyms(id) on delete set null;

create index if not exists idx_videos_gym
  on public.videos(gym_id);

-- clips: denormalized identity resolution written by Stage F
alter table public.clips
  add column if not exists fighter_a_profile_id uuid references public.profiles(id) on delete set null,
  add column if not exists fighter_b_profile_id uuid references public.profiles(id) on delete set null;

create index if not exists idx_clips_fighter_a
  on public.clips(fighter_a_profile_id);

create index if not exists idx_clips_fighter_b
  on public.clips(fighter_b_profile_id);
