alter table public.clips
  add column if not exists fighter_a_tag_id bigint,
  add column if not exists fighter_b_tag_id bigint;

create index if not exists idx_clips_fighter_tags
  on public.clips (fighter_a_tag_id, fighter_b_tag_id);