-- get_claimable_clips: returns clips with unresolved profile_ids for a given tag+gym.
-- claim_clip: allows an authenticated user to claim a clip as theirs.
-- Both SECURITY DEFINER to bypass RLS.
-- Applied: 2026-03-18

create or replace function public.get_claimable_clips(
  p_tag_id integer,
  p_gym_id uuid,
  p_window_hours integer default 24
)
returns table (
  clip_id uuid,
  match_id text,
  start_seconds float,
  end_seconds float,
  storage_object_path text,
  storage_bucket text,
  fighter_side text,
  created_at timestamptz
)
language plpgsql
security definer
as $$
begin
  return query
  select
    c.id,
    c.match_id,
    c.start_seconds::float,
    c.end_seconds::float,
    c.storage_object_path,
    c.storage_bucket,
    case
      when c.fighter_a_tag_id = p_tag_id then 'a'
      else 'b'
    end as fighter_side,
    c.created_at
  from clips c
  join videos v on c.video_id = v.id
  where v.gym_id = p_gym_id
    and (c.fighter_a_tag_id = p_tag_id or c.fighter_b_tag_id = p_tag_id)
    and (
      (c.fighter_a_tag_id = p_tag_id and c.fighter_a_profile_id is null)
      or
      (c.fighter_b_tag_id = p_tag_id and c.fighter_b_profile_id is null)
    )
    and c.created_at >= now() - (p_window_hours || ' hours')::interval;
end;
$$;


create or replace function public.claim_clip(
  p_clip_id uuid,
  p_fighter_side text  -- 'a' or 'b'
)
returns void
language plpgsql
security definer
as $$
declare
  v_profile_id uuid;
begin
  select id into v_profile_id
  from profiles
  where auth_user_id = auth.uid()
  limit 1;

  if v_profile_id is null then
    raise exception 'No profile found for current user';
  end if;

  if p_fighter_side = 'a' then
    update clips
    set fighter_a_profile_id = v_profile_id, status = 'uploaded'
    where id = p_clip_id and fighter_a_profile_id is null;
  elsif p_fighter_side = 'b' then
    update clips
    set fighter_b_profile_id = v_profile_id, status = 'uploaded'
    where id = p_clip_id and fighter_b_profile_id is null;
  else
    raise exception 'fighter_side must be a or b';
  end if;
end;
$$;
