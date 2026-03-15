-- Phase E supplement: check-in source tracking + auto tag_id assignment

begin;

-- ============================================================
-- 1. Check-in source column
-- ============================================================

alter table public.gym_checkins
  add column source text not null default 'manual';

comment on column public.gym_checkins.source is
  'How the check-in was created: manual or wifi_auto';

-- ============================================================
-- 2. Tag ID auto-assignment via cycling sequence (36h11: 0–586)
-- ============================================================

create sequence public.tag_id_seq
  minvalue 0 maxvalue 586 start 0 cycle;

-- Update handle_new_user to assign tag_id on sign-up
create or replace function public.handle_new_user()
returns trigger as $$
begin
  insert into public.profiles (auth_user_id, email, tag_id, tag_assigned_at)
  values (new.id, new.email, nextval('public.tag_id_seq'), now());
  return new;
end;
$$ language plpgsql security definer;

-- Backfill any existing profiles missing a tag_id
update public.profiles
set tag_id = nextval('public.tag_id_seq'),
    tag_assigned_at = now()
where tag_id is null
  and auth_user_id is not null;

commit;
