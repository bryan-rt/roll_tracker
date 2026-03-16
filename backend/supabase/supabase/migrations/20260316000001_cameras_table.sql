-- cameras table: auto-registered by nest_recorder on discovery
-- Applied: 2026-03-16

begin;

create table if not exists public.cameras (
  id              uuid primary key default gen_random_uuid(),
  gym_id          uuid not null references public.gyms(id) on delete cascade,
  cam_id          text not null,
  device_path     text not null,
  display_name    text,
  is_active       boolean not null default true,
  first_seen_at   timestamptz not null default now(),
  last_seen_at    timestamptz not null default now()
);

-- Unique constraint required for PostgREST upsert (Prefer: resolution=merge-duplicates)
alter table public.cameras
  add constraint cameras_gym_cam_unique unique (gym_id, cam_id);

create index if not exists idx_cameras_gym_id on public.cameras(gym_id);

-- RLS: service role bypasses automatically (recorder uses service key)
alter table public.cameras enable row level security;

create policy "Gym owners can read their cameras"
  on public.cameras for select
  using (
    gym_id in (
      select id from public.gyms
      where owner_profile_id in (
        select id from public.profiles where auth_user_id = auth.uid()
      )
    )
  );

create policy "Gym owners can update their cameras"
  on public.cameras for update
  using (
    gym_id in (
      select id from public.gyms
      where owner_profile_id in (
        select id from public.profiles where auth_user_id = auth.uid()
      )
    )
  )
  with check (
    gym_id in (
      select id from public.gyms
      where owner_profile_id in (
        select id from public.profiles where auth_user_id = auth.uid()
      )
    )
  );

commit;
