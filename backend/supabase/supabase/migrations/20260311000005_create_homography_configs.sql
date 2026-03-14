create table if not exists public.homography_configs (
  id          uuid primary key default gen_random_uuid(),
  gym_id      uuid not null references public.gyms(id) on delete cascade,
  camera_id   text not null,
  config_data jsonb not null default '{}'::jsonb,
  created_at  timestamptz default now(),
  updated_at  timestamptz default now(),

  unique(gym_id, camera_id)
);

create index if not exists idx_homography_configs_gym
  on public.homography_configs(gym_id);

create trigger set_homography_configs_updated_at
before update on public.homography_configs
for each row
execute function public.set_updated_at();
