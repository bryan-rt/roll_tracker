create table if not exists public.gyms (
  id               uuid primary key default gen_random_uuid(),
  name             text not null,
  owner_profile_id uuid not null references public.profiles(id) on delete restrict,
  address          text,
  wifi_ssid        text,
  wifi_bssid       text,
  created_at       timestamptz default now(),
  updated_at       timestamptz default now()
);

create index if not exists idx_gyms_owner
  on public.gyms(owner_profile_id);

create trigger set_gyms_updated_at
before update on public.gyms
for each row
execute function public.set_updated_at();
