create type subscription_tier as enum ('free', 'pro', 'enterprise');

create table if not exists public.gym_subscriptions (
  id         uuid primary key default gen_random_uuid(),
  gym_id     uuid not null references public.gyms(id) on delete cascade,
  tier       subscription_tier not null default 'free',
  started_at timestamptz default now(),
  ended_at   timestamptz,
  is_current boolean not null default true,
  created_at timestamptz default now()
);

create index if not exists idx_gym_subscriptions_gym
  on public.gym_subscriptions(gym_id);

-- Only one active subscription per gym at a time
create unique index if not exists idx_gym_subscriptions_current
  on public.gym_subscriptions(gym_id)
  where is_current = true;
