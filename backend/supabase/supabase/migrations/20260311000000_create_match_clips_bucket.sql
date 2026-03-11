-- Ensure storage bucket exists for match clips
insert into storage.buckets (id, name, public)
values ('match-clips', 'match-clips', false)
on conflict (id) do nothing;

