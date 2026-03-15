-- Storage policies for match-clips bucket.
-- Authenticated users can read (download/signed URL) objects in the bucket.
-- Only service role can insert/update/delete (uploader service).

create policy "Authenticated users can read match clips"
  on storage.objects for select
  using (
    bucket_id = 'match-clips'
    and auth.role() = 'authenticated'
  );
