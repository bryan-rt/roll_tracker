-- CP14f: cross-camera identity merge — global person IDs on clips
ALTER TABLE clips
  ADD COLUMN IF NOT EXISTS global_person_id_a text,
  ADD COLUMN IF NOT EXISTS global_person_id_b text;
