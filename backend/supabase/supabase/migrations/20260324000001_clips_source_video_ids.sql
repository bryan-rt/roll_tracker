-- Add source_video_ids array column to clips
-- Handles matches spanning multiple 2.5-minute source segments (CP14e)
ALTER TABLE clips ADD COLUMN source_video_ids text[];

-- Backfill from existing video_id where present
UPDATE clips
SET source_video_ids = ARRAY[video_id::text]
WHERE video_id IS NOT NULL
  AND source_video_ids IS NULL;

-- video_id remains nullable for backward compatibility
-- source_video_ids is the canonical field going forward
