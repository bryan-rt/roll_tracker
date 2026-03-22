# Processor Service Contracts — Input/Output

Version: 1.0
Status: Implemented

## Filesystem Contracts

### Input (recordings from nest_recorder)

Production (gym-scoped):
```
data/raw/nest/{gym_id}/{cam_id}/{YYYY-MM-DD}/{HH}/{cam_id}-{timestamp}.mp4
```

Legacy:
```
data/raw/nest/{cam_id}/{YYYY-MM-DD}/{HH}/{cam_id}-{timestamp}.mp4
```

The processor scans `SCAN_ROOT` (default: `data/raw/nest/`) for `.mp4` files.
If `GYM_ID` is set, scanning is restricted to that gym's subdirectory.
Files under `diag/` are excluded.

### Output (pipeline artifacts for uploader)

Gym-scoped:
```
outputs/{gym_id}/{cam_id}/{YYYY-MM-DD}/{HH}/{clip_id}/
  clip_manifest.json
  stage_A/
  stage_B/
  stage_C/
  stage_D/
  stage_E/
  stage_F/
    export_manifest.jsonl   ← uploader watches for this
    exports/
      {clip_files}.mp4
```

Legacy (no gym_id in ingest path):
```
outputs/legacy/{cam_id}/{YYYY-MM-DD}/{HH}/{clip_id}/
  ...same stage structure...
```

### Already-processed guard

A clip is skipped if its expected output directory already contains
`stage_F/export_manifest.jsonl`. The output path is derived deterministically
from the ingest path using `validate_ingest_path()`.

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `SCAN_ROOT` | No | `data/raw/nest` | Root directory to scan for MP4s |
| `OUTPUT_ROOT` | No | `outputs` | Root directory for pipeline outputs |
| `POLL_INTERVAL_SECONDS` | No | `30` | Seconds between scan cycles |
| `GYM_ID` | No | None | Restrict scanning to one gym |
| `SUPABASE_URL` | Yes | — | Supabase instance URL |
| `SUPABASE_SERVICE_ROLE_KEY` | Yes | — | Service role key for DB writes |

## Notes

- The processor does not open its own Supabase client. All DB writes
  go through Stage F's existing path.
- Communication with other services is filesystem-only (hub rule:
  all services communicate via Supabase or shared filesystem).
- `validate_ingest_path()` is imported from `bjj_pipeline` — path
  parsing logic is not duplicated.
