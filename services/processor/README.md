# Processor Service

Polls `data/raw/nest/` for new MP4 recordings from `nest_recorder`, invokes the
`bjj_pipeline` (stages A→F) on each clip, and writes gym-scoped outputs that
the `uploader` service can consume.

## How it works

1. Scans `SCAN_ROOT` for `.mp4` files (optionally filtered by `GYM_ID`)
2. Skips clips whose output already contains `stage_F/export_manifest.jsonl`
3. Invokes `run_pipeline()` from `bjj_pipeline` programmatically
4. Emits structured JSON log lines to stdout for each event

## Output path convention

```
outputs/{gym_id}/{cam_id}/{YYYY-MM-DD}/{HH}/{clip_id}/
  stage_A/ stage_B/ stage_C/ stage_D/ stage_E/ stage_F/
```

Legacy (no gym_id): `outputs/legacy/{cam_id}/{date}/{hour}/{clip_id}/`

## Configuration

Copy `.env.example` to `.env`. See `contracts/input_output.md` for full env var docs.

## Running

```bash
# Build and run
docker compose up --build -d

# Or run locally (with bjj_pipeline installed)
SUPABASE_URL=http://... SUPABASE_SERVICE_ROLE_KEY=... python processor.py
```

## Docker volumes

| Mount | Container path | Mode |
|---|---|---|
| `../../data/raw/nest` | `/app/data/raw/nest` | read-only |
| `../../outputs` | `/app/outputs` | read-write |
| `../../configs` | `/app/configs` | read-only |
