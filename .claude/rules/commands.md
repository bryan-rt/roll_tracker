---
description: Common development, pipeline, and service commands
---

# Common Commands

## Install (clean)
```bash
rm -rf .venv && python3.12 -m venv .venv && source .venv/bin/activate
pip install -U pip setuptools wheel && pip install -r requirements.txt
pip install --no-deps ultralytics boxmot && pip install -e .
```

## Camera Calibration (wizard)
```bash
# Full 3-step calibration (initial H → lens cal → final H refinement)
python -m bjj_pipeline.tools.calibrate_camera --camera J_EDEw \
  --video data/raw/nest/calibration_test/J_EDEw/<mp4>

# H-only recalibration (skip lens cal)
python -m bjj_pipeline.tools.calibrate_camera --camera J_EDEw \
  --video ... --skip-lens

# All cameras in one go
python -m bjj_pipeline.tools.calibrate_camera \
  --camera FP7oJQ J_EDEw PPDmUg --video <fp7.mp4> <jed.mp4> <ppd.mp4>
```

## Pipeline
```bash
# Full run
python -m bjj_pipeline.stages.orchestration.cli run --input <clip.mp4> --camera cam03
# Stage-specific (e.g. stop after D1)
python -m bjj_pipeline.stages.orchestration.cli run --input <clip> --camera cam03 \
  --config '{"stages": {"stage_D": {"run_until": "D1"}}}'
# Status / validate
python -m bjj_pipeline.stages.orchestration.cli status --clip-id <clip_id>
python -m bjj_pipeline.stages.orchestration.cli validate --clip-id <clip_id>
```

## Supabase Local Dev
```bash
cd backend/supabase/supabase
npx supabase start     # Start local instance
npx supabase db reset   # Reset + replay migrations
```

## Uploader (local, against local Supabase)
```bash
SUPABASE_URL=http://127.0.0.1:54321 \
SUPABASE_SERVICE_ROLE_KEY=<from-npx-supabase-status> \
SUPABASE_DB_URL=postgresql://postgres:postgres@127.0.0.1:54322/postgres \
SUPABASE_STORAGE_BUCKET=match-clips UPLOADER_DELETE_LOCAL=false \
python -c "import sys; sys.path.insert(0,'services/uploader'); \
from uploader.cli import main; sys.argv=['u','--manifest','<path>']; main()"
```

## Flutter
```bash
~/development/flutter/bin/flutter pub get
~/development/flutter/bin/flutter run -d 2A191FDH300C9Z  # Pixel 7 Pro
```

## Docker Services
```bash
cd services/nest_recorder && docker compose up
cd services/uploader && docker compose up
```

## Processor (native Mac)
```bash
caffeinate -is bash -c 'time bash services/processor/run_local.sh'
```

## Camera Geometry Analysis
```bash
# All 4 phases: height surface → ROI mask → detectability → coverage optimization
python tools/camera_geometry_analysis.py all \
  --outputs outputs --gym-id <uuid>

# Individual phases
python tools/camera_geometry_analysis.py phase1 --outputs outputs --gym-id <uuid>
python tools/camera_geometry_analysis.py phase2 --outputs outputs --gym-id <uuid>
python tools/camera_geometry_analysis.py phase3 --outputs outputs --gym-id <uuid>
python tools/camera_geometry_analysis.py phase4 --outputs outputs --gym-id <uuid>

# With known height reference
python tools/camera_geometry_analysis.py all \
  --outputs outputs --gym-id <uuid> --reference-height-m 1.80
```

## Audio Survey
```bash
python tools/detect_buzzer.py --input <mp4_or_dir> --survey
```
