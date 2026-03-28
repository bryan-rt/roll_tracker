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

## Audio Survey
```bash
python tools/detect_buzzer.py --input <mp4_or_dir> --survey
```
