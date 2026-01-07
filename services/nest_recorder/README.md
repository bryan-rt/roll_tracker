# Nest Recorder — Stage 0 Ingest

Stage 0 ingest service that records Nest camera streams on a schedule. EntryPoint runs `/app/entrypoint.sh`; dev mode keeps the container idle for interactive scripts under `/app/*.sh`.

## Outputs (host paths)
- Recordings: `data/raw/nest/<CAM_ID>/YYYY-MM-DD/HH/<CAM_ID>-YYYYmmdd-HHMMSS.mp4`
- Logs: `data/raw/nest/logs/...`
- Token cache: `data/raw/nest/secrets/` (e.g., `access_token.json`, rotated `refresh_token.txt`)

## Ops Mode (scheduled recording)
```bash
cd services/nest_recorder
cp .env.example .env
docker compose up -d
```

## Dev Mode (sleep infinity)
```bash
cd services/nest_recorder
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d --build
docker compose exec recorder bash
# inside container:
/app/diag_v8.sh
# or other scripts in /app/*.sh
```
Why both files: the dev YAML is override-only and relies on the base compose for `build` and `env_file`.

## Troubleshooting
- Container logs: `docker compose logs -f recorder`
- Script logs: check `data/raw/nest/logs/...`
- Token issues: inspect `data/raw/nest/secrets/access_token.json` and `refresh_token.txt`
- Permissions: ensure `services/nest_recorder/secrets/*` exist (plain text) or set env vars in `.env`
- Timezone: verify `TZ` in `.env`

Historical compose files are archived under `services/nest_recorder/docs/archive/` for reference.
