# Roll Tracker – Camera API Setup Guide (For Gym Admins)

This guide shows you how to connect your Nest cameras to the Roll Tracker recorder service.
You only need to do this once to generate your OAuth refresh token. Cameras are discovered
automatically on every recording run — no manual camera configuration required.

---

## Step 1: Prerequisites
- You've already been added as a **test user** for the Roll Tracker Google Cloud project.
- You have access to the gym's cameras through your Google Home app.
- Docker Desktop (Mac/Windows) or Docker Engine (Linux) is installed.
- Local Supabase is running (`cd backend/supabase/supabase && npx supabase start`).

---

## Step 2: Authorize your account
1. Visit this special URL (replace with your project details, which we'll provide):

   ```
   https://nestservices.google.com/partnerconnections/<PROJECT_ID>/auth
   ?redirect_uri=https://www.google.com
   &access_type=offline&prompt=consent
   &client_id=<YOUR_WEB_OAUTH_CLIENT_ID>
   &response_type=code
   &scope=https://www.googleapis.com/auth/sdm.service
   ```

   - Make sure the whole link is on one line.
   - You'll see a Google screen asking you to approve camera access.

2. Select the cameras you want to allow.
3. After approval, you'll be redirected to `https://www.google.com/?code=...`.
   - Copy the long `code` value from the URL bar.

---

## Step 3: Exchange code for refresh token
Run this command in your terminal (replace values with your actual `CLIENT_ID`, `CLIENT_SECRET`, and the `code` from Step 2):

```bash
curl -s -X POST "https://www.googleapis.com/oauth2/v4/token" \
  -d client_id="<CLIENT_ID>" \
  -d client_secret="<CLIENT_SECRET>" \
  -d code="<AUTH_CODE_FROM_URL>" \
  -d grant_type="authorization_code" \
  -d redirect_uri="https://www.google.com"
```

Output will look like:

```json
{
  "access_token": "...",
  "expires_in": 3599,
  "refresh_token": "YOUR_REFRESH_TOKEN",
  "scope": "https://www.googleapis.com/auth/sdm.service",
  "token_type": "Bearer"
}
```

**Save the `refresh_token`.** This is what your recorder uses for long-term access.

**Important:** If the Google Cloud project's OAuth consent screen is in "Testing" mode,
refresh tokens expire after 7 days. Publishing the app to "Production" mode gives
non-expiring tokens.

---

## Step 4: Configure secrets and `.env`

### Secrets files

Place your credentials in `services/nest_recorder/secrets/`:

```
secrets/
  client_id.txt        # OAuth client ID (plain text)
  client_secret.txt    # OAuth client secret (plain text)
  refresh_token.txt    # refresh token from Step 3
  project_id.txt       # SDM project UUID (e.g., 56095b82-7cf1-...)
```

These are mounted read-only into the container at `/secrets`.

### `.env` file

Copy `.env.example` to `.env` and fill in the values:

```
TZ=America/New_York

# Supabase connection (for camera auto-registration)
GYM_ID=<your-gym-uuid-from-supabase>
SUPABASE_URL=http://192.168.0.66:54321
SUPABASE_SERVICE_ROLE_KEY=<service-role-key-from-supabase-status>

# Recording settings
WINDOW_MINUTES=30
PRE_ROLL_SECONDS=10
SEG_SECONDS=300
```

No manual camera/device configuration is needed. The service discovers
all cameras linked to your Google account via the Nest SDM API on every run.

---

## Step 5: Run a smoke test

Start the container in dev mode and run a short diagnostic recording:

```bash
cd services/nest_recorder

# Build and start in dev mode (sleep infinity — exec in to test)
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d --build recorder

# Verify token works
docker compose -f docker-compose.yml -f docker-compose.dev.yml exec recorder \
  bash -lc 'unset SDM_PROJECT_ID && /app/get_access_token.sh'

# Discover cameras
docker compose -f docker-compose.yml -f docker-compose.dev.yml exec recorder \
  bash -lc 'unset SDM_PROJECT_ID && /app/list_cameras.sh'

# Short 60-second test recording (production path, with camera registration)
docker compose -f docker-compose.yml -f docker-compose.dev.yml exec recorder \
  bash -lc 'unset SDM_PROJECT_ID && WINDOW_SECONDS=60 SEG_SECONDS=30 REENCODE=0 DISCOVER=1 /app/diag_v7_2.sh'
```

### What to expect

**Camera discovery** lists all cameras linked to your account, e.g.:
```
Matroom 1  (device suffix: J_EDEw)
Matroom 2  (device suffix: FP7oJQ)
Matroom 3  (device suffix: PPDmUg)
```

The `cam_id` is the last 6 characters of each camera's SDM device path. These
are auto-registered to the Supabase `cameras` table by `register_cameras.sh`.

**Recording output** (when GYM_ID is set) appears at:
```
data/raw/nest/{gym_id}/{cam_id}/{YYYY-MM-DD}/{HH}/{cam_id}-{timestamp}.mp4
```

Example:
```
data/raw/nest/00000000-.../J_EDEw/2026-03-17/07/J_EDEw-20260317-070708.mp4
```

**Diagnostic mode** (GYM_ID unset) writes to:
```
data/raw/nest/diag/{TS}/Cam_{name}__{cam_id}/
```

---

## Step 6: Start the scheduled recorder

The container entrypoint (`entrypoint.sh`) delegates to `diag_v8.sh`, which
includes its own scheduling loop. In simple mode (default), it runs daily at the
configured time.

```bash
cd services/nest_recorder
docker compose up --build -d
```

Schedule defaults are controlled by env vars in `.env`:
- `SCHED_DAILY_HHMM` — time to start recording (e.g., `20:00`)
- `SCHED_TZ` — timezone for the schedule (defaults to `TZ`)

---

## Step 7: Running the CV pipeline on recorded clips

The CV pipeline accepts recordings from both gym-scoped and legacy paths:

```bash
# Gym-scoped path (from production recorder)
python -m bjj_pipeline.stages.orchestration.cli run \
  --clip data/raw/nest/{gym_id}/J_EDEw/2026-03-17/07/J_EDEw-20260317-070708.mp4 \
  --camera J_EDEw

# Legacy path (from earlier test recordings)
python -m bjj_pipeline.stages.orchestration.cli run \
  --clip data/raw/nest/cam03/2026-01-03/12/cam03-20260103-124000.mp4 \
  --camera cam03
```

The `--camera` value must match the directory name in the path (the `cam_id`).
For discovery-based cameras, this is the 6-character device suffix (e.g., `J_EDEw`).
The pipeline extracts `gym_id` from the path automatically — no separate argument needed.

---

## Security Notes
- Keep your `.env` and `secrets/` files private. They contain credentials.
- Do **not** share your refresh token — every gym generates their own.
- If your token is ever compromised, revoke it in the [Google Account security page](https://myaccount.google.com/permissions) and generate a new one.
- Secrets files are gitignored — never commit them to the repository.

---

That's it! Once configured, the recorder discovers cameras, registers them to
Supabase, and writes gym-scoped recordings that the CV pipeline can ingest directly.
