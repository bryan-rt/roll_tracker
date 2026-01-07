# 🎥 Roll Tracker – Camera API Setup Guide (For Gym Admins)

This guide shows you how to connect your Nest cameras to the Roll Tracker recorder container.  
You only need to do this once to generate your own refresh token.

---

## ✅ Step 1: Prerequisites
- You’ve already been added as a **test user** for the Roll Tracker Google Cloud project.  
- You have access to the gym’s cameras through your Google Home app.  
- Docker Desktop (Mac/Windows) or Docker Engine (Linux) is installed.

---

## 🔑 Step 2: Authorize your account
1. Visit this special URL (replace with your project details, which I’ll provide):

   ```
   https://nestservices.google.com/partnerconnections/<PROJECT_ID>/auth
   ?redirect_uri=https://www.google.com
   &access_type=offline&prompt=consent
   &client_id=<YOUR_WEB_OAUTH_CLIENT_ID>
   &response_type=code
   &scope=https://www.googleapis.com/auth/sdm.service
   ```

   - Make sure the whole link is on one line.  
   - You’ll see a Google screen asking you to approve camera access.

2. Select the cameras you want to allow.  
3. After approval, you’ll be redirected to `https://www.google.com/?code=...`.  
   - Copy the long `code` value from the URL bar.

---

## 🔄 Step 3: Exchange code for refresh token
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

👉 **Save the `refresh_token`.** This is what your recorder uses for long-term access.

---

## ⚙️ Step 4: Configure your `.env`
In your local copy of the Roll Tracker repo, create a `.env` file with these values:

```
SDM_PROJECT_ID=<PROJECT_ID>
SDM_CLIENT_ID=<CLIENT_ID>
SDM_CLIENT_SECRET=<CLIENT_SECRET>
SDM_REFRESH_TOKEN=<YOUR_REFRESH_TOKEN>
TZ=America/New_York

# Camera devices (full paths from devices.list output)
DEVICE_1=enterprises/<PROJECT_ID>/devices/<DEVICE_ID>
CAM_ID_1=cam1

WINDOW_MINUTES=30
PRE_ROLL_SECONDS=30
SEG_SECONDS=120
```

---

## ▶️ Step 5: Run a smoke test
```bash
docker compose run --rm \
  -e WINDOW_MINUTES=5 \
  -e PRE_ROLL_SECONDS=0 \
  --entrypoint /app/record_window.sh recorder
```

You should see `.mp4` files appear under:
```
./data/recordings/cam1/YYYY-MM-DD/HH/
```

---

## 📅 Step 6: Start the scheduled recorder
To have the recorder automatically run Mon–Fri at 8:00–8:30 PM:

```bash
docker compose up --build -d
```

---

## 🔒 Security Notes
- Keep your `.env` private. It contains secrets just like a password.  
- Do **not** share your refresh token — every user generates their own.  
- If your token is ever compromised, revoke it in the [Google Account security page](https://myaccount.google.com/permissions) and generate a new one.

---

✅ That’s it! You’re set up to capture sparring footage during scheduled windows.
