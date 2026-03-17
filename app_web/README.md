# Roll Tracker — Web App

Vite + React. Admin dashboard for Roll Tracker gym owners.

## Local dev setup

Requires local Supabase to be running first:

    cd backend/supabase/supabase && npx supabase start

Then from the repo root:

    bash app_web/setup.sh   # first time only — creates .env, runs npm install
    cd app_web && npm run dev

Open http://localhost:5173
Admin dashboard: http://localhost:5173/admin/pricing
