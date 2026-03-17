#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# 1. npm install if needed
if [ ! -d "node_modules" ]; then
  echo "Installing dependencies..."
  npm install
else
  echo "node_modules/ exists, skipping npm install"
fi

# 2. .env setup
if [ -f ".env" ]; then
  echo ".env already exists, skipping"
else
  echo "Reading Supabase status..."
  SUPABASE_DIR="$SCRIPT_DIR/../backend/supabase/supabase"

  if ! STATUS_OUTPUT=$(cd "$SUPABASE_DIR" && npx supabase status 2>&1); then
    echo ""
    echo "Could not read Supabase status. Is local Supabase running?"
    echo "Start it with: cd backend/supabase/supabase && npx supabase start"
    exit 1
  fi

  API_URL=$(echo "$STATUS_OUTPUT" | grep -E '^\s*API URL:' | sed 's/.*API URL:[[:space:]]*//')
  ANON_KEY=$(echo "$STATUS_OUTPUT" | grep -E '^\s*anon key:' | sed 's/.*anon key:[[:space:]]*//')

  if [ -z "$API_URL" ] || [ -z "$ANON_KEY" ]; then
    echo ""
    echo "Could not read Supabase status. Is local Supabase running?"
    echo "Start it with: cd backend/supabase/supabase && npx supabase start"
    exit 1
  fi

  printf "Enter your admin email: "
  read -r ADMIN_EMAIL

  cat > .env <<EOF
VITE_SUPABASE_URL=$API_URL
VITE_SUPABASE_ANON_KEY=$ANON_KEY
VITE_ADMIN_EMAIL=$ADMIN_EMAIL
EOF

  echo ".env created successfully"
fi

echo ""
echo "Setup complete. Run: npm run dev"
