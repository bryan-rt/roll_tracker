# Nest Recorder Secrets

This folder is mounted into the container at `/secrets` (read-only) and holds static credentials or tokens. Rotated access tokens and refresh tokens are cached under `/recordings/secrets` inside the container (host: `data/raw/nest/secrets`).

Do NOT commit real secrets.

## Expected Files & Formats

- `client_id.txt`: plain text; contains the OAuth client ID.
- `client_secret.txt`: plain text; contains the OAuth client secret.
- `refresh_token.txt`: plain text; contains the OAuth refresh token.

The recorder scripts read environment variables first (e.g., `SDM_CLIENT_ID`, `SDM_CLIENT_SECRET`, `SDM_REFRESH_TOKEN`). If those are not set, they fall back to the files above. When Google returns a rotated refresh token, it is written into `/recordings/secrets/refresh_token.txt` inside the container.

## Runtime Caches (created automatically)

- `/recordings/secrets/access_token.json`: cached access token and expiry.
- `/recordings/secrets/refresh_token.txt`: rotated refresh token (if provided by Google).

These caches live under the mounted recordings path and should be treated as ephemeral runtime state.

## Notes

- The container also respects `SDM_CACHE_DIR` (optional), defaulting to `/recordings/secrets` for token caches.
- Ensure `services/nest_recorder/.env` contains the necessary non-secret config such as timezone (see `.env.example`).
- Keep file permissions restrictive if you place secrets here (the scripts use `umask 077`).
