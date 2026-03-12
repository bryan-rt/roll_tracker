FROM python:3.12-slim

WORKDIR /app
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

COPY services/uploader /app/services/uploader

RUN pip install --no-cache-dir psycopg[binary] supabase

CMD ["python", "-u", "-m", "services.uploader.uploader.cli", "--worker"]