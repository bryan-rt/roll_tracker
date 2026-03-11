FROM python:3.12-slim

WORKDIR /app

COPY services/uploader /app/services/uploader

RUN pip install --no-cache-dir \
    psycopg[binary] \
    supabase

ENV PYTHONPATH=/app

CMD ["python", "-m", "services.uploader.uploader.cli", "--worker"]

