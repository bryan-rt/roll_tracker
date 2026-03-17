FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libgl1 libglib2.0-0 \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir --no-deps ultralytics boxmot \
 && pip install --no-cache-dir pydantic-settings

# Install bjj_pipeline as editable package
COPY pyproject.toml ./
COPY src/ ./src/
RUN pip install --no-cache-dir -e .

# Copy configs (needed by _load_config)
COPY configs/ ./configs/

# Copy processor service code
COPY services/processor/config.py services/processor/processor.py ./

ENTRYPOINT ["python", "processor.py"]
