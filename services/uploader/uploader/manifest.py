from __future__ import annotations

import json
from pathlib import Path

from .models import ManifestRecord


def load_manifest(path: str | Path) -> list[ManifestRecord]:
    manifest_path = Path(path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    records: list[ManifestRecord] = []

    with manifest_path.open("r", encoding="utf-8") as f:
        for line_number, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue

            data = json.loads(line)
            inputs = data["inputs"]
            uploader_contract = inputs["uploader_contract"]
            clip_row = uploader_contract["clip_row"]
            log_events = uploader_contract.get("log_events", [])

            local_output_path = clip_row["metadata"]["local_output_path"]
            storage_bucket = clip_row["storage_bucket"]
            storage_object_path = clip_row["storage_object_path"]

            record = ManifestRecord(
                export_id=data["export_id"],
                camera_id=data["camera_id"],
                input_video_path=inputs["input_video_path"],
                local_output_path=local_output_path,
                storage_bucket=storage_bucket,
                storage_object_path=storage_object_path,
                clip_row=clip_row,
                log_events=log_events,
            )
            records.append(record)

    return records
