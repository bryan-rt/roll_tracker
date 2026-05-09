"""Training-data manifest read/write.

Loads per-model YAML manifests from configs/models/{model_id}.yaml.
Provides helpers to enumerate in-distribution vs held-out (val) frames
for a given camera and annotated range.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml

from pipeline_validation.common.schemas import ExportEntry, ModelManifest


def load_manifest(path: Path) -> ModelManifest:
    """Load a model manifest YAML file and validate via Pydantic."""
    with open(path) as f:
        data = yaml.safe_load(f)
    return ModelManifest(**data)


def enumerate_split_frames(
    export: ExportEntry, split: Literal["train", "val"]
) -> list[int]:
    """Return the exact frame indices for the given camera+split."""
    fr = getattr(export.splits, split)
    return list(range(fr.start, fr.stop + 1, fr.stride))


def enumerate_annotated_frames(export: ExportEntry) -> list[int]:
    """Return all annotated frame indices (train + val union)."""
    ar = export.annotated_range
    return list(range(ar.start, ar.stop + 1, ar.stride))
