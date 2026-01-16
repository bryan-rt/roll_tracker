"""
F0 — Parquet schema mappings (authoritative)

Design:
- Parquet artifacts store flattened columns (no nested structs), including bbox columns x1,y1,x2,y2.
- Base metadata lives in clip_manifest.json, not repeated in every row.
- Validation here is "schema-shape + dtype family" focused; deeper invariants live in f0_validate.py.

This module intentionally supports both:
- pandas DataFrames (local dev, tests)
- pyarrow schema objects (optional use by writers)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Literal, Optional, Sequence, Tuple

import pandas as pd


# ----------------------------
# Column specs
# ----------------------------

DTypeFamily = Literal["string", "int", "float", "bool"]


@dataclass(frozen=True)
class ColSpec:
    name: str
    family: DTypeFamily
    required: bool = True
    # Optional: allow nulls even if present (common in Parquet for optional cols)
    nullable: bool = True


def _spec_map(specs: Sequence[ColSpec]) -> Dict[str, ColSpec]:
    return {c.name: c for c in specs}


# ----------------------------
# Canonical schemas (flattened)
# ----------------------------

DETECTIONS_SPECS: List[ColSpec] = [
    ColSpec("clip_id", "string"),
    ColSpec("camera_id", "string"),
    ColSpec("frame_index", "int"),
    ColSpec("timestamp_ms", "int"),
    ColSpec("detection_id", "string"),
    ColSpec("class_name", "string"),
    ColSpec("confidence", "float"),
    # flattened bbox
    ColSpec("x1", "float"),
    ColSpec("y1", "float"),
    ColSpec("x2", "float"),
    ColSpec("y2", "float"),
    # optional
    ColSpec("tracklet_id", "string", required=False),
    # Optional lightweight segmentation mask reference (Stage A YOLO-seg, Stage B SAM refinement).
    # File-backed .npz is canonical; this stores the relative path (from outputs/<clip_id>/).
    ColSpec("mask_ref", "string", required=False),
    # Optional metadata about the mask producer/gates (e.g., "yolo_seg", "bbox_fallback", "sam").
    ColSpec("mask_source", "string", required=False),
    ColSpec("mask_quality", "float", required=False),
    ColSpec("source", "string", required=False),
    # debug intentionally excluded from Parquet (keep debug in audit.jsonl); if you really want it:
    ColSpec("debug_json", "string", required=False),
]

TRACKLET_FRAMES_SPECS: List[ColSpec] = [
    ColSpec("clip_id", "string"),
    ColSpec("camera_id", "string"),
    ColSpec("tracklet_id", "string"),
    ColSpec("frame_index", "int"),
    ColSpec("timestamp_ms", "int"),
    ColSpec("detection_id", "string"),
    ColSpec("local_track_conf", "float", required=False),
    # Optional per-frame contact point + projection.
    # These are produced by Stage A when homography is available (preflight-owned by D7/F1).
    ColSpec("u_px", "float", required=False),
    ColSpec("v_px", "float", required=False),
    ColSpec("x_m", "float", required=False),
    ColSpec("y_m", "float", required=False),
    ColSpec("vx_m", "float", required=False),
    ColSpec("vy_m", "float", required=False),
    ColSpec("on_mat", "bool", required=False),
    ColSpec("contact_conf", "float", required=False),
    ColSpec("contact_method", "string", required=False),
]

TRACKLET_SUMMARIES_SPECS: List[ColSpec] = [
    ColSpec("clip_id", "string"),
    ColSpec("camera_id", "string"),
    ColSpec("tracklet_id", "string"),
    ColSpec("start_frame", "int"),
    ColSpec("end_frame", "int"),
    ColSpec("n_frames", "int"),
    # optional mean bbox (flattened)
    ColSpec("mean_x1", "float", required=False),
    ColSpec("mean_y1", "float", required=False),
    ColSpec("mean_x2", "float", required=False),
    ColSpec("mean_y2", "float", required=False),
    ColSpec("quality_score", "float", required=False),
    # store reason codes as JSON string list for Parquet compatibility
    ColSpec("reason_codes_json", "string", required=False),
]

CONTACT_POINTS_SPECS: List[ColSpec] = [
    ColSpec("clip_id", "string"),
    ColSpec("camera_id", "string"),
    ColSpec("frame_index", "int"),
    ColSpec("timestamp_ms", "int"),
    ColSpec("detection_id", "string"),
    # Optional linkage (Stage A provides; Stage B may omit)
    ColSpec("tracklet_id", "string", required=False),
    # Contact pixel + ground-plane projection (Stage A baseline + Stage B refined)
    ColSpec("u_px", "float"),
    ColSpec("v_px", "float"),
    ColSpec("x_m", "float"),
    ColSpec("y_m", "float"),
    # Stage A baseline fields
    ColSpec("on_mat", "bool", required=False),
    ColSpec("contact_conf", "float", required=False),
    ColSpec("contact_method", "string", required=False),
    # Stage B legacy fields (kept for backward compatibility)
    ColSpec("method", "string", required=False),
    ColSpec("confidence", "float", required=False),
    ColSpec("homography_id", "string", required=False),
]

PERSON_TRACKS_SPECS: List[ColSpec] = [
    ColSpec("clip_id", "string"),
    ColSpec("camera_id", "string"),
    ColSpec("person_id", "string"),
    ColSpec("frame_index", "int"),
    ColSpec("timestamp_ms", "int"),
    ColSpec("detection_id", "string"),
    ColSpec("tracklet_id", "string"),
    # flattened bbox
    ColSpec("x1", "float"),
    ColSpec("y1", "float"),
    ColSpec("x2", "float"),
    ColSpec("y2", "float"),
    # ground-plane
    ColSpec("x_m", "float"),
    ColSpec("y_m", "float"),
    # optional
    ColSpec("mask_ref", "string", required=False),  # relative path
    ColSpec("reid_sim", "float", required=False),
    ColSpec("stitch_edge_type", "string", required=False),
]


# Registry by artifact key (used by manifest + validators)
PARQUET_SCHEMAS: Dict[str, List[ColSpec]] = {
    "detections": DETECTIONS_SPECS,
    "tracklet_frames": TRACKLET_FRAMES_SPECS,
    "tracklet_summaries": TRACKLET_SUMMARIES_SPECS,
    "contact_points": CONTACT_POINTS_SPECS,
    "person_tracks": PERSON_TRACKS_SPECS,
}


# ----------------------------
# dtype-family checks (pandas)
# ----------------------------

def _is_string_dtype(s: pd.Series) -> bool:
    return pd.api.types.is_string_dtype(s.dtype) or pd.api.types.is_object_dtype(s.dtype)


def _is_int_dtype(s: pd.Series) -> bool:
    # allow nullable Int64 and regular ints
    return pd.api.types.is_integer_dtype(s.dtype)


def _is_float_dtype(s: pd.Series) -> bool:
    return pd.api.types.is_float_dtype(s.dtype) or pd.api.types.is_integer_dtype(s.dtype)


def _is_bool_dtype(s: pd.Series) -> bool:
    return pd.api.types.is_bool_dtype(s.dtype)


_FAMILY_CHECK = {
    "string": _is_string_dtype,
    "int": _is_int_dtype,
    "float": _is_float_dtype,
    "bool": _is_bool_dtype,
}


class ParquetSchemaError(ValueError):
    pass


def validate_df_schema(df: pd.DataFrame, specs: Sequence[ColSpec], *, table_name: str) -> None:
    """
    Validates:
    - required columns exist
    - no unexpected columns (strict by default)
    - dtype families align (string/int/float/bool)
    """
    spec_by_name = _spec_map(specs)
    cols = list(df.columns)

    missing = [c.name for c in specs if c.required and c.name not in df.columns]
    if missing:
        raise ParquetSchemaError(f"{table_name}: missing required columns: {missing}")

    unexpected = [c for c in cols if c not in spec_by_name]
    if unexpected:
        raise ParquetSchemaError(f"{table_name}: unexpected columns: {unexpected}")

    # dtype family validation
    for name, spec in spec_by_name.items():
        if name not in df.columns:
            continue
        series = df[name]
        checker = _FAMILY_CHECK[spec.family]
        if not checker(series):
            raise ParquetSchemaError(
                f"{table_name}: column '{name}' expected family '{spec.family}', got dtype '{series.dtype}'"
            )


def validate_df_schema_by_key(df: pd.DataFrame, key: str) -> None:
    if key not in PARQUET_SCHEMAS:
        raise KeyError(f"Unknown parquet schema key: {key}")
    validate_df_schema(df, PARQUET_SCHEMAS[key], table_name=key)


# ----------------------------
# bbox flatten/unflatten helpers
# ----------------------------

def flatten_bbox_xyxy(df: pd.DataFrame, bbox_col: str = "bbox_xyxy") -> pd.DataFrame:
    """
    If df has a column 'bbox_xyxy' containing dict-like {x1,y1,x2,y2} or 4-seq,
    expand into x1,y1,x2,y2 columns and drop bbox_col.
    """
    if bbox_col not in df.columns:
        return df

    def _get(v, i, k):
        if v is None:
            return None
        if isinstance(v, dict):
            return v.get(k)
        if isinstance(v, (list, tuple)) and len(v) == 4:
            return v[i]
        # allow pydantic model .model_dump()
        if hasattr(v, "model_dump"):
            d = v.model_dump()
            return d.get(k)
        raise ValueError(f"Unsupported bbox value: {type(v)}")

    out = df.copy()
    out["x1"] = out[bbox_col].map(lambda v: _get(v, 0, "x1"))
    out["y1"] = out[bbox_col].map(lambda v: _get(v, 1, "y1"))
    out["x2"] = out[bbox_col].map(lambda v: _get(v, 2, "x2"))
    out["y2"] = out[bbox_col].map(lambda v: _get(v, 3, "y2"))
    out = out.drop(columns=[bbox_col])
    return out


def unflatten_bbox_xyxy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recompose bbox_xyxy as dict column from x1,y1,x2,y2.
    Useful for loading into Pydantic DetectionRow/PersonTrackPointRow.
    """
    required = ["x1", "y1", "x2", "y2"]
    if not all(c in df.columns for c in required):
        return df

    out = df.copy()
    out["bbox_xyxy"] = out.apply(lambda r: {"x1": r["x1"], "y1": r["y1"], "x2": r["x2"], "y2": r["y2"]}, axis=1)
    out = out.drop(columns=required)
    return out
