"""
F0 — Invariant validators (authoritative)

These validators enforce pipeline-wide contracts:
- frame_index monotonicity (where applicable)
- foreign key traceability between artifacts
- basic ID hygiene and consistency checks
- identity hint and assignment sanity checks

Policy:
- Contract violations raise ValidationError (hard fail).
- Quality warnings should be emitted as AuditEvents in stage code, not here.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import pandas as pd

from . import f0_parquet as pq
from .f0_paths import ClipOutputLayout


# ----------------------------
# Errors
# ----------------------------

class ValidationError(ValueError):
    pass


# ----------------------------
# Small helpers
# ----------------------------

def _require_non_empty_str(value: str, *, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{name} must be a non-empty string")


def _require_columns(df: pd.DataFrame, cols: Sequence[str], *, table_name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValidationError(f"{table_name}: missing required columns for validation: {missing}")


def _assert_singleton(df: pd.DataFrame, col: str, *, table_name: str) -> str:
    _require_columns(df, [col], table_name=table_name)
    vals = df[col].dropna().unique().tolist()
    if len(vals) != 1:
        raise ValidationError(f"{table_name}: expected exactly one distinct '{col}', got {vals}")
    return str(vals[0])


def _validate_frame_index_monotonic(
    df: pd.DataFrame,
    *,
    table_name: str,
    group_cols: Sequence[str],
    frame_col: str = "frame_index",
) -> None:
    _require_columns(df, list(group_cols) + [frame_col], table_name=table_name)
    if df.empty:
        return

    # For each group, frame_index must be non-decreasing (monotonic).
    # Note: "monotonic" here means not going backwards; duplicates are allowed.
    gb = df.sort_values(list(group_cols) + [frame_col]).groupby(list(group_cols), sort=False)
    for key, g in gb:
        frames = g[frame_col].to_numpy()
        if (frames[1:] < frames[:-1]).any():
            raise ValidationError(
                f"{table_name}: frame_index not monotonic within group {key}. "
                f"Found a decrease."
            )


def _build_fk_set(
    df: pd.DataFrame,
    cols: Sequence[str],
    *,
    table_name: str,
) -> Set[Tuple]:
    _require_columns(df, list(cols), table_name=table_name)
    # Use tuples; drop rows with NA in any FK component.
    sub = df[list(cols)].dropna()
    return set(map(tuple, sub.itertuples(index=False, name=None)))


def _validate_fk_subset(
    child_keys: Set[Tuple],
    parent_keys: Set[Tuple],
    *,
    child_name: str,
    parent_name: str,
    example_limit: int = 10,
) -> None:
    missing = child_keys - parent_keys
    if missing:
        examples = list(missing)[:example_limit]
        raise ValidationError(
            f"Foreign key violation: {child_name} references keys not found in {parent_name}. "
            f"Missing count={len(missing)} examples={examples}"
        )


def _validate_bbox_xyxy(df: pd.DataFrame, *, table_name: str) -> None:
    for col in ("x1", "y1", "x2", "y2"):
        if col not in df.columns:
            return  # if bbox not present, nothing to validate here
    bad = df[(df["x2"] < df["x1"]) | (df["y2"] < df["y1"])]
    if not bad.empty:
        row = bad.iloc[0].to_dict()
        raise ValidationError(f"{table_name}: invalid bbox with x2<x1 or y2<y1. Example row={row}")


# ----------------------------
# Parquet table validators
# ----------------------------

def validate_detections_df(df: pd.DataFrame) -> None:
    """
    Validates detections.parquet table shape + basic invariants.
    """
    pq.validate_df_schema_by_key(df, "detections")

    if df.empty:
        return

    _assert_singleton(df, "clip_id", table_name="detections")
    _assert_singleton(df, "camera_id", table_name="detections")

    _validate_frame_index_monotonic(df, table_name="detections", group_cols=("clip_id", "camera_id"))
    _validate_bbox_xyxy(df, table_name="detections")

    # IDs non-empty
    if df["detection_id"].isna().any():
        raise ValidationError("detections: detection_id contains nulls")
    if (df["detection_id"].astype(str).str.len() == 0).any():
        raise ValidationError("detections: detection_id contains empty strings")

    # Optional mask references must be portable relative paths.
    # Canonical mask blobs are file-backed (e.g., stage_A/masks/*.npz).
    if "mask_ref" in df.columns:
        s = df["mask_ref"].dropna().astype(str)
        if (s.str.startswith("/") | s.str.contains(r"^[A-Za-z]:\\")).any():
            raise ValidationError("detections: mask_ref must be a relative path, not absolute")


def validate_tracklet_tables(tracklet_frames: pd.DataFrame, tracklet_summaries: pd.DataFrame) -> None:
    """
    Validates tracklet_frames.parquet and tracklet_summaries.parquet.
    """
    pq.validate_df_schema_by_key(tracklet_frames, "tracklet_frames")
    pq.validate_df_schema_by_key(tracklet_summaries, "tracklet_summaries")

    if tracklet_frames.empty and tracklet_summaries.empty:
        return

    # clip/cam consistency (each table individually singleton)
    if not tracklet_frames.empty:
        _assert_singleton(tracklet_frames, "clip_id", table_name="tracklet_frames")
        _assert_singleton(tracklet_frames, "camera_id", table_name="tracklet_frames")
        _validate_frame_index_monotonic(
            tracklet_frames, table_name="tracklet_frames", group_cols=("clip_id", "camera_id", "tracklet_id")
        )

    if not tracklet_summaries.empty:
        _assert_singleton(tracklet_summaries, "clip_id", table_name="tracklet_summaries")
        _assert_singleton(tracklet_summaries, "camera_id", table_name="tracklet_summaries")

        # start/end ordering
        bad = tracklet_summaries[tracklet_summaries["end_frame"] < tracklet_summaries["start_frame"]]
        if not bad.empty:
            raise ValidationError(
                f"tracklet_summaries: end_frame < start_frame for some rows. Example={bad.iloc[0].to_dict()}"
            )

        # n_frames sanity (>=1)
        bad2 = tracklet_summaries[tracklet_summaries["n_frames"] < 1]
        if not bad2.empty:
            raise ValidationError(
                f"tracklet_summaries: n_frames < 1 for some rows. Example={bad2.iloc[0].to_dict()}"
            )

        # optional mean bbox validity if present
        if all(c in tracklet_summaries.columns for c in ("mean_x1", "mean_y1", "mean_x2", "mean_y2")):
            sub = tracklet_summaries.dropna(subset=["mean_x1", "mean_y1", "mean_x2", "mean_y2"])
            if not sub.empty:
                badbb = sub[(sub["mean_x2"] < sub["mean_x1"]) | (sub["mean_y2"] < sub["mean_y1"])]
                if not badbb.empty:
                    raise ValidationError(
                        f"tracklet_summaries: invalid mean bbox. Example={badbb.iloc[0].to_dict()}"
                    )

    # Cross-table: every tracklet_id in frames should exist in summaries (recommended hard invariant)
    if not tracklet_frames.empty and not tracklet_summaries.empty:
        frame_ids = set(tracklet_frames["tracklet_id"].dropna().astype(str).unique().tolist())
        sum_ids = set(tracklet_summaries["tracklet_id"].dropna().astype(str).unique().tolist())
        missing = frame_ids - sum_ids
        if missing:
            ex = list(missing)[:10]
            raise ValidationError(
                f"tracklet_tables: tracklet_frames has tracklet_id(s) not present in tracklet_summaries. "
                f"Missing count={len(missing)} examples={ex}"
            )


def validate_tracklet_bank_tables(bank_frames: pd.DataFrame, bank_summaries: pd.DataFrame) -> None:
    """
    Validates Stage D bank tables (tracklet_bank_frames.parquet and tracklet_bank_summaries.parquet).
    These are D0+ evolving "master tables" with strict explicit schemas.
    """
    pq.validate_df_schema_by_key(bank_frames, "tracklet_bank_frames")
    pq.validate_df_schema_by_key(bank_summaries, "tracklet_bank_summaries")

    if bank_frames.empty and bank_summaries.empty:
        return

    if not bank_frames.empty:
        _assert_singleton(bank_frames, "clip_id", table_name="tracklet_bank_frames")
        _assert_singleton(bank_frames, "camera_id", table_name="tracklet_bank_frames")
        _validate_frame_index_monotonic(
            bank_frames,
            table_name="tracklet_bank_frames",
            group_cols=("clip_id", "camera_id", "tracklet_id"),
        )

    if not bank_summaries.empty:
        _assert_singleton(bank_summaries, "clip_id", table_name="tracklet_bank_summaries")
        _assert_singleton(bank_summaries, "camera_id", table_name="tracklet_bank_summaries")
        bad = bank_summaries[bank_summaries["end_frame"] < bank_summaries["start_frame"]]
        if not bad.empty:
            raise ValidationError(
                f"tracklet_bank_summaries: end_frame < start_frame for some rows. Example={bad.iloc[0].to_dict()}"
            )

    # Cross-table: every tracklet_id in frames should exist in summaries
    if not bank_frames.empty and not bank_summaries.empty:
        frame_ids = set(bank_frames["tracklet_id"].dropna().astype(str).unique().tolist())
        sum_ids = set(bank_summaries["tracklet_id"].dropna().astype(str).unique().tolist())
        missing = frame_ids - sum_ids
        if missing:
            ex = list(missing)[:10]
            raise ValidationError(
                "tracklet_bank_tables: tracklet_bank_frames has tracklet_id(s) not present in "
                f"tracklet_bank_summaries. Missing count={len(missing)} examples={ex}"
            )

    # If D0 repair columns are present, enforce basic invariants.
    if not bank_frames.empty and "is_repaired" in bank_frames.columns:
        repaired = bank_frames[bank_frames["is_repaired"] == True]  # noqa: E712
        if not repaired.empty:
            for c in ["x_m_repaired", "y_m_repaired", "repair_span_id"]:
                if c not in repaired.columns:
                    raise ValidationError(
                        f"tracklet_bank_frames: missing column {c!r} required when is_repaired exists"
                    )
                if repaired[c].isna().any():
                    ex = repaired[repaired[c].isna()].iloc[0].to_dict()
                    raise ValidationError(
                        f"tracklet_bank_frames: repaired row has null {c!r}. Example row={ex}"
                    )


def validate_tracklet_frames_fk_to_detections(tracklet_frames: pd.DataFrame, detections: pd.DataFrame) -> None:
    """
    FK invariant: each (clip_id,camera_id,frame_index,detection_id) in tracklet_frames must exist in detections.
    """
    if tracklet_frames.empty:
        return
    if detections.empty:
        raise ValidationError("FK validation: detections is empty but tracklet_frames is not")

    # Build keys
    child = _build_fk_set(
        tracklet_frames,
        ("clip_id", "camera_id", "frame_index", "detection_id"),
        table_name="tracklet_frames",
    )
    parent = _build_fk_set(
        detections,
        ("clip_id", "camera_id", "frame_index", "detection_id"),
        table_name="detections",
    )
    _validate_fk_subset(child, parent, child_name="tracklet_frames", parent_name="detections")


def validate_contact_points_df(df: pd.DataFrame) -> None:
    """Validate the generic (cross-stage) contact_points table shape.

    Note: Stage-specific requirements (e.g., Stage A requiring on_mat/contact_conf/contact_method
    and deterministic ordering by (frame_index, detection_id)) are enforced in dedicated validators.
    """
    pq.validate_df_schema_by_key(df, "contact_points")

    if df.empty:
        return

    _assert_singleton(df, "clip_id", table_name="contact_points")
    _assert_singleton(df, "camera_id", table_name="contact_points")
    _validate_frame_index_monotonic(df, table_name="contact_points", group_cols=("clip_id", "camera_id"))

    # confidence range sanity (Stage B legacy column; may be absent)
    if "confidence" in df.columns:
        s = df["confidence"]
        # ignore nulls
        s = s.dropna()
        if not s.empty:
            mask = (df["confidence"] < 0.0) | (df["confidence"] > 1.0)
            try:
                mask = mask.fillna(False)
            except Exception:
                pass
            bad = df[mask]
            if not bad.empty:
                raise ValidationError(
                    f"contact_points: confidence outside [0,1]. Example={bad.iloc[0].to_dict()}"
                )


def validate_stage_A_contact_points_df(df: pd.DataFrame) -> None:
    """Stage A-specific contact_points validator (F0G).

    Requirements:
    - Must include join keys: clip_id,camera_id,frame_index,timestamp_ms,detection_id
    - Must include tracklet_id
    - Must include geometry: u_px,v_px,x_m,y_m,on_mat
    - Must include contact fields: contact_conf,contact_method
    - Deterministic ordering by (frame_index asc, detection_id asc)
    """
    validate_contact_points_df(df)

    required_cols = [
        "clip_id",
        "camera_id",
        "frame_index",
        "timestamp_ms",
        "detection_id",
        "tracklet_id",
        "u_px",
        "v_px",
        "x_m",
        "y_m",
        "on_mat",
        "contact_conf",
        "contact_method",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValidationError(f"stage_A: contact_points missing required columns: {missing}")

    if df.empty:
        return

    # Deterministic ordering: frame_index asc, detection_id asc (stable lexicographic)
    cur = df[["frame_index", "detection_id"]].reset_index(drop=True)
    sorted_df = df.sort_values(["frame_index", "detection_id"], kind="mergesort")
    exp = sorted_df[["frame_index", "detection_id"]].reset_index(drop=True)
    if not cur.equals(exp):
        ex = df.iloc[0].to_dict() if not df.empty else {}
        raise ValidationError(
            "stage_A: contact_points must be sorted by (frame_index asc, detection_id asc). "
            f"Example_first_row={ex}"
        )


def validate_person_tracks_df(df: pd.DataFrame) -> None:
    pq.validate_df_schema_by_key(df, "person_tracks")

    if df.empty:
        return

    _assert_singleton(df, "clip_id", table_name="person_tracks")
    _assert_singleton(df, "camera_id", table_name="person_tracks")

    # Monotonic per person_id
    _validate_frame_index_monotonic(
        df, table_name="person_tracks", group_cols=("clip_id", "camera_id", "person_id")
    )
    _validate_bbox_xyxy(df, table_name="person_tracks")


def validate_person_tracks_traceability(
    person_tracks: pd.DataFrame,
    detections: pd.DataFrame,
    tracklet_frames: pd.DataFrame,
) -> None:
    """
    FK invariants:
    - person_tracks references valid detections via (clip_id,camera_id,frame_index,detection_id)
    - person_tracks references valid tracklet_id via (clip_id,camera_id,tracklet_id)
    """
    if person_tracks.empty:
        return
    if detections.empty:
        raise ValidationError("traceability: detections is empty but person_tracks is not")
    if tracklet_frames.empty:
        raise ValidationError("traceability: tracklet_frames is empty but person_tracks is not")

    # detection fk
    child_det = _build_fk_set(
        person_tracks,
        ("clip_id", "camera_id", "frame_index", "detection_id"),
        table_name="person_tracks",
    )
    parent_det = _build_fk_set(
        detections,
        ("clip_id", "camera_id", "frame_index", "detection_id"),
        table_name="detections",
    )
    _validate_fk_subset(child_det, parent_det, child_name="person_tracks(detection)", parent_name="detections")

    # tracklet fk (by id only; frame-level existence is ensured by the detection fk + tracklet_frames fk elsewhere)
    child_trk = _build_fk_set(
        person_tracks,
        ("clip_id", "camera_id", "tracklet_id"),
        table_name="person_tracks",
    )
    parent_trk = _build_fk_set(
        tracklet_frames,
        ("clip_id", "camera_id", "tracklet_id"),
        table_name="tracklet_frames",
    )
    _validate_fk_subset(child_trk, parent_trk, child_name="person_tracks(tracklet)", parent_name="tracklet_frames")


def validate_d2_edge_costs_df(df: pd.DataFrame) -> None:
    """Validate Stage D2 edge costs table (solver-agnostic).

    D2 is intermediate (run_until=D2) and is validated by stage code.
    """
    pq.validate_df_schema_by_key(df, "d2_edge_costs")
    if df.empty:
        return

    if df["edge_id"].isna().any():
        raise ValidationError("d2_edge_costs: edge_id contains nulls")
    if not df["edge_id"].is_unique:
        raise ValidationError("d2_edge_costs: edge_id must be unique")

    # is_allowed / reasons must be consistent
    if df["is_allowed"].isna().any():
        raise ValidationError("d2_edge_costs: is_allowed contains nulls")
    if df["disallow_reasons_json"].isna().any():
        raise ValidationError("d2_edge_costs: disallow_reasons_json contains nulls")

    # Must be valid JSON lists; disallowed rows must have >=1 reason.
    for i, (allowed, reasons_raw) in enumerate(zip(df["is_allowed"].tolist(), df["disallow_reasons_json"].tolist())):
        try:
            reasons = json.loads(reasons_raw)
        except Exception as e:
            raise ValidationError(f"d2_edge_costs: disallow_reasons_json is not valid JSON at row {i}: {e}") from e
        if not isinstance(reasons, list):
            raise ValidationError(f"d2_edge_costs: disallow_reasons_json must decode to a list (row {i})")
        if (not allowed) and len(reasons) < 1:
            raise ValidationError(f"d2_edge_costs: disallowed edge must include at least one reason (row {i})")

    # total_cost must be finite (no NaN/inf); disallowed edges should use a large finite sentinel.
    total = df["total_cost"].to_numpy(dtype=float)
    if not (pd.notna(total).all() and (total == total).all()):
        raise ValidationError("d2_edge_costs: total_cost contains NaN")
    if not (total < float("inf")).all():
        raise ValidationError("d2_edge_costs: total_cost contains inf")


def validate_d2_constraints_json(obj: Dict[str, Any]) -> None:
    """Validate normalized identity constraint spec produced by D2."""
    if not isinstance(obj, dict):
        raise ValidationError("d2_constraints: expected a dict")

    must = obj.get("must_link_groups")
    cannot = obj.get("cannot_link_pairs")
    if must is None or cannot is None:
        raise ValidationError("d2_constraints: missing must_link_groups or cannot_link_pairs")

    if not isinstance(must, list):
        raise ValidationError("d2_constraints: must_link_groups must be a list")
    if not isinstance(cannot, list):
        raise ValidationError("d2_constraints: cannot_link_pairs must be a list")

    # determinism: groups sorted by anchor_key; tracklet_ids sorted
    anchor_keys = []
    for g in must:
        if not isinstance(g, dict):
            raise ValidationError("d2_constraints: must_link_groups entries must be dicts")
        ak = g.get("anchor_key")
        tids = g.get("tracklet_ids")
        _require_non_empty_str(str(ak), name="d2_constraints.must_link_groups.anchor_key")
        if not isinstance(tids, list) or not all(isinstance(t, str) for t in tids):
            raise ValidationError("d2_constraints: must_link_groups.tracklet_ids must be a list[str]")
        if tids != sorted(tids):
            raise ValidationError(f"d2_constraints: tracklet_ids must be sorted for anchor_key={ak!r}")
        anchor_keys.append(str(ak))
    if anchor_keys != sorted(anchor_keys):
        raise ValidationError("d2_constraints: must_link_groups must be sorted by anchor_key")

    # cannot_link pairs: canonical order and unique, sorted
    pairs = []
    for p in cannot:
        if not (isinstance(p, list) or isinstance(p, tuple)) or len(p) != 2:
            raise ValidationError("d2_constraints: cannot_link_pairs entries must be [a,b]")
        a, b = p
        if not isinstance(a, str) or not isinstance(b, str):
            raise ValidationError("d2_constraints: cannot_link_pairs ids must be strings")
        if not (a < b):
            raise ValidationError("d2_constraints: cannot_link_pairs must have a < b canonical ordering")
        pairs.append((a, b))
    if pairs != sorted(pairs):
        raise ValidationError("d2_constraints: cannot_link_pairs must be sorted")
    if len(set(pairs)) != len(pairs):
        raise ValidationError("d2_constraints: cannot_link_pairs contains duplicates")


# ----------------------------
# JSONL validators (shape + key invariants)
# ----------------------------

def _validate_jsonl_required_fields(records: List[Dict], required: Sequence[str], *, name: str) -> None:
    for i, r in enumerate(records):
        for f in required:
            if f not in r:
                raise ValidationError(f"{name}: record[{i}] missing required field '{f}'")
            if r[f] is None:
                raise ValidationError(f"{name}: record[{i}] has null required field '{f}'")


def validate_identity_hints_records(records: List[Dict], *, expected_clip_id: Optional[str] = None) -> None:
    """
    Validate loaded JSONL identity_hints as dict records.
    Enforces:
    - base metadata present
    - keyed by tracklet_id
    - anchor_key namespaced (contains ':')
    - constraint in {must_link,cannot_link}
    """
    base = ["schema_version", "artifact_type", "clip_id", "camera_id", "pipeline_version", "created_at_ms"]
    req = base + ["tracklet_id", "anchor_key", "constraint", "confidence", "evidence"]
    _validate_jsonl_required_fields(records, req, name="identity_hints")

    for i, r in enumerate(records):
        if r["artifact_type"] != "identity_hint":
            raise ValidationError(f"identity_hints: record[{i}] artifact_type must be 'identity_hint'")

        if expected_clip_id is not None and r["clip_id"] != expected_clip_id:
            raise ValidationError(
                f"identity_hints: record[{i}] clip_id mismatch expected={expected_clip_id} got={r['clip_id']}"
            )

        _require_non_empty_str(r["tracklet_id"], name="identity_hints.tracklet_id")
        _require_non_empty_str(r["anchor_key"], name="identity_hints.anchor_key")

        if ":" not in str(r["anchor_key"]):
            raise ValidationError(f"identity_hints: record[{i}] anchor_key must be namespaced like 'tag:<id>'")

        if r["constraint"] not in ("must_link", "cannot_link"):
            raise ValidationError(f"identity_hints: record[{i}] invalid constraint={r['constraint']}")

        conf = float(r["confidence"])
        if conf < 0.0 or conf > 1.0:
            raise ValidationError(f"identity_hints: record[{i}] confidence outside [0,1]")

        # f0_models.IdentityHint declares evidence: Dict[str, Any]
        if not isinstance(r.get("evidence"), dict):
            raise ValidationError(f"identity_hints: record[{i}] evidence must be a dict")


def validate_tag_observations_records(
    records: List[Dict],
    *,
    expected_clip_id: Optional[str] = None,
    expected_tag_family: Optional[str] = None,
) -> None:
    """Validate loaded JSONL tag_observations as dict records.

    Enforces:
    - base metadata present
    - join-friendly keys present (frame_index, timestamp_ms, detection_id)
    - tag_family matches configured expected value (Option B), else defaults to "36h11"
    """

    base = ["schema_version", "artifact_type", "clip_id", "camera_id", "pipeline_version", "created_at_ms"]
    req = base + [
        "frame_index",
        "timestamp_ms",
        "detection_id",
        "tag_id",
        "tag_family",
        "confidence",
        "roi_method",
    ]
    _validate_jsonl_required_fields(records, req, name="tag_observations")

    fam = expected_tag_family or "36h11"

    for i, r in enumerate(records):
        if r["artifact_type"] != "tag_observation":
            raise ValidationError(f"tag_observations: record[{i}] artifact_type must be 'tag_observation'")

        if expected_clip_id is not None and r["clip_id"] != expected_clip_id:
            raise ValidationError(
                f"tag_observations: record[{i}] clip_id mismatch expected={expected_clip_id} got={r['clip_id']}"
            )

        _require_non_empty_str(r["detection_id"], name="tag_observations.detection_id")
        _require_non_empty_str(r["tag_id"], name="tag_observations.tag_id")
        _require_non_empty_str(r["tag_family"], name="tag_observations.tag_family")

        if str(r["tag_family"]) != fam:
            raise ValidationError(
                f"tag_observations: record[{i}] tag_family mismatch expected={fam} got={r['tag_family']}"
            )

        conf = float(r["confidence"])
        if conf < 0.0 or conf > 1.0:
            raise ValidationError(f"tag_observations: record[{i}] confidence outside [0,1]")


def validate_identity_assignments_records(records: List[Dict], *, expected_clip_id: Optional[str] = None) -> None:
    base = ["schema_version", "artifact_type", "clip_id", "camera_id", "pipeline_version", "created_at_ms"]
    req = base + ["person_id", "tag_id", "assignment_confidence", "evidence"]
    _validate_jsonl_required_fields(records, req, name="identity_assignments")

    for i, r in enumerate(records):
        if r["artifact_type"] != "identity_assignment":
            raise ValidationError(f"identity_assignments: record[{i}] artifact_type must be 'identity_assignment'")

        if expected_clip_id is not None and r["clip_id"] != expected_clip_id:
            raise ValidationError(
                f"identity_assignments: record[{i}] clip_id mismatch expected={expected_clip_id} got={r['clip_id']}"
            )

        _require_non_empty_str(r["person_id"], name="identity_assignments.person_id")
        _require_non_empty_str(r["tag_id"], name="identity_assignments.tag_id")

        conf = float(r["assignment_confidence"])
        if conf < 0.0 or conf > 1.0:
            raise ValidationError(f"identity_assignments: record[{i}] assignment_confidence outside [0,1]")


def validate_match_sessions_records(records: List[Dict], *, expected_clip_id: Optional[str] = None) -> None:
    base = ["schema_version", "artifact_type", "clip_id", "camera_id", "pipeline_version", "created_at_ms"]
    req = base + [
        "match_id",
        "person_id_a",
        "person_id_b",
        "start_frame",
        "end_frame",
        "start_ts_ms",
        "end_ts_ms",
        "method",
        "confidence",
        "evidence",
    ]
    _validate_jsonl_required_fields(records, req, name="match_sessions")

    for i, r in enumerate(records):
        if r["artifact_type"] != "match_session":
            raise ValidationError(f"match_sessions: record[{i}] artifact_type must be 'match_session'")

        if expected_clip_id is not None and r["clip_id"] != expected_clip_id:
            raise ValidationError(
                f"match_sessions: record[{i}] clip_id mismatch expected={expected_clip_id} got={r['clip_id']}"
            )

        if int(r["end_frame"]) < int(r["start_frame"]):
            raise ValidationError(f"match_sessions: record[{i}] end_frame < start_frame")

        conf = float(r["confidence"])
        if conf < 0.0 or conf > 1.0:
            raise ValidationError(f"match_sessions: record[{i}] confidence outside [0,1]")

        # v2 schema: optional fields (backward compatible)
        if "partial_start" in r and not isinstance(r["partial_start"], bool):
            raise ValidationError(f"match_sessions: record[{i}] partial_start must be bool")
        if "partial_end" in r and not isinstance(r["partial_end"], bool):
            raise ValidationError(f"match_sessions: record[{i}] partial_end must be bool")

        ev = r.get("evidence")
        if isinstance(ev, dict) and "sources" in ev:
            if not isinstance(ev["sources"], list):
                raise ValidationError(f"match_sessions: record[{i}] evidence.sources must be a list")


def validate_export_manifest_records(records: List[Dict], *, expected_clip_id: Optional[str] = None) -> None:
    base = ["schema_version", "artifact_type", "clip_id", "camera_id", "pipeline_version", "created_at_ms"]
    req = base + ["export_id", "match_id", "output_video_path", "crop_mode", "privacy", "inputs"]
    _validate_jsonl_required_fields(records, req, name="export_manifest")

    for i, r in enumerate(records):
        if r["artifact_type"] != "export_manifest":
            raise ValidationError(f"export_manifest: record[{i}] artifact_type must be 'export_manifest'")

        if expected_clip_id is not None and r["clip_id"] != expected_clip_id:
            raise ValidationError(
                f"export_manifest: record[{i}] clip_id mismatch expected={expected_clip_id} got={r['clip_id']}"
            )

        _require_non_empty_str(r["output_video_path"], name="export_manifest.output_video_path")


# ----------------------------
# End-to-end contract validation entrypoint (per-clip)
# ----------------------------

@dataclass(frozen=True)
class ClipTables:
    """
    Convenience bundle for validating a clip’s core Parquet tables together.
    (JSONL records validated separately.)
    """
    detections: pd.DataFrame
    tracklet_frames: pd.DataFrame
    tracklet_summaries: pd.DataFrame
    contact_points: Optional[pd.DataFrame] = None
    person_tracks: Optional[pd.DataFrame] = None


def read_stage_B_contact_points_df(layout: ClipOutputLayout) -> pd.DataFrame:
    """Load Stage B contact points (refined preferred, legacy fallback)."""
    refined = layout.stage_B_contact_points_refined_parquet()
    legacy = layout.stage_B_contact_points_parquet_legacy()
    if refined.exists():
        return pd.read_parquet(refined)
    if legacy.exists():
        return pd.read_parquet(legacy)
    raise FileNotFoundError(
        "Stage B contact points parquet not found. Expected either "
        f"{refined.as_posix()} or {legacy.as_posix()}"
    )


def read_stage_A_clip_tables(layout: ClipOutputLayout) -> ClipTables:
    """Load the core Stage A tables for contract validation.

    Critical invariant: Stage A contact points must be loaded from
    stage_A/contact_points.parquet (not Stage B).
    """
    return ClipTables(
        detections=pd.read_parquet(layout.detections_parquet()),
        tracklet_frames=pd.read_parquet(layout.tracklet_frames_parquet()),
        tracklet_summaries=pd.read_parquet(layout.tracklet_summaries_parquet()),
        contact_points=pd.read_parquet(layout.stage_A_contact_points_parquet()),
    )


def validate_stage_A_contract(tables: ClipTables) -> None:
    validate_detections_df(tables.detections)
    validate_tracklet_tables(tables.tracklet_frames, tables.tracklet_summaries)
    validate_tracklet_frames_fk_to_detections(tables.tracklet_frames, tables.detections)

    # F0G: Stage A owns baseline contact points
    if tables.contact_points is None:
        raise ValidationError("stage_A: contact_points table is required for validation")
    validate_stage_A_contact_points_df(tables.contact_points)


def validate_stage_B_contract(tables: ClipTables) -> None:
    """Stage B contact points remain a supported artifact, but are no longer required
    for end-to-end validation unless Stage B is explicitly validated.
    """
    if tables.contact_points is None:
        raise ValidationError("stage_B: contact_points table is required for validation")
    validate_contact_points_df(tables.contact_points)


def validate_stage_D_contract(tables: ClipTables) -> None:
    if tables.person_tracks is None:
        raise ValidationError("stage_D: person_tracks table is required for validation")
    validate_person_tracks_df(tables.person_tracks)
    validate_person_tracks_traceability(tables.person_tracks, tables.detections, tables.tracklet_frames)
