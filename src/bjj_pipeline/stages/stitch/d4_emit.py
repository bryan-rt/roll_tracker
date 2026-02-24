from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import pandas as pd

from bjj_pipeline.contracts.f0_paths import ClipOutputLayout
from bjj_pipeline.contracts.f0_models import SCHEMA_VERSION_DEFAULT
from bjj_pipeline.contracts.f0_validate import (
    validate_identity_assignments_records,
    validate_person_tracks_df,
)
from bjj_pipeline.stages.stitch.d3_audit import append_audit_event


# --------------------------------------------------------------------------------------
# Internal data structures
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class NodeSpan:
    person_id: str
    node_id: str
    tracklet_ids: Tuple[str, ...]
    start_frame: int
    end_frame: int
    effective_cap: int


# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------


def _extract_entity_paths_format_a(
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    flow_by_edge_id: Dict[str, int],
) -> List[Dict[str, Any]]:
    """
    Lightweight reimplementation of entity extraction using selected edges.

    Decompose integer flows into SOURCE->...->SINK paths by repeatedly extracting
    one unit of flow at a time (consuming remaining edge flow).

    This respects multiplicity (e.g. an edge with flow=2 contributes to 2 paths).
    """

    def _flow(eid: Any) -> int:
        try:
            return int(flow_by_edge_id.get(str(eid), 0))
        except Exception:
            return 0

    # Keep only edges with positive integer flow
    sel_edges = edges_df.copy()
    sel_edges["_flow"] = sel_edges["edge_id"].map(_flow)
    sel_edges = sel_edges[sel_edges["_flow"] > 0].copy()

    if sel_edges.empty:
        return []

    node_meta: Dict[str, Dict[str, Any]] = {}
    for _, row in nodes_df.iterrows():
        node_meta[str(row["node_id"])] = dict(row)

    # adjacency: u -> list of edge records (we will consume remaining flow)
    adj: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    remaining: Dict[str, int] = {}

    def _node_start_frame(nid: str) -> int:
        d = node_meta.get(nid, {})
        try:
            sf = d.get("start_frame")
            if sf is None or (isinstance(sf, float) and math.isnan(sf)):
                return 10**12
            return int(sf)
        except Exception:
            return 10**12

    # Deterministic ordering: prefer edges whose destination starts earlier,
    # then by edge_id as a stable tie-break.
    def _edge_sort_key(er: Dict[str, Any]) -> Tuple[int, str]:
        return (_node_start_frame(er["v"]), str(er["edge_id"]))

    for _, row in sel_edges.iterrows():
        eid = str(row["edge_id"])
        u = str(row["u"])
        v = str(row["v"])
        remaining[eid] = int(row["_flow"])
        adj[u].append(
            {
                "edge_id": eid,
                "edge_type": str(row.get("edge_type")),
                "u": u,
                "v": v,
            }
        )

    for u in list(adj.keys()):
        adj[u].sort(key=_edge_sort_key)

    # Total number of path units is sum of flow on SOURCE outgoing edges (k definition)
    k = 0
    for er in adj.get("SOURCE", []):
        k += remaining.get(er["edge_id"], 0)

    entities: List[Dict[str, Any]] = []

    for _ in range(k):
        cur = "SOURCE"
        steps: List[Dict[str, Any]] = []

        # Walk until SINK or dead-end
        guard = 0
        while cur != "SINK" and guard < 100000:
            guard += 1

            outs = adj.get(cur, [])
            # pick first edge with remaining flow
            chosen = None
            for er in outs:
                if remaining.get(er["edge_id"], 0) > 0:
                    chosen = er
                    break

            if chosen is None:
                # dead-end for this unit; stop path (shouldn't happen if graph is consistent)
                break

            # consume 1 unit of flow
            remaining[chosen["edge_id"]] = remaining.get(chosen["edge_id"], 0) - 1

            u = chosen["u"]
            v = chosen["v"]
            steps.append(
                {
                    "edge_id": chosen["edge_id"],
                    "edge_type": chosen["edge_type"],
                    "u": u,
                    "v": v,
                    "u_node": node_meta.get(u, {"node_id": u}),
                    "v_node": node_meta.get(v, {"node_id": v}),
                }
            )
            cur = v

        entities.append({"steps": steps})

    return entities


def _node_effective_capacity(
    node_row: Dict[str, Any],
    edges_df: pd.DataFrame,
) -> int:
    """
    Effective capacity = max(node.capacity, any incident edge payload.desired_capacity)
    """

    base_cap = int(node_row.get("capacity", 1) or 1)
    node_id = str(node_row["node_id"])

    eff = base_cap

    # scan incident edges
    for _, row in edges_df.iterrows():
        if str(row["u"]) == node_id or str(row["v"]) == node_id:
            payload = None
            if "payload_json" in row and isinstance(row["payload_json"], str) and row["payload_json"]:
                try:
                    payload = json.loads(row["payload_json"])
                except Exception:
                    payload = None
            # Some codepaths may still provide a decoded dict under "payload"
            if payload is None and isinstance(row.get("payload"), dict):
                payload = row.get("payload")

            if isinstance(payload, dict):
                desired = payload.get("desired_capacity")
                if desired is not None:
                    try:
                        eff = max(eff, int(desired))
                    except Exception:
                        pass

    return max(1, eff)


def _collect_tracklet_ids(node_row: Dict[str, Any]) -> Set[str]:
    """
    Collect all tracklet identifiers from node metadata.
    """

    tids: Set[str] = set()
    for key in (
        "base_tracklet_id",
        "carrier_tracklet_id",
        "disappearing_tracklet_id",
        "new_tracklet_id",
    ):
        val = node_row.get(key)
        if isinstance(val, str) and val:
            tids.add(val)
    return tids


# --------------------------------------------------------------------------------------
# Main D4 entrypoint
# --------------------------------------------------------------------------------------


def run_d4_emit(
    *,
    config: Dict[str, Any],
    inputs: Dict[str, Any],
    compiled: Any,
    res: Any,
    checkpoint: str,
) -> Dict[str, Any]:
    """
    Emit canonical Stage D outputs:

    - person_tracks.parquet
    - person_spans.parquet (helper)
    - identity_assignments.jsonl
    """

    layout: ClipOutputLayout = inputs["layout"]
    manifest = inputs["manifest"]

    if res is None:
        raise ValueError("D4 requires a solved ILP result.")

    nodes_df: pd.DataFrame = compiled.nodes_df
    edges_df: pd.DataFrame = compiled.edges_df
    flow_by_edge_id: Dict[str, int] = res.flow_by_edge_id

    # ------------------------------------------------------------------
    # 1) Extract entity paths
    # ------------------------------------------------------------------
    entities = _extract_entity_paths_format_a(
        nodes_df=nodes_df,
        edges_df=edges_df,
        flow_by_edge_id=flow_by_edge_id,
    )

    # ------------------------------------------------------------------
    # 2) Build person spans from D1 nodes (respecting effective capacity)
    # ------------------------------------------------------------------
    person_spans: List[NodeSpan] = []
    person_index = 1

    for ent in entities:
        person_id = f"p{person_index:04d}"
        person_index += 1

        visited_node_ids: Set[str] = set()

        for step in ent["steps"]:
            visited_node_ids.add(str(step["u_node"]["node_id"]))
            visited_node_ids.add(str(step["v_node"]["node_id"]))

        for node_id in visited_node_ids:
            node_row = nodes_df.loc[nodes_df["node_id"] == node_id]
            if node_row.empty:
                continue

            row_dict = dict(node_row.iloc[0])
            tracklets = tuple(sorted(_collect_tracklet_ids(row_dict)))
            if not tracklets:
                continue

            start_frame = int(row_dict["start_frame"])
            end_frame = int(row_dict["end_frame"])

            eff_cap = _node_effective_capacity(row_dict, edges_df)

            person_spans.append(
                NodeSpan(
                    person_id=person_id,
                    node_id=node_id,
                    tracklet_ids=tracklets,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    effective_cap=eff_cap,
                )
            )

    # ------------------------------------------------------------------
    # 3) Emit person_spans.parquet (helper artifact)
    # ------------------------------------------------------------------
    if person_spans:
        spans_df = pd.DataFrame(
            [
                {
                    "person_id": s.person_id,
                    "node_id": s.node_id,
                    "start_frame": s.start_frame,
                    "end_frame": s.end_frame,
                    "effective_cap": s.effective_cap,
                }
                for s in person_spans
            ]
        ).sort_values(["person_id", "start_frame", "node_id"])
    else:
        spans_df = pd.DataFrame(
            columns=["person_id", "node_id", "start_frame", "end_frame", "effective_cap"]
        )

    spans_df.to_parquet(layout.person_spans_parquet(), index=False)

    # ------------------------------------------------------------------
    # 4) Build person_tracks.parquet
    # ------------------------------------------------------------------
    bank_frames = pd.read_parquet(layout.tracklet_bank_frames_parquet())
    detections = pd.read_parquet(layout.detections_parquet())

    # join detections
    merged = bank_frames.merge(
        detections[
            [
                "clip_id",
                "camera_id",
                "frame_index",
                "detection_id",
                "x1",
                "y1",
                "x2",
                "y2",
                "mask_ref",
            ]
        ],
        on=["clip_id", "camera_id", "frame_index", "detection_id"],
        how="left",
        validate="many_to_one",
    )

    # select repaired world coords if available
    if "x_m_repaired" in merged.columns:
        merged["x_m"] = merged["x_m_repaired"].combine_first(merged["x_m"])
        merged["y_m"] = merged["y_m_repaired"].combine_first(merged["y_m"])

    rows: List[Dict[str, Any]] = []

    for span in person_spans:
        for tracklet_id in span.tracklet_ids:
            subset = merged[
                (merged["tracklet_id"] == tracklet_id)
                & (merged["frame_index"] >= span.start_frame)
                & (merged["frame_index"] <= span.end_frame)
            ]

            for _, r in subset.iterrows():
                rows.append(
                    {
                        "clip_id": r["clip_id"],
                        "camera_id": r["camera_id"],
                        "person_id": span.person_id,
                        "frame_index": r["frame_index"],
                        "timestamp_ms": r["timestamp_ms"],
                        "detection_id": r["detection_id"],
                        "tracklet_id": r["tracklet_id"],
                        "x1": r["x1"],
                        "y1": r["y1"],
                        "x2": r["x2"],
                        "y2": r["y2"],
                        "x_m": r.get("x_m"),
                        "y_m": r.get("y_m"),
                        "mask_ref": r.get("mask_ref"),
                        "reid_sim": np.nan,
                        "stitch_edge_type": None,
                    }
                )

    person_tracks_df = pd.DataFrame(rows)

    if not person_tracks_df.empty:
        person_tracks_df = person_tracks_df.sort_values(
            ["person_id", "frame_index", "detection_id"]
        ).reset_index(drop=True)

        # Enforce schema dtypes expected by F0:
        # - reid_sim must be float family (not object from None)
        if "reid_sim" not in person_tracks_df.columns:
            person_tracks_df["reid_sim"] = np.nan
        person_tracks_df["reid_sim"] = pd.to_numeric(
            person_tracks_df["reid_sim"], errors="coerce"
        ).astype("float64")

    validate_person_tracks_df(person_tracks_df)

    person_tracks_df.to_parquet(layout.person_tracks_parquet(), index=False)

    # ------------------------------------------------------------------
    # 5) Identity assignments (dominant tag + conflict logging)
    # ------------------------------------------------------------------
    tag_groups_all = compiled.constraints.get("must_link_groups", [])
    tag_groups = [
        g
        for g in tag_groups_all
        if isinstance(g.get("anchor_key"), str)
        and g["anchor_key"].startswith("tag:")
    ]

    identity_records: List[Dict[str, Any]] = []
    conflicts = 0

    # Diagnostics to explain empty outputs
    tag_tracklet_ids: Set[str] = set()
    tag_anchor_keys_sample: List[str] = []
    for g in tag_groups:
        ak = g.get("anchor_key")
        if isinstance(ak, str) and len(tag_anchor_keys_sample) < 5:
            tag_anchor_keys_sample.append(ak)
        for tid in g.get("tracklet_ids", []) or []:
            if isinstance(tid, str):
                tag_tracklet_ids.add(tid)

    person_tracklet_ids: Set[str] = set()
    if not person_tracks_df.empty and "tracklet_id" in person_tracks_df.columns:
        person_tracklet_ids = set(
            str(x) for x in person_tracks_df["tracklet_id"].dropna().unique()
        )

    overlap_tracklet_ids = sorted(tag_tracklet_ids.intersection(person_tracklet_ids))
    overlap_sample = overlap_tracklet_ids[:10]
    person_tracklet_sample = sorted(list(person_tracklet_ids))[:10]
    tag_tracklet_sample = sorted(list(tag_tracklet_ids))[:10]

    identity_assignment_reason = "emitted"
    if person_tracks_df.empty:
        identity_assignment_reason = "no_person_tracks"
    elif not tag_groups:
        identity_assignment_reason = "no_tag_groups"
    elif not overlap_tracklet_ids:
        identity_assignment_reason = "no_tracklet_overlap_between_tags_and_tracks"

    if identity_assignment_reason == "emitted":
        frame_counts = (
            person_tracks_df.groupby(["person_id", "tracklet_id"])
            .size()
            .to_dict()
        )

        created_at_ms = int(time.time() * 1000)

        for person_id in person_tracks_df["person_id"].unique():
            tag_scores: Dict[str, int] = {}
            for g in tag_groups:
                anchor_key = g["anchor_key"]  # e.g. "tag:1"
                overlap = set(g.get("tracklet_ids", []) or [])
                score = 0
                for (pid, tid), cnt in frame_counts.items():
                    if pid == person_id and tid in overlap:
                        score += int(cnt)
                if score > 0:
                    tag_scores[anchor_key] = score

            if not tag_scores:
                continue

            sorted_tags = sorted(
                tag_scores.items(),
                key=lambda kv: (-kv[1], kv[0]),
            )

            dominant_anchor_key, dominant_score = sorted_tags[0]
            total = sum(tag_scores.values())

            # anchor_key="tag:1" -> tag_id="1" (string)
            tag_id = (
                dominant_anchor_key.split("tag:", 1)[1]
                if "tag:" in dominant_anchor_key
                else dominant_anchor_key
            )

            evidence: Dict[str, Any] = {
                "anchor_key": dominant_anchor_key,
                "tag_scores": tag_scores,
                "dominant_score": int(dominant_score),
                "total_tag_frames": int(total),
            }

            if len(sorted_tags) > 1:
                conflicts += 1
                evidence["conflicts"] = [
                    {"anchor_key": t, "score": int(s)} for t, s in sorted_tags[1:]
                ]

            identity_records.append(
                {
                    "schema_version": SCHEMA_VERSION_DEFAULT,
                    "artifact_type": "identity_assignment",
                    "clip_id": manifest.clip_id,
                    "camera_id": manifest.camera_id,
                    "pipeline_version": manifest.pipeline_version,
                    "created_at_ms": created_at_ms,
                    "person_id": str(person_id),
                    "tag_id": str(tag_id),
                    "assignment_confidence": float(dominant_score) / float(total)
                    if total > 0
                    else 1.0,
                    "evidence": evidence,
                }
            )

    # Validate + write JSONL (empty list is allowed)
    validate_identity_assignments_records(identity_records, expected_clip_id=manifest.clip_id)

    with open(layout.identity_assignments_jsonl(), "w", encoding="utf-8") as f:
        for rec in identity_records:
            f.write(json.dumps(rec, sort_keys=True) + "\n")

    # ------------------------------------------------------------------
    # 6) Audit summary
    # ------------------------------------------------------------------
    event = {
        "event_type": "d4_emit_summary",
        "checkpoint": checkpoint,
        "n_person_ids": person_tracks_df["person_id"].nunique()
        if not person_tracks_df.empty
        else 0,
        "n_person_tracks_rows": len(person_tracks_df),
        "n_identity_assignments": len(identity_records),
        "n_tag_conflicts": conflicts,

        # Diagnostics for identity assignment emptiness
        "identity_assignment_reason": identity_assignment_reason,
        "n_must_link_groups_total": len(tag_groups_all) if isinstance(tag_groups_all, list) else 0,
        "n_tag_groups_tag_prefix": len(tag_groups),
        "tag_group_anchor_keys_sample": tag_anchor_keys_sample,
        "n_tag_group_unique_tracklet_ids": len(tag_tracklet_ids),
        "tag_group_tracklet_ids_sample": tag_tracklet_sample,
        "n_person_tracks_unique_tracklet_ids": len(person_tracklet_ids),
        "person_tracks_tracklet_ids_sample": person_tracklet_sample,
        "n_overlap_tracklet_ids": len(overlap_tracklet_ids),
        "overlap_tracklet_ids_sample": overlap_sample,
    }
    # append_audit_event in this repo is keyword-only; support the common signatures.
    try:
        append_audit_event(layout=layout, event=event)
    except TypeError:
        # Some implementations take an explicit audit_path instead of layout.
        append_audit_event(audit_path=layout.audit_jsonl("D"), event=event)

    return {
        "n_person_ids": person_tracks_df["person_id"].nunique()
        if not person_tracks_df.empty
        else 0,
        "n_rows": len(person_tracks_df),
    }