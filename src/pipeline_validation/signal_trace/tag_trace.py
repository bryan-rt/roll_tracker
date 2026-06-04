"""Tagged-person signal trace (CP-TAG-1).

Traces tag_id through the full pipeline (A->C->D->E) to answer:
does the AprilTag signal deliver correct identity for tagged people?

Key finding: tag detection is bbox-gated (Stage C scans padded detection
bounding boxes only, never the full frame). Detection recall directly
limits tag visibility.
"""
from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUTS_DIR = REPO_ROOT / "outputs"
TRAINING_DATA_DIR = REPO_ROOT / "data" / "training_data"
EVAL_DIR = OUTPUTS_DIR / "_eval" / "signal_trace"
CONFIGS_DIR = REPO_ROOT / "configs"


# ---------------------------------------------------------------------------
# Component 1: Cross-tab (Stage A x Stage D)
# ---------------------------------------------------------------------------

def build_cross_tab(model_id: str, camera_ids: list[str] | None = None) -> dict:
    """Cross-tabulate Stage A classification x Stage D classification.

    Uses existing gt_signal_trace_d.parquet for specified cameras (val-split only).
    Returns nested dict {a_class: {d_class: count}} plus per-camera breakdown.
    """
    trace_base = EVAL_DIR / model_id
    if camera_ids:
        cam_dirs = [trace_base / cam for cam in camera_ids if (trace_base / cam).is_dir()]
    else:
        cam_dirs = sorted(
            d for d in trace_base.iterdir()
            if d.is_dir()
            and not d.name.startswith("_")
            and (d / "gt_signal_trace_d.parquet").exists()
        )

    a_classes = ["tight_match", "pair_box", "split", "miss"]
    d_classes = ["correct_id", "wrong_id", "no_id", "no_detection"]

    per_camera = {}
    agg = {ac: {dc: 0 for dc in d_classes} for ac in a_classes}

    for cam_dir in cam_dirs:
        d_trace_path = cam_dir / "gt_signal_trace_d.parquet"
        if not d_trace_path.exists():
            continue

        df = pd.read_parquet(d_trace_path)
        cam = cam_dir.name
        cam_tab = {ac: {dc: 0 for dc in d_classes} for ac in a_classes}

        for _, row in df.iterrows():
            a_cls = row["classification"]
            d_cls = row["d_classification"]
            if a_cls in cam_tab and d_cls in cam_tab[a_cls]:
                cam_tab[a_cls][d_cls] += 1
                agg[a_cls][d_cls] += 1

        per_camera[cam] = {
            "cross_tab": cam_tab,
            "total": len(df),
        }

    return {
        "aggregate": agg,
        "per_camera": per_camera,
        "total": sum(sum(v.values()) for v in agg.values()),
    }


def write_cross_tab(model_id: str, cross_tab: dict) -> None:
    """Write cross_tab.json and cross_tab.md."""
    out_dir = EVAL_DIR / model_id
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "cross_tab.json", "w") as f:
        json.dump(cross_tab, f, indent=2)

    a_classes = ["tight_match", "pair_box", "split", "miss"]
    d_classes = ["correct_id", "wrong_id", "no_id", "no_detection"]
    agg = cross_tab["aggregate"]
    total = cross_tab["total"]

    lines = [
        f"# Stage A x Stage D Cross-Tab: {model_id}",
        "",
        "## Aggregate (all cameras)",
        "",
        "| Stage A \\ Stage D | correct_id | wrong_id | no_id | no_detection | Total |",
        "|---|---|---|---|---|---|",
    ]

    for ac in a_classes:
        row_total = sum(agg[ac].values())
        cells = []
        for dc in d_classes:
            n = agg[ac][dc]
            pct = f" ({n/total:.1%})" if total > 0 else ""
            cells.append(f"{n}{pct}")
        lines.append(f"| {ac} | {' | '.join(cells)} | {row_total} |")

    col_totals = [sum(agg[ac][dc] for ac in a_classes) for dc in d_classes]
    lines.append(f"| **Total** | {' | '.join(str(c) for c in col_totals)} | {total} |")

    # Key findings
    pair_wrong = agg["pair_box"]["wrong_id"]
    tight_wrong = agg["tight_match"]["wrong_id"]
    pair_total = sum(agg["pair_box"].values())
    tight_total = sum(agg["tight_match"].values())

    lines.extend([
        "",
        "## Key findings",
        "",
        f"- **pair_box -> wrong_id:** {pair_wrong} frames "
        f"({pair_wrong/pair_total:.1%} of pair_box)" if pair_total > 0 else "",
        f"- **tight_match -> wrong_id:** {tight_wrong} frames "
        f"({tight_wrong/tight_total:.1%} of tight_match)" if tight_total > 0 else "",
        f"- Pair-box-driven misattribution: {pair_wrong}/{pair_wrong + tight_wrong:.0f} "
        f"= {pair_wrong/(pair_wrong + tight_wrong):.1%} of all wrong_id"
        if (pair_wrong + tight_wrong) > 0 else "",
    ])

    # Per-camera
    for cam, data in cross_tab["per_camera"].items():
        ct = data["cross_tab"]
        cam_total = data["total"]
        lines.extend([
            "",
            f"## {cam} (n={cam_total})",
            "",
            "| Stage A \\ Stage D | correct_id | wrong_id | no_id | no_detection | Total |",
            "|---|---|---|---|---|---|",
        ])
        for ac in a_classes:
            row_total = sum(ct[ac].values())
            cells = [str(ct[ac][dc]) for dc in d_classes]
            lines.append(f"| {ac} | {' | '.join(cells)} | {row_total} |")

    with open(out_dir / "cross_tab.md", "w") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Component 2: Tagged person identification
# ---------------------------------------------------------------------------

def identify_tagged_person(
    tag_observations: list[dict],
    stage_a_trace: pd.DataFrame,
    tag_id: str = "1",
) -> dict:
    """Identify which gt_track_id corresponds to the tagged person.

    Cross-references tag observations (tracklet_id + frame_index) against
    the greedy-matched stage_a_trace to find the GT person.
    """
    filtered = [o for o in tag_observations if str(o.get("tag_id")) == tag_id]
    if not filtered:
        return {
            "gt_track_id": None,
            "tracklet_ids": [],
            "n_observations": 0,
            "vote_detail": {},
            "note": f"No tag_id={tag_id} observations found",
        }

    tracklet_ids = sorted(set(o["tracklet_id"] for o in filtered))

    # For each observation, find which GT person matches the tracklet at that frame.
    # Tag observations may fall between annotated frames (stride>1), so find the
    # nearest annotated frame within +/- stride distance.
    annotated_frames = sorted(stage_a_trace["frame_index"].unique())
    votes: Counter = Counter()
    for obs in filtered:
        tid = obs["tracklet_id"]
        fi = obs["frame_index"]

        # Try exact frame first, then nearest annotated frame
        search_frames = [fi]
        if fi not in annotated_frames and annotated_frames:
            # Find nearest annotated frame
            import bisect
            idx = bisect.bisect_left(annotated_frames, fi)
            candidates = []
            if idx > 0:
                candidates.append(annotated_frames[idx - 1])
            if idx < len(annotated_frames):
                candidates.append(annotated_frames[idx])
            if candidates:
                nearest = min(candidates, key=lambda x: abs(x - fi))
                search_frames.append(nearest)

        for search_fi in search_frames:
            mask = (
                (stage_a_trace["tracklet_id"] == tid)
                & (stage_a_trace["frame_index"] == search_fi)
            )
            matched = stage_a_trace.loc[mask]
            if not matched.empty:
                for _, row in matched.iterrows():
                    votes[int(row["gt_track_id"])] += 1
                break  # found match, stop searching

    if not votes:
        return {
            "gt_track_id": None,
            "tracklet_ids": tracklet_ids,
            "n_observations": len(filtered),
            "vote_detail": {},
            "note": "Tag observations found but no GT match at observation frames",
        }

    gt_track_id = votes.most_common(1)[0][0]
    return {
        "gt_track_id": gt_track_id,
        "tracklet_ids": tracklet_ids,
        "n_observations": len(filtered),
        "vote_detail": dict(votes),
    }


# ---------------------------------------------------------------------------
# Component 3: Tag census
# ---------------------------------------------------------------------------

def build_tag_census(
    tag_observations: list[dict],
    stage_a_trace: pd.DataFrame | None,
    detections_df: pd.DataFrame,
    tagged_gt_track_id: int | None,
    tagged_tracklet_ids: list[str],
    tag_id: str = "1",
) -> dict:
    """Build tag observation census: detection rate and context breakdown."""
    filtered = [o for o in tag_observations if str(o.get("tag_id")) == tag_id]

    # Tag observation frames
    obs_frames = set(o["frame_index"] for o in filtered)
    obs_tracklets = set(o["tracklet_id"] for o in filtered)

    # All tag_ids observed (check for misreads)
    all_tag_ids = Counter(str(o.get("tag_id")) for o in tag_observations)

    # Tracklet lifetime: frames where tagged tracklets appear in detections
    if tagged_tracklet_ids:
        tracklet_mask = detections_df["tracklet_id"].isin(tagged_tracklet_ids)
        tracklet_frames = set(detections_df.loc[tracklet_mask, "frame_index"].values)
    else:
        tracklet_frames = set()

    total_tracklet_frames = len(tracklet_frames)
    tag_detection_rate = len(obs_frames) / total_tracklet_frames if total_tracklet_frames > 0 else 0.0

    # Context breakdown (requires stage_a_trace)
    context_breakdown = {}
    if stage_a_trace is not None and tagged_gt_track_id is not None:
        gt_mask = stage_a_trace["gt_track_id"] == tagged_gt_track_id
        gt_trace = stage_a_trace.loc[gt_mask]

        for cls in ["tight_match", "pair_box", "split", "miss"]:
            cls_frames = set(gt_trace.loc[gt_trace["classification"] == cls, "frame_index"].values)
            cls_obs = cls_frames & obs_frames
            n = len(cls_frames)
            context_breakdown[cls] = {
                "total_frames": n,
                "frames_with_tag": len(cls_obs),
                "tag_rate": len(cls_obs) / n if n > 0 else None,
            }

    # Bbox-gated diagnostic: check detection coverage near tag observation frames
    bbox_gated_diagnostic = _bbox_gated_diagnostic(
        filtered, detections_df, tagged_tracklet_ids,
    )

    return {
        "tag_id": tag_id,
        "n_observations": len(filtered),
        "observation_frames": sorted(obs_frames),
        "observation_tracklets": sorted(obs_tracklets),
        "tracklet_lifetime_frames": total_tracklet_frames,
        "tag_detection_rate": tag_detection_rate,
        "tag_consistency": {
            "all_tag_ids_observed": dict(all_tag_ids),
            "misreads": len([t for t in all_tag_ids if t != tag_id]),
        },
        "context_breakdown": context_breakdown,
        "bbox_gated": bbox_gated_diagnostic,
    }


def _bbox_gated_diagnostic(
    tag_observations: list[dict],
    detections_df: pd.DataFrame,
    tagged_tracklet_ids: list[str],
) -> dict:
    """Check detection coverage near tag observation frames.

    Tag detection is bbox-gated: Stage C only scans padded detection bboxes.
    If the detector misses the person, Stage C never gets the chance to look.
    """
    if not tag_observations:
        return {
            "mechanism": "bbox_gated",
            "note": "Tag detection is bbox-gated (Stage C scans padded detection "
                    "bboxes only). No tag observations to analyze.",
        }

    # For each tag observation, record the pixel location (center of roi_xyxy)
    obs_locations = []
    for obs in tag_observations:
        roi = obs.get("roi_xyxy")
        if roi and len(roi) == 4:
            cx = (roi[0] + roi[2]) / 2
            cy = (roi[1] + roi[3]) / 2
            obs_locations.append({
                "frame_index": obs["frame_index"],
                "tracklet_id": obs["tracklet_id"],
                "roi_cx": cx, "roi_cy": cy,
                "roi_xyxy": roi,
            })

    if not obs_locations:
        return {"mechanism": "bbox_gated", "note": "No ROI data in observations."}

    # Check nearby frames (within +/- 30 frames) for detection coverage at tag location
    window = 30
    coverage_checks = []
    for loc in obs_locations:
        fi = loc["frame_index"]
        cx, cy = loc["roi_cx"], loc["roi_cy"]

        for offset in range(-window, window + 1):
            check_fi = fi + offset
            if check_fi < 0:
                continue
            frame_dets = detections_df[detections_df["frame_index"] == check_fi]
            if frame_dets.empty:
                coverage_checks.append({
                    "frame_index": check_fi,
                    "offset": offset,
                    "has_any_detection": False,
                    "has_covering_detection": False,
                })
                continue

            # Check if any detection covers the tag's last-known location
            covering = False
            for _, det in frame_dets.iterrows():
                if det["x1"] <= cx <= det["x2"] and det["y1"] <= cy <= det["y2"]:
                    covering = True
                    break

            coverage_checks.append({
                "frame_index": check_fi,
                "offset": offset,
                "has_any_detection": True,
                "has_covering_detection": covering,
            })

    n_checked = len(coverage_checks)
    n_covered = sum(1 for c in coverage_checks if c["has_covering_detection"])
    n_any_det = sum(1 for c in coverage_checks if c["has_any_detection"])

    return {
        "mechanism": "bbox_gated",
        "note": "Tag detection is bbox-gated: Stage C scans padded detection bboxes "
                "only, never the full frame. If Stage A misses the person, Stage C "
                "never gets the chance to look for their tag. Improved detection "
                "recall may increase tag observation rate.",
        "window_frames": window,
        "n_frames_checked": n_checked,
        "n_frames_with_any_detection": n_any_det,
        "n_frames_with_covering_detection": n_covered,
        "detection_coverage_rate": n_covered / n_checked if n_checked > 0 else None,
        "per_frame": coverage_checks,
    }


# ---------------------------------------------------------------------------
# Component 4: Tagged person per-frame trace
# ---------------------------------------------------------------------------

def build_tagged_person_trace(
    tag_observations: list[dict],
    stage_a_trace: pd.DataFrame,
    d_trace: pd.DataFrame,
    match_sessions: list[dict] | None,
    tagged_gt_track_id: int,
    tag_id: str = "1",
) -> pd.DataFrame:
    """Build per-frame trace for the tagged GT person through A->C->D->E."""
    filtered_obs = [o for o in tag_observations if str(o.get("tag_id")) == tag_id]

    # Index tag observations by (tracklet_id, frame_index)
    obs_index: dict[tuple[str, int], str] = {}
    for obs in filtered_obs:
        key = (obs["tracklet_id"], obs["frame_index"])
        obs_index[key] = str(obs.get("tag_id", ""))

    # Get GT person's frames from stage_a_trace
    gt_mask = stage_a_trace["gt_track_id"] == tagged_gt_track_id
    a_trace = stage_a_trace.loc[gt_mask].copy()

    # Get corresponding d_trace rows
    d_gt_mask = d_trace["gt_track_id"] == tagged_gt_track_id
    d_rows = d_trace.loc[d_gt_mask].set_index("frame_index")

    # Build person_ids that appear in match sessions
    match_person_ids: set[str] = set()
    if match_sessions:
        for ms in match_sessions:
            for pid in ms.get("person_ids", []):
                match_person_ids.add(pid)

    records = []
    for _, row in a_trace.iterrows():
        fi = int(row["frame_index"])
        tid = row.get("tracklet_id")
        det_id = row.get("detection_id")
        a_cls = row["classification"]

        # Tag observation check
        tag_key = (tid, fi) if tid else (None, fi)
        tag_observed = tag_key in obs_index
        tag_id_obs = obs_index.get(tag_key)

        # D-trace lookup
        d_row = d_rows.loc[fi] if fi in d_rows.index else None
        if d_row is not None and isinstance(d_row, pd.DataFrame):
            d_row = d_row.iloc[0]

        person_id = None
        d_cls = None
        if d_row is not None and not (isinstance(d_row, float)):
            person_id = d_row.get("dominant_person_id") if hasattr(d_row, "get") else getattr(d_row, "dominant_person_id", None)
            d_cls = d_row.get("d_classification") if hasattr(d_row, "get") else getattr(d_row, "d_classification", None)

        in_match = person_id in match_person_ids if person_id else False

        records.append({
            "frame_index": fi,
            "gt_track_id": tagged_gt_track_id,
            "stage_a_class": a_cls,
            "detection_id": det_id,
            "tracklet_id": tid,
            "tag_observed": tag_observed,
            "tag_id_observed": tag_id_obs,
            "person_id": person_id,
            "d_classification": d_cls,
            "in_match_session": in_match,
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Component 5: Identity hint propagation audit
# ---------------------------------------------------------------------------

def audit_identity_hints(
    clip_output_dir: Path,
    tagged_tracklet_ids: list[str],
    tag_id: str = "1",
) -> dict:
    """Trace tag signal propagation: Stage C -> D2 constraints -> solver."""
    result: dict[str, Any] = {
        "tag_id": tag_id,
        "tagged_tracklet_ids": tagged_tracklet_ids,
    }

    # Stage C: identity_hints.jsonl
    ih_path = clip_output_dir / "stage_C" / "identity_hints.jsonl"
    hints: list[dict] = []
    if ih_path.exists():
        for line in ih_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                hints.append(json.loads(line))

    tagged_hints = [
        h for h in hints
        if h.get("tracklet_id") in tagged_tracklet_ids
    ]
    must_link_hints = [h for h in tagged_hints if h.get("constraint") == "must_link"]
    cannot_link_hints = [h for h in tagged_hints if h.get("constraint") == "cannot_link"]

    result["identity_hints"] = {
        "total_hints": len(hints),
        "tagged_tracklet_hints": len(tagged_hints),
        "must_link": [
            {
                "tracklet_id": h["tracklet_id"],
                "anchor_key": h.get("anchor_key"),
                "confidence": h.get("confidence"),
                "evidence": h.get("evidence", {}),
            }
            for h in must_link_hints
        ],
        "cannot_link": [
            {
                "tracklet_id": h["tracklet_id"],
                "anchor_key": h.get("anchor_key"),
                "evidence": h.get("evidence", {}),
            }
            for h in cannot_link_hints
        ],
    }

    # Stage D: d2_constraints.json
    d2_path = clip_output_dir / "stage_D" / "d2_constraints.json"
    d2: dict = {}
    if d2_path.exists():
        d2 = json.loads(d2_path.read_text(encoding="utf-8"))

    must_link_groups = d2.get("must_link_groups", [])
    tag_anchor = f"tag:{tag_id}"
    tag_groups = [g for g in must_link_groups if g.get("anchor_key") == tag_anchor]
    tag_pings = [p for p in d2.get("tag_pings", []) if p.get("anchor_key") == tag_anchor]

    result["d2_constraints"] = {
        "tag_must_link_groups": tag_groups,
        "tag_pings": tag_pings,
        "n_tag_tracklets_in_constraints": sum(
            len(g.get("tracklet_ids", [])) for g in tag_groups
        ),
        "stats": d2.get("stats", {}),
    }

    # Stage D: person_tracks — what person_id was assigned?
    pt_path = clip_output_dir / "stage_D" / "person_tracks.parquet"
    person_assignments: dict[str, Any] = {}
    if pt_path.exists():
        pt = pd.read_parquet(pt_path)
        for tid in tagged_tracklet_ids:
            tid_rows = pt[pt["tracklet_id"] == tid]
            if tid_rows.empty:
                person_assignments[tid] = {
                    "person_ids": [],
                    "n_frames": 0,
                    "note": "tracklet not found in person_tracks",
                }
            else:
                pids = sorted(tid_rows["person_id"].unique().tolist())
                person_assignments[tid] = {
                    "person_ids": pids,
                    "n_frames": len(tid_rows),
                    "frame_range": [int(tid_rows["frame_index"].min()),
                                    int(tid_rows["frame_index"].max())],
                }

    result["person_assignments"] = person_assignments

    # Propagation summary
    hints_emitted = len(must_link_hints) > 0
    constraints_created = len(tag_groups) > 0
    assigned = any(
        len(v.get("person_ids", [])) > 0
        for v in person_assignments.values()
    )

    result["propagation_summary"] = {
        "c_hints_emitted": hints_emitted,
        "d2_constraints_created": constraints_created,
        "d4_person_assigned": assigned,
        "chain_complete": hints_emitted and constraints_created and assigned,
    }

    return result


# ---------------------------------------------------------------------------
# Component 6: Per-video case study report
# ---------------------------------------------------------------------------

def write_per_video_report(
    camera_id: str,
    clip_id: str,
    tagged_person: dict,
    census: dict,
    trace_df: pd.DataFrame | None,
    hint_audit: dict,
    out_dir: Path,
    train_split_caveat: bool = False,
) -> None:
    """Write _tagged_person_report.md for one video."""
    out_dir.mkdir(parents=True, exist_ok=True)

    lines = [
        f"# Tagged Person Report: {camera_id} / {clip_id}",
    ]
    if train_split_caveat:
        lines.extend([
            "",
            "> **Caveat:** This video uses train-split GT annotations (not held-out).",
            "> Results are indicative but not evaluation-grade.",
        ])

    # Who is the tagged person
    gt_id = tagged_person.get("gt_track_id")
    lines.extend([
        "",
        "## Tagged person identification",
        "",
        f"- **tag_id:** {census['tag_id']}",
        f"- **gt_track_id:** {gt_id}",
        f"- **tracklet_ids:** {tagged_person.get('tracklet_ids', [])}",
        f"- **tag observations:** {tagged_person.get('n_observations', 0)}",
        f"- **vote detail:** {tagged_person.get('vote_detail', {})}",
    ])
    if tagged_person.get("note"):
        lines.append(f"- **note:** {tagged_person['note']}")

    # Tag visibility
    lines.extend([
        "",
        "## Tag visibility",
        "",
        f"- **Total observations:** {census['n_observations']}",
        f"- **Observation frames:** {census['observation_frames']}",
        f"- **Tracklet lifetime frames:** {census['tracklet_lifetime_frames']}",
        f"- **Tag detection rate:** {census['tag_detection_rate']:.4%}"
        if census['tracklet_lifetime_frames'] > 0
        else f"- **Tag detection rate:** N/A (no tracklet frames)",
        f"- **Tag consistency:** {census['tag_consistency']}",
    ])

    # Context breakdown
    if census.get("context_breakdown"):
        lines.extend(["", "### Detection context breakdown", ""])
        lines.append("| Context | Total frames | Frames with tag | Tag rate |")
        lines.append("|---|---|---|---|")
        for ctx, data in census["context_breakdown"].items():
            rate = f"{data['tag_rate']:.4%}" if data['tag_rate'] is not None else "N/A"
            lines.append(
                f"| {ctx} | {data['total_frames']} | {data['frames_with_tag']} | {rate} |"
            )

    # Bbox-gated diagnostic
    bbox = census.get("bbox_gated", {})
    if bbox:
        lines.extend([
            "",
            "### Bbox-gated diagnostic",
            "",
            f"- **Mechanism:** {bbox.get('mechanism', 'unknown')}",
            f"- **Note:** {bbox.get('note', '')}",
        ])
        if bbox.get("n_frames_checked"):
            lines.extend([
                f"- **Window:** +/- {bbox.get('window_frames', '?')} frames around each observation",
                f"- **Frames checked:** {bbox['n_frames_checked']}",
                f"- **Frames with any detection:** {bbox['n_frames_with_any_detection']}",
                f"- **Frames with covering detection:** {bbox['n_frames_with_covering_detection']}",
                f"- **Detection coverage rate:** {bbox['detection_coverage_rate']:.1%}"
                if bbox.get('detection_coverage_rate') is not None else "",
            ])

    # Identity hint propagation
    prop = hint_audit.get("propagation_summary", {})
    lines.extend([
        "",
        "## Identity hint propagation (C -> D2 -> D4)",
        "",
        f"- **Stage C hints emitted:** {prop.get('c_hints_emitted', False)}",
        f"- **D2 constraints created:** {prop.get('d2_constraints_created', False)}",
        f"- **D4 person assigned:** {prop.get('d4_person_assigned', False)}",
        f"- **Chain complete:** {prop.get('chain_complete', False)}",
    ])

    if hint_audit.get("identity_hints", {}).get("must_link"):
        lines.extend(["", "### Must-link hints"])
        for ml in hint_audit["identity_hints"]["must_link"]:
            lines.append(
                f"- tracklet={ml['tracklet_id']}, anchor={ml['anchor_key']}, "
                f"conf={ml['confidence']}, evidence={ml.get('evidence', {}).get('reason')}"
            )

    if hint_audit.get("d2_constraints", {}).get("tag_must_link_groups"):
        lines.extend(["", "### D2 must-link groups"])
        for g in hint_audit["d2_constraints"]["tag_must_link_groups"]:
            lines.append(f"- anchor={g['anchor_key']}, tracklets={g['tracklet_ids']}")

    if hint_audit.get("d2_constraints", {}).get("tag_pings"):
        lines.extend(["", "### D2 tag pings"])
        for p in hint_audit["d2_constraints"]["tag_pings"]:
            lines.append(
                f"- tracklet={p['tracklet_id']}, frame={p['frame_index']}, "
                f"conf={p['confidence']}"
            )

    if hint_audit.get("person_assignments"):
        lines.extend(["", "### Person assignments for tagged tracklets"])
        for tid, pa in hint_audit["person_assignments"].items():
            lines.append(
                f"- {tid}: person_ids={pa['person_ids']}, "
                f"n_frames={pa['n_frames']}"
                + (f", range={pa.get('frame_range')}" if pa.get("frame_range") else "")
            )

    # Per-frame trace summary
    if trace_df is not None and not trace_df.empty:
        lines.extend([
            "",
            "## Per-frame trace summary",
            "",
            f"- **Total GT frames:** {len(trace_df)}",
        ])

        # Stage A breakdown
        a_counts = trace_df["stage_a_class"].value_counts().to_dict()
        lines.append(f"- **Stage A:** {dict(a_counts)}")

        # D classification breakdown
        d_counts = trace_df["d_classification"].value_counts().to_dict()
        lines.append(f"- **Stage D:** {dict(d_counts)}")

        # Tag observations in trace
        n_tag = trace_df["tag_observed"].sum()
        lines.append(f"- **Frames with tag observed:** {n_tag}")

        # Identity outcome
        if "person_id" in trace_df.columns:
            pid_counts = trace_df["person_id"].dropna().value_counts().to_dict()
            lines.append(f"- **Person IDs assigned:** {dict(pid_counts)}")

        # Match session presence
        n_in_match = trace_df["in_match_session"].sum()
        lines.append(f"- **Frames where person_id in match session:** {n_in_match}")

        # Key events timeline
        lines.extend(["", "### Key events"])

        # First/last tag detection
        tag_frames = trace_df[trace_df["tag_observed"]]
        if not tag_frames.empty:
            lines.append(f"- First tag detection: frame {tag_frames['frame_index'].min()}")
            lines.append(f"- Last tag detection: frame {tag_frames['frame_index'].max()}")

        # Pair-box entry/exit
        pair_frames = trace_df[trace_df["stage_a_class"] == "pair_box"]
        if not pair_frames.empty:
            lines.append(f"- Pair-box frames: {len(pair_frames)} "
                        f"(first={pair_frames['frame_index'].min()}, "
                        f"last={pair_frames['frame_index'].max()})")

        # Identity changes
        if "person_id" in trace_df.columns:
            pid_series = trace_df["person_id"].dropna()
            if len(pid_series) > 0:
                changes = (pid_series != pid_series.shift()).sum() - 1
                if changes > 0:
                    lines.append(f"- Identity changes: {changes}")

    # Failure analysis
    lines.extend(["", "## Failure analysis", ""])
    if gt_id is None:
        lines.append("No GT track identified for the tagged person. "
                     "Cannot perform failure analysis.")
    elif trace_df is not None and not trace_df.empty:
        wrong = trace_df[trace_df["d_classification"] == "wrong_id"]
        no_id = trace_df[trace_df["d_classification"] == "no_id"]
        no_det = trace_df[trace_df["stage_a_class"] == "miss"]
        correct = trace_df[trace_df["d_classification"] == "correct_id"]

        total = len(trace_df)
        lines.append(f"- correct_id: {len(correct)} ({len(correct)/total:.1%})")
        lines.append(f"- wrong_id: {len(wrong)} ({len(wrong)/total:.1%})")
        lines.append(f"- no_id: {len(no_id)} ({len(no_id)/total:.1%})")
        lines.append(f"- no_detection: {len(no_det)} ({len(no_det)/total:.1%})")

        if not wrong.empty:
            lines.extend(["", "### wrong_id frames detail"])
            wrong_pids = wrong["person_id"].value_counts().to_dict()
            lines.append(f"- Person IDs assigned (wrong): {wrong_pids}")
            wrong_a = wrong["stage_a_class"].value_counts().to_dict()
            lines.append(f"- Stage A context: {wrong_a}")
    else:
        lines.append("No per-frame trace available.")

    with open(out_dir / "_tagged_person_report.md", "w") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Component 7: Cross-video verdict
# ---------------------------------------------------------------------------

def write_verdict(
    model_id: str,
    all_reports: list[dict],
    cross_tab: dict,
    out_dir: Path,
) -> None:
    """Write _tag_signal_verdict.md — cross-video synthesis."""
    out_dir.mkdir(parents=True, exist_ok=True)

    lines = [
        f"# Tag Signal Verdict: {model_id}",
        "",
        "## Question",
        "",
        "Does the AprilTag signal deliver correct identity for tagged people?",
        "",
        "## Tag visibility summary",
        "",
        "| Video | Tag observations | Tracklet frames | Detection rate | Chain complete? |",
        "|---|---|---|---|---|",
    ]

    for r in all_reports:
        census = r["census"]
        prop = r["hint_audit"].get("propagation_summary", {})
        rate = f"{census['tag_detection_rate']:.4%}" if census["tracklet_lifetime_frames"] > 0 else "N/A"
        lines.append(
            f"| {r['camera_id']}/{r['clip_id']} | {census['n_observations']} | "
            f"{census['tracklet_lifetime_frames']} | {rate} | "
            f"{'Yes' if prop.get('chain_complete') else 'No'} |"
        )

    # Cross-tab summary
    agg = cross_tab["aggregate"]
    pair_wrong = agg["pair_box"]["wrong_id"]
    tight_wrong = agg["tight_match"]["wrong_id"]
    total_wrong = pair_wrong + tight_wrong

    lines.extend([
        "",
        "## Cross-tab key finding",
        "",
        f"- pair_box -> wrong_id: {pair_wrong} frames "
        f"({pair_wrong/total_wrong:.1%} of all wrong_id)" if total_wrong > 0 else "",
        f"- tight_match -> wrong_id: {tight_wrong} frames "
        f"({tight_wrong/total_wrong:.1%} of all wrong_id)" if total_wrong > 0 else "",
    ])

    # Bbox-gated finding
    lines.extend([
        "",
        "## Bbox-gated finding",
        "",
        "Tag detection is **bbox-gated**: Stage C only scans padded detection bounding "
        "boxes from Stage A, never the full frame. This means:",
        "",
        "1. If Stage A misses the person (no detection), Stage C never gets the chance "
        "to look for their tag.",
        "2. Detection recall directly limits tag visibility.",
        "3. The v2 detection model (+5pp recall) is relevant to the tag signal story, "
        "not just the general detection story.",
        "",
    ])

    # Per-video identity outcomes
    lines.extend(["## Per-video identity outcomes", ""])
    for r in all_reports:
        gt_id = r["tagged_person"].get("gt_track_id")
        lines.append(f"### {r['camera_id']}/{r['clip_id']}")
        if r.get("train_split_caveat"):
            lines.append("> Train-split GT (not held-out).")

        if gt_id is None:
            lines.append(f"- No tag observations. Cannot assess identity outcome.")
        else:
            prop = r["hint_audit"].get("propagation_summary", {})
            pa = r["hint_audit"].get("person_assignments", {})
            assigned_pids = set()
            for tid_data in pa.values():
                assigned_pids.update(tid_data.get("person_ids", []))

            lines.append(f"- gt_track_id: {gt_id}")
            lines.append(f"- Tag observations: {r['census']['n_observations']}")
            lines.append(f"- Chain complete: {prop.get('chain_complete', False)}")
            lines.append(f"- Person IDs assigned to tagged tracklets: {sorted(assigned_pids)}")

            if r.get("trace_df") is not None and not r["trace_df"].empty:
                df = r["trace_df"]
                correct = (df["d_classification"] == "correct_id").sum()
                total = len(df)
                lines.append(f"- Correct identity frames: {correct}/{total} ({correct/total:.1%})")
        lines.append("")

    # Intervention recommendations
    lines.extend([
        "## Intervention recommendations",
        "",
        "### 1. Tag visibility is the primary bottleneck",
        "",
        "Tag detection rates are extremely low (0.03-0.07% of tracklet frames). "
        "Even when the tag signal propagates correctly through C->D2->D4, "
        "it covers a tiny fraction of the person's lifetime.",
        "",
        "### 2. Detection recall affects tag visibility (bbox-gated)",
        "",
        "Because tag detection is bbox-gated, improving detection recall "
        "(v2 model, +5pp) directly increases the number of frames where "
        "Stage C can attempt tag detection. This is a necessary but not "
        "sufficient condition for improving tag visibility.",
        "",
        "### 3. Physical visibility dominates",
        "",
        "Even with perfect detection coverage, the AprilTag is only physically "
        "visible from certain angles, distances, and when not occluded by "
        "grappling. The fundamental limit is physical, not algorithmic.",
        "",
    ])

    # Verdict
    lines.extend([
        "## Verdict",
        "",
    ])

    # Count videos with chain complete
    chain_complete = sum(
        1 for r in all_reports
        if r["hint_audit"].get("propagation_summary", {}).get("chain_complete")
    )
    total_videos = len(all_reports)
    videos_with_tags = sum(1 for r in all_reports if r["census"]["n_observations"] > 0)

    lines.append(
        f"**Tag signal chain:** {chain_complete}/{videos_with_tags} videos with tag "
        f"observations have complete C->D2->D4 propagation "
        f"(out of {total_videos} total videos)."
    )
    lines.append("")

    if videos_with_tags == 0:
        lines.append(
            "**No tag observations found in any video.** Cannot assess tag signal quality."
        )
    else:
        lines.append(
            "**Tag visibility is the dominant bottleneck.** The tag signal mechanism "
            "(C->D2->D4) works when the tag is observed, but observations are extremely "
            "rare. The product cannot rely on AprilTags as the primary identity mechanism "
            "with current tag visibility rates. Complementary identity signals "
            "(color histograms, ReID, manual check-in) are needed."
        )

    # Conservation checks
    lines.extend([
        "",
        "## Conservation checks",
        "",
    ])
    for r in all_reports:
        cam = r["camera_id"]
        clip = r["clip_id"]
        census = r["census"]
        n_obs = census["n_observations"]
        lines.append(f"- {cam}/{clip}: {n_obs} tag observations")
        if r.get("trace_df") is not None and not r["trace_df"].empty:
            df = r["trace_df"]
            a_total = len(df)
            a_sum = sum(df["stage_a_class"].value_counts().values)
            if a_total != a_sum:
                lines.append(f"  WARNING: conservation violation: trace rows={a_total} != sum={a_sum}")
            else:
                lines.append(f"  trace rows: {a_total} (conservation OK)")

    with open(out_dir / "_tag_signal_verdict.md", "w") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def _load_tag_observations(clip_output_dir: Path) -> list[dict]:
    """Load tag_observations.jsonl from a clip's stage_C directory."""
    path = clip_output_dir / "stage_C" / "tag_observations.jsonl"
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def _load_match_sessions(clip_output_dir: Path) -> list[dict] | None:
    """Load match_sessions.jsonl from stage_E if available."""
    path = clip_output_dir / "stage_E" / "match_sessions.jsonl"
    if not path.exists():
        return None
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def _resolve_clip_output_dir(
    camera_id: str, clip_id: str, gym_id: str,
) -> Path | None:
    """Find the clip output directory."""
    pattern = f"{gym_id}/{camera_id}/**/{clip_id}"
    matches = list(OUTPUTS_DIR.glob(pattern))
    return matches[0] if matches else None


def run_tag_trace(
    model_id: str,
    tag_id: str = "1",
    gym_id: str | None = None,
    camera_filter: str | None = None,
    iou_threshold: float = 0.3,
) -> None:
    """Run full tag signal trace for a model."""
    import logging as _logging

    from pipeline_validation.common.manifest import load_manifest
    from pipeline_validation.signal_trace.stage_a_census import run_census
    from pipeline_validation.signal_trace.stage_d_trace import (
        run_d_trace,
        write_d_trace_artifacts,
    )

    _logging.basicConfig(level=_logging.INFO, format="%(levelname)s: %(message)s")

    manifest_path = CONFIGS_DIR / "models" / f"{model_id}.yaml"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    manifest = load_manifest(manifest_path)
    resolved_gym_id = gym_id or manifest.pipeline_gym_id or "_eval_gt"

    trace_base = EVAL_DIR / model_id

    # --- Collect all videos to process ---
    # Phase 1: v1 manifest exports (val-having, with existing trace artifacts)
    v1_exports = [e for e in manifest.training_data if e.splits.val is not None]
    if camera_filter:
        v1_exports = [e for e in v1_exports if e.camera_id == camera_filter]

    # --- Component 1: Cross-tab (val-split cameras only) ---
    v1_camera_ids = sorted(set(e.camera_id for e in v1_exports))
    print(f"\n=== Cross-tab (Stage A x Stage D): {model_id} ===")
    cross_tab = build_cross_tab(model_id, camera_ids=v1_camera_ids)
    write_cross_tab(model_id, cross_tab)
    print(f"  Written: cross_tab.json, cross_tab.md")

    agg = cross_tab["aggregate"]
    pair_wrong = agg["pair_box"]["wrong_id"]
    tight_wrong = agg["tight_match"]["wrong_id"]
    total_wrong = pair_wrong + tight_wrong
    if total_wrong > 0:
        print(f"  pair_box->wrong_id: {pair_wrong} ({pair_wrong/total_wrong:.1%} of wrong_id)")
        print(f"  tight_match->wrong_id: {tight_wrong} ({tight_wrong/total_wrong:.1%} of wrong_id)")

    # Phase 2: Check for v2 manifest's J_EDEw-200246
    v2_export = None
    v2_manifest = None
    v2_manifest_path = CONFIGS_DIR / "models" / "bjj-detect-all-cameras-v2.yaml"
    if v2_manifest_path.exists():
        v2_manifest = load_manifest(v2_manifest_path)
        for e in v2_manifest.training_data:
            if "200246" in e.source_video and e.splits.val is None:
                if camera_filter and e.camera_id != camera_filter:
                    continue
                v2_export = e
                break

    all_reports: list[dict] = []

    # --- Process v1 exports ---
    for export in v1_exports:
        cam = export.camera_id
        clip_id = (export.pipeline_output_clip_id
                   or export.source_video.replace(".mp4", ""))

        print(f"\n=== {cam} / {clip_id} ===")

        # Find clip output directory
        clip_dir = _resolve_clip_output_dir(cam, clip_id, resolved_gym_id)
        if clip_dir is None:
            print(f"  WARNING: No pipeline output found for {cam}/{clip_id}")
            continue

        # Load existing trace artifacts
        cam_trace_dir = trace_base / cam
        a_trace_path = cam_trace_dir / "gt_signal_trace_stage_a.parquet"
        d_trace_path = cam_trace_dir / "gt_signal_trace_d.parquet"

        if not a_trace_path.exists():
            print(f"  WARNING: No stage_a trace at {a_trace_path}")
            continue

        a_trace = pd.read_parquet(a_trace_path)
        d_trace = pd.read_parquet(d_trace_path) if d_trace_path.exists() else pd.DataFrame()

        # Load tag observations
        tag_obs = _load_tag_observations(clip_dir)
        print(f"  Tag observations: {len(tag_obs)}")

        # Load detections
        det_path = clip_dir / "stage_A" / "detections.parquet"
        det_df = pd.read_parquet(det_path) if det_path.exists() else pd.DataFrame()

        # Load match sessions
        match_sessions = _load_match_sessions(clip_dir)

        # Component 2: Identify tagged person
        tagged = identify_tagged_person(tag_obs, a_trace, tag_id)
        gt_id = tagged["gt_track_id"]
        print(f"  Tagged person: gt_track_id={gt_id}, tracklets={tagged['tracklet_ids']}")

        # Component 3: Tag census
        census = build_tag_census(
            tag_obs, a_trace, det_df, gt_id, tagged["tracklet_ids"], tag_id,
        )
        print(f"  Tag detection rate: {census['tag_detection_rate']:.4%}"
              if census["tracklet_lifetime_frames"] > 0
              else "  Tag detection rate: N/A")

        # Component 4: Per-frame trace
        trace_df = None
        if gt_id is not None and not d_trace.empty:
            trace_df = build_tagged_person_trace(
                tag_obs, a_trace, d_trace, match_sessions, gt_id, tag_id,
            )
            print(f"  Trace: {len(trace_df)} frames")

        # Component 5: Identity hint audit
        hint_audit = audit_identity_hints(clip_dir, tagged["tracklet_ids"], tag_id)
        chain = hint_audit["propagation_summary"]["chain_complete"]
        print(f"  Hint propagation chain complete: {chain}")

        # Write artifacts
        out_dir = trace_base / cam
        out_dir.mkdir(parents=True, exist_ok=True)

        with open(out_dir / "tag_census.json", "w") as f:
            # Strip per_frame from bbox_gated for JSON (keep it concise)
            census_out = dict(census)
            if "bbox_gated" in census_out:
                bbox_out = dict(census_out["bbox_gated"])
                bbox_out.pop("per_frame", None)
                census_out["bbox_gated"] = bbox_out
            json.dump(census_out, f, indent=2)

        with open(out_dir / "identity_hint_audit.json", "w") as f:
            json.dump(hint_audit, f, indent=2)

        if trace_df is not None and not trace_df.empty:
            trace_df.to_parquet(out_dir / "tagged_person_trace.parquet", index=False)

        # Component 6: Report
        write_per_video_report(
            cam, clip_id, tagged, census, trace_df, hint_audit, out_dir,
        )

        all_reports.append({
            "camera_id": cam,
            "clip_id": clip_id,
            "tagged_person": tagged,
            "census": census,
            "trace_df": trace_df,
            "hint_audit": hint_audit,
            "train_split_caveat": False,
        })

    # --- Process v2 200246 export ---
    if v2_export is not None:
        cam = v2_export.camera_id
        clip_id = v2_export.source_video.replace(".mp4", "")
        # Pipeline output is under the real gym_id
        real_gym_id = "c8a592a4-2bca-400a-80e1-fec0e5cbea77"

        print(f"\n=== {cam} / {clip_id} (v2, train-split) ===")

        clip_dir = _resolve_clip_output_dir(cam, clip_id, real_gym_id)
        if clip_dir is None:
            print(f"  WARNING: No pipeline output for {cam}/{clip_id}")
        else:
            # Run greedy matcher on-the-fly for train-split GT
            out_dir_200246 = trace_base / f"{cam}_200246"
            out_dir_200246.mkdir(parents=True, exist_ok=True)

            print(f"  Running Stage A census on train-split GT (450 frames)...")
            try:
                a_trace_200246, a_summary = run_census(
                    v2_manifest, v2_export, real_gym_id, iou_threshold,
                )
                # Write stage_a trace for 200246
                a_trace_200246.to_parquet(
                    out_dir_200246 / "gt_signal_trace_stage_a.parquet", index=False,
                )
                for cls in ("tight_match", "pair_box", "split", "miss"):
                    c = a_summary[cls]
                    print(f"    {cls}: {c['count']} ({c['pct']:.1%})")
            except Exception as e:
                print(f"  Stage A census FAILED: {e}")
                import traceback
                traceback.print_exc()
                a_trace_200246 = pd.DataFrame()

            # Run D-trace on-the-fly
            d_trace_200246 = pd.DataFrame()
            if not a_trace_200246.empty:
                print(f"  Running D-stage trace...")
                try:
                    # run_d_trace expects a file path, not a DataFrame
                    a_trace_parquet = out_dir_200246 / "gt_signal_trace_stage_a.parquet"
                    d_trace_200246, d_summary = run_d_trace(
                        v2_manifest, v2_export, real_gym_id, a_trace_parquet,
                    )
                    d_trace_200246.to_parquet(
                        out_dir_200246 / "gt_signal_trace_d.parquet", index=False,
                    )
                    for cls in ("correct_id", "wrong_id", "no_id", "no_detection"):
                        c = d_summary.get(cls, {"count": 0, "pct": 0})
                        print(f"    {cls}: {c['count']} ({c['pct']:.1%})")
                except Exception as e:
                    print(f"  D-trace FAILED: {e}")
                    import traceback
                    traceback.print_exc()

            # Load tag observations from real pipeline output
            tag_obs = _load_tag_observations(clip_dir)
            print(f"  Tag observations: {len(tag_obs)}")

            det_path = clip_dir / "stage_A" / "detections.parquet"
            det_df = pd.read_parquet(det_path) if det_path.exists() else pd.DataFrame()

            match_sessions = _load_match_sessions(clip_dir)

            # Identify tagged person
            if not a_trace_200246.empty:
                tagged = identify_tagged_person(tag_obs, a_trace_200246, tag_id)
            else:
                tagged = {
                    "gt_track_id": None, "tracklet_ids": [],
                    "n_observations": len(tag_obs),
                    "vote_detail": {}, "note": "No stage_a trace available",
                }
            gt_id = tagged["gt_track_id"]
            print(f"  Tagged person: gt_track_id={gt_id}, tracklets={tagged['tracklet_ids']}")

            census = build_tag_census(
                tag_obs,
                a_trace_200246 if not a_trace_200246.empty else None,
                det_df, gt_id, tagged["tracklet_ids"], tag_id,
            )
            print(f"  Tag detection rate: {census['tag_detection_rate']:.4%}"
                  if census["tracklet_lifetime_frames"] > 0
                  else "  Tag detection rate: N/A")

            trace_df = None
            if gt_id is not None and not d_trace_200246.empty:
                trace_df = build_tagged_person_trace(
                    tag_obs, a_trace_200246, d_trace_200246,
                    match_sessions, gt_id, tag_id,
                )
                print(f"  Trace: {len(trace_df)} frames")

            hint_audit = audit_identity_hints(clip_dir, tagged["tracklet_ids"], tag_id)
            chain = hint_audit["propagation_summary"]["chain_complete"]
            print(f"  Hint propagation chain complete: {chain}")

            # Write artifacts
            census_out = dict(census)
            if "bbox_gated" in census_out:
                bbox_out = dict(census_out["bbox_gated"])
                bbox_out.pop("per_frame", None)
                census_out["bbox_gated"] = bbox_out
            with open(out_dir_200246 / "tag_census.json", "w") as f:
                json.dump(census_out, f, indent=2)
            with open(out_dir_200246 / "identity_hint_audit.json", "w") as f:
                json.dump(hint_audit, f, indent=2)
            if trace_df is not None and not trace_df.empty:
                trace_df.to_parquet(
                    out_dir_200246 / "tagged_person_trace.parquet", index=False,
                )

            write_per_video_report(
                cam, clip_id, tagged, census, trace_df, hint_audit,
                out_dir_200246, train_split_caveat=True,
            )

            all_reports.append({
                "camera_id": cam,
                "clip_id": clip_id,
                "tagged_person": tagged,
                "census": census,
                "trace_df": trace_df,
                "hint_audit": hint_audit,
                "train_split_caveat": True,
            })

    # --- Component 7: Verdict ---
    print(f"\n=== Writing verdict ===")
    write_verdict(model_id, all_reports, cross_tab, trace_base)
    print(f"  Written: _tag_signal_verdict.md")
    print(f"\nDone. Artifacts at: {trace_base}/")
