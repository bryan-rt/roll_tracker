"""CP-GT2ACTUALS-2: Dense join artifact builder.

Joins GT annotations against actual pipeline outputs into a dense
per-(frame_index, gt_track_id) error map. One row per GT person per annotated
frame, with all pipeline signals joined: detection match, tracklet, person_id,
D1 node info, world coords, velocity, appearance (HSV histogram + is_isolated),
and tag observations.

State column classifies each row's identity outcome (no jumps in CP-2 scope).

METRIC-BASIS DISCIPLINE: The artifact records which manifest and cadence
produced it (manifest_path + manifest_stride per row + metadata.json), so no
one compares a dense-built map against a stride-10-built one.

HSV COLUMNS: Explicitly nullable. NULL histogram = not isolated = entangled,
never interpolated across. is_isolated is the companion flag. Color is sparsest
during scrambles (where swaps happen) — a real ceiling on the color channel's
swap-detection contribution.

CANONICAL_PERSON_ID: Majority-vote convenience column. For FRAGMENTED GT tracks
(diluted entities), majority vote may pick a canonical that's mostly another
person's frames. correct/wrong_id are only as trustworthy as canonical.
"""
from __future__ import annotations

import json
import logging
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline_validation.common.gt_loader import load_gt_for_split
from pipeline_validation.common.manifest import (
    enumerate_annotated_frames,
    load_manifest,
)
from pipeline_validation.common.schemas import ExportEntry, ModelManifest
from pipeline_validation.signal_trace.greedy_matcher import greedy_match

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUTS_DIR = REPO_ROOT / "outputs"
TRAINING_DATA_DIR = REPO_ROOT / "data" / "training_data"
EVAL_DIR = OUTPUTS_DIR / "_eval" / "gt2actuals"

# State classification priority (first match wins)
STATES = (
    "no_canonical",
    "miss",
    "untracked",
    "no_id",
    "wrong_id",
    "correct",
)


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def _resolve_clip_dir(
    export: ExportEntry, gym_id: str,
) -> Path | None:
    clip_id = export.pipeline_output_clip_id or export.source_video.replace(".mp4", "")
    cam = export.camera_id
    pattern = f"{gym_id}/{cam}/**/{clip_id}"
    matches = sorted(OUTPUTS_DIR.glob(pattern))
    return matches[0] if matches else None


# ---------------------------------------------------------------------------
# GT loading (train + val merged)
# ---------------------------------------------------------------------------

def _load_gt_all(zip_path: Path, export: ExportEntry) -> dict[int, list]:
    gt = load_gt_for_split(zip_path, export, "train")
    if export.splits.val is not None:
        gt_val = load_gt_for_split(zip_path, export, "val")
        gt.update(gt_val)
    return gt


# ---------------------------------------------------------------------------
# Canonical person_id derivation (majority vote)
# ---------------------------------------------------------------------------

def _derive_canonical_mapping(
    frame_gt_pids: list[tuple[int, list[str]]],
) -> dict[int, str]:
    """Majority-vote: gt_track_id -> canonical_person_id.

    CAVEAT: For FRAGMENTED GT tracks (diluted entities), majority vote may
    pick a canonical that's mostly another person's frames. correct/wrong_id
    are only as trustworthy as this mapping. CP-3 must spot-check canonical
    assignments on known-fragmented tracks specifically.

    Args:
        frame_gt_pids: list of (gt_track_id, [person_ids]) from matched frames.
    """
    counters: dict[int, Counter] = defaultdict(Counter)
    for gt_tid, pids in frame_gt_pids:
        for pid in pids:
            counters[gt_tid][pid] += 1

    result: dict[int, str] = {}
    for gt_tid, counter in counters.items():
        if not counter:
            continue
        max_count = max(counter.values())
        candidates = sorted(pid for pid, c in counter.items() if c == max_count)
        result[gt_tid] = candidates[0]
    return result


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------

def run_dense_join(
    manifest_path: Path,
    gym_id: str | None = None,
    camera_filter: str | None = None,
    iou_threshold: float = 0.3,
) -> list[Path]:
    """Build dense GT-to-actuals join artifact for all cameras in manifest.

    Args:
        manifest_path: Explicit path to manifest YAML (not model_id).
        gym_id: Override gym_id (default: from manifest.pipeline_gym_id).
        camera_filter: Restrict to one camera_id (default: all).
        iou_threshold: Greedy match threshold (default 0.3).

    Returns:
        List of output parquet paths written.
    """
    manifest = load_manifest(manifest_path)
    if gym_id is None:
        gym_id = manifest.pipeline_gym_id or "_eval_gt"

    manifest_path_str = str(manifest_path)
    output_paths: list[Path] = []

    for export in manifest.training_data:
        cam = export.camera_id
        if camera_filter and cam != camera_filter:
            continue

        logger.info("=== %s: starting dense join ===", cam)
        try:
            result_path = _build_one_camera(
                manifest, export, gym_id, manifest_path_str,
                iou_threshold,
            )
            output_paths.append(result_path)
        except Exception:
            logger.exception("Failed to build dense join for %s", cam)

    return output_paths


def _build_one_camera(
    manifest: ModelManifest,
    export: ExportEntry,
    gym_id: str,
    manifest_path_str: str,
    iou_threshold: float,
) -> Path:
    cam = export.camera_id
    stride = export.annotated_range.stride

    # --- Resolve paths ---
    clip_dir = _resolve_clip_dir(export, gym_id)
    if clip_dir is None:
        raise FileNotFoundError(f"Clip directory not found for {cam} under {gym_id}")

    stage_a_dir = clip_dir / "stage_A"
    stage_d_dir = clip_dir / "stage_D"
    for req_file in ["detections.parquet", "color_histograms.parquet"]:
        if not (stage_a_dir / req_file).exists():
            raise FileNotFoundError(f"{req_file} missing in {stage_a_dir}")
    for req_file in [
        "person_tracks.parquet", "d1_graph_nodes.parquet",
        "tracklet_bank_frames.parquet", "tracklet_bank_summaries.parquet",
    ]:
        if not (stage_d_dir / req_file).exists():
            raise FileNotFoundError(f"{req_file} missing in {stage_d_dir}")

    # --- Load GT ---
    zip_path = TRAINING_DATA_DIR / export.export
    gt_by_frame = _load_gt_all(zip_path, export)
    annotated_frames = sorted(enumerate_annotated_frames(export))
    logger.info("%s: %d annotated frames (stride=%d)", cam, len(annotated_frames), stride)

    # --- Load pipeline artifacts ---
    det_df = pd.read_parquet(stage_a_dir / "detections.parquet")
    hist_df = pd.read_parquet(stage_a_dir / "color_histograms.parquet")
    pt_df = pd.read_parquet(stage_d_dir / "person_tracks.parquet")
    d1_nodes = pd.read_parquet(stage_d_dir / "d1_graph_nodes.parquet")
    bank_frames = pd.read_parquet(stage_d_dir / "tracklet_bank_frames.parquet")

    # Tag observations (sparse, may be empty)
    tag_obs_path = clip_dir / "stage_C" / "tag_observations.jsonl"
    tag_obs_by_frame_det: dict[tuple[int, str], str] = {}
    if tag_obs_path.exists():
        for line in tag_obs_path.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            key = (int(rec["frame_index"]), str(rec["detection_id"]))
            tag_obs_by_frame_det[key] = str(rec["tag_id"])

    clip_id = str(det_df["clip_id"].iloc[0]) if len(det_df) > 0 else "unknown"
    logger.info("%s: %d detections, %d person_tracks, %d d1_nodes",
                cam, len(det_df), len(pt_df), len(d1_nodes))

    # --- Build indexes ---

    # Detection index: frame_index -> [(x1,y1,x2,y2, detection_id, tracklet_id)]
    det_by_frame: dict[int, list] = defaultdict(list)
    for _, r in det_df.iterrows():
        det_by_frame[int(r.frame_index)].append((
            float(r.x1), float(r.y1), float(r.x2), float(r.y2),
            str(r.detection_id), str(r.tracklet_id),
        ))

    # Histogram index: (frame_index, track_id) -> row
    hist_idx: dict[tuple[int, str], int] = {}
    for i, r in hist_df.iterrows():
        hist_idx[(int(r.frame_index), str(r.track_id))] = i

    hist_cols = [c for c in hist_df.columns if c.startswith("hist_")]

    # Bank frames index: (tracklet_id, frame_index) -> row index
    bank_idx: dict[tuple[str, int], int] = {}
    for i, r in bank_frames.iterrows():
        bank_idx[(str(r.tracklet_id), int(r.frame_index))] = i

    # D0.5 split resolution
    from pipeline_validation.signal_trace.stage_d_trace import (
        _build_split_resolution,
        _resolve_tracklet_id,
    )
    split_map, tid_frame_range = _build_split_resolution(stage_d_dir)

    # Person tracks index: (resolved_tracklet_id, frame_index) -> [person_ids]
    pt_grouped = pt_df.groupby(["tracklet_id", "frame_index"])["person_id"].apply(list)
    pt_lookup = pt_grouped.to_dict()

    # D1 lookup: (tracklet_id, frame_index) -> [node_info]
    from pipeline_validation.gt_person_trace import _build_d1_lookup
    d1_lookup = _build_d1_lookup(d1_nodes)

    # D3 status
    from pipeline_validation.gt_person_trace import _build_d3_status
    d3_status = _build_d3_status(clip_dir / "_debug" / "d3_solution_ledger.json")

    # --- Phase 1: Greedy match + build frame_gt_tracklets for node inversion ---

    # frame_gt_tracklets: (frame_index, resolved_tracklet_id) -> {gt_track_ids}
    # Used for node_gt_set inversion. Both sides resolved through split lineage.
    frame_gt_tracklets: dict[tuple[int, str], set[int]] = defaultdict(set)

    # Accumulate (gt_track_id, [person_ids]) for canonical mapping
    canonical_votes: list[tuple[int, list[str]]] = []

    # Per-frame match results: list of dicts for final DataFrame
    match_rows: list[dict] = []

    for fi in annotated_frames:
        gt_boxes_raw = gt_by_frame.get(fi, [])
        if not gt_boxes_raw:
            continue

        gt_tuples = [(b.x1, b.y1, b.x2, b.y2) for b in gt_boxes_raw]
        gt_track_ids = [b.track_id for b in gt_boxes_raw]

        frame_dets = det_by_frame.get(fi, [])
        det_tuples = [(d[0], d[1], d[2], d[3]) for d in frame_dets]
        det_ids = [d[4] for d in frame_dets]
        det_tids = [d[5] for d in frame_dets]

        matches = greedy_match(gt_tuples, det_tuples, iou_threshold)

        # Count GT sharing per detection
        det_to_gt_count: dict[int, int] = defaultdict(int)
        gt_best_det: dict[int, tuple[int, float]] = {}
        for gi, di, iou in matches:
            det_to_gt_count[di] += 1
            if gi not in gt_best_det or iou > gt_best_det[gi][1]:
                gt_best_det[gi] = (di, iou)

        for gi, gt_box in enumerate(gt_boxes_raw):
            gt_tid_val = gt_track_ids[gi]

            if gi not in gt_best_det:
                match_rows.append(_make_row_miss(
                    cam, clip_id, fi, gt_tid_val, gt_box,
                    manifest_path_str, stride, hist_cols,
                ))
                continue

            best_di, best_iou = gt_best_det[gi]
            n_sharing = det_to_gt_count[best_di]
            det_id = det_ids[best_di]
            raw_tid = det_tids[best_di]

            # Classify topology
            gt_to_dets_count = sum(1 for g, d, _ in matches if g == gi)
            if gt_to_dets_count >= 2:
                topology = "split"
            elif n_sharing >= 2:
                topology = "pair_box"
            else:
                topology = "tight_match"

            # Resolve tracklet through split lineage
            resolved_tid = _resolve_tracklet_id(
                raw_tid, fi, split_map, tid_frame_range,
            )

            # Record for node_gt_set inversion (resolved tracklet)
            frame_gt_tracklets[(fi, resolved_tid)].add(gt_tid_val)

            # Person IDs for this detection (via resolved tracklet)
            pids = pt_lookup.get((resolved_tid, fi), [])
            if pids:
                canonical_votes.append((gt_tid_val, pids))

            # D1 node info (via resolved tracklet)
            d1_infos = d1_lookup.get((resolved_tid, fi), [])
            d1_node_id = d1_infos[0]["node_id"] if d1_infos else None
            d1_node_type = d1_infos[0]["node_type"] if d1_infos else None
            d1_is_group = any("GROUP" in info["node_type"] for info in d1_infos) if d1_infos else False
            d1_carrier = None
            if d1_infos:
                for info in d1_infos:
                    ct = info.get("carrier_tracklet_id")
                    if ct and ct == resolved_tid:
                        d1_carrier = "carrier"
                        break
                if d1_carrier is None:
                    d1_carrier = "non_carrier"

            # D3 status
            d3_st = d3_status.get(resolved_tid)

            # Bank frames: world coords + velocity
            bk_i = bank_idx.get((resolved_tid, fi))
            x_m = y_m = x_m_eff = y_m_eff = None
            is_repaired = None
            speed = vx = vy = None
            if bk_i is not None:
                bk_row = bank_frames.iloc[bk_i]
                x_m = _safe_float(bk_row.get("x_m"))
                y_m = _safe_float(bk_row.get("y_m"))
                is_repaired = bool(bk_row.get("is_repaired", False))
                x_rep = _safe_float(bk_row.get("x_m_repaired"))
                y_rep = _safe_float(bk_row.get("y_m_repaired"))
                x_m_eff = x_rep if is_repaired and x_rep is not None else x_m
                y_m_eff = y_rep if is_repaired and y_rep is not None else y_m
                speed = _safe_float(bk_row.get("speed_mps_k"))
                vx = _safe_float(bk_row.get("vx_mps_k"))
                vy = _safe_float(bk_row.get("vy_mps_k"))

            # Histogram: is_isolated + hist values
            h_i = hist_idx.get((fi, resolved_tid))
            # Fall back to raw tracklet ID for histogram (track_id in hist is
            # the original Stage A tracklet, not the D0.5 product).
            if h_i is None:
                h_i = hist_idx.get((fi, raw_tid))
            is_isolated = None
            crop_method = None
            hist_vals: dict[str, float | None] = {}
            if h_i is not None:
                h_row = hist_df.iloc[h_i]
                is_isolated = bool(h_row.get("is_isolated", False))
                crop_method = str(h_row.get("crop_method", "")) or None
                for hc in hist_cols:
                    v = h_row.get(hc)
                    hist_vals[hc] = float(v) if pd.notna(v) else None
            else:
                for hc in hist_cols:
                    hist_vals[hc] = None

            # Tag observation
            tag_key = (fi, det_id)
            has_tag = tag_key in tag_obs_by_frame_det
            tag_id = tag_obs_by_frame_det.get(tag_key)

            row = {
                "camera_id": cam,
                "clip_id": clip_id,
                "frame_index": fi,
                "gt_track_id": gt_tid_val,
                "gt_x1": gt_box.x1, "gt_y1": gt_box.y1,
                "gt_x2": gt_box.x2, "gt_y2": gt_box.y2,
                "manifest_path": manifest_path_str,
                "manifest_stride": stride,
                "detection_id": det_id,
                "tracklet_id": raw_tid,
                "resolved_tracklet_id": resolved_tid,
                "match_iou": round(best_iou, 4),
                "match_topology": topology,
                "n_gt_sharing_det": n_sharing,
                "person_ids": json.dumps(sorted(pids)),
                "n_person_ids": len(pids),
                "canonical_person_id": None,  # filled after canonical derivation
                "d1_node_id": d1_node_id,
                "d1_node_type": d1_node_type,
                "d1_carrier_status": d1_carrier,
                "d1_is_group": d1_is_group,
                "node_gt_set": None,  # filled after inversion
                "node_gt_set_size": 0,
                "x_m": x_m,
                "y_m": y_m,
                "x_m_eff": x_m_eff,
                "y_m_eff": y_m_eff,
                "is_repaired": is_repaired,
                "speed_mps_k": speed,
                "vx_mps_k": vx,
                "vy_mps_k": vy,
                "is_isolated": is_isolated,
                "crop_method": crop_method,
                **hist_vals,
                "has_tag_obs": has_tag,
                "tag_id": tag_id,
                "state": None,  # filled after canonical derivation
                "is_group_ambiguous": False,
                "d3_status": d3_st,
            }
            match_rows.append(row)

    if not match_rows:
        raise ValueError(f"No GT-person-frame rows generated for {cam}")

    # --- Phase 2: Canonical mapping + node_gt_set + state ---

    canonical_map = _derive_canonical_mapping(canonical_votes)
    logger.info("%s: %d GT tracks with canonical person_id", cam, len(canonical_map))

    # Node GT-identity SET inversion
    from pipeline_validation.gt2actuals.node_gt_set import build_node_gt_sets
    node_gt_sets = build_node_gt_sets(d1_lookup, dict(frame_gt_tracklets))

    # Fill canonical, node_gt_set, state
    for row in match_rows:
        gt_tid = row["gt_track_id"]
        row["canonical_person_id"] = canonical_map.get(gt_tid)

        # Node GT set
        nid = row.get("d1_node_id")
        fi = row["frame_index"]
        if nid is not None:
            gt_set = node_gt_sets.get((nid, fi), set())
            row["node_gt_set"] = json.dumps(sorted(gt_set))
            row["node_gt_set_size"] = len(gt_set)
            row["is_group_ambiguous"] = row["d1_is_group"] and len(gt_set) >= 2

        # State classification (priority order)
        row["state"] = _classify_state(row)

    # --- Phase 3: Write output ---
    df = pd.DataFrame(match_rows)

    out_dir = _output_dir(cam, export)
    out_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = out_dir / "gt2actuals_dense.parquet"
    df.to_parquet(parquet_path, index=False)

    # Metadata
    state_counts = df["state"].value_counts().to_dict()
    metadata = {
        "manifest_path": manifest_path_str,
        "manifest_stride": int(export.annotated_range.stride),
        "camera_id": cam,
        "clip_id": clip_id,
        "gym_id": gym_id,
        "n_annotated_frames": len(annotated_frames),
        "n_gt_person_frames": len(df),
        "n_gt_tracks": int(df["gt_track_id"].nunique()),
        "n_canonical_mappings": len(canonical_map),
        "n_states": {s: state_counts.get(s, 0) for s in STATES},
        "canonical_note": (
            "Majority-vote convenience column. For fragmented GT tracks, "
            "canonical may be dominated by another person's frames. "
            "correct/wrong_id trustworthiness depends on canonical quality."
        ),
        "hsv_note": (
            "Histogram columns are nullable. NULL = not isolated (entangled), "
            "never interpolated. is_isolated is the companion flag."
        ),
        "ppd_note": (
            "PPDmUg has no dense manifest variant. Cross-camera validation "
            "at stride-10 only."
        ),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(
        "%s: wrote %d rows to %s | states: %s",
        cam, len(df), parquet_path,
        " ".join(f"{s}={state_counts.get(s, 0)}" for s in STATES),
    )
    return parquet_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _output_dir(camera_id: str, export: ExportEntry) -> Path:
    clip_id = export.pipeline_output_clip_id or export.source_video.replace(".mp4", "")
    return EVAL_DIR / camera_id / clip_id


def _safe_float(v) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
        return f if np.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def _make_row_miss(
    cam: str, clip_id: str, fi: int, gt_tid: int, gt_box,
    manifest_path_str: str, stride: int, hist_cols: list[str],
) -> dict:
    row: dict = {
        "camera_id": cam,
        "clip_id": clip_id,
        "frame_index": fi,
        "gt_track_id": gt_tid,
        "gt_x1": gt_box.x1, "gt_y1": gt_box.y1,
        "gt_x2": gt_box.x2, "gt_y2": gt_box.y2,
        "manifest_path": manifest_path_str,
        "manifest_stride": stride,
        "detection_id": None,
        "tracklet_id": None,
        "resolved_tracklet_id": None,
        "match_iou": None,
        "match_topology": "miss",
        "n_gt_sharing_det": 0,
        "person_ids": "[]",
        "n_person_ids": 0,
        "canonical_person_id": None,
        "d1_node_id": None,
        "d1_node_type": None,
        "d1_carrier_status": None,
        "d1_is_group": False,
        "node_gt_set": None,
        "node_gt_set_size": 0,
        "x_m": None,
        "y_m": None,
        "x_m_eff": None,
        "y_m_eff": None,
        "is_repaired": None,
        "speed_mps_k": None,
        "vx_mps_k": None,
        "vy_mps_k": None,
        "is_isolated": None,
        "crop_method": None,
        "has_tag_obs": False,
        "tag_id": None,
        "state": "miss",
        "is_group_ambiguous": False,
        "d3_status": None,
    }
    for hc in hist_cols:
        row[hc] = None
    return row


def _classify_state(row: dict) -> str:
    """State classification — priority order, first match wins."""
    canonical = row.get("canonical_person_id")
    if canonical is None:
        return "no_canonical"

    det_id = row.get("detection_id")
    if det_id is None:
        return "miss"

    tid = row.get("tracklet_id")
    if tid is None:
        return "untracked"

    n_pids = row.get("n_person_ids", 0)
    if n_pids == 0:
        return "no_id"

    pids = json.loads(row.get("person_ids", "[]"))
    if canonical in pids:
        return "correct"

    return "wrong_id"
