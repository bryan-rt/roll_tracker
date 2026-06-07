#!/usr/bin/env python3
"""CP-TAG-3: Two-clip tag identity baseline evidence extraction.

Three subcommands:
  tag-trace    Per-clip tag trace for vid2 (J_EDEw-200246) under _eval_gt.
               Imports signal trace modules directly with gym_id=_eval_gt.
               Includes faithfulness self-check against vid1 official numbers.
  session      Fresh session-level Stage D on both J_EDEw clips + targeted queries.
  carrier      t99/t143 carrier geometry evidence at frames 1781-1782.

Usage:
  PYTHONPATH=src python tools/cp_tag_3_evidence.py tag-trace
  PYTHONPATH=src python tools/cp_tag_3_evidence.py session
  PYTHONPATH=src python tools/cp_tag_3_evidence.py carrier
  PYTHONPATH=src python tools/cp_tag_3_evidence.py all

Session scoping: two-clip, single-camera (J_EDEw only). No Tier 3 cross-camera
evidence. Post-CP-TAG-4 re-measure gate MUST reuse this exact scope for
apples-to-apples comparison.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = REPO_ROOT / "outputs"
EVIDENCE_DIR = REPO_ROOT / "docs" / "evidence" / "cp_tag_3_baseline"
CONFIGS_DIR = REPO_ROOT / "configs"

# Both clips
VID1_CLIP_ID = "J_EDEw-20260318-200015"
VID2_CLIP_ID = "J_EDEw-20260318-200246"
VID1_MP4 = REPO_ROOT / "data/raw/nest/_eval_gt/J_EDEw/2026-03-18/20" / f"{VID1_CLIP_ID}.mp4"
VID2_MP4 = REPO_ROOT / "data/raw/nest/_eval_gt/J_EDEw/2026-03-18/20" / f"{VID2_CLIP_ID}.mp4"
GYM_ID = "_eval_gt"
CAM_ID = "J_EDEw"
TAG_ID = "1"

# GT track IDs for the tagged person (same physical person in both clips)
VID1_GT_TRACK_ID = 24
VID2_GT_TRACK_ID = 8


def _clip_dir(clip_id: str) -> Path | None:
    pattern = f"{GYM_ID}/{CAM_ID}/**/{clip_id}"
    matches = list(OUTPUTS_DIR.glob(pattern))
    return matches[0] if matches else None


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


# -----------------------------------------------------------------------
# Subcommand: tag-trace
# -----------------------------------------------------------------------

def cmd_tag_trace():
    """Run tag trace for vid2 under _eval_gt, with vid1 faithfulness self-check."""
    from pipeline_validation.common.manifest import (
        enumerate_annotated_frames,
        load_manifest,
    )
    from pipeline_validation.signal_trace.stage_a_census import run_census
    from pipeline_validation.signal_trace.stage_d_trace import run_d_trace
    from pipeline_validation.signal_trace.tag_trace import (
        audit_identity_hints,
        build_tag_census,
        build_tagged_person_trace,
        identify_tagged_person,
        write_per_video_report,
    )

    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

    # Load manifests
    v1_manifest = load_manifest(CONFIGS_DIR / "models" / "bjj-detect-all-cameras.yaml")
    v2_manifest = load_manifest(CONFIGS_DIR / "models" / "bjj-detect-all-cameras-v2.yaml")

    # Find vid1 and vid2 exports
    vid1_export = None
    for e in v1_manifest.training_data:
        if VID1_CLIP_ID in e.source_video and e.camera_id == CAM_ID:
            vid1_export = e
            break
    if vid1_export is None:
        for e in v2_manifest.training_data:
            if VID1_CLIP_ID in e.source_video and e.camera_id == CAM_ID:
                vid1_export = e
                break

    vid2_export = None
    for e in v2_manifest.training_data:
        if VID2_CLIP_ID in e.source_video and e.camera_id == CAM_ID:
            vid2_export = e
            break

    if vid1_export is None:
        print("ERROR: vid1 export not found in manifests")
        return
    if vid2_export is None:
        print("ERROR: vid2 export not found in manifests")
        return

    # -------------------------------------------------------------------
    # Faithfulness self-check: run our code path against vid1 and compare
    # to the official signal-trace numbers.
    # -------------------------------------------------------------------
    print("=" * 60)
    print("FAITHFULNESS SELF-CHECK: vid1 (J_EDEw-200015)")
    print("=" * 60)

    # Use vid1's manifest for census (it has the right GT)
    vid1_manifest_for_census = v1_manifest

    print("  Running Stage A census...")
    a_trace_v1, a_summary_v1 = run_census(
        vid1_manifest_for_census, vid1_export, GYM_ID, iou_threshold=0.3,
    )
    print(f"  Stage A: {len(a_trace_v1)} rows")

    # Write temporary parquet for d_trace
    tmp_a_path = EVIDENCE_DIR / "_tmp_vid1_a_trace.parquet"
    a_trace_v1.to_parquet(tmp_a_path, index=False)

    print("  Running D-stage trace...")
    d_trace_v1, d_summary_v1 = run_d_trace(
        vid1_manifest_for_census, vid1_export, GYM_ID, tmp_a_path,
    )

    # Extract tagged person trace
    vid1_clip_dir = _clip_dir(VID1_CLIP_ID)
    tag_obs_v1 = _load_jsonl(vid1_clip_dir / "stage_C" / "tag_observations.jsonl")
    tagged_v1 = identify_tagged_person(tag_obs_v1, a_trace_v1, TAG_ID)
    gt_id_v1 = tagged_v1["gt_track_id"]

    match_sessions_v1 = _load_jsonl(vid1_clip_dir / "stage_E" / "match_sessions.jsonl") or None
    trace_v1 = build_tagged_person_trace(
        tag_obs_v1, a_trace_v1, d_trace_v1, match_sessions_v1, gt_id_v1, TAG_ID,
    )

    # Compute failure breakdown for tagged person
    v1_d_counts = trace_v1["d_classification"].value_counts().to_dict()
    v1_total = len(trace_v1)

    print(f"\n  Tagged person gt_track_id={gt_id_v1}, {v1_total} frames:")
    for cls in ("correct_id", "wrong_id", "no_id", "no_detection"):
        n = v1_d_counts.get(cls, 0)
        pct = n / v1_total * 100 if v1_total > 0 else 0
        print(f"    {cls}: {n} ({pct:.1f}%)")

    # Official numbers from the signal trace report
    official_v1 = {"correct_id": 77, "wrong_id": 152, "no_id": 3, "no_detection": 69}
    print(f"\n  Official signal-trace numbers:")
    for cls, n in official_v1.items():
        pct = n / 301 * 100
        print(f"    {cls}: {n} ({pct:.1f}%)")

    # Compare
    faithful = True
    for cls in ("correct_id", "wrong_id", "no_id", "no_detection"):
        ours = v1_d_counts.get(cls, 0)
        theirs = official_v1[cls]
        if ours != theirs:
            print(f"\n  MISMATCH: {cls}: ours={ours}, official={theirs}")
            faithful = False

    if faithful:
        print("\n  FAITHFULNESS CHECK PASSED: all numbers match official signal-trace.")
    else:
        print("\n  FAITHFULNESS CHECK FAILED: numbers differ from official signal-trace.")
        print("  Investigate before trusting vid2 numbers.")

    # Clean up tmp
    tmp_a_path.unlink(missing_ok=True)

    # -------------------------------------------------------------------
    # Vid2 tag trace (the actual new baseline)
    # -------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("VID2 TAG TRACE: J_EDEw-200246 (under _eval_gt)")
    print("=" * 60)

    vid2_clip_dir = _clip_dir(VID2_CLIP_ID)
    if vid2_clip_dir is None:
        print("ERROR: No pipeline output for vid2 under _eval_gt")
        return

    print(f"  Pipeline output: {vid2_clip_dir}")

    print("  Running Stage A census...")
    a_trace_v2, a_summary_v2 = run_census(
        v2_manifest, vid2_export, GYM_ID, iou_threshold=0.3,
    )
    for cls in ("tight_match", "pair_box", "split", "miss"):
        c = a_summary_v2[cls]
        print(f"    {cls}: {c['count']} ({c['pct']:.1%})")

    # Write for d_trace
    a_trace_v2_path = EVIDENCE_DIR / "vid2_a_trace.parquet"
    a_trace_v2.to_parquet(a_trace_v2_path, index=False)

    print("  Running D-stage trace...")
    d_trace_v2, d_summary_v2 = run_d_trace(
        v2_manifest, vid2_export, GYM_ID, a_trace_v2_path,
    )
    for cls in ("correct_id", "wrong_id", "no_id", "no_detection"):
        c = d_summary_v2[cls]
        print(f"    {cls}: {c['count']} ({c['pct']:.1%})")

    d_trace_v2.to_parquet(EVIDENCE_DIR / "vid2_d_trace.parquet", index=False)

    # Tag census
    tag_obs_v2 = _load_jsonl(vid2_clip_dir / "stage_C" / "tag_observations.jsonl")
    det_v2 = pd.read_parquet(vid2_clip_dir / "stage_A" / "detections.parquet")
    tagged_v2 = identify_tagged_person(tag_obs_v2, a_trace_v2, TAG_ID)
    gt_id_v2 = tagged_v2["gt_track_id"]
    print(f"\n  Tagged person: gt_track_id={gt_id_v2}, tracklets={tagged_v2['tracklet_ids']}")
    print(f"  Tag observations: {tagged_v2['n_observations']}")

    census_v2 = build_tag_census(
        tag_obs_v2, a_trace_v2, det_v2, gt_id_v2, tagged_v2["tracklet_ids"], TAG_ID,
    )
    if census_v2["tracklet_lifetime_frames"] > 0:
        print(f"  Tag detection rate: {census_v2['tag_detection_rate']:.4%}")

    # Tagged person trace
    match_sessions_v2 = _load_jsonl(vid2_clip_dir / "stage_E" / "match_sessions.jsonl") or None
    trace_v2 = None
    if gt_id_v2 is not None and not d_trace_v2.empty:
        trace_v2 = build_tagged_person_trace(
            tag_obs_v2, a_trace_v2, d_trace_v2, match_sessions_v2, gt_id_v2, TAG_ID,
        )
        print(f"  Trace: {len(trace_v2)} frames")

    # Identity hint audit
    hint_audit_v2 = audit_identity_hints(vid2_clip_dir, tagged_v2["tracklet_ids"], TAG_ID)
    chain = hint_audit_v2["propagation_summary"]["chain_complete"]
    print(f"  Hint propagation chain complete: {chain}")

    # Failure analysis
    if trace_v2 is not None and not trace_v2.empty:
        v2_d_counts = trace_v2["d_classification"].value_counts().to_dict()
        v2_total = len(trace_v2)
        print(f"\n  Failure analysis ({v2_total} frames):")
        for cls in ("correct_id", "wrong_id", "no_id", "no_detection"):
            n = v2_d_counts.get(cls, 0)
            pct = n / v2_total * 100 if v2_total > 0 else 0
            print(f"    {cls}: {n} ({pct:.1f}%)")

        # Person ID details
        if "person_id" in trace_v2.columns:
            pid_counts = trace_v2["person_id"].dropna().value_counts()
            print(f"\n  Person IDs assigned: {pid_counts.to_dict()}")
            print(f"  Frames in match session: {trace_v2['in_match_session'].sum()}")

    # Write report
    out_dir = EVIDENCE_DIR / "vid2_tag_trace"
    out_dir.mkdir(parents=True, exist_ok=True)

    if trace_v2 is not None and not trace_v2.empty:
        trace_v2.to_parquet(out_dir / "tagged_person_trace.parquet", index=False)

    write_per_video_report(
        CAM_ID, VID2_CLIP_ID, tagged_v2, census_v2, trace_v2, hint_audit_v2,
        out_dir, train_split_caveat=True,
    )

    # Write summary JSON
    summary = {
        "vid1_faithfulness_check": {
            "passed": faithful,
            "ours": {cls: v1_d_counts.get(cls, 0) for cls in official_v1},
            "official": official_v1,
        },
        "vid2_baseline": {
            "gt_track_id": gt_id_v2,
            "tracklet_ids": tagged_v2["tracklet_ids"],
            "n_tag_observations": tagged_v2["n_observations"],
            "tag_detection_rate": census_v2.get("tag_detection_rate"),
            "chain_complete": chain,
            "failure_analysis": {
                cls: v2_d_counts.get(cls, 0) for cls in ("correct_id", "wrong_id", "no_id", "no_detection")
            } if trace_v2 is not None else None,
            "total_frames": v2_total if trace_v2 is not None else 0,
        },
        "stage_a_summary": {
            cls: a_summary_v2[cls] for cls in ("tight_match", "pair_box", "split", "miss")
        },
        "stage_d_summary": {
            cls: d_summary_v2[cls] for cls in ("correct_id", "wrong_id", "no_id", "no_detection")
        },
    }
    with open(EVIDENCE_DIR / "tag_trace_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Artifacts written to: {out_dir}")
    print(f"  Summary: {EVIDENCE_DIR / 'tag_trace_summary.json'}")


# -----------------------------------------------------------------------
# Subcommand: session
# -----------------------------------------------------------------------

def cmd_session():
    """Fresh session-level Stage D on both J_EDEw clips + targeted queries.

    Session scope: two-clip, single-camera (J_EDEw only). No Tier 3
    cross-camera evidence. Post-CP-TAG-4 re-measure gate MUST reuse this
    exact scope for apples-to-apples comparison.
    """
    from bjj_pipeline.config.loader import load_yaml
    from bjj_pipeline.contracts.f0_paths import SessionOutputLayout
    from bjj_pipeline.stages.stitch.session_d_run import run_session_d

    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("SESSION-LEVEL EVIDENCE: two-clip J_EDEw session")
    print("=" * 60)
    print("  Scope: J_EDEw only, two clips, no cross-camera evidence")
    print("  Clips:")
    print(f"    {VID1_MP4.name}")
    print(f"    {VID2_MP4.name}")

    # Verify both clips exist
    for mp4 in (VID1_MP4, VID2_MP4):
        if not mp4.exists():
            print(f"ERROR: {mp4} not found")
            return

    # Load config
    config_dict = load_yaml(CONFIGS_DIR / "default.yaml")

    # Build SessionOutputLayout
    session_layout = SessionOutputLayout(
        gym_id=GYM_ID,
        date="2026-03-18",
        session_id="cp_tag_3_baseline",
        root=OUTPUTS_DIR,
    )

    session_clips = [
        (VID1_MP4, CAM_ID),
        (VID2_MP4, CAM_ID),
    ]

    print(f"\n  Session output: {session_layout.session_root}")
    print("  Running session_d (D1->D4)...")

    session_manifest = run_session_d(
        config=config_dict,
        session_layout=session_layout,
        session_clips=session_clips,
        cam_id=CAM_ID,
        output_root=OUTPUTS_DIR,
    )

    if session_manifest is None:
        print("ERROR: run_session_d returned None")
        return

    print(f"  Session D completed. fps={session_manifest.fps}")

    # --- Targeted queries ---
    stage_d_dir = session_layout.stage_dir("D")
    pt_path = stage_d_dir / f"person_tracks_{CAM_ID}.parquet"
    ia_path = stage_d_dir / f"identity_assignments_{CAM_ID}.jsonl"

    if not pt_path.exists():
        print(f"ERROR: {pt_path} not found")
        return

    pt_df = pd.read_parquet(pt_path)
    ia_records = _load_jsonl(ia_path)

    print(f"\n  person_tracks: {len(pt_df)} rows")
    print(f"  identity_assignments: {len(ia_records)} records")

    results = {}

    # Discover actual tagged tracklets from identity_hints
    # (session-namespaced: {clip_id}:{tid})
    hints_path = stage_d_dir / f"identity_hints_{CAM_ID}.jsonl"
    session_hints = _load_jsonl(hints_path)
    tag1_hints = [h for h in session_hints
                  if h.get("constraint") == "must_link"
                  and str(h.get("evidence", {}).get("tag_id", "")) == TAG_ID]
    session_tagged_tids = sorted(set(h["tracklet_id"] for h in tag1_hints))
    print(f"\n  Session-level tagged tracklets (from hints): {session_tagged_tids}")
    results["session_tagged_tracklets"] = session_tagged_tids

    # (a) Are tagged tracklets kept or dropped?
    print(f"\n  (a) Tagged tracklet drop status:")
    for stid in session_tagged_tids:
        tid_rows = pt_df[pt_df["tracklet_id"] == stid]
        # Also check split products
        tid_products = pt_df[pt_df["tracklet_id"].str.startswith(f"{stid}_s")]
        all_rows = pd.concat([tid_rows, tid_products]) if not tid_products.empty else tid_rows
        dropped = len(all_rows) == 0
        print(f"      {stid}: {'DROPPED' if dropped else 'KEPT'} "
              f"({len(all_rows)} rows)")
        if not all_rows.empty:
            pids = all_rows["person_id"].unique().tolist()
            print(f"        person_ids: {pids}")
        results[f"tagged_{stid}_dropped"] = dropped
        results[f"tagged_{stid}_rows"] = len(all_rows)

    # Legacy check: old t99/t366 IDs (for comparison with prior diagnostics)
    for legacy_label, legacy_id in [("t99", f"{VID2_CLIP_ID}:t99"),
                                     ("t366", f"{VID1_CLIP_ID}:t366")]:
        legacy_rows = pt_df[pt_df["tracklet_id"] == legacy_id]
        if not legacy_rows.empty:
            print(f"      [legacy] {legacy_id}: {len(legacy_rows)} rows")
        results[f"legacy_{legacy_label}_exists"] = len(legacy_rows) > 0

    # (b) Which person_ids does tag:1 receive? Cross-clip boundary?
    tag1_assignments = [r for r in ia_records if r.get("tag_id") == 1 or str(r.get("tag_id")) == "1"]
    print(f"\n  (b) tag:1 identity_assignments: {len(tag1_assignments)}")
    for a in tag1_assignments:
        print(f"      person_id={a.get('person_id')}, "
              f"tag_id={a.get('tag_id')}, "
              f"confidence={a.get('confidence')}")
    results["tag1_identity_assignments"] = tag1_assignments

    # Check if any tag:1 person_id spans the clip boundary
    # Clip boundary: vid1 ends at its last frame offset, vid2 starts at its first frame offset
    # Parse clip timestamps to find the boundary frame
    from bjj_pipeline.stages.stitch.session_d_run import (
        derive_clip_frame_offset,
        parse_clip_timestamp,
    )
    vid1_dt = parse_clip_timestamp(VID1_MP4)
    vid2_dt = parse_clip_timestamp(VID2_MP4)
    session_start = min(vid1_dt, vid2_dt) if vid1_dt and vid2_dt else vid1_dt
    fps = session_manifest.fps

    vid1_offset = derive_clip_frame_offset(VID1_MP4, session_start, fps) if session_start else 0
    vid2_offset = derive_clip_frame_offset(VID2_MP4, session_start, fps) if session_start else 0
    print(f"\n      Frame offsets: vid1={vid1_offset}, vid2={vid2_offset}")
    results["frame_offsets"] = {"vid1": vid1_offset, "vid2": vid2_offset}

    # For each tag:1 person_id, check if frames span the boundary
    for a in tag1_assignments:
        pid = a.get("person_id")
        if pid:
            pid_frames = pt_df[pt_df["person_id"] == pid]["frame_index"]
            if not pid_frames.empty:
                min_f, max_f = int(pid_frames.min()), int(pid_frames.max())
                spans_boundary = min_f < vid2_offset and max_f >= vid2_offset
                print(f"      {pid}: frames [{min_f}, {max_f}], spans_boundary={spans_boundary}")
                a["frame_range"] = [min_f, max_f]
                a["spans_clip_boundary"] = spans_boundary

    # (c) identity_assignment count for tag:1
    results["tag1_assignment_count"] = len(tag1_assignments)
    print(f"\n  (c) tag:1 assignment count: {len(tag1_assignments)}")

    # (d) Person_id transition count for tagged tracklets
    print(f"\n  (d) Tagged tracklet person_id transitions:")
    results["tagged_tracklet_transitions"] = {}
    for stid in session_tagged_tids:
        tid_rows = pt_df[pt_df["tracklet_id"] == stid].sort_values("frame_index")
        # Also include split products
        tid_products = pt_df[pt_df["tracklet_id"].str.startswith(f"{stid}_s")]
        if not tid_products.empty:
            tid_rows = pd.concat([tid_rows, tid_products]).sort_values("frame_index")

        if tid_rows.empty:
            print(f"      {stid}: NOT FOUND (dropped)")
            results["tagged_tracklet_transitions"][stid] = {
                "dropped": True, "transitions": None, "person_ids": [],
            }
            continue

        pids_list = tid_rows["person_id"].tolist()
        transitions = sum(1 for i in range(1, len(pids_list))
                          if pids_list[i] != pids_list[i - 1])
        unique_pids = sorted(set(pids_list))
        pid_counts = Counter(pids_list)
        print(f"      {stid}: {len(tid_rows)} rows, {transitions} transitions")
        print(f"        person_ids: {unique_pids}")
        for pid, cnt in pid_counts.most_common():
            print(f"          {pid}: {cnt} frames")
        results["tagged_tracklet_transitions"][stid] = {
            "dropped": False,
            "transitions": transitions,
            "person_ids": unique_pids,
            "pid_counts": dict(pid_counts),
            "n_rows": len(tid_rows),
        }

    # --- Write session evidence ---
    with open(EVIDENCE_DIR / "session_evidence.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Session evidence: {EVIDENCE_DIR / 'session_evidence.json'}")

    # Write session_evidence.md
    lines = [
        "# CP-TAG-3 Session-Level Evidence",
        "",
        "## Session Scope",
        "",
        "Two-clip, single-camera (J_EDEw only). No Tier 3 cross-camera evidence.",
        "This is a controlled experiment for the clip-boundary question.",
        "**Post-CP-TAG-4 re-measure gate MUST reuse this exact session scope**",
        "for apples-to-apples comparison.",
        "",
        f"- Clip 1: {VID1_CLIP_ID} (offset {vid1_offset} frames)",
        f"- Clip 2: {VID2_CLIP_ID} (offset {vid2_offset} frames)",
        f"- FPS: {fps}",
        f"- Session output: `{session_layout.session_root.relative_to(REPO_ROOT)}`",
        "",
        "## Results",
        "",
        f"### (a) Tagged tracklet drop status",
        f"- Session-level tagged tracklets: {session_tagged_tids}",
    ]
    for stid in session_tagged_tids:
        dropped = results.get(f"tagged_{stid}_dropped", True)
        n_rows = results.get(f"tagged_{stid}_rows", 0)
        lines.append(f"- `{stid}`: {'DROPPED' if dropped else 'KEPT'} ({n_rows} rows)")

    lines.extend([
        "",
        f"### (b) tag:1 person_ids: {len(tag1_assignments)} assignments",
    ])
    for a in tag1_assignments:
        pid = a.get("person_id", "?")
        spans = a.get("spans_clip_boundary", "N/A")
        fr = a.get("frame_range", "?")
        lines.append(f"- {pid}: frames {fr}, spans_boundary={spans}")

    lines.extend([
        "",
        f"### (c) tag:1 assignment count: {len(tag1_assignments)}",
        "",
        "### (d) Tagged tracklet person_id transitions",
    ])
    for stid, tinfo in results.get("tagged_tracklet_transitions", {}).items():
        if tinfo.get("dropped"):
            lines.append(f"- `{stid}`: DROPPED")
        else:
            lines.append(f"- `{stid}`: {tinfo['transitions']} transitions, "
                        f"person_ids={tinfo['person_ids']}")
            if tinfo.get("pid_counts"):
                for pid, cnt in sorted(tinfo["pid_counts"].items(), key=lambda x: -x[1]):
                    lines.append(f"  - {pid}: {cnt} frames")

    with open(EVIDENCE_DIR / "session_evidence.md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Session evidence report: {EVIDENCE_DIR / 'session_evidence.md'}")


# -----------------------------------------------------------------------
# Subcommand: carrier
# -----------------------------------------------------------------------

def cmd_carrier():
    """Extract tagged-tracklet carrier geometry evidence at tag observation frames.

    Adaptive: discovers actual tagged tracklets from current pipeline output
    rather than hardcoding stale IDs. The old t99/t143 nesting from the pre-CP5
    real-gym-id run no longer exists in the v2-model _eval_gt run.
    """
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("CARRIER EVIDENCE: tagged tracklet geometry at tag observation frames")
    print("=" * 60)

    vid2_clip_dir = _clip_dir(VID2_CLIP_ID)
    if vid2_clip_dir is None:
        print("ERROR: No pipeline output for vid2 under _eval_gt")
        return

    # Load artifacts
    det_df = pd.read_parquet(vid2_clip_dir / "stage_A" / "detections.parquet")
    seg_path = vid2_clip_dir / "stage_D" / "d1_segments.parquet"
    seg_df = pd.read_parquet(seg_path) if seg_path.exists() else pd.DataFrame()
    pt_path = vid2_clip_dir / "stage_D" / "person_tracks.parquet"
    pt_df = pd.read_parquet(pt_path) if pt_path.exists() else pd.DataFrame()
    tag_obs = _load_jsonl(vid2_clip_dir / "stage_C" / "tag_observations.jsonl")

    # Discover tagged tracklets from observations
    tag1_obs = [o for o in tag_obs if str(o.get("tag_id")) == TAG_ID]
    tagged_tids = sorted(set(o["tracklet_id"] for o in tag1_obs))
    obs_frames = sorted(set(o["frame_index"] for o in tag1_obs))

    print(f"  Tag observations: {len(tag1_obs)}")
    print(f"  Tagged tracklets: {tagged_tids}")
    print(f"  Observation frames: {obs_frames}")

    evidence = {
        "tagged_tracklets": tagged_tids,
        "observation_frames": obs_frames,
        "frames": {},
    }

    # Per-frame geometry at each observation frame
    for frame_idx in obs_frames:
        frame_dets = det_df[det_df["frame_index"] == frame_idx]
        frame_data = {"frame_index": frame_idx, "detections": {}, "tag_observations": []}

        # All detections at this frame
        for _, row in frame_dets.iterrows():
            tid = row["tracklet_id"]
            bbox = {
                "x1": float(row["x1"]), "y1": float(row["y1"]),
                "x2": float(row["x2"]), "y2": float(row["y2"]),
            }
            bbox["width"] = bbox["x2"] - bbox["x1"]
            bbox["height"] = bbox["y2"] - bbox["y1"]
            bbox["area"] = bbox["width"] * bbox["height"]
            bbox["center_x"] = (bbox["x1"] + bbox["x2"]) / 2
            bbox["center_y"] = (bbox["y1"] + bbox["y2"]) / 2
            is_tagged = tid in tagged_tids
            frame_data["detections"][tid] = bbox
            if is_tagged:
                print(f"  Frame {frame_idx}: TAGGED {tid} bbox=[{bbox['x1']:.1f}, "
                      f"{bbox['y1']:.1f}, {bbox['x2']:.1f}, {bbox['y2']:.1f}] "
                      f"area={bbox['area']:.0f}")

        # Tag observation details
        frame_tag_obs = [o for o in tag1_obs if o["frame_index"] == frame_idx]
        for obs in frame_tag_obs:
            roi = obs.get("roi_xyxy", [])
            tag_tid = obs.get("tracklet_id")
            entry = {"tracklet_id": tag_tid, "tag_id": obs.get("tag_id")}
            if roi and len(roi) == 4:
                tag_cx = (roi[0] + roi[2]) / 2
                tag_cy = (roi[1] + roi[3]) / 2
                entry["tag_center"] = {"x": tag_cx, "y": tag_cy}
                entry["roi_xyxy"] = roi
                print(f"  Frame {frame_idx}: tag on {tag_tid}, "
                      f"center=({tag_cx:.1f}, {tag_cy:.1f})")

                # Check containment in each detection
                for det_tid, bbox in frame_data["detections"].items():
                    inside = (bbox["x1"] <= tag_cx <= bbox["x2"] and
                              bbox["y1"] <= tag_cy <= bbox["y2"])
                    if inside:
                        print(f"    Tag center inside {det_tid}")
            frame_data["tag_observations"].append(entry)

        # Check for nested/overlapping detections with tagged tracklets
        overlaps = []
        for tagged_tid in tagged_tids:
            if tagged_tid not in frame_data["detections"]:
                continue
            tagged_bbox = frame_data["detections"][tagged_tid]
            for other_tid, other_bbox in frame_data["detections"].items():
                if other_tid == tagged_tid:
                    continue
                ix1 = max(tagged_bbox["x1"], other_bbox["x1"])
                iy1 = max(tagged_bbox["y1"], other_bbox["y1"])
                ix2 = min(tagged_bbox["x2"], other_bbox["x2"])
                iy2 = min(tagged_bbox["y2"], other_bbox["y2"])
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                if inter <= 0:
                    continue
                union = tagged_bbox["area"] + other_bbox["area"] - inter
                iou = inter / union if union > 0 else 0
                smaller_area = min(tagged_bbox["area"], other_bbox["area"])
                containment = inter / smaller_area if smaller_area > 0 else 0
                if iou > 0.05:
                    overlaps.append({
                        "tagged_tid": tagged_tid,
                        "other_tid": other_tid,
                        "iou": round(iou, 4),
                        "containment": round(containment, 4),
                    })
                    print(f"  Frame {frame_idx}: overlap {tagged_tid}/{other_tid} "
                          f"IoU={iou:.4f} containment={containment:.4f}")
        frame_data["overlaps"] = overlaps

        evidence["frames"][str(frame_idx)] = frame_data

    # Tagged tracklet details
    print(f"\n  Tagged tracklet details:")
    for tid in tagged_tids:
        tid_frames = det_df[det_df["tracklet_id"] == tid]
        n_frames = len(tid_frames["frame_index"].unique())
        if not tid_frames.empty:
            min_f = int(tid_frames["frame_index"].min())
            max_f = int(tid_frames["frame_index"].max())
            print(f"    {tid}: {n_frames} frames [{min_f}, {max_f}]")
            evidence[f"{tid}_length"] = {
                "n_frames": n_frames, "min_frame": min_f, "max_frame": max_f,
            }
        else:
            print(f"    {tid}: NOT FOUND in detections")
            evidence[f"{tid}_length"] = None

        # GROUP participation
        if not seg_df.empty:
            tid_segs = seg_df[seg_df["base_tracklet_id"] == tid]
            group_segs = tid_segs[tid_segs["segment_type"] == "GROUP"]
            solo_segs = tid_segs[tid_segs["segment_type"] == "SOLO"]
            print(f"    {tid}: {len(group_segs)} GROUP, {len(solo_segs)} SOLO segments")
            evidence[f"{tid}_segments"] = {
                "group_count": int(len(group_segs)),
                "solo_count": int(len(solo_segs)),
                "group_nodes": group_segs["node_id"].tolist() if not group_segs.empty else [],
            }

        # Also check split products
        tid_split_ids = [tid]
        if not seg_df.empty:
            products = seg_df[seg_df["base_tracklet_id"].str.startswith(f"{tid}_s")]
            if not products.empty:
                tid_split_ids.extend(products["base_tracklet_id"].unique().tolist())

        # Person_id assignment
        if not pt_df.empty:
            tid_pt = pt_df[pt_df["tracklet_id"].isin(tid_split_ids)]
            if tid_pt.empty:
                print(f"    {tid}: DROPPED (not in person_tracks)")
                evidence[f"{tid}_person_ids"] = []
            else:
                pids = tid_pt["person_id"].unique().tolist()
                n_transitions = 0
                if len(tid_pt) > 1:
                    sorted_pt = tid_pt.sort_values("frame_index")
                    pid_list = sorted_pt["person_id"].tolist()
                    n_transitions = sum(1 for i in range(1, len(pid_list))
                                        if pid_list[i] != pid_list[i - 1])
                print(f"    {tid}: person_ids={pids}, {len(tid_pt)} rows, "
                      f"{n_transitions} transitions")
                evidence[f"{tid}_person_ids"] = pids
                evidence[f"{tid}_person_id_transitions"] = n_transitions

    # Comparison with prior diagnostic (old t99/t143)
    has_nesting = any(
        len(fd.get("overlaps", [])) > 0
        for fd in evidence["frames"].values()
    )

    evidence["comparison_with_prior"] = {
        "prior_tagged_tracklets": ["t99", "t143"],
        "prior_nesting": True,
        "prior_t99_length": 862,
        "prior_t143_length": 17,
        "prior_t99_dropped": True,
        "current_tagged_tracklets": tagged_tids,
        "current_nesting": has_nesting,
        "note": ("The v2-model _eval_gt run produces different tracklet IDs and "
                 "topology from the stale real-gym-id run. The old t99/t143 nested "
                 "detection (where the tag observation was captured by the short "
                 "nested tracklet and routed to the wrong person when t99 was "
                 "dropped) no longer occurs. The carrier-selection rule question "
                 "from the task brief may need reframing for the current pipeline "
                 "state."),
    }

    # Write evidence
    with open(EVIDENCE_DIR / "carrier_evidence.json", "w") as f:
        json.dump(evidence, f, indent=2, default=str)

    # Write carrier_evidence.md
    lines = [
        "# CP-TAG-3 Carrier Evidence: Tagged Tracklet Geometry",
        "",
        "## Context",
        "",
        f"In vid2 ({VID2_CLIP_ID}), tag_id=1 is observed at frame(s) {obs_frames}.",
        f"Tagged tracklet(s): {tagged_tids}.",
        "",
        "## Comparison with Prior Diagnostic",
        "",
        "The prior cross-tracklet diagnostic (pre-CP5, real gym_id) found:",
        "- t99 (862 frames) and t143 (17 frames, nested bbox inside t99)",
        "- Tag obs captured by BOTH t99 and t143 at frames 1781-1782",
        "- t99 DROPPED by solver; tag identity routed to wrong person via t143",
        "",
        "**Current state (v2 model, _eval_gt, post-CP5/CP-SPLIT-1):**",
        f"- Tagged tracklet(s): {tagged_tids}",
        f"- Nesting detected at observation frames: {has_nesting}",
    ]
    if not has_nesting:
        lines.append("- The old t99/t143 nested detection no longer occurs")
        lines.append("- Carrier-selection rule question may need reframing")
    lines.append("")

    # Tracklet details
    lines.append("## Tagged Tracklet Details")
    lines.append("")
    for tid in tagged_tids:
        info = evidence.get(f"{tid}_length")
        if info:
            lines.append(f"### {tid}")
            lines.append(f"- Length: {info['n_frames']} frames [{info['min_frame']}, {info['max_frame']}]")
            seg_info = evidence.get(f"{tid}_segments")
            if seg_info:
                lines.append(f"- GROUP segments: {seg_info['group_count']}, SOLO: {seg_info['solo_count']}")
                if seg_info.get("group_nodes"):
                    lines.append(f"- GROUP nodes: {seg_info['group_nodes']}")
            pids = evidence.get(f"{tid}_person_ids", [])
            if pids:
                lines.append(f"- Person IDs: {pids}")
                trans = evidence.get(f"{tid}_person_id_transitions", 0)
                lines.append(f"- Person ID transitions: {trans}")
            else:
                lines.append("- **DROPPED** (not in person_tracks)")
            lines.append("")

    # Per-frame geometry
    lines.append("## Observation Frame Geometry")
    lines.append("")
    for fi_str, fd in evidence["frames"].items():
        lines.append(f"### Frame {fi_str}")
        lines.append(f"- Detections: {len(fd['detections'])}")
        for det_tid, bbox in fd["detections"].items():
            marker = " **[TAGGED]**" if det_tid in tagged_tids else ""
            lines.append(f"  - {det_tid}: [{bbox['x1']:.1f}, {bbox['y1']:.1f}, "
                        f"{bbox['x2']:.1f}, {bbox['y2']:.1f}] "
                        f"area={bbox['area']:.0f}{marker}")
        for obs in fd.get("tag_observations", []):
            tc = obs.get("tag_center")
            if tc:
                lines.append(f"- Tag center: ({tc['x']:.1f}, {tc['y']:.1f}) on {obs['tracklet_id']}")
        if fd.get("overlaps"):
            for ov in fd["overlaps"]:
                lines.append(f"- Overlap: {ov['tagged_tid']}/{ov['other_tid']} "
                            f"IoU={ov['iou']:.4f} containment={ov['containment']:.4f}")
        else:
            lines.append("- No overlapping detections with tagged tracklet")
        lines.append("")

    with open(EVIDENCE_DIR / "carrier_evidence.md", "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\n  Carrier evidence: {EVIDENCE_DIR / 'carrier_evidence.json'}")
    print(f"  Carrier report: {EVIDENCE_DIR / 'carrier_evidence.md'}")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "tag-trace":
        cmd_tag_trace()
    elif cmd == "session":
        cmd_session()
    elif cmd == "carrier":
        cmd_carrier()
    elif cmd == "all":
        cmd_tag_trace()
        cmd_session()
        cmd_carrier()
    else:
        print(f"Unknown subcommand: {cmd}")
        print("Usage: python tools/cp_tag_3_evidence.py {tag-trace|session|carrier|all}")
        sys.exit(1)
