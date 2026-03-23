"""CP14 Validation: adapter paths, session bank aggregation, and summary report.

Runs read-only validation against the 36 Alpha BJJ clips. Writes temp outputs
to /tmp/roll_tracker_validation/ — never modifies production outputs.

Usage:
    python tools/validate_cp14_session_bank.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# Ensure repo root is importable
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import pandas as pd

from bjj_pipeline.contracts.f0_paths import SessionOutputLayout
from bjj_pipeline.stages.stitch.session_d_run import (
    SessionStageLayoutAdapter,
    aggregate_session_bank,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GYM_ID = "c8a592a4-2bca-400a-80e1-fec0e5cbea77"
DATE = "2026-03-18"
SESSION_ID = "2026-03-18T2000"
RAW_ROOT = REPO / "data" / "raw" / "nest" / GYM_ID
OUTPUT_ROOT = REPO / "outputs"
TEMP_ROOT = Path("/tmp/roll_tracker_validation")
CAM_IDS = sorted([p.name for p in RAW_ROOT.iterdir() if p.is_dir()])


def _discover_clips() -> list[tuple[Path, str]]:
    """Find all (mp4_path, cam_id) pairs under the gym_id raw directory."""
    clips = []
    for cam_id in CAM_IDS:
        cam_dir = RAW_ROOT / cam_id
        for mp4 in sorted(cam_dir.rglob("*.mp4")):
            clips.append((mp4, cam_id))
    return clips


# ---------------------------------------------------------------------------
# Test 1: SessionStageLayoutAdapter path resolution
# ---------------------------------------------------------------------------

def test_adapter_paths() -> bool:
    print("=" * 70)
    print("TEST 1: SessionStageLayoutAdapter path resolution")
    print("=" * 70)

    passed = 0
    failed = 0

    for cam_id in CAM_IDS:
        print(f"\n--- cam_id: {cam_id} ---")
        layout = SessionOutputLayout(
            gym_id=GYM_ID, date=DATE, session_id=SESSION_ID, root=TEMP_ROOT,
        )
        adapter = SessionStageLayoutAdapter(layout, cam_id=cam_id)

        def check(name: str, condition: bool, detail: str = ""):
            nonlocal passed, failed
            status = "PASS" if condition else "FAIL"
            if not condition:
                failed += 1
            else:
                passed += 1
            print(f"  [{status}] {name}  {detail}")

        # All methods return Path objects
        methods = [
            ("tracklet_bank_frames_parquet", adapter.tracklet_bank_frames_parquet()),
            ("tracklet_bank_summaries_parquet", adapter.tracklet_bank_summaries_parquet()),
            ("tracklet_frames_parquet", adapter.tracklet_frames_parquet()),
            ("tracklet_summaries_parquet", adapter.tracklet_summaries_parquet()),
            ("identity_hints_jsonl", adapter.identity_hints_jsonl()),
            ("detections_parquet", adapter.detections_parquet()),
            ("d1_graph_nodes_parquet", adapter.d1_graph_nodes_parquet()),
            ("d1_graph_edges_parquet", adapter.d1_graph_edges_parquet()),
            ("d1_segments_parquet", adapter.d1_segments_parquet()),
            ("d2_edge_costs_parquet", adapter.d2_edge_costs_parquet()),
            ("d2_constraints_json", adapter.d2_constraints_json()),
            ("person_tracks_parquet", adapter.person_tracks_parquet()),
            ("person_spans_parquet", adapter.person_spans_parquet()),
            ("identity_assignments_jsonl", adapter.identity_assignments_jsonl()),
            ("audit_jsonl('D')", adapter.audit_jsonl("D")),
            ("clip_manifest_path", adapter.clip_manifest_path()),
            ("stage_dir('D')", adapter.stage_dir("D")),
            ("clip_root", adapter.clip_root),
        ]
        for name, val in methods:
            check(f"{name} returns Path", isinstance(val, Path), f"→ {val}")

        # Specific assertions
        bf = adapter.tracklet_bank_frames_parquet()
        check("bank_frames contains cam_id", cam_id in str(bf), str(bf))

        bs = adapter.tracklet_bank_summaries_parquet()
        check("bank_summaries contains cam_id", cam_id in str(bs), str(bs))

        check(
            "tracklet_frames == bank_frames (alias)",
            adapter.tracklet_frames_parquet() == adapter.tracklet_bank_frames_parquet(),
        )
        check(
            "tracklet_summaries == bank_summaries (alias)",
            adapter.tracklet_summaries_parquet() == adapter.tracklet_bank_summaries_parquet(),
        )
        check(
            "clip_root == session_root",
            adapter.clip_root == layout.session_root,
            f"{adapter.clip_root} == {layout.session_root}",
        )

        rel = adapter.rel_to_clip_root(adapter.d1_graph_nodes_parquet())
        check(
            "rel_to_clip_root is relative",
            not rel.startswith("/") and not Path(rel).is_absolute(),
            f"→ {rel!r}",
        )

    print(f"\nTest 1 totals: {passed} passed, {failed} failed")
    return failed == 0


# ---------------------------------------------------------------------------
# Test 2: aggregate_session_bank() against real clips
# ---------------------------------------------------------------------------

def test_bank_aggregation() -> bool:
    print("\n" + "=" * 70)
    print("TEST 2: aggregate_session_bank() against real clips")
    print("=" * 70)

    session_clips = _discover_clips()
    print(f"\nDiscovered {len(session_clips)} clips across {len(CAM_IDS)} cameras")

    all_ok = True

    for cam_id in CAM_IDS:
        print(f"\n--- cam_id: {cam_id} ---")

        session_layout = SessionOutputLayout(
            gym_id=GYM_ID, date=DATE, session_id=SESSION_ID, root=TEMP_ROOT,
        )
        adapter = SessionStageLayoutAdapter(session_layout, cam_id=cam_id)

        try:
            frames_p, summ_p, det_p, hints_p = aggregate_session_bank(
                session_layout=session_layout,
                adapter=adapter,
                session_clips=session_clips,
                cam_id=cam_id,
                output_root=OUTPUT_ROOT,
            )
        except Exception as exc:
            print(f"  [FAIL] aggregate_session_bank raised: {exc}")
            all_ok = False
            continue

        # --- Frames parquet ---
        frames_df = pd.read_parquet(frames_p)
        print(f"\n  Frames parquet: {len(frames_df)} rows, {len(frames_df.columns)} cols")
        print(f"    Columns: {list(frames_df.columns)}")

        if "tracklet_id" in frames_df.columns:
            tids = frames_df["tracklet_id"].unique()
            has_colon = [t for t in tids if ":" in str(t)]
            no_colon = [t for t in tids if ":" not in str(t)]
            prefixes = set(str(t).split(":")[0] for t in has_colon)
            print(f"    Unique tracklet_ids: {len(tids)}")
            print(f"    With ':' separator: {len(has_colon)}")
            print(f"    WITHOUT ':' (bugs): {len(no_colon)}")
            if no_colon:
                print(f"    *** BUG — un-prefixed IDs: {list(no_colon)[:10]}")
                all_ok = False
            print(f"    Unique clip prefixes: {len(prefixes)} → {sorted(prefixes)[:5]}...")
            print(f"    Sample tracklet_ids: {list(tids[:5])}")
        else:
            print("    [FAIL] No tracklet_id column!")
            all_ok = False

        # --- Summaries parquet ---
        summ_df = pd.read_parquet(summ_p)
        print(f"\n  Summaries parquet: {len(summ_df)} rows")
        if "tracklet_id" in summ_df.columns:
            stids = summ_df["tracklet_id"].unique()
            s_no_colon = [t for t in stids if ":" not in str(t)]
            print(f"    Unique tracklet_ids: {len(stids)}")
            print(f"    WITHOUT ':' (bugs): {len(s_no_colon)}")
            if s_no_colon:
                print(f"    *** BUG — un-prefixed IDs: {list(s_no_colon)[:10]}")
                all_ok = False
        else:
            print("    [FAIL] No tracklet_id column!")
            all_ok = False

        # --- Detections parquet ---
        det_df = pd.read_parquet(det_p)
        print(f"\n  Detections parquet: {len(det_df)} rows, {len(det_df.columns)} cols")
        print(f"    Columns: {list(det_df.columns)}")

        # --- Identity hints ---
        hint_count = 0
        hints_with_tid = 0
        hints_without_tid = 0
        sample_hints: list[dict] = []
        with open(hints_p, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                hint_count += 1
                rec = json.loads(line)
                if "tracklet_id" in rec:
                    hints_with_tid += 1
                    if ":" not in str(rec["tracklet_id"]):
                        hints_without_tid += 1
                if len(sample_hints) < 3:
                    sample_hints.append(rec)

        print(f"\n  Identity hints: {hint_count} records")
        print(f"    With tracklet_id: {hints_with_tid}")
        print(f"    Missing ':' in tracklet_id (bugs): {hints_without_tid}")
        if hints_without_tid:
            all_ok = False
        for i, h in enumerate(sample_hints):
            print(f"    Sample {i+1}: {json.dumps(h)[:120]}...")

        # Camera summary
        clips_for_cam = [c for c in session_clips if c[1] == cam_id]
        print(f"\n  Clips contributing: {len(prefixes) if 'tracklet_id' in frames_df.columns else '?'}/{len(clips_for_cam)}")

    print(f"\n{'=' * 70}")
    verdict = "PASS" if all_ok else "FAIL"
    print(f"Test 2 overall: [{verdict}]")
    return all_ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print(f"CP14 Validation Script")
    print(f"Gym ID: {GYM_ID}")
    print(f"Cameras: {CAM_IDS}")
    print(f"Temp output: {TEMP_ROOT}")
    print()

    t1 = test_adapter_paths()
    t2 = test_bank_aggregation()

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"  Test 1 (adapter paths):     {'PASS' if t1 else 'FAIL'}")
    print(f"  Test 2 (bank aggregation):  {'PASS' if t2 else 'FAIL'}")
    print()
    print("Test 3 (buzzer survey) — run separately:")
    print(f"  python tools/detect_buzzer.py \\")
    print(f"    --input {RAW_ROOT} \\")
    print(f"    --survey \\")
    print(f"    --output-dir {TEMP_ROOT}/audio_events")

    return 0 if (t1 and t2) else 1


if __name__ == "__main__":
    sys.exit(main())
