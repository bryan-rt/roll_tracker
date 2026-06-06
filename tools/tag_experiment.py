"""CP-TAG-2 experiment orchestrator: dense GT + full-frame tag scan comparison.

Runs:
1. Dense GT Stage A census (stride=1, 10x more evaluation points)
2. Dense GT Stage D trace
3. Loads full-frame tag scan observations (from tag_fullscan.py)
4. Three-way comparison: baseline vs dense-GT vs dense-GT+full-scan
5. Generates _tag_experiment_report.md

Usage:
    PYTHONPATH=src python tools/tag_experiment.py
"""
from __future__ import annotations

import json
import logging
import sys
import textwrap
import time
from pathlib import Path

import pandas as pd

from pipeline_validation.common.manifest import load_manifest
from pipeline_validation.signal_trace.stage_a_census import run_census
from pipeline_validation.signal_trace.stage_d_trace import run_d_trace

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = REPO_ROOT / "outputs"
EVAL_DIR = OUTPUTS_DIR / "_eval" / "signal_trace"
FULLSCAN_DIR = OUTPUTS_DIR / "_experiments" / "tag_fullscan"

DENSE_MANIFEST_PATH = REPO_ROOT / "configs" / "models" / "bjj-detect-all-cameras-dense.yaml"
V1_MODEL_ID = "bjj-detect-all-cameras"
REAL_GYM_ID_200246 = "c8a592a4-2bca-400a-80e1-fec0e5cbea77"

# Pipeline clip directories for loading tag observations
PIPELINE_CLIP_DIRS = {
    "J_EDEw-20260318-200015": OUTPUTS_DIR / "_eval_gt" / "J_EDEw" / "2026-03-18" / "20" / "J_EDEw-20260318-200015",
    "J_EDEw-20260318-200246": OUTPUTS_DIR / REAL_GYM_ID_200246 / "J_EDEw" / "2026-03-18" / "20" / "J_EDEw-20260318-200246",
}


def load_tag_observations(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def run_dense_trace(export, manifest, gym_id: str, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run Stage A census + D trace with dense GT for one export."""
    out_dir.mkdir(parents=True, exist_ok=True)

    cam = export.camera_id
    clip_id = export.source_video.replace(".mp4", "")

    # Stage A census
    print(f"\n  Running dense Stage A census for {cam}/{clip_id}...")
    t0 = time.time()
    a_trace, a_summary = run_census(manifest, export, gym_id, iou_threshold=0.3)
    a_path = out_dir / "gt_signal_trace_stage_a.parquet"
    a_trace.to_parquet(a_path, index=False)
    print(f"  Stage A: {len(a_trace)} rows in {time.time()-t0:.1f}s")
    for cls in ("tight_match", "pair_box", "split", "miss"):
        c = a_summary[cls]
        print(f"    {cls}: {c['count']} ({c['pct']:.1%})")

    # Write summary
    with open(out_dir / "stage_a_summary.json", "w") as f:
        json.dump(a_summary, f, indent=2)

    # Stage D trace
    d_trace = pd.DataFrame()
    print(f"  Running dense D-stage trace for {cam}/{clip_id}...")
    t0 = time.time()
    try:
        d_trace, d_summary = run_d_trace(manifest, export, gym_id, a_path)
        d_trace.to_parquet(out_dir / "gt_signal_trace_d.parquet", index=False)
        print(f"  Stage D: {len(d_trace)} rows in {time.time()-t0:.1f}s")
        for cls in ("correct_id", "wrong_id", "no_id", "no_detection"):
            c = d_summary.get(cls, {"count": 0, "pct": 0})
            print(f"    {cls}: {c['count']} ({c['pct']:.1%})")
        with open(out_dir / "stage_d_summary.json", "w") as f:
            json.dump(d_summary, f, indent=2)
    except Exception as e:
        print(f"  D-trace FAILED: {e}")
        import traceback
        traceback.print_exc()

    return a_trace, d_trace


def load_baseline_trace(camera_dir_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load existing stride-10 baseline trace artifacts."""
    base = EVAL_DIR / V1_MODEL_ID / camera_dir_name
    a_path = base / "gt_signal_trace_stage_a.parquet"
    d_path = base / "gt_signal_trace_d.parquet"
    a_trace = pd.read_parquet(a_path) if a_path.exists() else pd.DataFrame()
    d_trace = pd.read_parquet(d_path) if d_path.exists() else pd.DataFrame()
    return a_trace, d_trace


def summarize_trace(a_trace: pd.DataFrame, d_trace: pd.DataFrame, tag_obs: list[dict], label: str) -> dict:
    """Compute summary stats from trace dataframes."""
    n_frames = len(a_trace) if not a_trace.empty else 0
    n_gt_person_frames = len(a_trace) if not a_trace.empty else 0

    a_stats = {}
    if not a_trace.empty and "classification" in a_trace.columns:
        counts = a_trace["classification"].value_counts()
        total = len(a_trace)
        for cls in ("tight_match", "pair_box", "split", "miss"):
            c = int(counts.get(cls, 0))
            a_stats[cls] = {"count": c, "pct": c / total if total > 0 else 0}
        a_stats["total"] = total

    d_stats = {}
    if not d_trace.empty and "d_classification" in d_trace.columns:
        counts = d_trace["d_classification"].value_counts()
        total = len(d_trace)
        for cls in ("correct_id", "wrong_id", "no_id", "no_detection"):
            c = int(counts.get(cls, 0))
            d_stats[cls] = {"count": c, "pct": c / total if total > 0 else 0}
        d_stats["total"] = total

    return {
        "label": label,
        "gt_frames": n_frames,
        "gt_person_frames": n_gt_person_frames,
        "tag_observations": len(tag_obs),
        "unique_tag_frames": len(set(o.get("frame_index", -1) for o in tag_obs)),
        "stage_a": a_stats,
        "stage_d": d_stats,
    }


def generate_report(
    results: list[dict],
    fullscan_summary: dict | None,
    output_path: Path,
) -> None:
    """Generate _tag_experiment_report.md."""
    lines = [
        "# CP-TAG-2: Full-Frame Tag Scan + Dense GT Evaluation",
        "",
        "## Experiment Setup",
        "",
        "- **Videos:** J_EDEw-20260318-200015 (val-split, 3001 frames), "
        "J_EDEw-20260318-200246 (train-split, 4491 frames)",
        "- **Tag family:** 36h11",
        "- **Model:** bjj-detect-all-cameras (same as CP-TAG-1)",
        "- **Dense manifest:** configs/models/bjj-detect-all-cameras-dense.yaml (stride=1)",
        "",
    ]

    # Full-frame scan results
    if fullscan_summary:
        lines.append("## Full-Frame Scan Results")
        lines.append("")
        lines.append("| Video | Frames Scanned | Full-scan Obs | Frames with Tags | Detection Rate | Pipeline Obs | Multiplier |")
        lines.append("|-------|---------------|--------------|-----------------|---------------|-------------|------------|")
        for vs in fullscan_summary.get("videos", []):
            mult = f"{vs['multiplier']:.0f}x" if vs.get("multiplier") else "N/A"
            lines.append(
                f"| {vs['clip_id']} | {vs['total_frames_scanned']} | "
                f"{vs['fullscan_observations']} | {vs['fullscan_frames_with_tags']} | "
                f"{vs['fullscan_detection_rate']:.4%} | "
                f"{vs['original_pipeline_observations']} | {mult} |"
            )
        lines.append("")
        for vs in fullscan_summary.get("videos", []):
            if vs.get("fullscan_tag_ids"):
                lines.append(f"**{vs['clip_id']}** tag IDs seen: {vs['fullscan_tag_ids']}")
        lines.append("")

    # Per-video three-way comparison
    for video_results in results:
        clip_id = video_results["clip_id"]
        lines.append(f"## {clip_id}")
        lines.append("")

        configs = video_results.get("configs", [])
        if configs:
            lines.append("### Three-Way Comparison")
            lines.append("")
            lines.append("| Config | GT Person-Frames | Tag Obs | tight_match | pair_box | miss | correct_id | wrong_id | no_id |")
            lines.append("|--------|-----------------|---------|-------------|----------|------|-----------|---------|-------|")
            for cfg in configs:
                a = cfg.get("stage_a", {})
                d = cfg.get("stage_d", {})
                tm = a.get("tight_match", {})
                pb = a.get("pair_box", {})
                ms = a.get("miss", {})
                ci = d.get("correct_id", {})
                wi = d.get("wrong_id", {})
                ni = d.get("no_id", {})

                def fmt(stat):
                    if not stat:
                        return "—"
                    return f"{stat.get('count', 0)} ({stat.get('pct', 0):.1%})"

                lines.append(
                    f"| {cfg['label']} | {cfg['gt_person_frames']} | "
                    f"{cfg['tag_observations']} | "
                    f"{fmt(tm)} | {fmt(pb)} | {fmt(ms)} | "
                    f"{fmt(ci)} | {fmt(wi)} | {fmt(ni)} |"
                )
            lines.append("")

    # Verdict section (placeholder — filled after running)
    lines.extend([
        "## Verdict",
        "",
        "**Tag detection ceiling:** [filled after experiment]",
        "",
        "**Bottleneck:** [physical occlusion vs pipeline restriction]",
        "",
        "**Implications for identity strategy:** [filled after experiment]",
        "",
    ])

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport written: {output_path}")


def main():
    if not DENSE_MANIFEST_PATH.exists():
        print(f"ERROR: Dense manifest not found: {DENSE_MANIFEST_PATH}")
        sys.exit(1)

    manifest = load_manifest(DENSE_MANIFEST_PATH)

    # Find J_EDEw exports
    jedew_200015 = None
    jedew_200246 = None
    for e in manifest.training_data:
        if e.camera_id == "J_EDEw" and "200015" in e.source_video:
            jedew_200015 = e
        elif e.camera_id == "J_EDEw" and "200246" in e.source_video:
            jedew_200246 = e

    if jedew_200015 is None:
        print("ERROR: J_EDEw-200015 not found in dense manifest")
        sys.exit(1)

    trace_base = EVAL_DIR / V1_MODEL_ID

    all_results = []

    # --- Video 1: J_EDEw-200015 ---
    print("=" * 60)
    print("Video 1: J_EDEw-20260318-200015")
    print("=" * 60)

    clip_id_200015 = "J_EDEw-20260318-200015"
    clip_dir_200015 = PIPELINE_CLIP_DIRS[clip_id_200015]

    # Config 1: Baseline (stride-10, existing artifacts)
    baseline_a, baseline_d = load_baseline_trace("J_EDEw")
    pipeline_tags_200015 = load_tag_observations(
        clip_dir_200015 / "stage_C" / "tag_observations.jsonl"
    )
    baseline_summary = summarize_trace(
        baseline_a, baseline_d, pipeline_tags_200015, "Baseline (stride-10, pipeline tags)"
    )

    # Config 2: Dense GT (stride-1, pipeline tags)
    dense_out_200015 = trace_base / "J_EDEw" / "dense_gt_trace"
    dense_a_200015, dense_d_200015 = run_dense_trace(
        jedew_200015, manifest, "_eval_gt", dense_out_200015,
    )
    dense_summary = summarize_trace(
        dense_a_200015, dense_d_200015, pipeline_tags_200015,
        "Dense GT (stride-1, pipeline tags)"
    )

    # Config 3: Dense GT + full-scan tags
    fullscan_tags_200015 = load_tag_observations(
        FULLSCAN_DIR / clip_id_200015 / "tag_observations.jsonl"
    )
    fullscan_summary = summarize_trace(
        dense_a_200015, dense_d_200015, fullscan_tags_200015,
        "Dense GT + full-scan tags"
    )

    all_results.append({
        "clip_id": clip_id_200015,
        "configs": [baseline_summary, dense_summary, fullscan_summary],
    })

    # --- Video 2: J_EDEw-200246 ---
    if jedew_200246 is not None:
        print("\n" + "=" * 60)
        print("Video 2: J_EDEw-20260318-200246")
        print("=" * 60)

        clip_id_200246 = "J_EDEw-20260318-200246"
        clip_dir_200246 = PIPELINE_CLIP_DIRS[clip_id_200246]

        # Config 1: Baseline (stride-10, existing artifacts)
        baseline_a_246, baseline_d_246 = load_baseline_trace("J_EDEw_200246")
        pipeline_tags_200246 = load_tag_observations(
            clip_dir_200246 / "stage_C" / "tag_observations.jsonl"
        )
        baseline_summary_246 = summarize_trace(
            baseline_a_246, baseline_d_246, pipeline_tags_200246,
            "Baseline (stride-10, pipeline tags)"
        )

        # Config 2: Dense GT (stride-1, pipeline tags)
        dense_out_200246 = trace_base / "J_EDEw_200246" / "dense_gt_trace"
        dense_a_200246, dense_d_200246 = run_dense_trace(
            jedew_200246, manifest, REAL_GYM_ID_200246, dense_out_200246,
        )
        dense_summary_246 = summarize_trace(
            dense_a_200246, dense_d_200246, pipeline_tags_200246,
            "Dense GT (stride-1, pipeline tags)"
        )

        # Config 3: Dense GT + full-scan tags
        fullscan_tags_200246 = load_tag_observations(
            FULLSCAN_DIR / clip_id_200246 / "tag_observations.jsonl"
        )
        fullscan_summary_246 = summarize_trace(
            dense_a_200246, dense_d_200246, fullscan_tags_200246,
            "Dense GT + full-scan tags"
        )

        all_results.append({
            "clip_id": clip_id_200246,
            "configs": [baseline_summary_246, dense_summary_246, fullscan_summary_246],
        })

    # Load full-scan experiment summary
    fullscan_exp_summary = None
    fullscan_exp_path = FULLSCAN_DIR / "experiment_summary.json"
    if fullscan_exp_path.exists():
        with open(fullscan_exp_path) as f:
            fullscan_exp_summary = json.load(f)

    # Generate report
    report_path = trace_base / V1_MODEL_ID / "_tag_experiment_report.md"
    # Put report at the model level
    report_path = trace_base / "_tag_experiment_report.md"
    generate_report(all_results, fullscan_exp_summary, report_path)

    # Write results JSON
    results_path = trace_base / "_tag_experiment_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results JSON: {results_path}")


if __name__ == "__main__":
    main()
