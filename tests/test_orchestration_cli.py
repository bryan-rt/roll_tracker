import json
import os
import shutil
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from bjj_pipeline.stages.orchestration.cli import app
from bjj_pipeline.stages.orchestration.pipeline import required_outputs_for_stage
from bjj_pipeline.contracts.f0_paths import ClipOutputLayout


runner = CliRunner()


def make_ingest_path(tmp_path: Path, cam: str, stem: str) -> Path:
    # .../data/raw/nest/<cam>/2026-01-03/12/<stem>.mp4
    p = tmp_path / "data" / "raw" / "nest" / cam / "2026-01-03" / "12"
    p.mkdir(parents=True, exist_ok=True)
    clip = p / f"{stem}.mp4"
    clip.write_bytes(b"\x00\x00\x00")
    return clip


def write_stage_a(layout: ClipOutputLayout, clip_id: str, camera_id: str):
    layout.ensure_dirs_for_stage("A")
    det = pd.DataFrame([
        {
            "clip_id": clip_id,
            "camera_id": camera_id,
            "frame_index": 0,
            "timestamp_ms": 0,
            "detection_id": "d1",
            "class_name": "person",
            "confidence": 0.9,
            "x1": 0.0,
            "y1": 0.0,
            "x2": 1.0,
            "y2": 1.0,
            "tracklet_id": "t1",
            "source": "unit",
            "debug_json": "{}",
        }
    ])
    tf = pd.DataFrame([
        {
            "clip_id": clip_id,
            "camera_id": camera_id,
            "tracklet_id": "t1",
            "frame_index": 0,
            "timestamp_ms": 0,
            "detection_id": "d1",
            "local_track_conf": 0.5,
        }
    ])
    ts = pd.DataFrame([
        {
            "clip_id": clip_id,
            "camera_id": camera_id,
            "tracklet_id": "t1",
            "start_frame": 0,
            "end_frame": 0,
            "n_frames": 1,
            "quality_score": 0.5,
            "reason_codes_json": "[]",
        }
    ])
    det.to_parquet(layout.detections_parquet())
    tf.to_parquet(layout.tracklet_frames_parquet())
    ts.to_parquet(layout.tracklet_summaries_parquet())

    # Stage A now requires baseline contact_points.parquet
    cp = pd.DataFrame([
        {
            "clip_id": clip_id,
            "camera_id": camera_id,
            "frame_index": 0,
            "timestamp_ms": 0,
            "detection_id": "d1",
            "tracklet_id": "t1",
            "u_px": 0.5,
            "v_px": 1.0,
            "x_m": 0.1,
            "y_m": 0.2,
            "on_mat": True,
            "contact_conf": 0.5,
            "contact_method": "unit",
        }
    ]).sort_values(["frame_index", "detection_id"], kind="mergesort")
    cp.to_parquet(layout.stage_A_contact_points_parquet(), index=False)

    (layout.audit_jsonl("A")).write_text("{}\n", encoding="utf-8")


def write_stage_b(layout: ClipOutputLayout, clip_id: str, camera_id: str):
    layout.ensure_dirs_for_stage("B")
    cp = pd.DataFrame([
        {
            "clip_id": clip_id,
            "camera_id": camera_id,
            "frame_index": 0,
            "timestamp_ms": 0,
            "detection_id": "d1",
            "u_px": 10.0,
            "v_px": 20.0,
            "x_m": 0.1,
            "y_m": 0.2,
            "method": "proj",
            "confidence": 0.8,
            "homography_id": "H1",
        }
    ])
    cp.to_parquet(layout.contact_points_parquet())
    (layout.audit_jsonl("B")).write_text("{}\n", encoding="utf-8")


def write_stage_c(layout: ClipOutputLayout, clip_id: str, camera_id: str):
    layout.ensure_dirs_for_stage("C")
    # Stage C outputs are allowed to be empty in early slices; ensure validator-safe.
    Path(layout.identity_hints_jsonl()).write_text("", encoding="utf-8")
    Path(layout.tag_observations_jsonl()).write_text("", encoding="utf-8")
    (layout.audit_jsonl("C")).write_text("{}\n", encoding="utf-8")


def write_stage_d(layout: ClipOutputLayout, clip_id: str, camera_id: str):
    layout.ensure_dirs_for_stage("D")
    pt = pd.DataFrame([
        {
            "clip_id": clip_id,
            "camera_id": camera_id,
            "person_id": "p1",
            "frame_index": 0,
            "timestamp_ms": 0,
            "detection_id": "d1",
            "tracklet_id": "t1",
            "x1": 0.0,
            "y1": 0.0,
            "x2": 1.0,
            "y2": 1.0,
            "x_m": 0.0,
            "y_m": 0.0,
            "mask_ref": "",
            "reid_sim": 0.1,
            "stitch_edge_type": "none",
        }
    ])
    pt.to_parquet(layout.person_tracks_parquet())
    assigns = [{
        "schema_version": "0",
        "artifact_type": "identity_assignment",
        "clip_id": clip_id,
        "camera_id": camera_id,
        "pipeline_version": "dev",
        "created_at_ms": 0,
        "person_id": "p1",
        "tag_id": "123",
        "assignment_confidence": 0.5,
        "evidence": "unit",
    }]
    Path(layout.identity_assignments_jsonl()).write_text("\n".join(json.dumps(r) for r in assigns)+"\n", encoding="utf-8")
    (layout.audit_jsonl("D")).write_text("{}\n", encoding="utf-8")


def write_stage_e(layout: ClipOutputLayout, clip_id: str, camera_id: str):
    layout.ensure_dirs_for_stage("E")
    ms = [{
        "schema_version": "0",
        "artifact_type": "match_session",
        "clip_id": clip_id,
        "camera_id": camera_id,
        "pipeline_version": "dev",
        "created_at_ms": 0,
        "match_id": "m1",
        "person_id_a": "p1",
        "person_id_b": "p2",
        "start_frame": 0,
        "end_frame": 0,
        "start_ts_ms": 0,
        "end_ts_ms": 0,
        "method": "heuristic",
        "confidence": 0.7,
        "evidence": "unit",
    }]
    Path(layout.match_sessions_jsonl()).write_text("\n".join(json.dumps(r) for r in ms)+"\n", encoding="utf-8")
    (layout.audit_jsonl("E")).write_text("{}\n", encoding="utf-8")


def write_stage_f(layout: ClipOutputLayout, clip_id: str, camera_id: str):
    layout.ensure_dirs_for_stage("F")
    em = [{
        "schema_version": "0",
        "artifact_type": "export_manifest",
        "clip_id": clip_id,
        "camera_id": camera_id,
        "pipeline_version": "dev",
        "created_at_ms": 0,
        "export_id": "e1",
        "match_id": "m1",
        "output_video_path": "exports/out.mp4",
        "crop_mode": "full",
        "privacy": "normal",
        "inputs": [],
    }]
    Path(layout.export_manifest_jsonl()).write_text("\n".join(json.dumps(r) for r in em)+"\n", encoding="utf-8")
    (layout.audit_jsonl("F")).write_text("{}\n", encoding="utf-8")


def fake_run_factory(letter: str):
    def _run(config, inputs):
        layout: ClipOutputLayout = inputs["layout"]
        clip_id = layout.clip_id
        camera_id = config.get("camera_id", "camXX")
        if letter == "A":
            write_stage_a(layout, clip_id, camera_id)
        elif letter == "B":
            write_stage_b(layout, clip_id, camera_id)
        elif letter == "C":
            write_stage_c(layout, clip_id, camera_id)
        elif letter == "D":
            write_stage_d(layout, clip_id, camera_id)
        elif letter == "E":
            write_stage_e(layout, clip_id, camera_id)
        elif letter == "F":
            write_stage_f(layout, clip_id, camera_id)
    return _run


@pytest.fixture
def clip_env(tmp_path: Path):
    cam = "cam03"
    stem = "cam03-20260103-124000"
    ingest = make_ingest_path(tmp_path, cam, stem)
    outputs = tmp_path / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"camera_id": cam, "param": 1}), encoding="utf-8")
    return {
        "cam": cam,
        "stem": stem,
        "ingest": ingest,
        "outputs": outputs,
        "config_path": config_path,
    }

def _write_homography(tmp_path: Path, cam: str):
    # Homography stored at configs/cameras/<camera_id>/homography.json relative to cwd
    p = tmp_path / "configs" / "cameras" / cam
    p.mkdir(parents=True, exist_ok=True)
    (p / "homography.json").write_text(json.dumps({"H": [[1,0,0],[0,1,0],[0,0,1]]}), encoding="utf-8")

def test_missing_homography_fails_fast_noninteractive(monkeypatch, clip_env, tmp_path):
    # Ensure we run in tmp_path so relative configs/ resolves there
    monkeypatch.chdir(tmp_path)
    res = runner.invoke(app, [
        "run",
        "--clip", str(clip_env["ingest"]),
        "--camera", clip_env["cam"],
        "--config", str(clip_env["config_path"]),
        "--to-stage", "A",
    ])
    assert res.exit_code != 0
    assert "Missing homography" in res.stdout

def test_homography_present_allows_stage_a(monkeypatch, clip_env, tmp_path):
    monkeypatch.chdir(tmp_path)
    _write_homography(tmp_path, clip_env["cam"])
    from bjj_pipeline.stages.detect_track import run as mA
    monkeypatch.setattr(mA, "run", fake_run_factory("A"))
    res = runner.invoke(app, [
        "run",
        "--clip", str(clip_env["ingest"]),
        "--camera", clip_env["cam"],
        "--config", str(clip_env["config_path"]),
        "--to-stage", "A",
    ])
    assert res.exit_code == 0


def test_clip_id_from_stem(clip_env):
    ingest = clip_env["ingest"]
    assert ingest.stem == clip_env["stem"]


def test_ingest_path_validation_camera_mismatch(clip_env):
    # wrong camera id
    res = runner.invoke(app, [
        "run",
        "--clip", str(clip_env["ingest"]),
        "--camera", "cam02",
        "--config", str(clip_env["config_path"]),
    ])
    assert res.exit_code != 0
    assert "camera_id directory mismatch" in res.stdout


def test_run_creates_manifest_and_audit(monkeypatch, clip_env, tmp_path):
    # Patch all stage runs to write minimal outputs
    # Ensure homography present for Stage A preflight
    monkeypatch.chdir(tmp_path)
    _write_homography(tmp_path, clip_env["cam"])
    from bjj_pipeline.stages.detect_track import run as mA
    from bjj_pipeline.stages.masks import run as mB
    from bjj_pipeline.stages.tags import run as mC
    from bjj_pipeline.stages.stitch import run as mD
    from bjj_pipeline.stages.matches import run as mE
    from bjj_pipeline.stages.export import run as mF
    monkeypatch.setattr(mA, "run", fake_run_factory("A"))
    monkeypatch.setattr(mB, "run", fake_run_factory("B"))
    monkeypatch.setattr(mC, "run", fake_run_factory("C"))
    monkeypatch.setattr(mD, "run", fake_run_factory("D"))
    monkeypatch.setattr(mE, "run", fake_run_factory("E"))
    monkeypatch.setattr(mF, "run", fake_run_factory("F"))

    res = runner.invoke(app, [
        "run",
        "--clip", str(clip_env["ingest"]),
        "--camera", clip_env["cam"],
        "--config", str(clip_env["config_path"]),
    ])
    assert res.exit_code == 0

    layout = ClipOutputLayout(clip_id=clip_env["stem"])
    assert (layout.clip_manifest_path()).exists()
    audit = layout.clip_root / "orchestration_audit.jsonl"
    assert audit.exists()
    txt = audit.read_text(encoding="utf-8")
    assert "run_started" in txt


def test_skip_when_complete_and_same_config(monkeypatch, clip_env, tmp_path):
    # First run to populate A
    monkeypatch.chdir(tmp_path)
    _write_homography(tmp_path, clip_env["cam"])
    from bjj_pipeline.stages.detect_track import run as mA
    monkeypatch.setattr(mA, "run", fake_run_factory("A"))
    res1 = runner.invoke(app, [
        "run",
        "--clip", str(clip_env["ingest"]),
        "--camera", clip_env["cam"],
        "--config", str(clip_env["config_path"]),
        "--to-stage", "A",
    ])
    assert res1.exit_code == 0

    # Second run with same config should skip A
    res2 = runner.invoke(app, [
        "run",
        "--clip", str(clip_env["ingest"]),
        "--camera", clip_env["cam"],
        "--config", str(clip_env["config_path"]),
        "--to-stage", "A",
    ])
    assert res2.exit_code == 0
    audit = (ClipOutputLayout(clip_id=clip_env["stem"]).clip_root / "orchestration_audit.jsonl").read_text(encoding="utf-8")
    assert "stage_skipped" in audit


def test_force_reruns(monkeypatch, clip_env, tmp_path):
    monkeypatch.chdir(tmp_path)
    _write_homography(tmp_path, clip_env["cam"])
    from bjj_pipeline.stages.detect_track import run as mA
    monkeypatch.setattr(mA, "run", fake_run_factory("A"))
    # initial run
    runner.invoke(app, [
        "run",
        "--clip", str(clip_env["ingest"]),
        "--camera", clip_env["cam"],
        "--config", str(clip_env["config_path"]),
        "--to-stage", "A",
    ])
    # force rerun
    res = runner.invoke(app, [
        "run",
        "--clip", str(clip_env["ingest"]),
        "--camera", clip_env["cam"],
        "--config", str(clip_env["config_path"]),
        "--to-stage", "A",
        "--force",
    ])
    assert res.exit_code == 0
    audit = (ClipOutputLayout(clip_id=clip_env["stem"]).clip_root / "orchestration_audit.jsonl").read_text(encoding="utf-8")
    assert "stage_started" in audit


def test_failure_writes_stage_failed(monkeypatch, clip_env):
    # Make stage A run raise
    def bad_run(config, inputs):
        raise RuntimeError("boom")
    from bjj_pipeline.stages.detect_track import run as mA
    monkeypatch.setattr(mA, "run", bad_run)

    res = runner.invoke(app, [
        "run",
        "--clip", str(clip_env["ingest"]),
        "--camera", clip_env["cam"],
        "--config", str(clip_env["config_path"]),
        "--to-stage", "A",
        "--force",
    ])
    assert res.exit_code != 0
    audit = (ClipOutputLayout(clip_id=clip_env["stem"]).clip_root / "orchestration_audit.jsonl").read_text(encoding="utf-8")
    assert "stage_failed" in audit


def test_status_reports_correctly(monkeypatch, clip_env):
    # Populate A outputs
    from bjj_pipeline.stages.detect_track import run as mA
    monkeypatch.setattr(mA, "run", fake_run_factory("A"))
    runner.invoke(app, [
        "run",
        "--clip", str(clip_env["ingest"]),
        "--camera", clip_env["cam"],
        "--config", str(clip_env["config_path"]),
        "--to-stage", "A",
    ])

    res = runner.invoke(app, [
        "status",
        "--clip", str(clip_env["ingest"]),
        "--camera", clip_env["cam"],
    ])
    assert res.exit_code == 0
    out = res.stdout
    assert "A" in out and "yes" in out  # complete yes


def test_validate_nonzero_on_problem(monkeypatch, clip_env):
    # Write invalid detections (bbox x2 < x1) to trigger validation failure
    layout = ClipOutputLayout(clip_id=clip_env["stem"])
    layout.ensure_dirs_for_stage("A")
    det = pd.DataFrame([
        {
            "clip_id": clip_env["stem"],
            "camera_id": clip_env["cam"],
            "frame_index": 0,
            "timestamp_ms": 0,
            "detection_id": "d1",
            "class_name": "person",
            "confidence": 0.9,
            "x1": 1.0,
            "y1": 0.0,
            "x2": 0.0,  # invalid bbox
            "y2": 1.0,
        }
    ])
    det.to_parquet(layout.detections_parquet())
    # minimal other tables for stage A validator
    tf = pd.DataFrame([], columns=["clip_id","camera_id","tracklet_id","frame_index","timestamp_ms","detection_id"])  # empty ok
    ts = pd.DataFrame([], columns=["clip_id","camera_id","tracklet_id","start_frame","end_frame","n_frames"])  # empty ok
    tf.to_parquet(layout.tracklet_frames_parquet())
    ts.to_parquet(layout.tracklet_summaries_parquet())

    res = runner.invoke(app, [
        "validate",
        "--clip", str(clip_env["ingest"]),
        "--camera", clip_env["cam"],
        "--stage", "A",
    ])
    assert res.exit_code != 0
