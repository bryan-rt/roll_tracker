"""Orchestration runner and stage registry for offline pipeline (fixed).

Implements:
- Stage registry for: detect_track (A), masks (B), tags (C), stitch (D), matches (E), export (F)
- Ingest path validation and clip_id extraction
- Outputs root creation and manifest lifecycle via bjj_pipeline.contracts.f0_manifest
- Orchestration audit JSONL append with deterministic events
- Resume logic based on required outputs, validators, and config hash stability
- Stage execution contract: run(config, inputs) per stage
"""

from __future__ import annotations

import importlib
import json
import time
from dataclasses import dataclass
import subprocess
import sys
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import math

from bjj_pipeline.contracts.f0_manifest import (
    ClipManifest,
    init_manifest,
    load_manifest,
    register_stage_A_defaults,
    register_stage_B_defaults,
    write_manifest,
)
from bjj_pipeline.contracts.f0_paths import ClipOutputLayout, StageLetter
from bjj_pipeline.contracts import f0_validate as v
from bjj_pipeline.config.loader import to_runtime_config


class PipelineError(RuntimeError):
    pass


@dataclass(frozen=True)
class StageSpec:
    key: str
    letter: StageLetter
    module_path: str


STAGES: List[StageSpec] = [
    StageSpec("detect_track", "A", "bjj_pipeline.stages.detect_track.run"),
    StageSpec("masks", "B", "bjj_pipeline.stages.masks.run"),
    StageSpec("tags", "C", "bjj_pipeline.stages.tags.run"),
    StageSpec("stitch", "D", "bjj_pipeline.stages.stitch.run"),
    StageSpec("matches", "E", "bjj_pipeline.stages.matches.run"),
    StageSpec("export", "F", "bjj_pipeline.stages.export.run"),
]


def required_outputs_for_stage(layout: ClipOutputLayout, letter: StageLetter) -> List[str]:
    if letter == "A":
        return [
            layout.rel_to_clip_root(layout.detections_parquet()),
            layout.rel_to_clip_root(layout.tracklet_frames_parquet()),
            layout.rel_to_clip_root(layout.tracklet_summaries_parquet()),
            layout.rel_to_clip_root(layout.stage_A_contact_points_parquet()),
            layout.rel_to_clip_root(layout.audit_jsonl("A")),
        ]
    if letter == "B":
        return [
            # Accept either refined (canonical) or legacy filename for completion checks.
            f"glob:{layout.rel_to_clip_root(layout.stage_dir('B'))}/contact_points*.parquet",
            f"glob:{layout.rel_to_clip_root(layout.masks_dir())}/*.npz",
            layout.rel_to_clip_root(layout.audit_jsonl("B")),
        ]
    if letter == "C":
        return [
            layout.rel_to_clip_root(layout.tag_observations_jsonl()),
            layout.rel_to_clip_root(layout.identity_hints_jsonl()),
            layout.rel_to_clip_root(layout.audit_jsonl("C")),
        ]
    if letter == "D":
        return [
            layout.rel_to_clip_root(layout.person_tracks_parquet()),
            layout.rel_to_clip_root(layout.identity_assignments_jsonl()),
            layout.rel_to_clip_root(layout.audit_jsonl("D")),
        ]
    if letter == "E":
        return [
            layout.rel_to_clip_root(layout.match_sessions_jsonl()),
            layout.rel_to_clip_root(layout.audit_jsonl("E")),
        ]
    if letter == "F":
        return [
            layout.rel_to_clip_root(layout.export_manifest_jsonl()),
            layout.rel_to_clip_root(layout.audit_jsonl("F")),
        ]
    return []


def extract_clip_id(ingest_path: Path) -> str:
    return ingest_path.stem


def validate_ingest_path(ingest_path: Path, camera_id: str) -> None:
    parts = ingest_path.resolve().parts
    try:
        idx = parts.index("nest")
    except ValueError:
        raise PipelineError("ingest path missing 'nest' component under data/raw")
    if idx < 2 or parts[idx - 2] != "data" or parts[idx - 1] != "raw":
        raise PipelineError("ingest path must be under data/raw/nest/...")
    try:
        cam = parts[idx + 1]
        date_str = parts[idx + 2]
        hour_str = parts[idx + 3]
    except IndexError:
        raise PipelineError("ingest path incomplete; expected data/raw/nest/<cam>/<YYYY-MM-DD>/<HH>/...")
    if cam != camera_id:
        raise PipelineError(f"camera_id directory mismatch: path={cam} cli={camera_id}")
    stem = ingest_path.stem
    if not stem.startswith(f"{camera_id}-"):
        raise PipelineError(f"clip filename stem must start with '{camera_id}-' (got '{stem}')")
    if len(date_str) != 10 or date_str[4] != "-" or date_str[7] != "-":
        raise PipelineError(f"date folder not in YYYY-MM-DD format: {date_str}")
    if len(hour_str) != 2 or not hour_str.isdigit():
        raise PipelineError(f"hour folder not a 2-digit hour: {hour_str}")


def hash_config(config: Dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return sha256(payload.encode("utf-8")).hexdigest()


def orchestration_audit_path(layout: ClipOutputLayout) -> Path:
    return layout.clip_root / "orchestration_audit.jsonl"


def _now_ms() -> int:
    return int(time.time() * 1000)


def append_audit(layout: ClipOutputLayout, event: Dict[str, Any]) -> None:
    path = orchestration_audit_path(layout)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n")

def _homography_path(camera_id: str) -> Path:
    # Canonical per-manager decision: configs/cameras/<camera_id>/homography.json
    return Path("configs") / "cameras" / camera_id / "homography.json"

def _validate_homography_json(obj: Any) -> None:
    if not isinstance(obj, dict):
        raise PipelineError("homography.json must be a JSON object")
    H = obj.get("H")
    if H is None:
        raise PipelineError("homography.json missing required field 'H'")
    if not (isinstance(H, list) and len(H) == 3 and all(isinstance(r, list) and len(r) == 3 for r in H)):
        raise PipelineError("homography.json field 'H' must be a 3x3 list")
    for r in H:
        for x in r:
            if not isinstance(x, (int, float)):
                raise PipelineError("homography.json field 'H' must contain only numbers")

def _is_identity_homography(obj: Any, *, tol: float = 1e-6) -> bool:
    """
    Best-effort identity check for a homography.json payload.
    Returns True if obj contains a numeric 3x3 'H' that is ~I.
    """
    if not isinstance(obj, dict):
        return False
    H = obj.get("H")
    if not (isinstance(H, list) and len(H) == 3 and all(isinstance(r, list) and len(r) == 3 for r in H)):
        return False
    try:
        vals = [[float(H[i][j]) for j in range(3)] for i in range(3)]
    except Exception:
        return False
    I = [[1.0, 0.0, 0.0],
         [0.0, 1.0, 0.0],
         [0.0, 0.0, 1.0]]
    for i in range(3):
        for j in range(3):
            if abs(vals[i][j] - I[i][j]) > tol:
                return False
    return True

def _is_placeholder_homography(obj: Dict[str, Any]) -> bool:
    """
    Explicit placeholder marker to support tests/onboarding without 'magic' numeric heuristics.
    """
    src = obj.get("source")
    if isinstance(src, dict) and src.get("type") == "placeholder_identity":
        return True
    return False

def ensure_homography_preflight(layout: ClipOutputLayout, *, camera_id: str, interactive: bool, config_hash: str,
                                ingest_path: Path) -> None:
    append_audit(layout, {
        "event": "homography_preflight_started",
        "timestamp": _now_ms(),
        "clip_id": layout.clip_id,
        "camera_id": camera_id,
        "config_hash": config_hash,
        "homography_path": str(_homography_path(camera_id)),
        "interactive": bool(interactive),
    })

    path = _homography_path(camera_id)
    if path.exists():
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
            _validate_homography_json(obj)
        except Exception as e:
            append_audit(layout, {
                "event": "homography_preflight_failed",
                "timestamp": _now_ms(),
                "clip_id": layout.clip_id,
                "camera_id": camera_id,
                "config_hash": config_hash,
                "error_summary": str(e),
            })
            raise

        # If the file is explicitly a placeholder or (near) identity, we treat it as "needs calibration"
        # in interactive runs, and fall through to launch the calibrator.
        if interactive:
            src = obj.get("source") if isinstance(obj, dict) else None
            src_type = src.get("type") if isinstance(src, dict) else None
            if src_type == "placeholder_identity" or _is_identity_homography(obj):
                append_audit(layout, {
                    "event": "homography_preflight_needs_calibration",
                    "timestamp": _now_ms(),
                    "clip_id": layout.clip_id,
                    "camera_id": camera_id,
                    "config_hash": config_hash,
                    "reason": "placeholder_or_identity",
                    "homography_path": str(path),
                })
                # proceed to interactive calibrator
            else:
                append_audit(layout, {
                    "event": "homography_preflight_succeeded",
                    "timestamp": _now_ms(),
                    "clip_id": layout.clip_id,
                    "camera_id": camera_id,
                    "config_hash": config_hash,
                })
                return
        else:
            append_audit(layout, {
                "event": "homography_preflight_succeeded",
                "timestamp": _now_ms(),
                "clip_id": layout.clip_id,
                "camera_id": camera_id,
                "config_hash": config_hash,
            })
            return

    if not path.exists():
        append_audit(layout, {
            "event": "homography_preflight_missing",
            "timestamp": _now_ms(),
            "clip_id": layout.clip_id,
            "camera_id": camera_id,
            "config_hash": config_hash,
            "homography_path": str(path),
        })

    if not interactive:
        raise PipelineError(
            f"Missing homography for camera '{camera_id}'. Expected: {path}. "
            "Create this file (or run the homography calibrator) and re-run."
        )

    # Launch interactive calibrator using the current clip as the video source.
    # NOTE: homography_calibrate requires a subcommand (`interactive` or `import`)
    # and interactive mode requires `--video`.
    print(f"[roll-tracker][D7] Launching homography calibrator for {camera_id} ...")
    CALIBRATOR_CMD = [
        sys.executable,
        "-m", "bjj_pipeline.tools.homography_calibrate",
        "--camera", camera_id,
        "interactive",
        "--video", str(ingest_path),
        "--mat-blueprint", str((Path("configs") / "mat_blueprint.json").resolve()),
        "--calibration-ui", "overlay_rect",
    ]
    subprocess.run(CALIBRATOR_CMD, check=True)
    print(f"[roll-tracker][D7] Homography calibration complete. Continuing pipeline ...")
    if not path.exists():
        raise PipelineError(f"Homography calibrator did not create expected file: {path}")

    obj = json.loads(path.read_text(encoding="utf-8"))
    _validate_homography_json(obj)
    append_audit(layout, {
        "event": "homography_preflight_succeeded",
        "timestamp": _now_ms(),
        "clip_id": layout.clip_id,
        "camera_id": camera_id,
        "config_hash": config_hash,
    })


def get_last_stage_success_config_hash(layout: ClipOutputLayout, stage_letter: StageLetter) -> Optional[str]:
    path = orchestration_audit_path(layout)
    if not path.exists():
        return None
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return None
    for line in reversed(lines):
        try:
            rec = json.loads(line)
        except Exception:
            continue
        if rec.get("event") == "stage_succeeded" and rec.get("stage") == stage_letter:
            return rec.get("config_hash")
    return None


def ensure_manifest(layout: ClipOutputLayout, clip_id: str, camera_id: str, input_video_path: Path, *,
                    pipeline_version: str = "dev") -> ClipManifest:
    mpath = layout.clip_manifest_path()
    audit_rel = layout.rel_to_clip_root(orchestration_audit_path(layout))
    if mpath.exists():
        manifest = load_manifest(mpath)
        try:
            manifest.get_misc_artifact_path(key="orchestration_audit_jsonl")
        except Exception:
            manifest.register_misc_artifact(
                key="orchestration_audit_jsonl",
                relpath=audit_rel,
                content_type="application/jsonl",
            )
            write_manifest(manifest, mpath)
        return manifest
    manifest = init_manifest(
        clip_id=clip_id,
        camera_id=camera_id,
        input_video_path=str(input_video_path),
        fps=0.0,
        frame_count=0,
        duration_ms=0,
        pipeline_version=pipeline_version,
        created_at_ms=_now_ms(),
    )
    manifest.register_misc_artifact(
        key="orchestration_audit_jsonl",
        relpath=audit_rel,
        content_type="application/jsonl",
    )
    write_manifest(manifest, mpath)
    return manifest


def _cfg_get(cfg: Any, path: str, default: Any = None) -> Any:
    """Lightweight dot-path getter used for validation-time config lookups."""
    cur: Any = cfg
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _validate_stage_outputs(
    manifest: ClipManifest,
    layout: ClipOutputLayout,
    letter: StageLetter,
    *,
    resolved_config: Dict[str, Any],
) -> None:
    root = layout.clip_root
    if letter == "A":
        tables = v.read_stage_A_clip_tables(layout)
        v.validate_stage_A_contract(tables)
        return
    if letter == "B":
        refined = root / "stage_B" / "contact_points_refined.parquet"
        legacy = root / "stage_B" / "contact_points.parquet"
        if refined.exists():
            cp_path = refined
        elif legacy.exists():
            cp_path = legacy
        else:
            raise PipelineError("Missing stage B contact points parquet (expected refined or legacy file)")

        cp = pd.read_parquet(cp_path)
        v.validate_contact_points_df(cp)
        return
    if letter == "C":
        # Stage C JSONL outputs are allowed to be empty (0 records) in early slices.
        # Validators should still run on any records that are present.
        expected_family = _cfg_get(resolved_config, "stages.stage_C.tag_family", "36h11")

        to_path = root / "stage_C" / "tag_observations.jsonl"
        if to_path.exists():
            records = [json.loads(line) for line in to_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            v.validate_tag_observations_records(
                records,
                expected_clip_id=manifest.clip_id,
                expected_tag_family=str(expected_family) if expected_family is not None else None,
            )

        ih_path = root / "stage_C" / "identity_hints.jsonl"
        if ih_path.exists():
            records = [json.loads(line) for line in ih_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            v.validate_identity_hints_records(records, expected_clip_id=manifest.clip_id)
        return
    if letter == "D":
        pt = pd.read_parquet(root / "stage_D" / "person_tracks.parquet")
        det = pd.read_parquet(root / "stage_A" / "detections.parquet")
        tf = pd.read_parquet(root / "stage_A" / "tracklet_frames.parquet")
        v.validate_person_tracks_df(pt)
        v.validate_person_tracks_traceability(pt, det, tf)
        return
    if letter == "E":
        ms_path = root / "stage_E" / "match_sessions.jsonl"
        if ms_path.exists():
            records = [json.loads(line) for line in ms_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            v.validate_match_sessions_records(records, expected_clip_id=manifest.clip_id)
        return
    if letter == "F":
        em_path = root / "stage_F" / "export_manifest.jsonl"
        if em_path.exists():
            records = [json.loads(line) for line in em_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            v.validate_export_manifest_records(records, expected_clip_id=manifest.clip_id)
        return


def _files_exist(layout: ClipOutputLayout, rels: List[str]) -> bool:
    for rel in rels:
        if rel.startswith("glob:"):
            pattern = rel[len("glob:") :]
            abs_pattern = layout.clip_root / pattern
            matches = list(abs_pattern.parent.glob(abs_pattern.name))
            if not matches:
                return False
            continue
        if not (layout.clip_root / rel).exists():
            return False
    return True


def _compute_stage_run_plan(manifest: ClipManifest, layout: ClipOutputLayout, letters: List[StageLetter], *,
                            cfg_hash: str, resolved_config: Dict[str, Any], force_stages: Optional[List[StageLetter]] = None) -> Dict[StageLetter, Dict[str, Any]]:
    """Compute per-stage run/skip decisions for a set of stage letters.

    This preserves existing multipass semantics:
    - If forced: run
    - If required outputs missing: run
    - If config hash matches last successful stage hash AND validation passes: skip
    - Otherwise: run (hash changed or validation failed)
    """
    plan: Dict[StageLetter, Dict[str, Any]] = {}
    forced = set(force_stages or [])
    for letter in letters:
        if letter in forced:
            plan[letter] = {"should_run": True, "is_complete": False, "reason": "forced"}
            continue
        required = required_outputs_for_stage(layout, letter)
        if not _files_exist(layout, required):
            plan[letter] = {"should_run": True, "is_complete": False, "reason": "missing_outputs"}
            continue
        last_hash = get_last_stage_success_config_hash(layout, letter)
        if last_hash != cfg_hash:
            plan[letter] = {"should_run": True, "is_complete": False, "reason": "hash_changed"}
            continue
        # Hash matches; validate to confirm completeness
        try:
            _validate_stage_outputs(manifest, layout, letter, resolved_config=resolved_config)
        except Exception:
            plan[letter] = {"should_run": True, "is_complete": False, "reason": "validate_failed"}
            continue
        plan[letter] = {"should_run": False, "is_complete": True, "reason": "resume_complete"}
    return plan


def _import_stage_run(module_path: str):
    mod = importlib.import_module(module_path)
    fn = getattr(mod, "run", None)
    if not callable(fn):
        raise PipelineError(f"stage module has no callable run(): {module_path}")
    return fn


def _resolve_inputs_for_stage(manifest: ClipManifest, layout: ClipOutputLayout, letter: StageLetter,
                              ingest_path: Path) -> Dict[str, Any]:
    if letter == "A":
        return {"clip_path": str(ingest_path), "layout": layout}
    return {"layout": layout, "manifest": manifest}


def run_pipeline(ingest_path: Path, camera_id: str, config: Dict[str, Any], *,
                 out_root: Optional[Path] = None,
                 force_stages: Optional[List[StageLetter]] = None,
                 pipeline_version: str = "dev",
                 from_stage: Optional[StageLetter] = None,
                 to_stage: Optional[StageLetter] = None,
                 interactive: bool = False,
                 mode: str = "multipass",
                 visualize: bool = False,
                 config_sources: Optional[List[str]] = None,
                 config_hash_override: Optional[str] = None) -> None:
    validate_ingest_path(ingest_path, camera_id)
    clip_id = extract_clip_id(ingest_path)
    layout = ClipOutputLayout(clip_id=clip_id, root=out_root or Path("outputs"))
    layout.clip_root.mkdir(parents=True, exist_ok=True)

    manifest = ensure_manifest(layout, clip_id, camera_id, ingest_path, pipeline_version=pipeline_version)

    # "config" is the resolved/merged dict. Create a backwards-compatible runtime
    # view for stage implementations that still expect certain top-level keys.
    resolved_config = config
    runtime_config = to_runtime_config(resolved_config, camera_id=camera_id)

    cfg_hash = config_hash_override or hash_config(resolved_config)
    run_start_ts = _now_ms()
    # Emit config resolution audit event
    try:
        append_audit(layout, {
            "event": "config_resolved",
            "event_type": "config_resolved",
            "timestamp": run_start_ts,
            "clip_id": clip_id,
            "camera_id": camera_id,
            "config_hash": cfg_hash,
            "config_sources": list(config_sources or []),
            "resolved_config": resolved_config,
            "runtime_config": runtime_config,
            "mode": mode,
            "visualize": visualize,
        })
    except Exception:
        # Do not fail the run if audit append fails for this event
        pass
    append_audit(layout, {
        "event": "run_started",
        "timestamp": run_start_ts,
        "clip_id": clip_id,
        "camera_id": camera_id,
        "config_hash": cfg_hash,
    })

    try:
        stage_list = STAGES
        if from_stage or to_stage:
            letters = [s.letter for s in STAGES]
            start_i = letters.index(from_stage) if from_stage else 0
            end_i = letters.index(to_stage) if to_stage else len(STAGES) - 1
            stage_list = STAGES[start_i: end_i + 1]

        # Enforce homography preflight when Stage A is in the window
        if any(s.letter == "A" for s in stage_list):
            ensure_homography_preflight(
                layout,
                camera_id=camera_id,
                interactive=interactive,
                config_hash=cfg_hash,
                ingest_path=ingest_path,
            )

        allowed_modes = {'multipass', 'multiplex_AC'}
        if mode not in allowed_modes:
            raise PipelineError(f"invalid mode: {mode}; expected one of {sorted(allowed_modes)}")

        # Optional single-pass multiplexer for stages A/C
        if mode == 'multiplex_AC':
            abc_letters = [s.letter for s in stage_list if s.letter in {'A','C'}]
            if abc_letters:
                from bjj_pipeline.stages.orchestration.multiplex_runner import run_multiplex_AC
                run_plan = _compute_stage_run_plan(
                    manifest,
                    layout,
                    abc_letters,
                    cfg_hash=cfg_hash,
                    resolved_config=resolved_config,
                    force_stages=force_stages,
                )
                stage_starts: Dict[StageLetter, Tuple[int, str]] = {}
                # Emit stage_started / stage_skipped events consistent with multipass behavior
                for letter in abc_letters:
                    spec0 = next(s for s in stage_list if s.letter == letter)
                    stage_start_ts0 = _now_ms()
                    stage_starts[letter] = (stage_start_ts0, spec0.key)
                    append_audit(layout, {
                        'event': 'stage_started',
                        'timestamp': stage_start_ts0,
                        'clip_id': clip_id,
                        'camera_id': camera_id,
                        'stage': letter,
                        'stage_key': spec0.key,
                        'config_hash': cfg_hash,
                    })
                    if not run_plan[letter]['should_run']:
                        append_audit(layout, {
                            'event': 'stage_skipped',
                            'timestamp': _now_ms(),
                            'clip_id': clip_id,
                            'camera_id': camera_id,
                            'stage': letter,
                            'stage_key': spec0.key,
                            'config_hash': cfg_hash,
                            'reason': run_plan[letter]['reason'],
                            'durations_ms': {'stage': _now_ms() - stage_start_ts0},
                        })

                letters_to_run = [l for l in abc_letters if run_plan[l]['should_run']]
                if letters_to_run or visualize:
                    try:
                        run_multiplex_AC(
                            ingest_path=ingest_path,
                            layout=layout,
                            manifest=manifest,
                            camera_id=camera_id,
                            runtime_config=runtime_config,
                            resolved_config=resolved_config,
                            cfg_hash=cfg_hash,
                            run_plan=run_plan,
                            visualize=visualize,
                        )
                        for letter in letters_to_run:
                            stage_start_ts0, stage_key0 = stage_starts[letter]
                            # Register canonical artifacts exactly as in multipass mode
                            if letter == 'A':
                                register_stage_A_defaults(manifest, layout)
                            elif letter == 'C':
                                manifest.register_artifact(
                                    stage='C', key='tag_observations_jsonl',
                                    relpath=layout.rel_to_clip_root(layout.tag_observations_jsonl()),
                                    content_type='application/jsonl',
                                )
                                manifest.register_artifact(
                                    stage='C', key='identity_hints_jsonl',
                                    relpath=layout.rel_to_clip_root(layout.identity_hints_jsonl()),
                                    content_type='application/jsonl',
                                )
                                manifest.register_artifact(
                                    stage='C', key='audit_jsonl',
                                    relpath=layout.rel_to_clip_root(layout.audit_jsonl('C')),
                                    content_type='application/jsonl',
                                )
                            _validate_stage_outputs(manifest, layout, letter, resolved_config=resolved_config)
                            write_manifest(manifest, layout.clip_manifest_path())

                            append_audit(layout, {
                                'event': 'stage_succeeded',
                                'timestamp': _now_ms(),
                                'clip_id': clip_id,
                                'camera_id': camera_id,
                                'stage': letter,
                                'stage_key': stage_key0,
                                'config_hash': cfg_hash,
                                'durations_ms': {'stage': _now_ms() - stage_start_ts0},
                            })
                    except Exception as e:
                        for letter in letters_to_run:
                            stage_start_ts0, stage_key0 = stage_starts[letter]
                            append_audit(layout, {
                                'event': 'stage_failed',
                                'timestamp': _now_ms(),
                                'clip_id': clip_id,
                                'camera_id': camera_id,
                                'stage': letter,
                                'stage_key': stage_key0,
                                'config_hash': cfg_hash,
                                'error_summary': str(e),
                                'durations_ms': {'stage': _now_ms() - stage_start_ts0},
                            })
                        raise

                # Remove handled stages so they are not executed again in multipass loop
                stage_list = [s for s in stage_list if s.letter not in {'A','C'}]

        for spec in stage_list:
            stage_letter = spec.letter
            stage_key = spec.key
            stage_start_ts = _now_ms()
            append_audit(layout, {
                "event": "stage_started",
                "timestamp": stage_start_ts,
                "clip_id": clip_id,
                "camera_id": camera_id,
                "stage": stage_letter,
                "stage_key": stage_key,
                "config_hash": cfg_hash,
            })
            if interactive:
                print(f"[roll-tracker] Stage {stage_letter} started ...")

            required_rels = required_outputs_for_stage(layout, stage_letter)
            last_success_hash = get_last_stage_success_config_hash(layout, stage_letter)
            should_force = force_stages and stage_letter in set(force_stages)

            is_complete = False
            try:
                if _files_exist(layout, required_rels):
                    _validate_stage_outputs(manifest, layout, stage_letter, resolved_config=resolved_config)
                    if last_success_hash == cfg_hash:
                        is_complete = True
            except Exception:
                is_complete = False

            if is_complete and not should_force:
                append_audit(layout, {
                    "event": "stage_skipped",
                    "timestamp": _now_ms(),
                    "clip_id": clip_id,
                    "camera_id": camera_id,
                    "stage": stage_letter,
                    "stage_key": stage_key,
                    "config_hash": cfg_hash,
                    "reason": "resume_complete",
                    "durations_ms": {"stage": _now_ms() - stage_start_ts},
                })
                if interactive:
                    print(f"[roll-tracker] Stage {stage_letter} skipped (resume_complete).")
                continue

            inputs = _resolve_inputs_for_stage(manifest, layout, stage_letter, ingest_path)
            try:
                run_fn = _import_stage_run(spec.module_path)
                run_fn(runtime_config, inputs)
                if stage_letter == "A":
                    register_stage_A_defaults(manifest, layout)
                elif stage_letter == "B":
                    register_stage_B_defaults(manifest, layout)
                elif stage_letter == "C":
                    manifest.register_artifact(
                        stage="C", key="tag_observations_jsonl",
                        relpath=layout.rel_to_clip_root(layout.tag_observations_jsonl()),
                        content_type="application/jsonl",
                    )
                    manifest.register_artifact(
                        stage="C", key="identity_hints_jsonl",
                        relpath=layout.rel_to_clip_root(layout.identity_hints_jsonl()),
                        content_type="application/jsonl",
                    )
                    manifest.register_artifact(
                        stage="C", key="audit_jsonl",
                        relpath=layout.rel_to_clip_root(layout.audit_jsonl("C")),
                        content_type="application/jsonl",
                    )
                elif stage_letter == "D":
                    manifest.register_artifact(
                        stage="D", key="person_tracks_parquet",
                        relpath=layout.rel_to_clip_root(layout.person_tracks_parquet()),
                        content_type="application/parquet",
                    )
                    manifest.register_artifact(
                        stage="D", key="identity_assignments_jsonl",
                        relpath=layout.rel_to_clip_root(layout.identity_assignments_jsonl()),
                        content_type="application/jsonl",
                    )
                    manifest.register_artifact(
                        stage="D", key="audit_jsonl",
                        relpath=layout.rel_to_clip_root(layout.audit_jsonl("D")),
                        content_type="application/jsonl",
                    )
                elif stage_letter == "E":
                    manifest.register_artifact(
                        stage="E", key="match_sessions_jsonl",
                        relpath=layout.rel_to_clip_root(layout.match_sessions_jsonl()),
                        content_type="application/jsonl",
                    )
                    manifest.register_artifact(
                        stage="E", key="audit_jsonl",
                        relpath=layout.rel_to_clip_root(layout.audit_jsonl("E")),
                        content_type="application/jsonl",
                    )
                elif stage_letter == "F":
                    manifest.register_artifact(
                        stage="F", key="export_manifest_jsonl",
                        relpath=layout.rel_to_clip_root(layout.export_manifest_jsonl()),
                        content_type="application/jsonl",
                    )
                    manifest.register_artifact(
                        stage="F", key="audit_jsonl",
                        relpath=layout.rel_to_clip_root(layout.audit_jsonl("F")),
                        content_type="application/jsonl",
                    )

                _validate_stage_outputs(manifest, layout, stage_letter, resolved_config=resolved_config)
                write_manifest(manifest, layout.clip_manifest_path())

                append_audit(layout, {
                    "event": "stage_succeeded",
                    "timestamp": _now_ms(),
                    "clip_id": clip_id,
                    "camera_id": camera_id,
                    "stage": stage_letter,
                    "stage_key": stage_key,
                    "config_hash": cfg_hash,
                    "durations_ms": {"stage": _now_ms() - stage_start_ts},
                })
                if interactive:
                    print(f"[roll-tracker] Stage {stage_letter} succeeded.")
            except Exception as e:
                err = {
                    "event": "stage_failed",
                    "timestamp": _now_ms(),
                    "clip_id": clip_id,
                    "camera_id": camera_id,
                    "stage": stage_letter,
                    "stage_key": stage_key,
                    "config_hash": cfg_hash,
                    "error_summary": str(e),
                    "durations_ms": {"stage": _now_ms() - stage_start_ts},
                }
                append_audit(layout, err)
                raise PipelineError(f"Stage {stage_letter}/{stage_key} failed: {e}")

        append_audit(layout, {
            "event": "run_finished",
            "timestamp": _now_ms(),
            "clip_id": clip_id,
            "camera_id": camera_id,
            "config_hash": cfg_hash,
            "durations_ms": {"run": _now_ms() - run_start_ts},
        })
    except Exception as e:
        append_audit(layout, {
            "event": "run_failed",
            "timestamp": _now_ms(),
            "clip_id": clip_id,
            "camera_id": camera_id,
            "config_hash": cfg_hash,
            "error_summary": str(e),
            "durations_ms": {"run": _now_ms() - run_start_ts},
        })
        if isinstance(e, PipelineError):
            raise
        raise PipelineError(f"Pipeline run failed: {e}")

