"""Stage F runner: Export match clips to cropped MP4 artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from bjj_pipeline.contracts.f0_models import ExportManifest, jsonl_serialize, SCHEMA_VERSION_DEFAULT
from bjj_pipeline.contracts.f0_validate import validate_export_manifest_records

from .consolidate import ExportSession, consolidate_export_sessions
from .redact import RedactionRenderError, build_redaction_plan, render_redacted_clip, summarize_redaction_plan
from .manifest import (
	build_supabase_clip_contract,
	build_supabase_log_contracts,
	compute_clip_seconds,
	derive_storage_target,
	get_file_stats,
)
from .cropper import CropPlanError, FixedRoiCropPlan, plan_crop_fixed_roi
from .ffmpeg import ExportClipError, export_clip, probe_video_metadata


def _now_ms() -> int:
	return int(time.time() * 1000)


def _stage_f_cfg(config: Dict[str, Any]) -> Dict[str, Any]:
	stages_blk = config.get("stages")
	if isinstance(stages_blk, dict):
		f_blk = stages_blk.get("stage_F")
		if isinstance(f_blk, dict):
			return f_blk
	blk = config.get("stage_F")
	return blk if isinstance(blk, dict) else {}


def _cfg_int(cfg: Dict[str, Any], key: str, default: int) -> int:
	try:
		return int(cfg.get(key, default))
	except Exception:
		return int(default)


def _cfg_float(cfg: Dict[str, Any], key: str, default: float) -> float:
	try:
		return float(cfg.get(key, default))
	except Exception:
		return float(default)


def _cfg_bool(cfg: Dict[str, Any], key: str, default: bool) -> bool:
	value = cfg.get(key, default)
	if isinstance(value, bool):
		return value
	txt = str(value).strip().lower()
	if txt in {"1", "true", "yes", "on"}:
		return True
	if txt in {"0", "false", "no", "off"}:
		return False
	return bool(default)


def _append_jsonl(path: Path, record: Dict[str, Any]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("a", encoding="utf-8") as f:
		f.write(json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n")


def _load_match_sessions(path: Path) -> List[Dict[str, Any]]:
	records: List[Dict[str, Any]] = []
	if not path.exists():
		raise FileNotFoundError(f"Stage F missing match sessions: {path}")
	with path.open("r", encoding="utf-8") as f:
		for line in f:
			line = line.strip()
			if not line:
				continue
			rec = json.loads(line)
			if isinstance(rec, dict) and rec.get("artifact_type") == "match_session":
				records.append(rec)
	records.sort(
		key=lambda r: (
			int(r.get("start_frame", 0)),
			int(r.get("end_frame", 0)),
			str(r.get("person_id_a", "")),
			str(r.get("person_id_b", "")),
			str(r.get("match_id", "")),
		)
	)
	return records


def _load_person_tracks_df(path: Path) -> pd.DataFrame:
	if not path.exists():
		raise FileNotFoundError(f"Stage F missing person tracks: {path}")
	df = pd.read_parquet(path)
	required = {"person_id", "frame_index", "x1", "y1", "x2", "y2"}
	missing = sorted(required - set(df.columns))
	if missing:
		raise ValueError(f"person_tracks.parquet missing required columns: {missing}")
	return df.sort_values(["frame_index", "person_id"], kind="mergesort").reset_index(drop=True)


def _sha256_file(path: Path) -> str:
	h = hashlib.sha256()
	with path.open("rb") as f:
		for chunk in iter(lambda: f.read(1024 * 1024), b""):
			h.update(chunk)
	return h.hexdigest()


def _infer_last_frame(video_meta: Any, fps: float, matches: List[Dict[str, Any]]) -> int:
	duration_sec = getattr(video_meta, "duration_sec", None)
	if duration_sec is not None and fps > 0.0:
		try:
			return max(0, int(math.floor(float(duration_sec) * float(fps))) - 1)
		except Exception:
			pass
	if matches:
		return max(int(match.get("end_frame", 0)) for match in matches)
	return 0


def _build_export_record(
	*,
	manifest: Any,
	input_video_path: Path,
	export_session: ExportSession,
	crop_plan: FixedRoiCropPlan,
	output_video_path: Path,
	ffmpeg_cmd: str,
	hash_sha256: str | None,
	privacy_render_applied: bool,
	n_mask_targets_applied: int,
	n_bbox_targets_applied: int,
	redaction_plan: Any,
	storage_target: Any,
	seconds_payload: Dict[str, float],
	file_size_bytes: int | None,
	clip_row_payload: Dict[str, Any],
	log_event_payloads: List[Dict[str, Any]],
	created_at_ms: int,
) -> ExportManifest:
	return ExportManifest(
		clip_id=manifest.clip_id,
		camera_id=manifest.camera_id,
		gym_id=getattr(manifest, "gym_id", None),
		pipeline_version=str(getattr(manifest, "pipeline_version", "dev")),
		created_at_ms=int(created_at_ms),
		export_id=str(export_session.export_id),
		match_id=str(export_session.source_match_ids[0]),
		output_video_path=str(output_video_path),
		crop_mode="fixed_roi",
		privacy={
			"redaction_enabled": bool(redaction_plan.enabled),
			"method": str(redaction_plan.mode) if redaction_plan.enabled else None,
		},
		inputs={
			"input_video_path": str(input_video_path),
			"person_id_a": str(export_session.person_id_a),
			"person_id_b": str(export_session.person_id_b),
			"april_tag_id_a": export_session.april_tag_id_a,
			"april_tag_id_b": export_session.april_tag_id_b,
			"match_start_frame": int(export_session.match_start_frame),
			"match_end_frame": int(export_session.match_end_frame),
			"match_start_ts_ms": int(export_session.match_start_ts_ms),
			"match_end_ts_ms": int(export_session.match_end_ts_ms),
			"export_start_frame": int(export_session.export_start_frame),
			"export_end_frame": int(export_session.export_end_frame),
			"source_match_ids": list(export_session.source_match_ids),
			"source_match_count": int(len(export_session.source_match_ids)),
			"resolved_pair_key": list(export_session.resolved_pair_key),
			"source_person_ids": list(export_session.source_person_ids),
			"gaps_merged": list(export_session.gaps_merged),
			"crop_rect_xywh": [crop_plan.x, crop_plan.y, crop_plan.width, crop_plan.height],
			"padding_px": int(crop_plan.padding_px),
			"envelope_method": str(crop_plan.envelope_method),
			"n_pair_frames": int(crop_plan.n_pair_frames),
			"privacy": summarize_redaction_plan(redaction_plan),
			"privacy_render_applied": bool(privacy_render_applied),
			"n_mask_targets_applied": int(n_mask_targets_applied),
			"n_bbox_targets_applied": int(n_bbox_targets_applied),
			"start_seconds": float(seconds_payload["start_seconds"]),
			"end_seconds": float(seconds_payload["end_seconds"]),
			"duration_seconds": float(seconds_payload["duration_seconds"]),
			"file_size_bytes": file_size_bytes,
			"storage_bucket": str(storage_target.bucket),
			"storage_object_path": str(storage_target.object_path),
			"uploader_contract": {
				"storage": {
					"bucket": str(storage_target.bucket),
					"object_path": str(storage_target.object_path),
					"file_name": str(storage_target.file_name),
				},
				"clip_row": clip_row_payload,
				"log_events": log_event_payloads,
			},
		},
		ffmpeg_cmd=ffmpeg_cmd,
		hash_sha256=hash_sha256,
		collision_hints=(
			{"same_tag_collision": True, "tag_id": int(export_session.april_tag_id_a)}
			if (export_session.april_tag_id_a is not None
				and export_session.april_tag_id_a == export_session.april_tag_id_b)
			else None
		),
	)


def run(config: Dict[str, Any], inputs: Dict[str, Any]) -> Dict[str, Any]:
	"""Orchestrator entrypoint for Stage F clip generation."""
	layout = inputs["layout"]
	manifest = inputs["manifest"]
	input_video_path = Path(str(manifest.input_video_path))

	stage_cfg = _stage_f_cfg(config)
	padding_px = _cfg_int(stage_cfg, "padding_px", 80)
	low_quantile = _cfg_float(stage_cfg, "low_quantile", 0.05)
	high_quantile = _cfg_float(stage_cfg, "high_quantile", 0.95)
	min_crop_width = _cfg_int(stage_cfg, "min_crop_width", 160)
	min_crop_height = _cfg_int(stage_cfg, "min_crop_height", 160)
	consolidate_sessions_enabled = _cfg_bool(stage_cfg, "consolidate_sessions", False)
	consolidate_max_gap_frames = _cfg_int(stage_cfg, "consolidate_max_gap_frames", 120)
	consolidate_buffer_sec = _cfg_float(stage_cfg, "consolidate_buffer_sec", 5.0)
	consolidate_require_nonconflicting_tags = _cfg_bool(
		stage_cfg, "consolidate_require_nonconflicting_tags", True
	)
	privacy_mode = str(stage_cfg.get("privacy_mode", "none"))
	redact_non_focus_people = _cfg_bool(stage_cfg, "redact_non_focus_people", False)
	redact_use_masks_when_available = _cfg_bool(stage_cfg, "redact_use_masks_when_available", True)
	redact_fallback_to_bbox = _cfg_bool(stage_cfg, "redact_fallback_to_bbox", True)
	blur_kernel_size = _cfg_int(stage_cfg, "blur_kernel_size", 31)
	gym_id = (
		str(manifest.gym_id)
		if getattr(manifest, "gym_id", None) is not None
		else stage_cfg.get("gym_id") or None
	)
	storage_bucket = str(stage_cfg.get("storage_bucket", "match-clips"))
	clip_type = str(stage_cfg.get("clip_type", "match"))
	initial_status = str(stage_cfg.get("initial_status", "exported_local"))

	layout.ensure_dirs_for_stage("F")
	layout.ensure_exports_dir()
	audit_path = layout.audit_jsonl("F")
	if audit_path.exists():
		audit_path.unlink()
	export_manifest_path = layout.export_manifest_jsonl()
	if export_manifest_path.exists():
		export_manifest_path.unlink()

	match_sessions_path = layout.match_sessions_jsonl()
	person_tracks_path = layout.person_tracks_parquet()
	missing_required = []
	if not match_sessions_path.exists():
		missing_required.append(str(match_sessions_path))
	if not person_tracks_path.exists():
		missing_required.append(str(person_tracks_path))
	if not input_video_path.exists():
		missing_required.append(str(input_video_path))

	_append_jsonl(
		audit_path,
		{
			"artifact_type": "stage_f_started",
			"created_at_ms": _now_ms(),
			"clip_id": manifest.clip_id,
			"camera_id": manifest.camera_id,
			"mode": "F_clip_export_v1",
			"inputs": {
				"match_sessions_jsonl": str(match_sessions_path),
				"person_tracks_parquet": str(person_tracks_path),
				"input_video_path": str(input_video_path),
			},
			"missing_required": missing_required,
		},
	)
	if missing_required:
		_append_jsonl(
			audit_path,
			{
				"artifact_type": "stage_f_error",
				"created_at_ms": _now_ms(),
				"clip_id": manifest.clip_id,
				"camera_id": manifest.camera_id,
				"error": "missing_required_inputs",
				"missing": missing_required,
			},
		)
		raise FileNotFoundError(f"Stage F missing required inputs: {missing_required}")

	matches = _load_match_sessions(match_sessions_path)

	if len(matches) == 0:
		_append_jsonl(
			audit_path,
			{
				"artifact_type": "stage_f_no_matches",
				"created_at_ms": _now_ms(),
				"clip_id": manifest.clip_id,
				"camera_id": manifest.camera_id,
				"message": "stage_F: no match sessions found, skipping export",
			},
		)
		# Write a no_matches record so the already-processed guard fires
		export_manifest_path.parent.mkdir(parents=True, exist_ok=True)
		with export_manifest_path.open("w", encoding="utf-8") as f:
			f.write(json.dumps({
				"schema_version": SCHEMA_VERSION_DEFAULT,
				"artifact_type": "export_manifest",
				"status": "no_matches",
				"clip_id": manifest.clip_id,
				"camera_id": manifest.camera_id,
				"pipeline_version": str(getattr(manifest, "pipeline_version", "dev")),
				"created_at_ms": _now_ms(),
			}) + "\n")
		return {
			"status": "no_matches",
			"clip_id": manifest.clip_id,
			"export_count": 0,
		}

	person_tracks_df = _load_person_tracks_df(person_tracks_path)
	video_meta = probe_video_metadata(input_video_path)
	fps = float(video_meta.fps if video_meta.fps > 0 else getattr(manifest, "fps", 0.0))
	if fps <= 0.0:
		raise ValueError("Stage F requires a positive fps from ffprobe/OpenCV or clip manifest")
	last_frame = _infer_last_frame(video_meta, fps, matches)
	buffer_frames = int(round(float(consolidate_buffer_sec) * float(fps)))
	export_sessions = consolidate_export_sessions(
		matches,
		enabled=consolidate_sessions_enabled,
		max_gap_frames=consolidate_max_gap_frames,
		buffer_frames=buffer_frames,
		last_frame=last_frame,
		require_nonconflicting_tags=consolidate_require_nonconflicting_tags,
	)

	_append_jsonl(
		audit_path,
		{
			"artifact_type": "stage_f_consolidation_summary",
			"created_at_ms": _now_ms(),
			"clip_id": manifest.clip_id,
			"camera_id": manifest.camera_id,
			"enabled": bool(consolidate_sessions_enabled),
			"input_match_sessions": int(len(matches)),
			"output_export_sessions": int(len(export_sessions)),
			"buffer_frames": int(buffer_frames),
			"buffer_sec": float(consolidate_buffer_sec),
			"max_gap_frames": int(consolidate_max_gap_frames),
			"require_nonconflicting_tags": bool(consolidate_require_nonconflicting_tags),
			"method": "resolved_pair_key_buffer_overlap_v1",
		},
	)
	for export_session in export_sessions:
		if len(export_session.source_match_ids) > 1:
			_append_jsonl(
				audit_path,
				{
					"artifact_type": "export_sessions_consolidated",
					"created_at_ms": _now_ms(),
					"clip_id": manifest.clip_id,
					"camera_id": manifest.camera_id,
					"export_id": export_session.export_id,
					"source_match_ids": list(export_session.source_match_ids),
					"source_match_count": int(len(export_session.source_match_ids)),
					"resolved_pair_key": list(export_session.resolved_pair_key),
					"match_start_frame": int(export_session.match_start_frame),
					"match_end_frame": int(export_session.match_end_frame),
					"export_start_frame": int(export_session.export_start_frame),
					"export_end_frame": int(export_session.export_end_frame),
					"gaps_merged": list(export_session.gaps_merged),
				},
			)

	export_records: List[ExportManifest] = []
	skipped_count = 0

	for export_session in export_sessions:
		export_id = str(export_session.export_id)
		try:
			crop_plan = plan_crop_fixed_roi(
				tracks_df=person_tracks_df,
				person_id_a=str(export_session.person_id_a),
				person_id_b=str(export_session.person_id_b),
				start_frame=int(export_session.export_start_frame),
				end_frame=int(export_session.export_end_frame),
				frame_width=int(video_meta.width),
				frame_height=int(video_meta.height),
				padding_px=padding_px,
				low_quantile=low_quantile,
				high_quantile=high_quantile,
				min_crop_width=min_crop_width,
				min_crop_height=min_crop_height,
			)
			redaction_plan = build_redaction_plan(
				export_session=export_session,
				crop_plan=crop_plan,
				person_tracks_df=person_tracks_df,
				privacy_mode=privacy_mode,
				redact_non_focus_people=redact_non_focus_people,
				redact_use_masks_when_available=redact_use_masks_when_available,
				redact_fallback_to_bbox=redact_fallback_to_bbox,
			)
			privacy_render_applied = bool(redaction_plan.enabled and redaction_plan.n_targets > 0)
			n_mask_targets_applied = 0
			n_bbox_targets_applied = 0

			output_abs = layout.exports_dir() / f"{export_id}.mp4"
			if privacy_render_applied:
				render_result = render_redacted_clip(
					input_video_path=input_video_path,
					output_video_path=output_abs,
					crop_plan=crop_plan,
					redaction_plan=redaction_plan,
					fps=fps,
					export_start_frame=int(export_session.export_start_frame),
					export_end_frame=int(export_session.export_end_frame),
					blur_kernel_size=blur_kernel_size,
				)
				export_cmd = "privacy_render_opencv"
				n_mask_targets_applied = int(render_result.n_mask_targets_applied)
				n_bbox_targets_applied = int(render_result.n_bbox_targets_applied)
			else:
				export_result = export_clip(
					input_video_path=input_video_path,
					output_video_path=output_abs,
					crop_plan=crop_plan,
					fps=fps,
					start_frame=int(export_session.export_start_frame),
					end_frame=int(export_session.export_end_frame),
				)
				export_cmd = export_result.ffmpeg_cmd
			file_hash = _sha256_file(output_abs)
			output_rel = Path(layout.rel_to_clip_root(output_abs))
			file_stats = get_file_stats(output_abs)
			file_size_bytes = file_stats.get("file_size_bytes")
			storage_target = derive_storage_target(
				gym_id=gym_id,
				camera_id=manifest.camera_id,
				clip_id=manifest.clip_id,
				export_id=export_session.export_id,
				storage_bucket=storage_bucket,
			)
			seconds_payload = compute_clip_seconds(
				fps=fps,
				export_start_frame=int(export_session.export_start_frame),
				export_end_frame=int(export_session.export_end_frame),
			)
			fighter_a_tag_id = export_session.april_tag_id_a
			fighter_b_tag_id = export_session.april_tag_id_b
			clip_row_payload = build_supabase_clip_contract(
				export_session=export_session,
				clip_id=manifest.clip_id,
				camera_id=manifest.camera_id,
				local_output_path=output_abs,
				storage_target=storage_target,
				clip_type=clip_type,
				initial_status=initial_status,
				fighter_a_tag_id=fighter_a_tag_id,
				fighter_b_tag_id=fighter_b_tag_id,
				seconds_payload=seconds_payload,
				pipeline_version=str(getattr(manifest, "pipeline_version", "dev")),
				crop_mode="fixed_roi",
				hash_sha256=file_hash,
				file_size_bytes=file_size_bytes,
			)
			log_event_payloads = build_supabase_log_contracts(
				export_session=export_session,
				clip_id=manifest.clip_id,
				camera_id=manifest.camera_id,
				storage_target=storage_target,
				clip_row_payload=clip_row_payload,
			)
			record = _build_export_record(
				manifest=manifest,
				input_video_path=input_video_path,
				export_session=export_session,
				crop_plan=crop_plan,
				output_video_path=output_rel,
				ffmpeg_cmd=export_cmd,
				hash_sha256=file_hash,
				privacy_render_applied=privacy_render_applied,
				n_mask_targets_applied=n_mask_targets_applied,
				n_bbox_targets_applied=n_bbox_targets_applied,
				redaction_plan=redaction_plan,
				storage_target=storage_target,
				seconds_payload=seconds_payload,
				file_size_bytes=file_size_bytes,
				clip_row_payload=clip_row_payload,
				log_event_payloads=log_event_payloads,
				created_at_ms=_now_ms(),
			)
			export_records.append(record)

			_append_jsonl(
				audit_path,
				{
					"artifact_type": "clip_exported",
					"created_at_ms": _now_ms(),
					"clip_id": manifest.clip_id,
					"camera_id": manifest.camera_id,
					"export_id": export_id,
					"source_match_ids": list(export_session.source_match_ids),
					"source_match_count": int(len(export_session.source_match_ids)),
					"person_id_a": export_session.person_id_a,
					"person_id_b": export_session.person_id_b,
					"match_start_frame": int(export_session.match_start_frame),
					"match_end_frame": int(export_session.match_end_frame),
					"export_start_frame": int(export_session.export_start_frame),
					"export_end_frame": int(export_session.export_end_frame),
					"output_video_path": str(output_rel),
					"crop_rect_xywh": [crop_plan.x, crop_plan.y, crop_plan.width, crop_plan.height],
					"privacy": summarize_redaction_plan(redaction_plan),
					"privacy_render_applied": bool(privacy_render_applied),
					"n_mask_targets_applied": int(n_mask_targets_applied),
					"n_bbox_targets_applied": int(n_bbox_targets_applied),
					"storage_bucket": str(storage_target.bucket),
					"storage_object_path": str(storage_target.object_path),
					"n_pair_frames": int(crop_plan.n_pair_frames),
				},
			)
		except (CropPlanError, ExportClipError, RedactionRenderError, FileNotFoundError, ValueError) as e:
			skipped_count += 1
			_append_jsonl(
				audit_path,
				{
					"artifact_type": "clip_skipped",
					"created_at_ms": _now_ms(),
					"clip_id": manifest.clip_id,
					"camera_id": manifest.camera_id,
					"export_id": export_id,
					"source_match_ids": list(export_session.source_match_ids),
					"reason": str(e),
				},
			)
			continue

	records_for_validation = [r.model_dump(mode="json") for r in export_records]
	validate_export_manifest_records(records_for_validation, expected_clip_id=manifest.clip_id)
	export_manifest_path.parent.mkdir(parents=True, exist_ok=True)
	with export_manifest_path.open("w", encoding="utf-8") as f:
		for record in export_records:
			f.write(jsonl_serialize(record) + "\n")

	_append_jsonl(
		audit_path,
		{
			"artifact_type": "stage_f_summary",
			"created_at_ms": _now_ms(),
			"clip_id": manifest.clip_id,
			"camera_id": manifest.camera_id,
			"n_input_match_sessions": int(len(matches)),
			"n_export_sessions": int(len(export_sessions)),
			"n_exports": int(len(export_records)),
			"n_skipped": int(skipped_count),
			"export_manifest_jsonl": str(export_manifest_path),
		},
	)

	return {
		"export_manifest_jsonl": str(export_manifest_path),
		"audit_jsonl": str(audit_path),
		"n_matches": int(len(matches)),
		"n_export_sessions": int(len(export_sessions)),
		"n_exports": int(len(export_records)),
		"n_skipped": int(skipped_count),
	}


def main() -> None:
	raise SystemExit(
		"Stage F (export) does not yet implement a standalone CLI; run via `roll-tracker`."
	)
