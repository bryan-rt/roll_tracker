from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import pandas as pd


@dataclass
class ClipContext:
	clip_root: Path
	clip_id: str
	camera_id: str
	stage_c_dir: Path
	identity_hints_path: Path
	bank_frames_json: Path | None


@dataclass
class TagAssignment:
	tag_id: str
	frame_index: Optional[int] = None


def _infer_clip_and_camera(clip_root: Path) -> Tuple[str, str]:
	"""Best-effort inference of clip_id and camera_id from path.

	clip_id: final path component (e.g., cam03-20260103-124000_0-30s)
	camera_id: prefix before first '-' in clip_id (e.g., cam03)
	"""
	clip_id = clip_root.name
	camera_id = clip_id.split("-")[0] if "-" in clip_id else "unknown"
	return clip_id, camera_id


def _load_bank_frames_json(clip_root: Path) -> Path | None:
	stage_d = clip_root / "stage_D"
	json_path = stage_d / "tracklet_bank_frames.json"
	if json_path.exists():
		return json_path
	parquet_path = stage_d / "tracklet_bank_frames.parquet"
	if parquet_path.exists():
		# Create a sidecar JSON once for convenience.
		df = pd.read_parquet(parquet_path)
		records = df.to_dict(orient="records")
		json_path.parent.mkdir(parents=True, exist_ok=True)
		json_path.write_text(json.dumps(records, sort_keys=True, indent=2), encoding="utf-8")
		return json_path
	return None


def _gather_clip_context(clip_root: Path) -> ClipContext:
	clip_root = clip_root.resolve()
	stage_c_dir = clip_root / "stage_C"
	identity_hints_path = stage_c_dir / "identity_hints.jsonl"
	clip_id, camera_id = _infer_clip_and_camera(clip_root)
	bank_frames_json = _load_bank_frames_json(clip_root)
	return ClipContext(
		clip_root=clip_root,
		clip_id=clip_id,
		camera_id=camera_id,
		stage_c_dir=stage_c_dir,
		identity_hints_path=identity_hints_path,
		bank_frames_json=bank_frames_json,
	)


def _parse_mapping_arg(raw: str | None) -> Dict[str, TagAssignment]:
	"""Parse mapping like 't1:23,t4:42' or 't1:23:120' into
	{"t1": TagAssignment(tag_id="23", frame_index=None), ...}.

	If a third colon-separated field is present, it is interpreted as
	"frame_index" (integer). If raw is None, returns empty dict.
	"""
	mapping: Dict[str, TagAssignment] = {}
	if not raw:
		return mapping
	parts = [p.strip() for p in raw.split(",") if p.strip()]
	for part in parts:
		pieces = [x.strip() for x in part.split(":")]
		if len(pieces) not in (2, 3):
			raise ValueError(
				f"Invalid mapping entry (expected tid:tag_id or tid:tag_id:frame_index): {part!r}"
			)
		tid, tag = pieces[0], pieces[1]
		frame_index: Optional[int]
		if len(pieces) == 3 and pieces[2]:
			try:
				frame_index = int(pieces[2])
			except ValueError as e:
				raise ValueError(
					f"Invalid frame_index in mapping entry (must be int): {part!r}"
				) from e
		else:
			frame_index = None
		if not tid or not tag:
			raise ValueError(f"Invalid mapping entry (empty tid/tag_id): {part!r}")
		mapping[tid] = TagAssignment(tag_id=tag, frame_index=frame_index)
	return mapping


def _load_mapping_json(path: Path | None) -> Dict[str, TagAssignment]:
	if path is None:
		return {}
	data = json.loads(path.read_text(encoding="utf-8"))
	if isinstance(data, dict):
		# assume {tid: tag_id} (no frame index information)
		return {str(k): TagAssignment(tag_id=str(v), frame_index=None) for k, v in data.items()}
	if isinstance(data, list):
		out: Dict[str, TagAssignment] = {}
		for item in data:
			if not isinstance(item, dict):
				continue
			tid = item.get("tracklet_id") or item.get("tid")
			tag = item.get("tag_id") or item.get("tag")
			if not (tid and tag):
				continue
			# Optional frame index in a few plausible keys
			frame_val = (
				item.get("frame_index")
				or item.get("frame")
				or item.get("decoded_frame_index")
			)
			frame_index: Optional[int] = None
			if frame_val is not None:
				try:
					frame_index = int(frame_val)
				except (TypeError, ValueError):
					frame_index = None
			out[str(tid)] = TagAssignment(tag_id=str(tag), frame_index=frame_index)
		return out
	raise ValueError("Unsupported mapping JSON format; expected dict or list of objects.")


def _load_existing_identity_hints(path: Path) -> List[Dict[str, Any]]:
	if not path.exists():
		return []
	rows: List[Dict[str, Any]] = []
	with path.open("r", encoding="utf-8") as f:
		for line in f:
			line = line.strip()
			if not line:
				continue
			try:
				rows.append(json.loads(line))
			except Exception:
				continue
	return rows


def _infer_metadata_from_existing(rows: List[Dict[str, Any]], ctx: ClipContext) -> Tuple[str, str, str]:
	"""Infer clip_id, camera_id, pipeline_version from existing hints if available."""
	clip_id = ctx.clip_id
	camera_id = ctx.camera_id
	pipeline_version = "dev"
	for r in rows:
		clip_id = str(r.get("clip_id", clip_id))
		camera_id = str(r.get("camera_id", camera_id))
		pipeline_version = str(r.get("pipeline_version", pipeline_version))
		break
	return clip_id, camera_id, pipeline_version


def _load_bank_frames(ctx: ClipContext) -> pd.DataFrame | None:
	if ctx.bank_frames_json and ctx.bank_frames_json.exists():
		try:
			records = json.loads(ctx.bank_frames_json.read_text(encoding="utf-8"))
			if isinstance(records, list) and records:
				return pd.DataFrame.from_records(records)
		except Exception:
			return None
	return None


def _summarize_tid_frames(bank_df: pd.DataFrame) -> Dict[str, Tuple[int, int]]:
	out: Dict[str, Tuple[int, int]] = {}
	if bank_df is None or bank_df.empty:
		return out
	if "tracklet_id" not in bank_df.columns or "frame_index" not in bank_df.columns:
		return out
	for tid, grp in bank_df.groupby("tracklet_id"):
		try:
			frames = grp["frame_index"].astype(int)
			out[str(tid)] = (int(frames.min()), int(frames.max()))
		except Exception:
			continue
	return out


def _build_synthetic_hints(
	*,
	ctx: ClipContext,
	mapping: Dict[str, TagAssignment],
	confidence: float,
	existing: List[Dict[str, Any]],
	bank_summary: Dict[str, Tuple[int, int]],
	overwrite: bool,
) -> List[Dict[str, Any]]:
	base_clip_id, base_camera_id, pipeline_version = _infer_metadata_from_existing(existing, ctx)

	# If not overwriting, keep non-synthetic hints as-is.
	out: List[Dict[str, Any]] = [] if overwrite else list(existing)

	for tid, assignment in mapping.items():
		tag_id = assignment.tag_id
		frame_index = assignment.frame_index
		start_end = bank_summary.get(tid)
		if frame_index is not None:
			frames = [int(frame_index)]
		elif start_end is not None:
			start_f, end_f = start_end
			frames = [int(start_f), int(end_f)]
		else:
			frames = []

		rec: Dict[str, Any] = {
			"schema_version": "0.3.0",
			"artifact_type": "identity_hint",
			"clip_id": base_clip_id,
			"camera_id": base_camera_id,
			"pipeline_version": pipeline_version,
			"created_at_ms": 0,
			"tracklet_id": str(tid),
			"anchor_key": f"tag:{tag_id}",
			"constraint": "must_link",
			"confidence": float(confidence),
			"evidence": {
				"reason": "synthetic_tag",
				"tag_id": str(tag_id),
				"source": "inject_synthetic_tags",
				"frames": frames,
			},
		}
		if frame_index is not None:
			# Make the decoded frame explicit for downstream tools that care.
			rec["evidence"]["frame_index"] = int(frame_index)
		out.append(rec)

	return out


def _write_identity_hints(path: Path, rows: List[Dict[str, Any]]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", encoding="utf-8") as f:
		for r in rows:
			f.write(json.dumps(r, sort_keys=True) + "\n")


def main(argv: List[str] | None = None) -> int:
	parser = argparse.ArgumentParser(
		description=(
			"Inject synthetic identity_hint records for AprilTags into a clip's Stage C artifacts "
			"so Stage D stitching behaves as if tags were decoded."
		),
	)
	parser.add_argument(
		"--clip-root",
		required=True,
		type=Path,
		help="Path to outputs/<clip_id> directory (e.g., outputs/cam03-...)",
	)
	parser.add_argument(
		"--mapping",
		type=str,
		default=None,
		help=(
			"Comma-separated tid:tag_id or tid:tag_id:frame_index pairs "
			"(e.g., 't1:23,t4:42' or 't1:23:120')"
		),
	)
	parser.add_argument(
		"--mapping-json",
		type=Path,
		default=None,
		help=(
			"Optional JSON file describing mapping: either {tid: tag_id} or "
			"list of objects with tracklet_id/tag_id[/frame_index]"
		),
	)
	parser.add_argument(
		"--confidence",
		type=float,
		default=0.95,
		help="Confidence to assign to synthetic must_link hints (0-1)",
	)
	parser.add_argument(
		"--overwrite",
		action="store_true",
		help="Overwrite existing identity_hints.jsonl instead of appending to it",
	)
	parser.add_argument(
		"--dry-run",
		action="store_true",
		help="Print planned changes without writing files",
	)

	args = parser.parse_args(argv)

	ctx = _gather_clip_context(args.clip_root)

	mapping_cli = _parse_mapping_arg(args.mapping)
	mapping_json = _load_mapping_json(args.mapping_json) if args.mapping_json else {}
	mapping: Dict[str, TagAssignment] = {}
	mapping.update(mapping_json)
	mapping.update(mapping_cli)

	if not mapping:
		print("No mapping provided; nothing to inject.", file=sys.stderr)
		return 1

	bank_df = _load_bank_frames(ctx)
	bank_summary = _summarize_tid_frames(bank_df) if bank_df is not None else {}

	existing = _load_existing_identity_hints(ctx.identity_hints_path)
	rows = _build_synthetic_hints(
		ctx=ctx,
		mapping=mapping,
		confidence=args.confidence,
		existing=existing,
		bank_summary=bank_summary,
		overwrite=args.overwrite,
	)

	print(f"Injecting {len(rows) - (0 if args.overwrite else len(existing))} synthetic identity_hint records "
		f"for {len(mapping)} tracklets into {ctx.identity_hints_path} (dry_run={args.dry_run})")

	if args.dry_run:
		for rec in rows:
			print(json.dumps(rec, sort_keys=True))
		return 0

	_write_identity_hints(ctx.identity_hints_path, rows)
	print(f"Wrote {len(rows)} identity_hint records to {ctx.identity_hints_path}")
	return 0


if __name__ == "__main__":  # pragma: no cover
	sys.exit(main())
