"""FFmpeg and FFprobe helpers for Stage F clip exports."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2  # type: ignore

from .cropper import FixedRoiCropPlan


@dataclass(frozen=True)
class VideoMetadata:
	width: int
	height: int
	fps: float
	duration_sec: float | None


@dataclass(frozen=True)
class ExportResult:
	output_video_path: Path
	ffmpeg_cmd: str
	return_code: int


class VideoProbeError(RuntimeError):
	pass


class ExportClipError(RuntimeError):
	pass


def _parse_fps(value: str) -> float:
	txt = str(value).strip()
	if not txt:
		return 0.0
	if "/" in txt:
		num_s, den_s = txt.split("/", 1)
		try:
			num = float(num_s)
			den = float(den_s)
			return num / den if den != 0.0 else 0.0
		except Exception:
			return 0.0
	try:
		return float(txt)
	except Exception:
		return 0.0


def _probe_video_metadata_cv2(input_video_path: Path) -> VideoMetadata:
	cap = cv2.VideoCapture(str(input_video_path))
	if not cap.isOpened():
		raise VideoProbeError(f"unable to open video via OpenCV: {input_video_path}")
	try:
		width = int(round(float(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0.0)))
		height = int(round(float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0.0)))
		fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
		frame_count = int(round(float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)))
	finally:
		cap.release()
	duration_sec = (frame_count / fps) if (fps > 0 and frame_count > 0) else None
	if width <= 0 or height <= 0 or fps <= 0.0:
		raise VideoProbeError(f"invalid OpenCV video metadata for: {input_video_path}")
	return VideoMetadata(width=width, height=height, fps=fps, duration_sec=duration_sec)


def probe_video_metadata(input_video_path: Path) -> VideoMetadata:
	try:
		proc = subprocess.run(
			[
				"ffprobe",
				"-v",
				"error",
				"-select_streams",
				"v:0",
				"-show_entries",
				"stream=width,height,r_frame_rate",
				"-show_entries",
				"format=duration",
				"-of",
				"json",
				str(input_video_path),
			],
			check=True,
			capture_output=True,
			text=True,
		)
		payload = json.loads(proc.stdout or "{}")
		stream = (payload.get("streams") or [{}])[0]
		width = int(stream.get("width") or 0)
		height = int(stream.get("height") or 0)
		fps = _parse_fps(str(stream.get("r_frame_rate") or "0"))
		duration_raw = (payload.get("format") or {}).get("duration")
		duration_sec = float(duration_raw) if duration_raw not in (None, "") else None
		if width > 0 and height > 0 and fps > 0.0:
			return VideoMetadata(width=width, height=height, fps=fps, duration_sec=duration_sec)
	except Exception:
		pass
	return _probe_video_metadata_cv2(input_video_path)


def build_export_command(
	*,
	input_video_path: Path,
	output_video_path: Path,
	crop_plan: FixedRoiCropPlan,
	fps: float,
	start_frame: int,
	end_frame: int,
	video_codec: str = "libx264",
	preset: str = "veryfast",
	crf: int = 23,
) -> list[str]:
	start_sec = float(start_frame) / float(fps)
	duration_sec = max(1.0 / float(fps), float(end_frame - start_frame + 1) / float(fps))
	vf = f"crop={int(crop_plan.width)}:{int(crop_plan.height)}:{int(crop_plan.x)}:{int(crop_plan.y)}"
	return [
		"ffmpeg",
		"-y",
		"-ss",
		f"{start_sec:.6f}",
		"-i",
		str(input_video_path),
		"-t",
		f"{duration_sec:.6f}",
		"-vf",
		vf,
		"-c:v",
		str(video_codec),
		"-preset",
		str(preset),
		"-crf",
		str(int(crf)),
		"-movflags",
		"+faststart",
		str(output_video_path),
	]


def _argv_to_cmd(argv: Sequence[str]) -> str:
	return subprocess.list2cmdline([str(x) for x in argv])


def export_clip(
	*,
	input_video_path: Path,
	output_video_path: Path,
	crop_plan: FixedRoiCropPlan,
	fps: float,
	start_frame: int,
	end_frame: int,
) -> ExportResult:
	argv = build_export_command(
		input_video_path=input_video_path,
		output_video_path=output_video_path,
		crop_plan=crop_plan,
		fps=fps,
		start_frame=start_frame,
		end_frame=end_frame,
	)
	output_video_path.parent.mkdir(parents=True, exist_ok=True)
	proc = subprocess.run(argv, capture_output=True, text=True)
	if proc.returncode != 0:
		stderr_tail = (proc.stderr or "").strip()[-1200:]
		raise ExportClipError(
			f"ffmpeg export failed for {output_video_path.name}: returncode={proc.returncode} stderr={stderr_tail}"
		)
	if not output_video_path.exists():
		raise ExportClipError(f"ffmpeg completed but output file was not created: {output_video_path}")
	return ExportResult(
		output_video_path=output_video_path,
		ffmpeg_cmd=_argv_to_cmd(argv),
		return_code=int(proc.returncode),
	)
