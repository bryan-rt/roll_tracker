"""Typer-based CLI for orchestrating pipeline stages.

Commands:
- run: run full pipeline or a stage window for a clip
- stage: run a single stage for a clip
- status: print status table per stage
- validate: run validators for a stage or all
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import typer
try:
	import yaml  # type: ignore
except Exception:  # pragma: no cover
	yaml = None

from .pipeline import (
	STAGES,
	PipelineError,
	run_pipeline,
	required_outputs_for_stage,
	_validate_stage_outputs,
	_files_exist,
	get_last_stage_success_config_hash,
)
from bjj_pipeline.contracts.f0_paths import ClipOutputLayout
from bjj_pipeline.config import load_config


app = typer.Typer(name="roll-tracker")


def _load_config(camera_id: str, overlay_path: Optional[Path]) -> Tuple[Dict[str, Any], str, List[str]]:
	"""Load merged config with precedence: default -> camera -> overlay.

	Precedence (PM-locked):
	  1) configs/default.yaml
	  2) configs/cameras/<camera_id>.yaml (if present)
	  3) CLI --config overlay (YAML or JSON)

	Additionally, if present, merges:
	  - configs/cameras/<camera_id>/homography.json

	Returns (resolved_dict, cfg_hash, sources).
	"""
	# Resolve paths relative to the repository root, not CWD.
	def _find_repo_root(start: Path) -> Path:
		cur = start
		for _ in range(8):  # walk up a handful of levels
			cfg = cur / "configs" / "default.yaml"
			if cfg.exists():
				return cur
			cur = cur.parent
		return start  # fallback

	project_root = _find_repo_root(Path(__file__).resolve().parent)
	default_yaml = project_root / "configs" / "default.yaml"
	cameras_dir = project_root / "configs" / "cameras"
	camera_yaml = cameras_dir / f"{camera_id}.yaml"

	typed_cfg, resolved, cfg_hash, sources = load_config(
		default_path=default_yaml,
		camera_path=camera_yaml if camera_yaml.exists() else None,
		overlay_path=overlay_path,
		camera_id=camera_id,
		cameras_dir=cameras_dir,
	)
	return resolved, cfg_hash, sources


@app.command()
def run(
	clip: Path = typer.Option(..., help="Path to clip file under data/raw/nest/..."),
	camera: str = typer.Option(..., help="Camera ID (e.g., cam03)"),
	out: Optional[Path] = typer.Option(None, help="Outputs root override (default: outputs/)"),
	config: Optional[Path] = typer.Option(None, help="Path to config overlay (.yaml/.yml/.json)"),
	from_stage: Optional[str] = typer.Option(None, help="Stage letter to start from (A..F)"),
	to_stage: Optional[str] = typer.Option(None, help="Stage letter to end at (A..F)"),
	force: bool = typer.Option(False, help="Force rerun all stages in window"),
	force_stage: List[str] = typer.Option([], help="Force specific stage letters (repeatable)"),
	mode: str = typer.Option("multipass", help="Execution mode: multipass|multiplex_AC"),
	visualize: bool = typer.Option(
		False,
		help="Write dev-only debug videos under outputs/<clip_id>/_debug/ (multiplex mode).",
	),
	interactive: bool = typer.Option(False, help="Enable interactive calibrators (e.g., homography preflight)"),
) -> None:
	"""Run pipeline orchestration for a clip."""
	try:
		cfg, cfg_hash, cfg_sources = _load_config(camera, config)

		# Default multiplex window to A..C unless user overrides to_stage
		effective_from = from_stage if from_stage in {"A","B","C","D","E","F"} else None
		effective_to = to_stage if to_stage in {"A","B","C","D","E","F"} else None
		if mode == "multiplex_AC" and effective_to is None:
			effective_to = "C"

		fl = [s for s in force_stage]
		fl_letters = [s for s in fl if s in {"A","B","C","D","E","F"}]
		# If --force is set without specific stages, force the current window
		if force and not fl_letters:
			letters = [s.letter for s in STAGES]
			start_i = letters.index(effective_from) if effective_from in letters else 0
			end_i = letters.index(effective_to) if effective_to in letters else len(letters) - 1
			fl_letters = letters[start_i: end_i + 1]
		run_pipeline(
			ingest_path=clip,
			camera_id=camera,
			config=cfg,
			out_root=out,
			force_stages=fl_letters or None,
			interactive=interactive,
			config_sources=cfg_sources,
			config_hash_override=cfg_hash,
			from_stage=effective_from,
			to_stage=effective_to,
			mode=mode,
			visualize=visualize,
		)
		return None
	except PipelineError as e:
		typer.echo(f"[roll-tracker] ERROR: {e}")
		raise typer.Exit(code=2)
	except typer.Exit:
		# allow explicit exits to pass through
		raise
	except Exception as e:
		typer.echo(f"[roll-tracker] FATAL: {e}")
		raise typer.Exit(code=1)


@app.command()
def stage(
	clip: Path = typer.Option(..., help="Path to clip file under data/raw/nest/..."),
	camera: str = typer.Option(..., help="Camera ID (e.g., cam03)"),
	stage: str = typer.Option(..., help="Stage key or letter (detect_track|masks|tags|stitch|matches|export or A..F)"),
	out: Optional[Path] = typer.Option(None, help="Outputs root override (default: outputs/)"),
	config: Optional[Path] = typer.Option(None, help="Path to config overlay (.yaml/.yml/.json)"),
	force: bool = typer.Option(False, help="Force rerun even if complete"),
	interactive: bool = typer.Option(False, help="Enable interactive calibrators (e.g., homography preflight)"),
) -> None:
	"""Run a single stage for a clip."""
	letter_map = {s.key: s.letter for s in STAGES}
	letter = stage if stage in {"A","B","C","D","E","F"} else letter_map.get(stage)
	if letter is None:
		typer.echo(f"Unknown stage: {stage}")
		raise typer.Exit(code=2)
	cfg, cfg_hash, cfg_sources = _load_config(camera, config)
	try:
		run_pipeline(
			ingest_path=clip,
			camera_id=camera,
			config=cfg,
			out_root=out,
			force_stages=[letter] if force else None,
			interactive=interactive,
			config_sources=cfg_sources,
			config_hash_override=cfg_hash,
			from_stage=letter,
			to_stage=letter,
		)
		return None
	except PipelineError as e:
		typer.echo(f"[roll-tracker] ERROR: {e}")
		raise typer.Exit(code=2)
	except typer.Exit:
		raise
	except Exception as e:
		typer.echo(f"[roll-tracker] FATAL: {e}")
		raise typer.Exit(code=1)


@app.command()
def status(
	clip: Path = typer.Option(..., help="Path to clip file under data/raw/nest/..."),
	camera: str = typer.Option(..., help="Camera ID (e.g., cam03)"),
	out: Optional[Path] = typer.Option(None, help="Outputs root override (default: outputs/)"),
) -> None:
	"""Print human-readable status table for all stages."""
	cfg, _, _ = _load_config(camera, None)
	layout = ClipOutputLayout(clip_id=clip.stem, root=out or Path("outputs"))
	typer.echo(f"clip_id={clip.stem} outputs={layout.clip_root}")
	header = ["stage", "complete", "validated", "last_success_ts", "last_config_hash"]
	rows: List[List[str]] = []
	for spec in STAGES:
		rels = required_outputs_for_stage(layout, spec.letter, resolved_config=cfg)
		complete = _files_exist(layout, rels)
		validated = False
		last_hash = get_last_stage_success_config_hash(layout, spec.letter) or ""
		last_ts = ""
		# scan audit to get last success ts
		audit = (layout.clip_root / "orchestration_audit.jsonl")
		if audit.exists():
			try:
				for line in reversed(audit.read_text(encoding="utf-8").splitlines()):
					rec = json.loads(line)
					if rec.get("event") == "stage_succeeded" and rec.get("stage") == spec.letter:
						last_ts = str(rec.get("timestamp", ""))
						break
			except Exception:
				pass
		# try validation if complete
		if complete:
			try:
				# cfg already loaded above
				# minimal manifest stub for validators that need clip_id
				from bjj_pipeline.contracts.f0_manifest import init_manifest
				m = init_manifest(
					clip_id=clip.stem,
					camera_id=camera,
					input_video_path=str(clip),
					fps=0.0,
					frame_count=0,
					duration_ms=0,
					pipeline_version="dev",
					created_at_ms=0,
				)
				_validate_stage_outputs(m, layout, spec.letter, resolved_config=cfg)
				validated = True
			except Exception:
				validated = False
		rows.append([
			spec.letter,
			"yes" if complete else "no",
			"yes" if validated else "no",
			last_ts,
			last_hash,
		])

	# Print table
	print("\nStatus for clip:", clip.stem)
	print(" ".join(h.ljust(18) for h in header))
	for r in rows:
		print(" ".join(str(c).ljust(18) for c in r))
	return None


@app.command()
def validate(
	clip: Path = typer.Option(..., help="Path to clip file under data/raw/nest/..."),
	camera: str = typer.Option(..., help="Camera ID (e.g., cam03)"),
	out: Optional[Path] = typer.Option(None, help="Outputs root override (default: outputs/)"),
	stage: Optional[str] = typer.Option(None, help="Stage letter to validate (A..F). If omitted, validate all (Stage B optional if missing)."),
) -> None:
	"""Run validators for a specific stage or all stages."""
	layout = ClipOutputLayout(clip_id=clip.stem, root=out or Path("outputs"))
	cfg, _, _ = _load_config(camera, None)
	from bjj_pipeline.contracts.f0_manifest import init_manifest
	m = init_manifest(
		clip_id=clip.stem,
		camera_id=camera,
		input_video_path=str(clip),
		fps=0.0,
		frame_count=0,
		duration_ms=0,
		pipeline_version="dev",
		created_at_ms=0,
	)
	try:
		letters = [stage] if stage in {"A","B","C","D","E","F"} else [s.letter for s in STAGES]
		for letter in letters:
			rels = required_outputs_for_stage(layout, letter, resolved_config=cfg)
			if stage is None and letter == "B" and not _files_exist(layout, rels):
				# Stage B is deferred; do not fail global validate when B outputs are absent
				continue
			if not _files_exist(layout, rels):
				raise PipelineError(f"Missing required outputs for stage {letter}: {rels}")
			_validate_stage_outputs(m, layout, letter, resolved_config=cfg)
		print("Validation OK")
		return None
	except Exception as e:
		print(f"Validation failed: {e}")
		raise typer.Exit(code=2)


def main(argv: List[str] | None = None) -> int:
	# Allow bin/run_pipeline.py to call into this entry
	return app(standalone_mode=False)

if __name__ == "__main__":
    raise SystemExit(main())
