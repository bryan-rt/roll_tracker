from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import hashlib
import json
import yaml

from .models import PipelineConfig


def deep_merge(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    """Deep-merge two dicts without mutating inputs.

    - Dicts: merge recursively
    - Lists: replaced by overlay (no concatenation)
    - Scalars/others: overlay wins
    """
    def _merge(a: Any, b: Any) -> Any:
        if isinstance(a, dict) and isinstance(b, dict):
            result: Dict[str, Any] = {}
            # keys from both, overlay wins when conflict
            for key in set(a.keys()) | set(b.keys()):
                if key in a and key in b:
                    result[key] = _merge(a[key], b[key])
                elif key in b:
                    result[key] = b[key]
                else:
                    result[key] = a[key]
            return result
        # lists: replace
        if isinstance(b, list):
            return list(b)
        # overlay wins otherwise
        return b if b is not None else a

    # copy base to avoid mutation
    base_copy = json.loads(json.dumps(base)) if base is not None else {}
    overlay_copy = json.loads(json.dumps(overlay)) if overlay is not None else {}
    return _merge(base_copy, overlay_copy)


def load_yaml(path: Path) -> Dict[str, Any]:
    """Load a YAML file and return a dict. Raises on missing or invalid type.

    Tabs in YAML cause parsing errors; normalize tabs to spaces for robustness.
    """
    if not path.exists():
        raise FileNotFoundError(f"YAML not found: {path}")
    text = path.read_text(encoding="utf-8")
    # Normalize tabs to two spaces to avoid YAML parser errors
    if "\t" in text:
        text = text.replace("\t", "  ")
    data = yaml.safe_load(text) or {}
    if not isinstance(data, dict):
        raise TypeError(f"YAML at {path} must be a mapping/dict")
    return data


def config_hash(resolved_dict: Dict[str, Any]) -> str:
    """Deterministic SHA256 of canonical JSON (sorted keys)."""
    # Use separators to avoid whitespace variance; sort keys for determinism.
    canonical = json.dumps(resolved_dict, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def to_runtime_config(resolved_dict: Dict[str, Any], *, camera_id: Optional[str] = None) -> Dict[str, Any]:
    """Create a backwards-compatible runtime config dict.

    The codebase historically expects some keys at the top level (e.g.
    config["camera_id"]). F2 introduces a typed/nested schema (e.g.
    config["camera"]["camera_id"]). To avoid churn across stages/tests,
    we synthesize a runtime view that preserves the nested structure while
    adding compatibility shims.

    This function is deterministic and side-effect free.
    """
    runtime: Dict[str, Any] = json.loads(json.dumps(resolved_dict or {}))

    cam_blk = runtime.get("camera")
    cam_id: Optional[str] = None
    if isinstance(cam_blk, dict):
        cam_id = cam_blk.get("camera_id")
    cam_id = cam_id or camera_id
    if cam_id:
        runtime.setdefault("camera_id", cam_id)

    # Back-compat for stage configs: some code may still look for stage_A keys
    # at the top level even if YAML groups them under "stages".
    stages_blk = runtime.get("stages")
    if isinstance(stages_blk, dict):
        for k, v in stages_blk.items():
            if k in {"stage_A", "stage_B", "stage_C", "stage_D", "stage_E", "stage_F"} and isinstance(v, dict):
                existing = runtime.get(k)
                if isinstance(existing, dict):
                    runtime[k] = deep_merge(existing, v)
                else:
                    runtime[k] = v

    return runtime


def apply_camera_id(resolved_dict: Dict[str, Any], camera_id: str) -> Dict[str, Any]:
    """Canonicalize camera_id in both nested and top-level views.

    Ensures:
      - resolved_dict["camera_id"] == camera_id
      - resolved_dict["camera"]["camera_id"] == camera_id (creating camera dict if needed)

    Returns a new dict (does not mutate input).
    """
    out: Dict[str, Any] = json.loads(json.dumps(resolved_dict or {}))
    out["camera_id"] = camera_id
    cam_blk = out.get("camera")
    if not isinstance(cam_blk, dict):
        cam_blk = {}
        out["camera"] = cam_blk
    cam_blk["camera_id"] = camera_id
    return out


def _load_homography_json(path: Path) -> Dict[str, Any]:
    """Load homography JSON and convert it into a config overlay dict.

    Expected shape: {"H": [[...],[...],[...]]}
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"homography.json at {path} must be a JSON object")
    H = data.get("H")
    if not isinstance(H, list):
        raise TypeError(f"homography.json at {path} must contain key 'H' as a 3x3 list")
    return {"camera": {"homography": H}}


def load_config(
    default_path: Path,
    camera_path: Optional[Path] = None,
    overlay_path: Optional[Path] = None,
    *,
    camera_id: Optional[str] = None,
    cameras_dir: Optional[Path] = None,
) -> Tuple[PipelineConfig, Dict[str, Any], str, List[str]]:
    """
    Load and deep-merge configuration in precedence order:
    default -> camera -> overlay.

    Returns (typed_config, resolved_dict, cfg_hash, sources).
    """
    sources: List[str] = []

    # Required default
    default_dict = load_yaml(default_path)
    sources.append(str(default_path))

    resolved: Dict[str, Any] = default_dict

    # Camera layer: required if provided path (callers decide presence)
    if camera_path is not None:
        if not camera_path.exists():
            raise FileNotFoundError(f"Camera config not found: {camera_path}")
        cam_dict = load_yaml(camera_path)
        resolved = deep_merge(resolved, cam_dict)
        sources.append(str(camera_path))

    # Optional overlay (YAML or JSON)
    if overlay_path is not None:
        if not overlay_path.exists():
            raise FileNotFoundError(f"Overlay config not found: {overlay_path}")
        suffix = overlay_path.suffix.lower()
        if suffix == ".json":
            data = json.loads(overlay_path.read_text(encoding="utf-8")) or {}
            if not isinstance(data, dict):
                raise TypeError(f"JSON config at {overlay_path} must be an object/dict")
            overlay_dict = data
        else:
            overlay_dict = load_yaml(overlay_path)
        resolved = deep_merge(resolved, overlay_dict)
        sources.append(str(overlay_path))

    # Optional per-camera homography JSON overlay.
    # We support: configs/cameras/<camera_id>/homography.json
    # This is kept here (not just CLI) so behavior is consistent across all callers.
    if camera_id:
        # Determine cameras_dir if not provided
        cam_root = cameras_dir
        if cam_root is None:
            if camera_path is not None:
                cam_root = camera_path.parent
            else:
                cam_root = default_path.parent / "cameras"
        homography_json = cam_root / camera_id / "homography.json"
        if homography_json.exists():
            try:
                overlay = _load_homography_json(homography_json)
            except Exception as e:
                raise RuntimeError(f"Failed to load homography JSON: {homography_json}: {e}") from e
            resolved = deep_merge(resolved, overlay)
            sources.append(str(homography_json))

        # Canonicalize camera id in resolved config for determinism/audit clarity.
        resolved = apply_camera_id(resolved, camera_id)

    # Build typed model from a flattened copy (keep original resolved intact)
    resolved_for_model = json.loads(json.dumps(resolved))
    stages_blk = resolved_for_model.get("stages")
    if isinstance(stages_blk, dict):
        for k, v in stages_blk.items():
            if k in {"stage_A", "stage_B", "stage_C", "stage_D", "stage_E", "stage_F"}:
                existing = resolved_for_model.get(k, {})
                if isinstance(existing, dict) and isinstance(v, dict):
                    resolved_for_model[k] = deep_merge(existing, v)
                else:
                    resolved_for_model[k] = v
        # Remove nested block for model validation to satisfy extra=forbid
        resolved_for_model.pop("stages", None)

    # Strip unknown top-level keys for typed validation (CLI overlays may include extra keys).
    allowed_top = {"paths", "compute", "camera", "stage_A", "stage_B", "stage_C", "stage_D", "stage_E", "stage_F"}
    for k in list(resolved_for_model.keys()):
        if k not in allowed_top:
            resolved_for_model.pop(k, None)

    typed = PipelineConfig.model_validate(resolved_for_model)

    # Compute deterministic hash from resolved dict
    cfg_hash = config_hash(resolved)

    return typed, resolved, cfg_hash, sources
