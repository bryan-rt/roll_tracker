from __future__ import annotations

from pathlib import Path, PurePath
from typing import List
import os


def get_clip_output_dir(outputs_root: Path, clip_id: str) -> Path:
    """Return outputs/<clip_id> without touching filesystem.

    This function does not create directories.
    """
    return (outputs_root / clip_id).resolve(strict=False)


def get_cache_dir(outputs_root: Path, clip_id: str, cache_dir_name: str = "_cache") -> Path:
    """Return outputs/<clip_id>/_cache (or custom cache dir name).

    The path is normalized and guaranteed to be under outputs/<clip_id>.
    """
    base = get_clip_output_dir(outputs_root, clip_id)
    cache_base = (base / cache_dir_name).resolve(strict=False)
    # Ensure cache_base is inside clip output dir
    base_abs = base.resolve(strict=False)
    cache_abs = cache_base
    if os.path.commonpath([str(base_abs), str(cache_abs)]) != str(base_abs):
        raise ValueError("Cache directory escapes clip output directory")
    return cache_base


def _sanitize_parts(key_parts: List[str]) -> List[str]:
    """Sanitize path segments and prevent traversal.

    - Reject absolute inputs and any segment that is '.' or '..'.
    - Flatten nested segments (e.g., 'foo/bar' -> ['foo', 'bar']).
    """
    sanitized: List[str] = []
    for raw in key_parts:
        if raw is None:
            raise ValueError("key_parts must not contain None")
        # Flatten using PurePath to split segments
        pp = PurePath(str(raw))
        if pp.is_absolute():
            raise ValueError("Absolute paths are not allowed in cache key parts")
        for seg in pp.parts:
            if seg in (".", ".."):
                raise ValueError("Path traversal segments are not allowed in cache key parts")
            if seg == "":
                continue
            sanitized.append(seg)
    return sanitized


def cache_path(
    outputs_root: Path,
    clip_id: str,
    key_parts: List[str],
    cache_dir_name: str = "_cache",
) -> Path:
    """Return a cache file/directory path under outputs/<clip_id>/_cache.

    The resulting path is normalized and verified to be contained within
    the cache directory to prevent traversal.
    """
    cache_base = get_cache_dir(outputs_root, clip_id, cache_dir_name)
    parts = _sanitize_parts(key_parts)
    candidate = cache_base.joinpath(*parts).resolve(strict=False)
    # Containment check
    cache_abs = cache_base.resolve(strict=False)
    cand_abs = candidate
    if os.path.commonpath([str(cache_abs), str(cand_abs)]) != str(cache_abs):
        raise ValueError("Cache path escapes cache directory")
    return candidate


def stage_cache_dir(outputs_root: Path, clip_id: str, stage_name: str, cache_dir_name: str = "_cache") -> Path:
    """Convenience: outputs/<clip_id>/_cache/<stage_name>.

    Stage name is treated as a single sanitized segment.
    """
    return cache_path(outputs_root, clip_id, [stage_name], cache_dir_name)


def tool_cache_dir(outputs_root: Path, clip_id: str, tool_name: str, cache_dir_name: str = "_cache") -> Path:
    """Convenience: outputs/<clip_id>/_cache/<tool_name>."""
    return cache_path(outputs_root, clip_id, [tool_name], cache_dir_name)
