"""Sidecar timing reader — schema v5.

Codes against docs/reference/sidecar_contract.md (schema 5, CP-R13b).
Two entry points:

  parse_sidecar(path)  — permissive about schema version and validity mode (accepts
      schema 4, source_pts=false, cfr_grid — all queryable). Strict about structural
      malformation (missing always-present fields, non-contiguous frame_index, unparseable
      rows — raises regardless of schema).
  load_sidecar(mp4_path) — strict about everything: schema 5 + passthrough + source_pts,
      plus all structural checks from parse_sidecar.

Schema-4 policy: parse_sidecar accepts (for tooling like probe_frame_index_join.py).
load_sidecar refuses with SidecarSchemaError. Schema-4 footage remains valid at the gap
level and as a regression baseline — refusing it in the pipeline reader is a policy about
the timing path, not a statement that the footage has no value.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class SidecarError(Exception):
    """Sidecar file missing or unreadable."""


class SidecarSchemaError(SidecarError):
    """Sidecar schema version not supported by this reader."""


class SidecarMalformedError(SidecarError):
    """Sidecar is structurally malformed — missing required fields, non-contiguous
    frame_index, or unparseable rows. Distinct from old-schema or wrong-mode: a
    malformed file cannot be used by anyone, not just the pipeline."""


class SidecarValidityError(SidecarError):
    """A required validity gate is not met (names the gate)."""


# ---------------------------------------------------------------------------
# Type-safe field readers
# ---------------------------------------------------------------------------

def _require_int(meta: dict, key: str, path: Path) -> int:
    if key not in meta:
        raise SidecarMalformedError(
            f"Sidecar malformed: required _meta field missing "
            f"(field={key!r}, value=<missing>, path={path})"
        )
    raw = meta[key]
    if isinstance(raw, bool):
        raise SidecarMalformedError(
            f"Sidecar malformed: _meta field must be int, got bool "
            f"(field={key!r}, value={raw!r}, path={path})"
        )
    if not isinstance(raw, (int, float)):
        raise SidecarMalformedError(
            f"Sidecar malformed: cannot convert _meta field to int "
            f"(field={key!r}, value={str(raw)[:50]!r}, path={path})"
        )
    if isinstance(raw, float) and raw != int(raw):
        raise SidecarMalformedError(
            f"Sidecar malformed: _meta field is non-integral float "
            f"(field={key!r}, value={raw!r}, path={path})"
        )
    return int(raw)


def _require_float(meta: dict, key: str, path: Path) -> float:
    if key not in meta:
        raise SidecarMalformedError(
            f"Sidecar malformed: required _meta field missing "
            f"(field={key!r}, value=<missing>, path={path})"
        )
    raw = meta[key]
    if isinstance(raw, bool):
        raise SidecarMalformedError(
            f"Sidecar malformed: _meta field must be numeric, got bool "
            f"(field={key!r}, value={raw!r}, path={path})"
        )
    if not isinstance(raw, (int, float)):
        raise SidecarMalformedError(
            f"Sidecar malformed: cannot convert _meta field to float "
            f"(field={key!r}, value={str(raw)[:50]!r}, path={path})"
        )
    return float(raw)


def _require_bool(meta: dict, key: str, path: Path) -> bool:
    if key not in meta:
        raise SidecarMalformedError(
            f"Sidecar malformed: required _meta field missing "
            f"(field={key!r}, value=<missing>, path={path})"
        )
    raw = meta[key]
    if not isinstance(raw, bool):
        raise SidecarMalformedError(
            f"Sidecar malformed: _meta field must be bool (true/false), got {type(raw).__name__} "
            f"(field={key!r}, value={str(raw)[:50]!r}, path={path})"
        )
    return raw


def _require_str(meta: dict, key: str, path: Path) -> str:
    if key not in meta:
        raise SidecarMalformedError(
            f"Sidecar malformed: required _meta field missing "
            f"(field={key!r}, value=<missing>, path={path})"
        )
    raw = meta[key]
    if not isinstance(raw, str):
        raise SidecarMalformedError(
            f"Sidecar malformed: _meta field must be string, got {type(raw).__name__} "
            f"(field={key!r}, value={str(raw)[:50]!r}, path={path})"
        )
    return raw


def _optional_int(meta: dict, key: str, path: Path) -> Optional[int]:
    if key not in meta:
        return None
    raw = meta[key]
    if isinstance(raw, bool):
        raise SidecarMalformedError(
            f"Sidecar malformed: _meta field must be int, got bool "
            f"(field={key!r}, value={raw!r}, path={path})"
        )
    if not isinstance(raw, (int, float)):
        raise SidecarMalformedError(
            f"Sidecar malformed: cannot convert _meta field to int "
            f"(field={key!r}, value={str(raw)[:50]!r}, path={path})"
        )
    return int(raw)


def _optional_float(meta: dict, key: str, path: Path) -> Optional[float]:
    if key not in meta:
        return None
    raw = meta[key]
    if isinstance(raw, bool):
        raise SidecarMalformedError(
            f"Sidecar malformed: _meta field must be numeric, got bool "
            f"(field={key!r}, value={raw!r}, path={path})"
        )
    if not isinstance(raw, (int, float)):
        raise SidecarMalformedError(
            f"Sidecar malformed: cannot convert _meta field to float "
            f"(field={key!r}, value={str(raw)[:50]!r}, path={path})"
        )
    return float(raw)


def _optional_bool(meta: dict, key: str, path: Path) -> Optional[bool]:
    if key not in meta:
        return None
    raw = meta[key]
    if not isinstance(raw, bool):
        raise SidecarMalformedError(
            f"Sidecar malformed: _meta field must be bool (true/false), got {type(raw).__name__} "
            f"(field={key!r}, value={str(raw)[:50]!r}, path={path})"
        )
    return raw


def _optional_str(meta: dict, key: str, path: Path) -> Optional[str]:
    if key not in meta:
        return None
    raw = meta[key]
    if not isinstance(raw, str):
        raise SidecarMalformedError(
            f"Sidecar malformed: _meta field must be string, got {type(raw).__name__} "
            f"(field={key!r}, value={str(raw)[:50]!r}, path={path})"
        )
    return raw


def _require_row_int(row: dict, key: str, line_num: int, path: Path) -> int:
    if key not in row:
        raise SidecarMalformedError(
            f"Sidecar malformed: required frame-row field missing "
            f"(field={key!r}, row={line_num}, value=<missing>, path={path})"
        )
    raw = row[key]
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise SidecarMalformedError(
            f"Sidecar malformed: cannot convert frame-row field to int "
            f"(field={key!r}, row={line_num}, value={str(raw)[:50]!r}, path={path})"
        )
    return int(raw)


def _require_row_float(row: dict, key: str, line_num: int, path: Path) -> float:
    if key not in row:
        raise SidecarMalformedError(
            f"Sidecar malformed: required frame-row field missing "
            f"(field={key!r}, row={line_num}, value=<missing>, path={path})"
        )
    raw = row[key]
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise SidecarMalformedError(
            f"Sidecar malformed: cannot convert frame-row field to float "
            f"(field={key!r}, row={line_num}, value={str(raw)[:50]!r}, path={path})"
        )
    return float(raw)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SidecarData:
    """Parsed sidecar data. Validity is queryable; field access raises on absent gates
    only via the accessor methods, not on construction."""

    # --- _meta: always present ---
    sidecar_schema: int
    timing_mode: str
    source_pts: bool
    row_source: str
    segment_start_epoch: int
    attempt: int
    input_frame_count: int
    output_frame_count: int
    mismatch: bool
    pts_timebase: int
    measured_fps_mean: float
    pts_tick_delta_median: float
    pts_tick_delta_mean: float
    pts_delta_trim_kept: int
    pts_delta_trim_total: int
    pts_mean_delta_ms: float
    pts_stdev_delta_ms: float

    # --- _meta: full raw dict for fields not explicitly modeled ---
    raw_meta: Dict[str, Any] = field(default_factory=dict)

    # --- _meta: gated on source_pts ---
    nominal_dt_s: Optional[float] = None
    measured_fps: Optional[float] = None
    measured_fps_median: Optional[float] = None
    is_bimodal: Optional[bool] = None
    """Advisory, not authoritative. Structurally cannot fire when the majority mode is the
    short one (contract section 5, 'Known limitation'). Consumers handling critical bimodal cases
    should also inspect the raw dt_s distribution."""

    # --- _meta: gated on showinfo availability ---
    showinfo_frame_count: Optional[int] = None
    showinfo_residual: Optional[int] = None
    showinfo_pts_offset: Optional[int] = None
    showinfo_matched_count: Optional[int] = None
    showinfo_unmatched_mp4_count: Optional[int] = None
    showinfo_surplus_count: Optional[int] = None
    showinfo_offset_status: Optional[str] = None

    # --- _meta: drift (gated on source_pts AND n_drift_windows >= 4) ---
    pts_wallclock_offset_s: Optional[float] = None
    drift_rate_s_per_s: Optional[float] = None
    drift_ppm: Optional[float] = None
    drift_flat: Optional[bool] = None
    n_drift_windows: Optional[int] = None

    # --- Frame rows (internal, indexed by frame_index) ---
    _frame_pts_time_s: List[float] = field(default_factory=list)
    _frame_dt_s: List[Optional[float]] = field(default_factory=list)
    _frame_host_arrival_s: List[Optional[float]] = field(default_factory=list)

    # -----------------------------------------------------------------------
    # Validity queries
    # -----------------------------------------------------------------------

    @property
    def has_source_pts(self) -> bool:
        return bool(self.source_pts)

    @property
    def has_showinfo(self) -> bool:
        """True when showinfo fields are available (row_source is not 'mp4_regenerated')."""
        return self.row_source != "mp4_regenerated"

    @property
    def is_passthrough(self) -> bool:
        return self.timing_mode == "passthrough"

    @property
    def frame_count(self) -> int:
        return len(self._frame_pts_time_s)

    # -----------------------------------------------------------------------
    # Derived scalars
    # -----------------------------------------------------------------------

    @property
    def nominal_fps(self) -> float:
        """1.0 / nominal_dt_s. Raises SidecarValidityError if nominal_dt_s is absent."""
        if self.nominal_dt_s is None or self.nominal_dt_s <= 0:
            raise SidecarValidityError(
                "nominal_fps unavailable: nominal_dt_s is absent (source_pts=false)"
            )
        return 1.0 / self.nominal_dt_s

    # -----------------------------------------------------------------------
    # Per-frame accessors
    # -----------------------------------------------------------------------

    def pts_time_s(self, frame_index: int) -> float:
        """Segment-relative PTS in seconds. Raises IndexError on out-of-range."""
        if frame_index < 0 or frame_index >= len(self._frame_pts_time_s):
            raise IndexError(
                f"frame_index {frame_index} out of range [0, {len(self._frame_pts_time_s)})"
            )
        return self._frame_pts_time_s[frame_index]

    def dt_s(self, frame_index: int) -> Optional[float]:
        """Inter-frame interval in seconds. None on frame 0 (no predecessor).
        Raises IndexError on out-of-range.
        Raises SidecarValidityError if dt_s is not available (source_pts=false or cfr_grid).
        """
        if frame_index < 0 or frame_index >= len(self._frame_dt_s):
            raise IndexError(
                f"frame_index {frame_index} out of range [0, {len(self._frame_dt_s)})"
            )
        if not self.has_source_pts:
            raise SidecarValidityError(
                "dt_s not available: source_pts=false"
            )
        if not self.is_passthrough:
            raise SidecarValidityError(
                f"dt_s not available: timing_mode={self.timing_mode!r}"
            )
        return self._frame_dt_s[frame_index]

    def host_arrival_s(self, frame_index: int) -> Optional[float]:
        """Host arrival time. None if absent (per-row join miss or wholesale absent).
        Raises IndexError on out-of-range."""
        if frame_index < 0 or frame_index >= len(self._frame_host_arrival_s):
            raise IndexError(
                f"frame_index {frame_index} out of range [0, {len(self._frame_host_arrival_s)})"
            )
        return self._frame_host_arrival_s[frame_index]


# ---------------------------------------------------------------------------
# Schema-conditional required fields
# ---------------------------------------------------------------------------

# Fields always present regardless of schema (contract §2 "Always present", cross-schema)
_ALWAYS_REQUIRED_META = [
    ("sidecar_schema", "int"),
    ("timing_mode", "str"),
    ("source_pts", "bool"),
    ("segment_start_epoch", "int"),
    ("attempt", "int"),
    ("input_frame_count", "int"),
    ("output_frame_count", "int"),
    ("mismatch", "bool"),
    ("pts_timebase", "int"),
    ("measured_fps_mean", "float"),
    ("pts_tick_delta_median", "float"),
    ("pts_tick_delta_mean", "float"),
    ("pts_delta_trim_kept", "int"),
    ("pts_delta_trim_total", "int"),
    ("pts_mean_delta_ms", "float"),
    ("pts_stdev_delta_ms", "float"),
]

# Fields required ONLY in schema >= 5 (contract §3 "Schema 5+")
_SCHEMA5_REQUIRED_META = [
    ("row_source", "str"),
]


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _parse_meta(meta: Dict[str, Any], path: Path) -> Dict[str, Any]:
    """Extract typed fields from _meta dict. Raises SidecarMalformedError on missing
    always-present fields or type failures."""

    # Determine schema first (needed for conditional required set)
    schema = _require_int(meta, "sidecar_schema", path)

    # Build the required set for this schema
    required = list(_ALWAYS_REQUIRED_META)
    if schema >= 5:
        required.extend(_SCHEMA5_REQUIRED_META)

    # Validate and extract required fields
    _require_str(meta, "timing_mode", path)
    _require_bool(meta, "source_pts", path)
    if schema >= 5:
        _require_str(meta, "row_source", path)
    _require_int(meta, "segment_start_epoch", path)
    _require_int(meta, "attempt", path)
    _require_int(meta, "input_frame_count", path)
    _require_int(meta, "output_frame_count", path)
    _require_bool(meta, "mismatch", path)
    _require_int(meta, "pts_timebase", path)
    _require_float(meta, "measured_fps_mean", path)
    _require_float(meta, "pts_tick_delta_median", path)
    _require_float(meta, "pts_tick_delta_mean", path)
    _require_int(meta, "pts_delta_trim_kept", path)
    _require_int(meta, "pts_delta_trim_total", path)
    _require_float(meta, "pts_mean_delta_ms", path)
    _require_float(meta, "pts_stdev_delta_ms", path)

    return dict(
        sidecar_schema=schema,
        timing_mode=meta["timing_mode"],
        source_pts=meta["source_pts"],
        row_source=meta.get("row_source", "showinfo_grid"),  # default for schema < 5
        segment_start_epoch=int(meta["segment_start_epoch"]),
        attempt=int(meta["attempt"]),
        input_frame_count=int(meta["input_frame_count"]),
        output_frame_count=int(meta["output_frame_count"]),
        mismatch=meta["mismatch"],
        pts_timebase=int(meta["pts_timebase"]),
        measured_fps_mean=float(meta["measured_fps_mean"]),
        pts_tick_delta_median=float(meta["pts_tick_delta_median"]),
        pts_tick_delta_mean=float(meta["pts_tick_delta_mean"]),
        pts_delta_trim_kept=int(meta["pts_delta_trim_kept"]),
        pts_delta_trim_total=int(meta["pts_delta_trim_total"]),
        pts_mean_delta_ms=float(meta["pts_mean_delta_ms"]),
        pts_stdev_delta_ms=float(meta["pts_stdev_delta_ms"]),
        raw_meta=dict(meta),
        # Gated fields — None if absent, raises if present but malformed
        nominal_dt_s=_optional_float(meta, "nominal_dt_s", path),
        measured_fps=_optional_float(meta, "measured_fps", path),
        measured_fps_median=_optional_float(meta, "measured_fps_median", path),
        is_bimodal=_optional_bool(meta, "is_bimodal", path),
        showinfo_frame_count=_optional_int(meta, "showinfo_frame_count", path),
        showinfo_residual=_optional_int(meta, "showinfo_residual", path),
        showinfo_pts_offset=_optional_int(meta, "showinfo_pts_offset", path),
        showinfo_matched_count=_optional_int(meta, "showinfo_matched_count", path),
        showinfo_unmatched_mp4_count=_optional_int(meta, "showinfo_unmatched_mp4_count", path),
        showinfo_surplus_count=_optional_int(meta, "showinfo_surplus_count", path),
        showinfo_offset_status=_optional_str(meta, "showinfo_offset_status", path),
        pts_wallclock_offset_s=_optional_float(meta, "pts_wallclock_offset_s", path),
        drift_rate_s_per_s=_optional_float(meta, "drift_rate_s_per_s", path),
        drift_ppm=_optional_float(meta, "drift_ppm", path),
        drift_flat=_optional_bool(meta, "drift_flat", path),
        n_drift_windows=_optional_int(meta, "n_drift_windows", path),
    )


def parse_sidecar(path: Path) -> SidecarData:
    """Permissive parse — for tooling. Accepts any schema/mode. Validity queryable.

    Permissive about schema version and validity mode: accepts schema 4,
    source_pts=false, cfr_grid, and makes their status queryable via has_source_pts,
    is_passthrough, etc.

    Strict about structural malformation: a file missing an always-present field,
    with non-contiguous frame_index values, or with unparseable rows is broken
    regardless of schema, and raises SidecarMalformedError.

    Raises:
        SidecarError: file missing or unreadable.
        SidecarMalformedError: structurally malformed (missing required field,
            non-contiguous frame_index, unparseable row, type mismatch).
    """
    path = Path(path)
    if not path.exists():
        raise SidecarError(f"Sidecar file not found: {path}")

    try:
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()
    except Exception as exc:
        raise SidecarError(f"Cannot read sidecar: {path}: {exc}") from exc

    if not lines:
        raise SidecarError(f"Sidecar is empty: {path}")

    try:
        meta = json.loads(lines[0])
    except json.JSONDecodeError as exc:
        raise SidecarMalformedError(f"Cannot parse _meta line: {path}: {exc}") from exc

    if not meta.get("_meta"):
        raise SidecarMalformedError(f"First line is not a _meta line: {path}")

    meta_fields = _parse_meta(meta, path)

    # Parse frame rows
    frame_pts: List[float] = []
    frame_dt: List[Optional[float]] = []
    frame_host: List[Optional[float]] = []

    for i, line in enumerate(lines[1:], start=1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SidecarMalformedError(
                f"Sidecar malformed: cannot parse frame row as JSON "
                f"(row={i}, path={path}): {exc}"
            ) from exc

        fi = _require_row_int(row, "frame_index", i, path)
        pts = _require_row_float(row, "pts_time_s", i, path)

        # Validate contiguity: frame_index must equal position
        expected_fi = len(frame_pts)
        if fi != expected_fi:
            raise SidecarMalformedError(
                f"Sidecar malformed: frame_index discontinuity "
                f"(field='frame_index', row={i}, expected={expected_fi}, got={fi}, path={path}) "
                f"— no gaps, no duplicates, monotonic required"
            )

        frame_pts.append(pts)
        frame_dt.append(row.get("dt_s"))  # None if absent or null — gated field
        frame_host.append(row.get("host_arrival_s"))  # None if absent — gated field

    # Assert row_count == output_frame_count (unconditional — field is guaranteed present)
    output_fc = meta_fields["output_frame_count"]
    if len(frame_pts) != output_fc:
        raise SidecarMalformedError(
            f"Sidecar malformed: row count ({len(frame_pts)}) != output_frame_count ({output_fc}) "
            f"(path={path})"
        )

    return SidecarData(
        _frame_pts_time_s=frame_pts,
        _frame_dt_s=frame_dt,
        _frame_host_arrival_s=frame_host,
        **meta_fields,
    )


def load_sidecar(mp4_path: Path) -> SidecarData:
    """Strict entry point — for pipeline consumers.

    Locates the sibling .timing.jsonl, parses, and enforces:
      - Schema >= 5
      - timing_mode == "passthrough"
      - source_pts == true

    Does NOT require the mp4 to exist — only derives the sidecar path from it.

    Raises:
        SidecarError: file missing or unreadable
        SidecarMalformedError: structurally malformed
        SidecarSchemaError: schema < 5
        SidecarValidityError: source_pts=false or timing_mode != passthrough
    """
    mp4_path = Path(mp4_path)
    sidecar_path = mp4_path.parent / (mp4_path.stem + ".timing.jsonl")

    data = parse_sidecar(sidecar_path)

    if data.sidecar_schema < 5:
        raise SidecarSchemaError(
            f"Sidecar schema {data.sidecar_schema} < 5 — pipeline requires schema 5+. "
            f"Schema-4 footage remains valid at the gap level and as a regression baseline, "
            f"but cannot serve as a timing source for the production pipeline. "
            f"File: {sidecar_path}"
        )

    if not data.has_source_pts:
        raise SidecarValidityError(
            f"source_pts=false — pipeline requires source-PTS capture timestamps. "
            f"CFR rollback footage (SOURCE_PTS=0) produces degraded timing unsuitable "
            f"for the production pipeline. File: {sidecar_path}"
        )

    if not data.is_passthrough:
        raise SidecarValidityError(
            f"timing_mode={data.timing_mode!r} — pipeline requires passthrough. "
            f"CFR footage cannot provide per-frame dt_s. File: {sidecar_path}"
        )

    return data
