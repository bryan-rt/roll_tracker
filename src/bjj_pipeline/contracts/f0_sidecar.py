"""Sidecar timing reader — schema v5.

Codes against docs/reference/sidecar_contract.md (schema 5, CP-R13b).
Two entry points:

  parse_sidecar(path)  — permissive, for tooling. Validity queryable, never raises on content.
  load_sidecar(mp4_path) — strict, for pipeline consumers. Enforces schema 5 + passthrough + source_pts.

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


class SidecarValidityError(SidecarError):
    """A required validity gate is not met (names the gate)."""


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
        if not self._frame_dt_s and not self.has_source_pts:
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
# Parsing
# ---------------------------------------------------------------------------

def _parse_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
    """Extract typed fields from _meta dict."""
    return dict(
        sidecar_schema=int(meta.get("sidecar_schema", 0)),
        timing_mode=str(meta.get("timing_mode", "")),
        source_pts=bool(meta.get("source_pts", False)),
        row_source=str(meta.get("row_source", "showinfo_grid")),
        segment_start_epoch=int(meta.get("segment_start_epoch", 0)),
        attempt=int(meta.get("attempt", 0)),
        input_frame_count=int(meta.get("input_frame_count", 0)),
        output_frame_count=int(meta.get("output_frame_count", 0)),
        mismatch=bool(meta.get("mismatch", False)),
        pts_timebase=int(meta.get("pts_timebase", 0)),
        measured_fps_mean=float(meta.get("measured_fps_mean", 0.0)),
        pts_tick_delta_median=float(meta.get("pts_tick_delta_median", 0.0)),
        pts_tick_delta_mean=float(meta.get("pts_tick_delta_mean", 0.0)),
        pts_delta_trim_kept=int(meta.get("pts_delta_trim_kept", 0)),
        pts_delta_trim_total=int(meta.get("pts_delta_trim_total", 0)),
        pts_mean_delta_ms=float(meta.get("pts_mean_delta_ms", 0.0)),
        pts_stdev_delta_ms=float(meta.get("pts_stdev_delta_ms", 0.0)),
        raw_meta=dict(meta),
        # Gated fields — None if absent
        nominal_dt_s=float(meta["nominal_dt_s"]) if "nominal_dt_s" in meta else None,
        measured_fps=float(meta["measured_fps"]) if "measured_fps" in meta else None,
        measured_fps_median=float(meta["measured_fps_median"]) if "measured_fps_median" in meta else None,
        is_bimodal=bool(meta["is_bimodal"]) if "is_bimodal" in meta else None,
        showinfo_frame_count=int(meta["showinfo_frame_count"]) if "showinfo_frame_count" in meta else None,
        showinfo_residual=int(meta["showinfo_residual"]) if "showinfo_residual" in meta else None,
        showinfo_pts_offset=int(meta["showinfo_pts_offset"]) if "showinfo_pts_offset" in meta else None,
        showinfo_matched_count=int(meta["showinfo_matched_count"]) if "showinfo_matched_count" in meta else None,
        showinfo_unmatched_mp4_count=int(meta["showinfo_unmatched_mp4_count"]) if "showinfo_unmatched_mp4_count" in meta else None,
        showinfo_surplus_count=int(meta["showinfo_surplus_count"]) if "showinfo_surplus_count" in meta else None,
        showinfo_offset_status=str(meta["showinfo_offset_status"]) if "showinfo_offset_status" in meta else None,
        pts_wallclock_offset_s=float(meta["pts_wallclock_offset_s"]) if "pts_wallclock_offset_s" in meta else None,
        drift_rate_s_per_s=float(meta["drift_rate_s_per_s"]) if "drift_rate_s_per_s" in meta else None,
        drift_ppm=float(meta["drift_ppm"]) if "drift_ppm" in meta else None,
        drift_flat=bool(meta["drift_flat"]) if "drift_flat" in meta else None,
        n_drift_windows=int(meta["n_drift_windows"]) if "n_drift_windows" in meta else None,
    )


def parse_sidecar(path: Path) -> SidecarData:
    """Permissive parse — for tooling. Accepts any schema/mode. Validity queryable.

    Raises SidecarError only if the file is missing or unparseable.
    Never raises on content (schema version, timing_mode, source_pts).
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
        raise SidecarError(f"Cannot parse _meta line: {path}: {exc}") from exc

    if not meta.get("_meta"):
        raise SidecarError(f"First line is not a _meta line: {path}")

    meta_fields = _parse_meta(meta)

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
        except json.JSONDecodeError:
            continue

        fi = int(row.get("frame_index", i - 1))

        # Validate contiguity: frame_index must equal position
        expected_fi = len(frame_pts)
        if fi != expected_fi:
            raise SidecarError(
                f"frame_index discontinuity at row {i}: expected {expected_fi}, got {fi} "
                f"(contiguity violation — no gaps, no duplicates, monotonic required)"
            )

        frame_pts.append(float(row.get("pts_time_s", 0.0)))
        frame_dt.append(row.get("dt_s"))  # None if absent or null
        frame_host.append(row.get("host_arrival_s"))  # None if absent

    # Assert row_count == output_frame_count (schema 5 guarantees by construction)
    output_fc = meta_fields["output_frame_count"]
    if output_fc > 0 and len(frame_pts) != output_fc:
        raise SidecarError(
            f"Row count ({len(frame_pts)}) != output_frame_count ({output_fc}) in {path}"
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
        SidecarError: file missing or unparseable
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
