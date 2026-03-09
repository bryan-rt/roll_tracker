"""Export-session consolidation helpers for Stage F."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from typing import Any, Dict, Iterable, List, Sequence


@dataclass(frozen=True)
class ExportSession:
    export_id: str
    clip_id: str
    camera_id: str

    match_start_frame: int
    match_end_frame: int
    match_start_ts_ms: int
    match_end_ts_ms: int

    export_start_frame: int
    export_end_frame: int

    person_id_a: str
    person_id_b: str

    april_tag_id_a: str | None
    april_tag_id_b: str | None

    resolved_pair_key: tuple[str, str]
    source_match_ids: tuple[str, ...]
    source_person_ids: tuple[str, ...]
    gaps_merged: tuple[int, ...]


def _norm_tag(value: Any) -> str | None:
    if value is None:
        return None
    txt = str(value).strip()
    return txt if txt else None


def _identity_token(person_id: Any, april_tag_id: Any) -> str:
    tag = _norm_tag(april_tag_id)
    if tag is not None:
        return f"tag:{tag}"
    return f"person:{str(person_id)}"


def _resolved_pair_key(match: Dict[str, Any]) -> tuple[str, str]:
    evidence = match.get("evidence") if isinstance(match.get("evidence"), dict) else {}
    t1 = _identity_token(match.get("person_id_a"), evidence.get("april_tag_id_a"))
    t2 = _identity_token(match.get("person_id_b"), evidence.get("april_tag_id_b"))
    return tuple(sorted((t1, t2)))


def _match_tags(match: Dict[str, Any]) -> frozenset[str]:
    evidence = match.get("evidence") if isinstance(match.get("evidence"), dict) else {}
    tags = []
    for key in ("april_tag_id_a", "april_tag_id_b"):
        tag = _norm_tag(evidence.get(key))
        if tag is not None:
            tags.append(tag)
    return frozenset(tags)


def _tags_conflict(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    ta = _match_tags(a)
    tb = _match_tags(b)
    if not ta or not tb:
        return False
    if len(ta) == 1 and len(tb) == 1:
        return ta != tb
    if len(ta) == 1 and len(tb) == 2:
        return not ta.issubset(tb)
    if len(ta) == 2 and len(tb) == 1:
        return not tb.issubset(ta)
    return ta != tb


def _stable_unique(values: Iterable[str]) -> tuple[str, ...]:
    seen = set()
    out: List[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return tuple(out)


def _make_export_id(source_match_ids: Sequence[str]) -> str:
    if len(source_match_ids) == 1:
        return str(source_match_ids[0])
    joined = "|".join(str(x) for x in source_match_ids)
    digest = hashlib.sha1(joined.encode("utf-8")).hexdigest()[:16]
    return f"exp_{digest}"


def _make_export_session(match: Dict[str, Any], *, buffer_frames: int, last_frame: int) -> ExportSession:
    evidence = match.get("evidence") if isinstance(match.get("evidence"), dict) else {}
    match_id = str(match["match_id"])
    match_start_frame = int(match["start_frame"])
    match_end_frame = int(match["end_frame"])
    export_start_frame = max(0, match_start_frame - int(buffer_frames))
    export_end_frame = min(int(last_frame), match_end_frame + int(buffer_frames))
    source_person_ids = _stable_unique(
        [
            str(match["person_id_a"]),
            str(match["person_id_b"]),
        ]
    )
    source_match_ids = (match_id,)
    return ExportSession(
        export_id=_make_export_id(source_match_ids),
        clip_id=str(match["clip_id"]),
        camera_id=str(match["camera_id"]),
        match_start_frame=match_start_frame,
        match_end_frame=match_end_frame,
        match_start_ts_ms=int(match["start_ts_ms"]),
        match_end_ts_ms=int(match["end_ts_ms"]),
        export_start_frame=export_start_frame,
        export_end_frame=export_end_frame,
        person_id_a=str(match["person_id_a"]),
        person_id_b=str(match["person_id_b"]),
        april_tag_id_a=_norm_tag(evidence.get("april_tag_id_a")),
        april_tag_id_b=_norm_tag(evidence.get("april_tag_id_b")),
        resolved_pair_key=_resolved_pair_key(match),
        source_match_ids=source_match_ids,
        source_person_ids=source_person_ids,
        gaps_merged=tuple(),
    )


def _merge_export_session_with_match(
    current: ExportSession,
    match: Dict[str, Any],
    *,
    buffer_frames: int,
    last_frame: int,
) -> ExportSession:
    next_match_start = int(match["start_frame"])
    next_match_end = int(match["end_frame"])
    next_export_end = min(int(last_frame), next_match_end + int(buffer_frames))
    gap = max(0, next_match_start - current.match_end_frame - 1)

    source_match_ids = current.source_match_ids + (str(match["match_id"]),)
    source_person_ids = _stable_unique(
        list(current.source_person_ids) + [str(match["person_id_a"]), str(match["person_id_b"])]
    )
    gaps_merged = current.gaps_merged + (int(gap),)

    return replace(
        current,
        export_id=_make_export_id(source_match_ids),
        match_end_frame=next_match_end,
        match_end_ts_ms=int(match["end_ts_ms"]),
        export_end_frame=max(current.export_end_frame, next_export_end),
        source_match_ids=source_match_ids,
        source_person_ids=source_person_ids,
        gaps_merged=gaps_merged,
    )


def consolidate_export_sessions(
    matches: list[dict[str, Any]],
    *,
    enabled: bool,
    max_gap_frames: int,
    buffer_frames: int,
    last_frame: int,
    require_nonconflicting_tags: bool = True,
) -> list[ExportSession]:
    if not matches:
        return []

    ordered = sorted(
        matches,
        key=lambda match: (
            str(match.get("clip_id", "")),
            _resolved_pair_key(match),
            int(match.get("start_frame", 0)),
            int(match.get("end_frame", 0)),
            str(match.get("match_id", "")),
        ),
    )

    if not enabled:
        return [
            _make_export_session(match, buffer_frames=buffer_frames, last_frame=last_frame)
            for match in ordered
        ]

    out: list[ExportSession] = []
    current: ExportSession | None = None
    current_match: Dict[str, Any] | None = None

    for match in ordered:
        candidate = _make_export_session(match, buffer_frames=buffer_frames, last_frame=last_frame)
        if current is None:
            current = candidate
            current_match = match
            continue

        same_clip = candidate.clip_id == current.clip_id
        same_pair_key = candidate.resolved_pair_key == current.resolved_pair_key
        raw_gap = int(match["start_frame"]) - int(current.match_end_frame) - 1
        raw_gap_ok = raw_gap <= int(max_gap_frames)
        buffered_overlap_ok = candidate.export_start_frame <= current.export_end_frame
        tag_ok = True
        if require_nonconflicting_tags and current_match is not None:
            tag_ok = not _tags_conflict(current_match, match)

        if same_clip and same_pair_key and raw_gap_ok and buffered_overlap_ok and tag_ok:
            current = _merge_export_session_with_match(
                current,
                match,
                buffer_frames=buffer_frames,
                last_frame=last_frame,
            )
            current_match = match
        else:
            out.append(current)
            current = candidate
            current_match = match

    if current is not None:
        out.append(current)
    return out