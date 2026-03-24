"""Cross-camera identity merge: link the same athlete across cameras via AprilTag co-observation.

CP14f: session-level cross-camera identity resolution. Uses identity_assignments
produced by per-camera D4 to find athletes observed on multiple cameras within
the same session. Produces a deterministic global_person_id map.

Algorithm (Option 1 — presence-based linking):
  1. Load identity_assignments_{cam_id}.jsonl for each camera
  2. Filter by min_tag_observations AND min_assignment_confidence
  2b. Intra-camera dedup: at most one (cam_id, person_id) per (cam_id, tag_id)
  3. Group by tag_id — any tag on 2+ cameras produces union-find links
  4. Union-find → connected components → deterministic gp_ IDs
  5. Every (cam_id, person_id) gets a global ID (linked or standalone)
"""
from __future__ import annotations

import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from loguru import logger

from bjj_pipeline.contracts.f0_paths import SessionOutputLayout
from bjj_pipeline.stages.stitch.session_d_run import SessionStageLayoutAdapter


# ---------------------------------------------------------------------------
# Union-Find
# ---------------------------------------------------------------------------

class _UnionFind:
    """Minimal union-find with path compression and union by rank."""

    def __init__(self) -> None:
        self._parent: Dict[str, str] = {}
        self._rank: Dict[str, int] = {}

    def find(self, x: str) -> str:
        if x not in self._parent:
            self._parent[x] = x
            self._rank[x] = 0
        if self._parent[x] != x:
            self._parent[x] = self.find(self._parent[x])
        return self._parent[x]

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self._rank[ra] < self._rank[rb]:
            ra, rb = rb, ra
        self._parent[rb] = ra
        if self._rank[ra] == self._rank[rb]:
            self._rank[ra] += 1

    def components(self) -> Dict[str, List[str]]:
        """Return {root → [members]} mapping."""
        groups: Dict[str, List[str]] = defaultdict(list)
        for x in self._parent:
            groups[self.find(x)].append(x)
        return dict(groups)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _deterministic_global_id(keys: List[str]) -> str:
    """Produce a deterministic gp_ prefixed ID from sorted member keys."""
    joined = "|".join(sorted(keys))
    digest = hashlib.sha256(joined.encode("utf-8")).hexdigest()[:12]
    return f"gp_{digest}"


def _load_identity_assignments(path: Path) -> List[Dict[str, Any]]:
    """Load identity_assignments JSONL, returning list of parsed records."""
    if not path.exists():
        return []
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if isinstance(rec, dict) and rec.get("artifact_type") == "identity_assignment":
                    records.append(rec)
            except json.JSONDecodeError:
                continue
    return records


def _append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_cross_camera_merge(
    *,
    config: dict,
    session_layout: SessionOutputLayout,
    cam_ids: List[str],
    adapter_map: Dict[str, SessionStageLayoutAdapter],
) -> Dict[str, str]:
    """Run cross-camera identity merge for a session.

    Returns {"{cam_id}:{person_id}" → "gp_..."} mapping. Every (cam_id, person_id)
    observed in identity assignments gets a global ID, whether linked or standalone.
    """
    cc_cfg = config.get("cross_camera", {})
    if not isinstance(cc_cfg, dict):
        cc_cfg = {}

    clock_sync = cc_cfg.get("clock_sync_method", "filename")
    if clock_sync != "filename":
        logger.warning(
            "cross_camera_merge: clock_sync_method={!r} not implemented, "
            "falling back to 'filename'",
            clock_sync,
        )

    # co_observation_window_frames: documented future hook for buzzer-based sync.
    # With clock_sync_method="filename", session boundary is the temporal constraint.
    # When buzzer sync is implemented, this window will gate temporal overlap checks.
    _co_obs_window = int(cc_cfg.get("co_observation_window_frames", 90))  # noqa: F841

    min_tag_obs = int(cc_cfg.get("min_tag_observations", 2))
    min_confidence = float(cc_cfg.get("min_assignment_confidence", 0.5))

    session_id = session_layout.session_id

    # --- Step 1: Load identity assignments for all cameras ---
    # Each record: {person_id, tag_id, assignment_confidence, evidence: {total_tag_frames, ...}, camera_id, ...}
    all_records: List[Tuple[str, Dict[str, Any]]] = []  # (cam_id, record)
    for cam_id in cam_ids:
        adapter = adapter_map.get(cam_id)
        if adapter is None:
            continue
        path = adapter.identity_assignments_jsonl()
        records = _load_identity_assignments(path)
        for rec in records:
            all_records.append((cam_id, rec))

    logger.info(
        "cross_camera_merge: session={} cameras={} raw_records={}",
        session_id, len(cam_ids), len(all_records),
    )

    # --- Step 2: Filter by min_tag_observations and min_assignment_confidence ---
    filtered: List[Tuple[str, str, str, float]] = []  # (cam_id, person_id, tag_id, confidence)
    for cam_id, rec in all_records:
        tag_id = rec.get("tag_id")
        person_id = rec.get("person_id")
        confidence = float(rec.get("assignment_confidence", 0.0))
        evidence = rec.get("evidence", {})
        total_tag_frames = int(evidence.get("total_tag_frames", 0))

        if tag_id is None or person_id is None:
            continue
        if total_tag_frames < min_tag_obs:
            continue
        if confidence < min_confidence:
            continue

        filtered.append((cam_id, str(person_id), str(tag_id), confidence))

    logger.info(
        "cross_camera_merge: after filter (min_obs={}, min_conf={:.2f}): {} records",
        min_tag_obs, min_confidence, len(filtered),
    )

    # --- Step 2b: Intra-camera dedup ---
    # For each (cam_id, tag_id), keep only the highest-confidence record.
    # Ties with different person_ids → skip that tag entirely (conflict).
    skip_tags: Set[str] = set()

    # Group: (cam_id, tag_id) → list of (person_id, confidence)
    cam_tag_index: Dict[Tuple[str, str], List[Tuple[str, float]]] = defaultdict(list)
    for cam_id, person_id, tag_id, confidence in filtered:
        cam_tag_index[(cam_id, tag_id)].append((person_id, confidence))

    # Deduped: (cam_id, tag_id) → person_id
    deduped: Dict[Tuple[str, str], str] = {}
    for (cam_id, tag_id), entries in cam_tag_index.items():
        if len(entries) == 1:
            deduped[(cam_id, tag_id)] = entries[0][0]
            continue

        # Multiple person_ids for same (cam_id, tag_id)
        entries.sort(key=lambda e: e[1], reverse=True)
        best_conf = entries[0][1]
        best_entries = [e for e in entries if e[1] == best_conf]

        if len(best_entries) == 1:
            # Clear winner by confidence
            deduped[(cam_id, tag_id)] = best_entries[0][0]
            logger.info(
                "cross_camera_merge: dedup cam={} tag={}: kept person={} (conf={:.2f}), "
                "dropped {} others",
                cam_id, tag_id, best_entries[0][0], best_conf, len(entries) - 1,
            )
        else:
            # Tie — conflicting person_ids with equal confidence
            conflicting_pids = [e[0] for e in best_entries]
            skip_tags.add(tag_id)
            logger.warning(
                "cross_camera_merge: CONFLICT cam={} tag={}: person_ids={} "
                "have equal confidence={:.2f} — skipping tag for linking",
                cam_id, tag_id, conflicting_pids, best_conf,
            )

    if skip_tags:
        logger.info(
            "cross_camera_merge: {} tags skipped due to intra-camera conflicts: {}",
            len(skip_tags), sorted(skip_tags),
        )

    # --- Step 3: Build tag_id → List[(cam_id, person_id)] index ---
    tag_index: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    for (cam_id, tag_id), person_id in deduped.items():
        if tag_id in skip_tags:
            continue
        tag_index[tag_id].append((cam_id, person_id))

    # --- Step 4: Union-find over cross-camera links ---
    uf = _UnionFind()

    # Ensure all (cam_id, person_id) keys are registered in union-find
    all_keys: Set[str] = set()
    for cam_id, person_id, tag_id, confidence in filtered:
        key = f"{cam_id}:{person_id}"
        all_keys.add(key)
        uf.find(key)  # register

    n_links = 0
    for tag_id, cam_person_pairs in tag_index.items():
        # Only link if tag appears on 2+ cameras
        cameras_seen = {cam_id for cam_id, _ in cam_person_pairs}
        if len(cameras_seen) < 2:
            continue

        # Link all pairs — but only cross-camera
        for i in range(len(cam_person_pairs)):
            for j in range(i + 1, len(cam_person_pairs)):
                cam_a, pid_a = cam_person_pairs[i]
                cam_b, pid_b = cam_person_pairs[j]

                if cam_a == cam_b:
                    # Should be impossible after dedup, but guard anyway
                    logger.warning(
                        "cross_camera_merge: same-camera link skipped: "
                        "cam={} person_a={} person_b={} tag={}",
                        cam_a, pid_a, pid_b, tag_id,
                    )
                    continue

                key_a = f"{cam_a}:{pid_a}"
                key_b = f"{cam_b}:{pid_b}"
                uf.union(key_a, key_b)
                n_links += 1
                logger.info(
                    "cross_camera_merge: linked {} ~ {} via tag={}",
                    key_a, key_b, tag_id,
                )

    # --- Step 5: Connected components → deterministic gp_ IDs ---
    components = uf.components()
    global_id_map: Dict[str, str] = {}
    n_linked_components = 0

    for root, members in components.items():
        gid = _deterministic_global_id(members)
        if len(members) > 1:
            n_linked_components += 1
        for key in members:
            global_id_map[key] = gid

    # Also assign standalone IDs for any (cam_id, person_id) that appeared in
    # the raw records but was filtered out (below min_obs or min_conf).
    # These still need a global ID for Stage F.
    for cam_id, rec in all_records:
        person_id = rec.get("person_id")
        if person_id is None:
            continue
        key = f"{cam_id}:{str(person_id)}"
        if key not in global_id_map:
            global_id_map[key] = _deterministic_global_id([key])

    logger.info(
        "cross_camera_merge: session={} links={} components_linked={} "
        "total_global_ids={}",
        session_id, n_links, n_linked_components, len(global_id_map),
    )

    # --- Step 6: Write cross_camera_identities.jsonl ---
    output_path = session_layout.session_cross_camera_identities_jsonl()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Clear previous run
    if output_path.exists():
        output_path.unlink()

    created_at_ms = int(time.time() * 1000)

    for key, gid in sorted(global_id_map.items()):
        cam_id, person_id = key.split(":", 1)
        _append_jsonl(output_path, {
            "artifact_type": "cross_camera_identity",
            "session_id": session_id,
            "cam_id": cam_id,
            "person_id": person_id,
            "global_person_id": gid,
            "created_at_ms": created_at_ms,
        })

    # Write audit summary
    audit_path = session_layout.session_audit_jsonl("D")
    _append_jsonl(audit_path, {
        "artifact_type": "cross_camera_merge_summary",
        "session_id": session_id,
        "created_at_ms": created_at_ms,
        "n_cameras": len(cam_ids),
        "n_raw_records": len(all_records),
        "n_filtered_records": len(filtered),
        "n_skip_tags": len(skip_tags),
        "skip_tags": sorted(skip_tags),
        "n_links": n_links,
        "n_linked_components": n_linked_components,
        "n_global_ids": len(global_id_map),
        "clock_sync_method": "filename",
        "min_tag_observations": min_tag_obs,
        "min_assignment_confidence": min_confidence,
    })

    return global_id_map
