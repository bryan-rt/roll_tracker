"""Stage C2 — Identity Registry (Voting + Conflicts).

C2 consumes flickery AprilTag observations emitted by C1 and deterministically
produces identity constraints usable by Stage D.

POC constraints (manager-locked):
- Online phase uses multiplex_AC (A + C). Stage B is deferred.
- C2 must function using only stage_C/tag_observations.jsonl.
- Canonical outputs are limited to F0 artifacts:
  - stage_C/identity_hints.jsonl
  - stage_C/audit.jsonl
- Conflicts: canonical cannot_link is tracklet-to-tracklet.
  Encode via IdentityHint.anchor_key = f"tracklet:{other_tracklet_id}".

Evidence model:
- Join key for tag observations is (frame_index, detection_id).
- Voting bucket is tracklet_id.

Determinism:
- Sort inputs deterministically (frame_index asc, detection_id asc, tag_id asc).
- Use fixed tie-breakers and stable pair iteration order.

Note on confidence:
- In current C1 multiplex implementation, confidence may be hard-coded (e.g., 1.0).
  We still apply the manager-approved gates; audit records make this behavior visible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def _safe_float(x: Any, default: float = 0.0) -> float:
	try:
		return float(x)
	except Exception:
		return default


def _safe_int(x: Any, default: int = 0) -> int:
	try:
		return int(x)
	except Exception:
		return default


@dataclass(frozen=True)
class CandidateAgg:
	tracklet_id: str
	tag_id: str
	vote_count: int
	total_weight: float
	max_conf: float
	first_seen_frame: int
	frames_seen: Tuple[int, ...]


class C2IdentityRegistry:
	"""Deterministic in-memory registry for C2 voting + conflicts."""

	def __init__(
		self,
		*,
		cfg: Dict[str, Any],
		clip_id: str,
		camera_id: str,
		pipeline_version: str,
		created_at_ms: int,
	) -> None:
		self.cfg = cfg or {}
		self.clip_id = clip_id
		self.camera_id = camera_id
		self.pipeline_version = pipeline_version
		self.created_at_ms = int(created_at_ms)

		# Manager-approved defaults
		self.min_obs = max(1, _safe_int(self.cfg.get("min_obs", 1), 1))
		self.min_conf = _safe_float(self.cfg.get("min_conf", 0.70), 0.70)
		self.min_weight = _safe_float(self.cfg.get("min_weight", 0.70), 0.70)
		self.min_conf_strong_single = _safe_float(self.cfg.get("min_conf_strong_single", 0.90), 0.90)
		self.min_margin_abs = _safe_float(self.cfg.get("min_margin_abs", 0.15), 0.15)

		# Optional weight multipliers keyed by roi_method (or roi_source). Default multiplier=1.0.
		self.roi_method_multipliers: Dict[str, float] = {
			str(k): _safe_float(v, 1.0) for k, v in (self.cfg.get("roi_method_multipliers", {}) or {}).items()
		}

		# Accumulate raw observations (dicts) for deterministic finalize.
		self._obs: List[Dict[str, Any]] = []

	def ingest_tag_observation(self, rec: Dict[str, Any]) -> None:
		"""Ingest a single tag observation record."""
		self._obs.append(dict(rec))

	def finalize(
		self,
		*,
		tracklet_spans: Optional[Dict[str, Tuple[int, int]]],
	) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
		"""Finalize voting + conflicts.

		Args:
			tracklet_spans: Optional mapping tracklet_id -> (start_frame,end_frame).
			If absent, overlap fallback uses observation frame_index sets.

		Returns:
			(identity_hints, audit_events)
		"""
		obs_sorted = sorted(
			self._obs,
			key=lambda r: (
				_safe_int(r.get("frame_index"), -1),
				str(r.get("detection_id") or ""),
				str(r.get("tag_id") or ""),
			),
		)

		# Group observations by (tracklet_id, tag_id)
		by_tracklet: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
		for r in obs_sorted:
			tid = str(r.get("tracklet_id") or "")
			tag = str(r.get("tag_id") or "")
			if not tid or not tag:
				continue
			by_tracklet.setdefault(tid, {}).setdefault(tag, []).append(r)

		# Precompute per-tracklet per-tag aggregates
		aggs_by_tracklet: Dict[str, List[CandidateAgg]] = {}
		audit_events: List[Dict[str, Any]] = []

		# For overlap fallback
		frames_by_tracklet: Dict[str, set[int]] = {}

		for tid in sorted(by_tracklet.keys()):
			cand_aggs: List[CandidateAgg] = []
			frames_set: set[int] = set()
			for tag in sorted(by_tracklet[tid].keys()):
				recs = by_tracklet[tid][tag]
				vote_count = 0
				total_weight = 0.0
				max_conf = 0.0
				first_seen = None
				frames_seen: List[int] = []

				for rr in recs:
					fi = _safe_int(rr.get("frame_index"), -1)
					conf = _safe_float(rr.get("confidence"), 0.0)
					roi_method = str(rr.get("roi_method") or rr.get("roi_source") or "")
					mult = self.roi_method_multipliers.get(roi_method, 1.0)
					w = conf * mult
					vote_count += 1
					total_weight += w
					max_conf = max(max_conf, conf)
					if fi >= 0:
						frames_set.add(fi)
						frames_seen.append(fi)
						if first_seen is None or fi < first_seen:
							first_seen = fi

				if vote_count == 0:
					continue
				if first_seen is None:
					first_seen = -1
				cand_aggs.append(
					CandidateAgg(
						tracklet_id=tid,
						tag_id=tag,
						vote_count=vote_count,
						total_weight=total_weight,
						max_conf=max_conf,
						first_seen_frame=first_seen,
						frames_seen=tuple(sorted(frames_seen)),
					)
				)

			frames_by_tracklet[tid] = frames_set
			aggs_by_tracklet[tid] = cand_aggs

		# Determine supported candidates per tracklet using FULL must-link gate.
		# Note: This is used BOTH for emitting must_link and for conflicts.
		must_link_hints: List[Dict[str, Any]] = []
		supported_by_tracklet: Dict[str, List[CandidateAgg]] = {}

		for tid in sorted(aggs_by_tracklet.keys()):
			cands = aggs_by_tracklet[tid]
			# Rank candidates using deterministic tie-breakers
			sorted_cands = sorted(
				cands,
				key=lambda a: (-a.total_weight, -a.max_conf, a.first_seen_frame, a.tag_id),
			)
			# Determine which candidates are "strong" under FULL gate
			strong: List[CandidateAgg] = []
			for a in sorted_cands:
				if self._passes_full_gate(a):
					strong.append(a)
			supported_by_tracklet[tid] = strong

			# Type A conflict: if two candidates are strong and margin < min_margin_abs, suppress must_link.
			if len(strong) >= 2:
				top = strong[0]
				runner = strong[1]
				margin = top.total_weight - runner.total_weight
				if margin < self.min_margin_abs:
					audit_events.append(
						self._audit_event(
							event="c2_conflict_two_tags_one_tracklet",
							payload={
								"tracklet_id": tid,
								"top": self._cand_summary(top),
								"runner_up": self._cand_summary(runner),
								"margin": margin,
								"min_margin_abs": self.min_margin_abs,
							},
						)
					)
					continue  # suppress must_link

			# Must-link emission: pick top candidate if it passes FULL gate and wins by margin.
			if not sorted_cands:
				continue
			top = sorted_cands[0]
			if not self._passes_full_gate(top):
				continue
			runner = sorted_cands[1] if len(sorted_cands) >= 2 else None
			if runner is not None:
				if (top.total_weight - runner.total_weight) < self.min_margin_abs:
					# Not enough separation; treat as insufficient consensus.
					audit_events.append(
						self._audit_event(
							event="c2_no_must_link_margin",
							payload={
								"tracklet_id": tid,
								"top": self._cand_summary(top),
								"runner_up": self._cand_summary(runner),
								"margin": top.total_weight - runner.total_weight,
								"min_margin_abs": self.min_margin_abs,
							},
						)
					)
					continue

			strength = "strong" if top.vote_count >= 2 else "tentative"
			must_link_hints.append(
				self._identity_hint(
					tracklet_id=tid,
					anchor_key=f"tag:{top.tag_id}",
					constraint="must_link",
					confidence=min(1.0, max(0.0, top.total_weight)),
					evidence={
						"reason": "tracklet_consensus_tag",
						"strength": strength,
						"tag_id": top.tag_id,
						"vote_count": top.vote_count,
						"total_weight": top.total_weight,
						"max_confidence": top.max_conf,
						"first_seen_frame": top.first_seen_frame,
					},
				)
			)
			audit_events.append(
				self._audit_event(
					event="c2_must_link_emitted",
					payload={
						"tracklet_id": tid,
						"tag_id": top.tag_id,
						"strength": strength,
						"top": self._cand_summary(top),
						"runner_up": self._cand_summary(runner) if runner is not None else None,
					},
				)
			)

		# Type B conflicts: same tag strongly supported by overlapping tracklets.
		cannot_link_hints: List[Dict[str, Any]] = []
		tag_to_tracklets: Dict[str, List[CandidateAgg]] = {}
		for tid in sorted(supported_by_tracklet.keys()):
			for a in supported_by_tracklet[tid]:
				tag_to_tracklets.setdefault(a.tag_id, []).append(a)

		for tag_id in sorted(tag_to_tracklets.keys()):
			supporters = sorted(tag_to_tracklets[tag_id], key=lambda a: a.tracklet_id)
			# deterministic pair iteration
			for i in range(len(supporters)):
				for j in range(i + 1, len(supporters)):
					a = supporters[i]
					b = supporters[j]
					if not self._overlaps(a.tracklet_id, b.tracklet_id, tracklet_spans, frames_by_tracklet):
						continue
					# Emit symmetric cannot-links (A->B and B->A) per manager decision.
					for src, dst in ((a.tracklet_id, b.tracklet_id), (b.tracklet_id, a.tracklet_id)):
						cannot_link_hints.append(
							self._identity_hint(
								tracklet_id=src,
								anchor_key=f"tracklet:{dst}",
								constraint="cannot_link",
								confidence=1.0,
								evidence={
									"reason": "conflict_same_tag_two_tracklets_overlap",
									"tag_id": tag_id,
									"other_tracklet_id": dst,
								},
							)
						)
					# Audit once per unordered pair
					audit_events.append(
						self._audit_event(
							event="c2_cannot_link_emitted",
							payload={
								"tag_id": tag_id,
								"tracklet_a": a.tracklet_id,
								"tracklet_b": b.tracklet_id,
								"a_support": self._cand_summary(a),
								"b_support": self._cand_summary(b),
							},
						)
					)

		# Add a config summary event so runs are interpretable.
		audit_events.insert(
			0,
			self._audit_event(
				event="c2_registry_header",
				payload={
					"min_obs": self.min_obs,
					"min_conf": self.min_conf,
					"min_weight": self.min_weight,
					"min_conf_strong_single": self.min_conf_strong_single,
					"min_margin_abs": self.min_margin_abs,
					"roi_method_multipliers": self.roi_method_multipliers,
					"n_observations": len(obs_sorted),
					"confidence_note": "C2 applies gates on observation.confidence; upstream may be hard-coded",
				},
			),
		)

		return (must_link_hints + cannot_link_hints), audit_events

	def _passes_full_gate(self, a: CandidateAgg) -> bool:
		"""FULL must-link gate used for must_link and for Type A/B support semantics."""
		if a.vote_count < self.min_obs:
			return False
		if a.total_weight < self.min_weight:
			return False
		if a.max_conf < self.min_conf:
			return False
		if a.vote_count >= 2:
			return True
		# vote_count == 1
		return a.max_conf >= self.min_conf_strong_single

	def _overlaps(
		self,
		tracklet_a: str,
		tracklet_b: str,
		spans: Optional[Dict[str, Tuple[int, int]]],
		frames_by_tracklet: Dict[str, set[int]],
	) -> bool:
		"""Determine overlap deterministically.

		Primary: spans overlap by >=1 frame if spans available for both.
		Fallback: observation frames share at least one frame_index.
		"""
		if spans is not None:
			a_span = spans.get(tracklet_a)
			b_span = spans.get(tracklet_b)
			if a_span is not None and b_span is not None:
				a0, a1 = a_span
				b0, b1 = b_span
				return not (a1 < b0 or b1 < a0)
		# fallback
		a_frames = frames_by_tracklet.get(tracklet_a, set())
		b_frames = frames_by_tracklet.get(tracklet_b, set())
		return len(a_frames.intersection(b_frames)) > 0

	def _identity_hint(
		self,
		*,
		tracklet_id: str,
		anchor_key: str,
		constraint: str,
		confidence: float,
		evidence: Dict[str, Any],
	) -> Dict[str, Any]:
		"""Create an IdentityHint JSONL record (schema-safe fields only)."""
		return {
			"schema_version": "0.3.0",
			"artifact_type": "identity_hint",
			"clip_id": self.clip_id,
			"camera_id": self.camera_id,
			"pipeline_version": self.pipeline_version,
			"created_at_ms": self.created_at_ms,
			"tracklet_id": tracklet_id,
			"anchor_key": anchor_key,
			"constraint": constraint,
			"confidence": float(confidence),
			"evidence": dict(evidence),
		}

	def _audit_event(self, *, event: str, payload: Dict[str, Any]) -> Dict[str, Any]:
		return {
			"event": event,
			"stage": "C",
			"clip_id": self.clip_id,
			"camera_id": self.camera_id,
			"pipeline_version": self.pipeline_version,
			"created_at_ms": self.created_at_ms,
			**payload,
		}

	def _cand_summary(self, a: Optional[CandidateAgg]) -> Optional[Dict[str, Any]]:
		if a is None:
			return None
		return {
			"tag_id": a.tag_id,
			"vote_count": a.vote_count,
			"total_weight": a.total_weight,
			"max_confidence": a.max_conf,
			"first_seen_frame": a.first_seen_frame,
		}
