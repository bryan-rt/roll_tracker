from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

from bjj_pipeline.stages.matches.hysteresis import EngagementInterval
from bjj_pipeline.stages.matches.seeds import SeedInterval


@dataclass(frozen=True)
class MergedInterval:
	person_id_a: str
	person_id_b: str
	start_frame: int
	end_frame: int
	seed_node_ids: List[str]
	seed_match_ids: List[str]


def merge_seeds_by_pair(
	*,
	seeds: List[SeedInterval],
	seed_match_id_by_seed: Dict[Tuple[str, str, int, int, str], str],
	max_gap_frames: int,
) -> List[MergedInterval]:
	"""Merge seed intervals per unordered pair (person_id_a, person_id_b).

	Merging rule:
	- Only merge within same (person_id_a, person_id_b)
	- Sort by start_frame then end_frame
	- Merge if next.start_frame <= current.end_frame + max_gap_frames

	seed_match_id_by_seed is keyed by (a,b,start,end,node_id) to avoid object identity issues.
	"""
	if not seeds:
		return []

	# Group by pair
	by_pair: Dict[Tuple[str, str], List[SeedInterval]] = {}
	for s in seeds:
		k = (s.person_id_a, s.person_id_b)
		by_pair.setdefault(k, []).append(s)

	out: List[MergedInterval] = []

	for (a, b), ss in sorted(by_pair.items(), key=lambda kv: kv[0]):
		ss_sorted = sorted(ss, key=lambda s: (s.start_frame, s.end_frame, s.node_id))

		cur_start = None
		cur_end = None
		cur_node_ids: List[str] = []
		cur_seed_ids: List[str] = []

		def flush() -> None:
			nonlocal cur_start, cur_end, cur_node_ids, cur_seed_ids
			if cur_start is None or cur_end is None:
				return
			# Deterministic evidence ordering
			node_ids = sorted(cur_node_ids)
			seed_ids = sorted(cur_seed_ids)
			out.append(
				MergedInterval(
					person_id_a=a,
					person_id_b=b,
					start_frame=int(cur_start),
					end_frame=int(cur_end),
					seed_node_ids=node_ids,
					seed_match_ids=seed_ids,
				)
			)
			cur_start = None
			cur_end = None
			cur_node_ids = []
			cur_seed_ids = []

		for s in ss_sorted:
			key = (s.person_id_a, s.person_id_b, int(s.start_frame), int(s.end_frame), s.node_id)
			seed_id = seed_match_id_by_seed.get(key)
			if seed_id is None:
				raise KeyError(f"Missing seed_match_id for seed key={key}")

			if cur_start is None:
				cur_start = int(s.start_frame)
				cur_end = int(s.end_frame)
				cur_node_ids = [s.node_id]
				cur_seed_ids = [seed_id]
				continue

			assert cur_end is not None
			if int(s.start_frame) <= int(cur_end) + int(max_gap_frames):
				# Merge/extend
				cur_end = max(int(cur_end), int(s.end_frame))
				cur_node_ids.append(s.node_id)
				cur_seed_ids.append(seed_id)
			else:
				flush()
				cur_start = int(s.start_frame)
				cur_end = int(s.end_frame)
				cur_node_ids = [s.node_id]
				cur_seed_ids = [seed_id]

		flush()

	# Deterministic global ordering
	out.sort(key=lambda m: (m.person_id_a, m.person_id_b, m.start_frame, m.end_frame))
	return out


# ---------------------------------------------------------------------------
# Union: proximity intervals + cap2 seed intervals
# ---------------------------------------------------------------------------

def union_engagement_intervals(
	proximity_intervals: List[EngagementInterval],
	seed_intervals: List[SeedInterval],
	*,
	max_gap_frames: int,
	session_start_frame: int,
	session_end_frame: int,
	clip_buffer_frames: int,
) -> List[EngagementInterval]:
	"""Union proximity intervals and cap2 seed intervals per pair.

	Steps:
	1. Convert SeedIntervals to EngagementIntervals (evidence=cap2_group)
	2. Per pair: merge all intervals where gap <= max_gap_frames
	3. Tag merged evidence_sources as union of constituent sources
	4. Apply clip_buffer_frames: expand start/end, clamp to session bounds
	5. Set partial_start/end flags based on proximity to session bounds
	"""
	# Step 1: convert seeds to EngagementIntervals
	seed_engagement: List[EngagementInterval] = []
	for s in seed_intervals:
		seed_engagement.append(EngagementInterval(
			person_id_a=s.person_id_a,
			person_id_b=s.person_id_b,
			start_frame=int(s.start_frame),
			end_frame=int(s.end_frame),
			evidence_sources=("cap2_group",),
			partial_start=False,
			partial_end=False,
		))

	# Pool all intervals
	all_intervals = list(proximity_intervals) + seed_engagement

	if not all_intervals:
		return []

	# Step 2: group by pair and merge
	by_pair: Dict[Tuple[str, str], List[EngagementInterval]] = {}
	for iv in all_intervals:
		k = (iv.person_id_a, iv.person_id_b)
		by_pair.setdefault(k, []).append(iv)

	result: List[EngagementInterval] = []

	for (a, b), ivs in sorted(by_pair.items()):
		ivs_sorted = sorted(ivs, key=lambda i: (i.start_frame, i.end_frame))

		# Merge overlapping/adjacent intervals
		merged_groups: List[List[EngagementInterval]] = []
		cur_group: List[EngagementInterval] = [ivs_sorted[0]]
		cur_end = ivs_sorted[0].end_frame

		for iv in ivs_sorted[1:]:
			if iv.start_frame <= cur_end + max_gap_frames:
				cur_group.append(iv)
				cur_end = max(cur_end, iv.end_frame)
			else:
				merged_groups.append(cur_group)
				cur_group = [iv]
				cur_end = iv.end_frame
		merged_groups.append(cur_group)

		# Step 3+4+5: build merged intervals with buffer and partial flags
		for group in merged_groups:
			raw_start = min(iv.start_frame for iv in group)
			raw_end = max(iv.end_frame for iv in group)

			# Union evidence sources
			sources: Set[str] = set()
			for iv in group:
				sources.update(iv.evidence_sources)

			# Inherit partial flags from constituents
			any_partial_start = any(iv.partial_start for iv in group)
			any_partial_end = any(iv.partial_end for iv in group)

			# Step 4: apply buffer and clamp
			buffered_start = max(raw_start - clip_buffer_frames, session_start_frame)
			buffered_end = min(raw_end + clip_buffer_frames, session_end_frame)

			# Step 5: partial flags
			partial_start = any_partial_start or (
				buffered_start == session_start_frame
				and raw_start - session_start_frame <= clip_buffer_frames
			)
			partial_end = any_partial_end or (
				buffered_end == session_end_frame
				and session_end_frame - raw_end <= clip_buffer_frames
			)

			result.append(EngagementInterval(
				person_id_a=a,
				person_id_b=b,
				start_frame=int(buffered_start),
				end_frame=int(buffered_end),
				evidence_sources=tuple(sorted(sources)),
				partial_start=partial_start,
				partial_end=partial_end,
			))

	result.sort(key=lambda i: (i.person_id_a, i.person_id_b, i.start_frame))
	return result
