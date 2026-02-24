from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

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
