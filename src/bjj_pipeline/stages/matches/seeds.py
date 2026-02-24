from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd


@dataclass(frozen=True)
class SeedInterval:
	person_id_a: str
	person_id_b: str
	start_frame: int
	end_frame: int
	node_id: str


def extract_cap2_seeds(spans_df: pd.DataFrame) -> List[SeedInterval]:
	"""Build cap2 seed intervals from stage_D/person_spans.parquet.

	Required columns:
		- person_id, node_id, start_frame, end_frame, effective_cap
	"""
	if spans_df is None or spans_df.empty:
		return []

	required = ["person_id", "node_id", "start_frame", "end_frame", "effective_cap"]
	missing = [c for c in required if c not in spans_df.columns]
	if missing:
		raise ValueError(f"person_spans missing required columns: {missing}")

	cap2 = spans_df[spans_df["effective_cap"] == 2].copy()
	if cap2.empty:
		return []

	# Deterministic ordering
	cap2 = cap2.sort_values(["node_id", "person_id", "start_frame", "end_frame"])

	out: List[SeedInterval] = []

	# Group by node_id; require exactly 2 distinct persons per node
	for node_id, g in cap2.groupby("node_id", sort=True):
		persons = list(g["person_id"].unique())
		if len(persons) != 2:
			continue

		p1, p2 = sorted([str(persons[0]), str(persons[1])])

		# For each person, compute the union span on that node.
		# (D4 usually emits one span per person per node, but we harden anyway.)
		g1 = g[g["person_id"] == p1]
		g2 = g[g["person_id"] == p2]

		s1 = int(g1["start_frame"].min())
		e1 = int(g1["end_frame"].max())
		s2 = int(g2["start_frame"].min())
		e2 = int(g2["end_frame"].max())

		start = max(s1, s2)
		end = min(e1, e2)
		if end < start:
			continue

		out.append(
			SeedInterval(
				person_id_a=p1,
				person_id_b=p2,
				start_frame=int(start),
				end_frame=int(end),
				node_id=str(node_id),
			)
		)

	out.sort(key=lambda s: (s.person_id_a, s.person_id_b, s.start_frame, s.end_frame, s.node_id))
	return out
