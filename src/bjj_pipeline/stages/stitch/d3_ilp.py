"""Stage D3 — ILP structure solve (POC_1).

This module solves the *structure-only* stitching problem on the pruned D1/D2 graph
produced by d3_compile.compile_solver_inputs().

POC_1 scope:
  - No must_link / cannot_link enforcement yet
  - No person_id extraction yet
  - Emit debug artifacts + audit summary only
"""

from __future__ import annotations

import time
import json
import math
import statistics
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import pandas as pd
from ortools.sat.python import cp_model

try:
	# Optional (dev-only): used to write a human-readable CP-SAT model dump.
	from google.protobuf import text_format  # type: ignore
except Exception:  # pragma: no cover
	text_format = None  # type: ignore

from bjj_pipeline.contracts.f0_manifest import ClipManifest
from bjj_pipeline.contracts.f0_paths import ClipOutputLayout
from bjj_pipeline.stages.stitch.d3_audit import append_audit_event
from bjj_pipeline.stages.stitch.d3_compile import CompiledInputs


@dataclass(frozen=True)
class ILPResult:
	status: str
	objective_scaled: int | None
	objective_value: float | None
	runtime_ms: int
	selected_edge_ids: List[str]
	flow_by_edge_id: Dict[str, int]
	cost_scale: int
	# Transparency/debugging for objective discretization and model constraints.
	enforced_min_one_path: bool
	rounding_n_edges: int
	rounding_n_edges_nonzero: int
	rounding_max_abs_scaled_error: float
	rounding_max_abs_cost_error: float
	# Tracklet "explain-or-penalize" diagnostics
	unexplained_tracklet_penalty: float | None
	n_tracklets_total: int
	n_tracklets_explained: int
	n_tracklets_unexplained: int
	# Deterministic lists for full transparency
	dropped_tracklet_ids: List[str]
	explained_tracklet_ids: List[str]


def _debug_dir(layout: ClipOutputLayout) -> Path:
	return layout.clip_root / "_debug"


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
	"""Read a JSONL file into a list of dicts.

	Determinism: preserves file order; callers may sort if desired.
	"""
	if not path.exists():
		return []
	text = path.read_text(encoding="utf-8").strip()
	if not text:
		return []
	recs: List[Dict[str, Any]] = []
	for i, line in enumerate(text.splitlines()):
		ln = line.strip()
		if not ln:
			continue
		try:
			recs.append(json.loads(ln))
		except Exception as e:
			raise ValueError(f"Failed to parse JSONL at {path} line {i+1}: {e}")
	return recs


def _stable_tag_sort_key(anchor_key: str) -> Tuple[int, str]:
	"""Sort tags deterministically: numeric tag id first when possible."""
	try:
		if anchor_key.startswith("tag:"):
			n = int(anchor_key.split(":", 1)[1])
			return (n, anchor_key)
	except Exception:
		pass
	return (10**18, anchor_key)


def _compute_graph_reachability(*, nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> Dict[str, Any]:
	"""Graph-only reachability diagnostics (ignores costs/ILP). Deterministic and cheap.

	Returns:
	  - source_id, sink_id
	  - graph_any_source_to_sink_path
	  - reachable_from_source_set, can_reach_sink_set (as python sets of node_id strings)
	"""
	if "node_id" not in nodes_df.columns or "node_type" not in nodes_df.columns:
		raise ValueError("nodes_df must include node_id and node_type")
	if "u" not in edges_df.columns or "v" not in edges_df.columns:
		raise ValueError("edges_df must include u and v")

	source_id = _find_unique_node_id(nodes_df=nodes_df, node_type="NodeType.SOURCE")
	sink_id = _find_unique_node_id(nodes_df=nodes_df, node_type="NodeType.SINK")

	out_adj: Dict[str, List[str]] = defaultdict(list)
	in_adj: Dict[str, List[str]] = defaultdict(list)
	for _, r in edges_df.iterrows():
		u = str(r["u"])
		v = str(r["v"])
		out_adj[u].append(v)
		in_adj[v].append(u)

	# Deterministic traversal order
	for k in list(out_adj.keys()):
		out_adj[k] = sorted(out_adj[k])
	for k in list(in_adj.keys()):
		in_adj[k] = sorted(in_adj[k])

	reachable: Set[str] = set()
	q: deque[str] = deque([str(source_id)])
	reachable.add(str(source_id))
	while q:
		cur = q.popleft()
		for nxt in out_adj.get(cur, []):
			if nxt in reachable:
				continue
			reachable.add(nxt)
			q.append(nxt)

	can_reach_sink: Set[str] = set()
	q2: deque[str] = deque([str(sink_id)])
	can_reach_sink.add(str(sink_id))
	while q2:
		cur = q2.popleft()
		for prv in in_adj.get(cur, []):
			if prv in can_reach_sink:
				continue
			can_reach_sink.add(prv)
			q2.append(prv)

	return {
		"source_id": str(source_id),
		"sink_id": str(sink_id),
		"graph_any_source_to_sink_path": bool(str(sink_id) in reachable),
		"reachable_from_source_set": reachable,
		"can_reach_sink_set": can_reach_sink,
	}


def _compute_k_metrics(*, nodes_df: pd.DataFrame, edges_df: pd.DataFrame, res: "ILPResult", k_paths_inferred: int | None) -> Dict[str, Any]:
	"""Make the meaning of 'K' / n_paths explicit for the ledger.

	In this codebase, n_paths is SOURCE outflow (sum of selected edge flows leaving SOURCE).
	We also compute a graph-only upper bound from SOURCE outgoing edge capacities.
	"""
	source_id = _find_unique_node_id(nodes_df=nodes_df, node_type="NodeType.SOURCE")
	sink_id = _find_unique_node_id(nodes_df=nodes_df, node_type="NodeType.SINK")

	# Graph-only capacity sums (upper bounds)
	k_max_from_source: int | None = None
	k_max_to_sink: int | None = None
	try:
		if "capacity" in edges_df.columns:
			src_out = edges_df[edges_df["u"].astype(str) == str(source_id)].copy()
			snk_in = edges_df[edges_df["v"].astype(str) == str(sink_id)].copy()
			k_max_from_source = int(src_out["capacity"].astype(int).sum()) if len(src_out) > 0 else 0
			k_max_to_sink = int(snk_in["capacity"].astype(int).sum()) if len(snk_in) > 0 else 0
	except Exception:
		# Keep as None; ledger will show unknown rather than guessing.
		k_max_from_source = None
		k_max_to_sink = None

	return {
		# What "K" means operationally here:
		"k_paths_definition": "sum(flow on edges leaving SOURCE)",
		"k_paths_inferred_from_solution": int(k_paths_inferred) if k_paths_inferred is not None else None,
		"k_min_required_by_constraints": 1 if bool(res.enforced_min_one_path) else 0,
		"k_max_possible_from_graph_source_cap": k_max_from_source,
		"k_max_possible_from_graph_sink_cap": k_max_to_sink,
		# Archive_4: there is no explicit max-paths parameter; record this to remove ambiguity.
		"k_is_explicitly_capped": False,
		"k_cap_value": None,
	}


def _forced_tag_overlap_and_counts(
	*,
	nodes_df: pd.DataFrame,
	forced_solo_node_labels: Dict[str, str],
	forced_group_node_labels: Dict[str, Set[str]],
) -> Dict[str, Any]:
	"""Pre-solve diagnostics for forced tag labels:
	- Detect any same-tag temporal overlap among forced nodes (guaranteed infeasible under no-overlap rule).
	- Report max simultaneous nodes per tag (same-tag overlap peak).
	- Report max simultaneous distinct tags (explicitly allowed).
	"""
	if "node_id" not in nodes_df.columns or "start_frame" not in nodes_df.columns or "end_frame" not in nodes_df.columns:
		raise ValueError("nodes_df must include node_id, start_frame, end_frame")

	# Index spans for quick lookup
	spans: Dict[str, Tuple[int, int]] = {}
	for _, r in nodes_df.iterrows():
		nid = str(r["node_id"])
		try:
			spans[nid] = (int(r["start_frame"]), int(r["end_frame"]))
		except Exception:
			continue

	# Build forced nodes by tag
	forced_nodes_by_tag: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
	for nid, lbl in forced_solo_node_labels.items():
		lbl_s = str(lbl)
		if not lbl_s.startswith("tag:"):
			continue
		if nid not in spans:
			continue
		s, e = spans[nid]
		forced_nodes_by_tag[lbl_s].append({"node_id": str(nid), "node_type": "SOLO", "start_frame": s, "end_frame": e})

	for gid, reqs in forced_group_node_labels.items():
		if gid not in spans:
			continue
		s, e = spans[gid]
		for k in reqs:
			k_s = str(k)
			if not k_s.startswith("tag:"):
				continue
			forced_nodes_by_tag[k_s].append({"node_id": str(gid), "node_type": "GROUP", "start_frame": s, "end_frame": e})

	# Same-tag overlap detection
	overlaps: List[Dict[str, Any]] = []
	max_simul_by_tag: Dict[str, int] = {}

	for k, lst in sorted(forced_nodes_by_tag.items(), key=lambda kv: _stable_tag_sort_key(kv[0])):
		lst2 = sorted(lst, key=lambda x: (int(x["start_frame"]), int(x["end_frame"]), str(x["node_id"])))
		# Check overlaps and compute peak overlap count via sweep
		active_ends: List[int] = []
		peak = 0
		for i in range(len(lst2)):
			a = lst2[i]
			a_s = int(a["start_frame"])
			a_e = int(a["end_frame"])
			# Evict ended
			active_ends = [ee for ee in active_ends if ee >= a_s]
			active_ends.append(a_e)
			peak = max(peak, len(active_ends))
			# Pairwise overlap report (small lists only)
			for j in range(i):
				b = lst2[j]
				b_s = int(b["start_frame"])
				b_e = int(b["end_frame"])
				if (a_s <= b_e) and (b_s <= a_e):
					overlaps.append(
						{
							"tag": str(k),
							"node_a": str(b["node_id"]),
							"span_a": [int(b_s), int(b_e)],
							"node_b": str(a["node_id"]),
							"span_b": [int(a_s), int(a_e)],
						}
					)
		max_simul_by_tag[str(k)] = int(peak)

	# Distinct tags simultaneous peak (allowed): sweep over forced intervals
	events: List[Tuple[int, int, str]] = []
	for k, lst in forced_nodes_by_tag.items():
		for x in lst:
			events.append((int(x["start_frame"]), +1, str(k)))
			events.append((int(x["end_frame"]) + 1, -1, str(k)))  # end inclusive → drop after end
	events.sort(key=lambda t: (t[0], -t[1], t[2]))
	active_tags: Dict[str, int] = defaultdict(int)
	peak_distinct = 0
	for _, delta, k_s in events:
		active_tags[k_s] += delta
		if active_tags[k_s] <= 0:
			active_tags.pop(k_s, None)
		peak_distinct = max(peak_distinct, len(active_tags))

	return {
		"forced_nodes_by_tag": {k: forced_nodes_by_tag[k] for k in sorted(forced_nodes_by_tag.keys(), key=_stable_tag_sort_key)},
		"forced_same_tag_overlaps": overlaps,
		"max_simultaneous_nodes_by_tag": max_simul_by_tag,
		"max_simultaneous_distinct_tags": int(peak_distinct),  # explicitly allowed
	}


def _extract_tag_hints(identity_hints: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
	"""Filter Stage C identity_hints down to tag must_link hints.

	Expected (injector) fields:
	  - artifact_type == "identity_hint"
	  - constraint == "must_link"
	  - anchor_key == "tag:<id>"
	  - tracklet_id
	  - evidence.frames (list) and optional evidence.frame_index
	"""
	out: List[Dict[str, Any]] = []
	for h in identity_hints:
		try:
			if str(h.get("artifact_type")) != "identity_hint":
				continue
			if str(h.get("constraint")) != "must_link":
				continue
			ak = str(h.get("anchor_key", ""))
			if not ak.startswith("tag:"):
				continue
			if "tracklet_id" not in h:
				continue
			out.append(h)
		except Exception:
			continue
	# Deterministic ordering (do not rely on JSONL write order)
	def _key(h: Dict[str, Any]) -> Tuple[str, str, int, str]:
		ak = str(h.get("anchor_key", ""))
		tid = str(h.get("tracklet_id", ""))
		ev = h.get("evidence") if isinstance(h.get("evidence"), dict) else {}
		fi = ev.get("frame_index", None)
		try:
			fi_i = int(fi) if fi is not None else -1
		except Exception:
			fi_i = -1
		frames = ev.get("frames", [])
		frames_s = ",".join(str(x) for x in frames) if isinstance(frames, list) else str(frames)
		return (tid, ak, fi_i, frames_s)
	out = sorted(out, key=_key)
	return out


def _infer_frame_span_from_hint(h: Dict[str, Any]) -> Tuple[int | None, int | None, int | None]:
	"""Extract (frame_index, start_frame, end_frame) from an identity_hint record.

	Injector behavior:
	- evidence.frame_index may be present for point pings
	- evidence.frames may be [frame] or [start,end]
	"""
	ev = h.get("evidence") if isinstance(h.get("evidence"), dict) else {}
	fi = ev.get("frame_index", None)
	fi_i: int | None
	try:
		fi_i = int(fi) if fi is not None else None
	except Exception:
		fi_i = None
	frames = ev.get("frames", None)
	if isinstance(frames, list) and len(frames) > 0:
		# frames could be [frame] or [start,end] or longer; interpret min/max for stability
		vals: List[int] = []
		for x in frames:
			try:
				vals.append(int(x))
			except Exception:
				continue
		if len(vals) == 1:
			return (fi_i if fi_i is not None else vals[0], vals[0], vals[0])
		if len(vals) >= 2:
			return (fi_i, min(vals), max(vals))
	return (fi_i, None, None)


def _bind_tag_hints_to_solo_nodes(
	*, nodes_df: pd.DataFrame, tag_hints: List[Dict[str, Any]]
) -> Tuple[List[str], Dict[str, str], List[Dict[str, Any]]]:
	"""Bind tag hints to SINGLE_TRACKLET nodes deterministically.

	Returns:
	- labels_domain: stable ordered list of labels (tag:<id> ... plus UNKNOWN)
	- forced_label_by_solo_node_id: mapping node_id -> required label
	- binding_ledger: records of how each hint was applied (including unbound)
	"""
	if len(tag_hints) == 0:
		return ([], {}, [])

	# Determine label domain from observed tags
	tags = sorted({str(h.get("anchor_key")) for h in tag_hints}, key=_stable_tag_sort_key)
	# UNKNOWN is optional but recommended; keep last for readability
	labels_domain = tags + ["UNKNOWN"]

	nodes = nodes_df.copy()
	for col in ("node_id", "node_type"):
		if col not in nodes.columns:
			raise ValueError(f"d1_graph_nodes missing required column for tag binding: {col}")
	for col in ("start_frame", "end_frame", "base_tracklet_id"):
		if col not in nodes.columns:
			raise ValueError(f"d1_graph_nodes missing required column for tag binding: {col}")
	nodes["node_id"] = nodes["node_id"].astype(str)
	nodes["node_type"] = nodes["node_type"].astype(str)
	nodes["base_tracklet_id"] = nodes["base_tracklet_id"].astype(str)
	# start/end may be floats if parquet has nulls; coerce carefully
	nodes["start_frame"] = pd.to_numeric(nodes["start_frame"], errors="coerce")
	nodes["end_frame"] = pd.to_numeric(nodes["end_frame"], errors="coerce")

	solo = nodes[nodes["node_type"] == "NodeType.SINGLE_TRACKLET"].copy()
	# Preindex by tracklet id for efficiency
	by_tid: Dict[str, pd.DataFrame] = {}
	for tid, g in solo.groupby("base_tracklet_id", sort=False):
		by_tid[str(tid)] = g.copy()

	forced: Dict[str, str] = {}
	# Track best (latest) hint per SOLO node for conflict resolution.
	best_by_node: Dict[str, Dict[str, Any]] = {}
	ledger: List[Dict[str, Any]] = []

	def _effective_frame(fi: int | None, f0: int | None, f1: int | None) -> int | None:
		"""Derive a single comparison frame for recency decisions.

		Preference order:
		- explicit frame_index when provided
		- range end_frame (most recent in the span)
		- range start_frame
		- None when no temporal info is available
		"""
		if fi is not None:
			return fi
		if f1 is not None:
			return f1
		if f0 is not None:
			return f0
		return None

	def _best_node(cands: pd.DataFrame, f: int) -> str:
		# stable tie-break: smallest span then node_id
		c = cands.copy()
		c["span"] = (c["end_frame"] - c["start_frame"]).fillna(10**9)
		c = c.sort_values(["span", "node_id"], kind="mergesort")
		return str(c.iloc[0]["node_id"])

	for h in tag_hints:
		ak = str(h.get("anchor_key"))
		tid = str(h.get("tracklet_id"))
		fi, f0, f1 = _infer_frame_span_from_hint(h)
		eff = _effective_frame(fi, f0, f1)
		rec: Dict[str, Any] = {
			"tracklet_id": tid,
			"anchor_key": ak,
			"frame_index": fi,
			"frames": [f0, f1] if f0 is not None and f1 is not None else None,
			"effective_frame": eff,
			"bound_solo_node_ids": [],
			"status": "unbound",
		}
		cands = by_tid.get(tid)
		if cands is None or len(cands) == 0:
			ledger.append(rec)
			continue

		bound_nodes: List[str] = []
		# Case A: point ping
		if fi is not None:
			cc = cands[(cands["start_frame"] <= fi) & (cands["end_frame"] >= fi)].copy()
			if len(cc) > 0:
				bound_nodes = [_best_node(cc, fi)]
		# Case B: range-only ping: bind-all intersecting nodes (strong truth)
		if len(bound_nodes) == 0 and f0 is not None and f1 is not None:
			cc = cands[(cands["end_frame"] >= f0) & (cands["start_frame"] <= f1)].copy()
			if len(cc) > 0:
				bound_nodes = [str(x) for x in cc["node_id"].astype(str).tolist()]
				bound_nodes = sorted(bound_nodes)
		# Case C: no frame info -> bind-all nodes for this tid
		if len(bound_nodes) == 0 and fi is None and f0 is None and f1 is None:
			bound_nodes = [str(x) for x in cands["node_id"].astype(str).tolist()]
			bound_nodes = sorted(bound_nodes)

		if len(bound_nodes) == 0:
			ledger.append(rec)
			continue

		rec["bound_solo_node_ids"] = bound_nodes
		rec["status"] = "bound"
		ledger.append(rec)
		for nid in bound_nodes:
			prev_meta = best_by_node.get(nid)
			if prev_meta is None:
				best_by_node[nid] = {"anchor_key": ak, "effective_frame": eff}
				forced[nid] = ak
				continue
			prev_label = str(prev_meta.get("anchor_key"))
			prev_eff = prev_meta.get("effective_frame")
			# Same label -> no conflict; keep existing best (may be earlier or later).
			if prev_label == ak:
				continue
			# Resolve conflict by recency when possible: latest ping wins.
			winner_label = prev_label
			winner_eff = prev_eff
			loser_label = ak
			loser_eff = eff
			if eff is not None and (prev_eff is None or eff > prev_eff):
				winner_label = ak
				winner_eff = eff
				loser_label = prev_label
				loser_eff = prev_eff
			best_by_node[nid] = {"anchor_key": winner_label, "effective_frame": winner_eff}
			forced[nid] = winner_label
			ledger.append(
				{
					"status": "conflict",
					"solo_node_id": str(nid),
					"required_labels": sorted([str(prev_label), str(ak)], key=_stable_tag_sort_key),
					"winner_label": str(winner_label),
					"loser_label": str(loser_label),
					"winner_effective_frame": int(winner_eff) if isinstance(winner_eff, int) else None,
					"loser_effective_frame": int(loser_eff) if isinstance(loser_eff, int) else None,
				}
			)

	return (labels_domain, forced, ledger)


def _parse_member_tracklet_ids(raw: Any) -> List[str]:
	"""Parse D1 node member_tracklet_ids value into a list[str].

	The node payload may round-trip through parquet as:
	- list[str]
	- JSON-encoded list[str]
	- a single string
	"""
	if raw is None:
		return []
	if isinstance(raw, list):
		return [str(x) for x in raw if str(x)]
	if isinstance(raw, str):
		s = raw.strip()
		if not s:
			return []
		try:
				obj = json.loads(s)
				if isinstance(obj, list):
					return [str(x) for x in obj if str(x)]
		except Exception:
			pass
		return [s]
	return [str(raw)]


def _parse_payload_obj(raw: Any) -> Dict[str, Any]:
	"""Parse a node payload field into a dict.

	The D1 node parquet includes a lossless payload column (typically `payload_json`)
	which may round-trip as:
	- dict (already parsed)
	- JSON-encoded string
	- None
	Return {} on failure.
	"""
	if raw is None:
		return {}
	if isinstance(raw, dict):
		return raw
	if isinstance(raw, str):
		s = raw.strip()
		if not s:
			return {}
		try:
				obj = json.loads(s)
				return obj if isinstance(obj, dict) else {}
		except Exception:
			return {}
	return {}


def _group_member_ids_from_row(r: "pd.Series") -> List[str]:
	"""Extract member tracklet ids for a GROUP node row deterministically.

	Fallback order:
	1) `member_tracklet_ids` column if present and non-empty
	2) `payload_json` (or `payload`) dict/json containing member_tracklet_ids
	3) explicit tracklet columns (carrier/new/disappearing/base)
	"""
	# 1) Direct column
	try:
		raw_members = r.get("member_tracklet_ids", None)
	except Exception:
		raw_members = None
	members = _parse_member_tracklet_ids(raw_members)
	if len(members) > 0:
		return sorted({m for m in members if m and m != "none"})

	# 2) Lossless payload (preferred)
	payload_raw = None
	if "payload_json" in getattr(r, "index", []):
		payload_raw = r.get("payload_json", None)
	elif "payload" in getattr(r, "index", []):
		payload_raw = r.get("payload", None)
	payload = _parse_payload_obj(payload_raw)
	pm = payload.get("member_tracklet_ids", None)
	members2 = _parse_member_tracklet_ids(pm)
	if len(members2) > 0:
		return sorted({m for m in members2 if m and m != "none"})

	# 3) Explicit columns fallback
	fallback: List[str] = []
	for k in ("carrier_tracklet_id", "new_tracklet_id", "disappearing_tracklet_id", "base_tracklet_id"):
		if k in getattr(r, "index", []):
			v = r.get(k, None)
			if v is None:
				continue
			s = str(v)
			if s and s != "none":
				fallback.append(s)
	# Deterministic dedup + sort
	return sorted({m for m in fallback if m and m != "none"})


def _bind_tag_hints_to_nodes(
	*, nodes_df: pd.DataFrame, tag_hints: List[Dict[str, Any]]
) -> Tuple[List[str], Dict[str, str], Dict[str, Set[str]], List[Dict[str, Any]]]:
	"""Bind tag hints to SOLO or GROUP nodes deterministically.

	Returns:
	- labels_domain: stable ordered list of labels (tag:<id> ... plus UNKNOWN)
	- forced_solo_by_node_id: mapping SOLO node_id -> required label
	- forced_group_by_node_id: mapping GROUP node_id -> set(required labels), size <= 2
	- binding_ledger: records of how each hint was applied (including unbound)
	"""
	if len(tag_hints) == 0:
		return ([], {}, {}, [])

	# Domain is derived from observed concrete tags. Keep UNKNOWN last for readability.
	tags = sorted({str(h.get("anchor_key")) for h in tag_hints}, key=_stable_tag_sort_key)
	labels_domain = tags + ["UNKNOWN"]

	nodes = nodes_df.copy()
	for col in ("node_id", "node_type", "start_frame", "end_frame"):
		if col not in nodes.columns:
			raise ValueError(f"d1_graph_nodes missing required column for tag binding: {col}")
	if "base_tracklet_id" not in nodes.columns:
		raise ValueError("d1_graph_nodes missing required column for tag binding: base_tracklet_id")
	if "member_tracklet_ids" not in nodes.columns:
		nodes["member_tracklet_ids"] = None

	nodes["node_id"] = nodes["node_id"].astype(str)
	nodes["node_type"] = nodes["node_type"].astype(str)
	nodes["base_tracklet_id"] = nodes["base_tracklet_id"].astype(str)
	nodes["start_frame"] = pd.to_numeric(nodes["start_frame"], errors="coerce")
	nodes["end_frame"] = pd.to_numeric(nodes["end_frame"], errors="coerce")

	solo = nodes[nodes["node_type"] == "NodeType.SINGLE_TRACKLET"].copy()
	group = nodes[nodes["node_type"] == "NodeType.GROUP_TRACKLET"].copy()

	# Preindex SOLO nodes by base_tracklet_id for efficiency.
	by_tid_solo: Dict[str, pd.DataFrame] = {}
	for tid, g in solo.groupby("base_tracklet_id", sort=False):
		by_tid_solo[str(tid)] = g.copy()

	# Preindex GROUP nodes by membership tid for efficiency.
	by_tid_group: Dict[str, pd.DataFrame] = {}
	if len(group) > 0:
		rows = []
		for _, r in group.iterrows():
			nid = str(r["node_id"])
			members = _group_member_ids_from_row(r)
			for tid in members:
				rows.append(
					{
						"tid": str(tid),
						"node_id": nid,
						"start_frame": r.get("start_frame", None),
						"end_frame": r.get("end_frame", None),
					}
				)
		if len(rows) > 0:
			m = pd.DataFrame(rows)
			m["start_frame"] = pd.to_numeric(m["start_frame"], errors="coerce")
			m["end_frame"] = pd.to_numeric(m["end_frame"], errors="coerce")
			for tid, g in m.groupby("tid", sort=False):
				by_tid_group[str(tid)] = g.copy()

	forced_solo: Dict[str, str] = {}
	forced_group: Dict[str, Set[str]] = {}
	ledger: List[Dict[str, Any]] = []

	def _best_node_from_df(cands: pd.DataFrame) -> str:
		c = cands.copy()
		c["span"] = (c["end_frame"] - c["start_frame"]).fillna(1e18)
		c = c.sort_values(["span", "node_id"], kind="mergesort")
		return str(c.iloc[0]["node_id"])

	def _span_for_node(nid: str) -> float:
		r = nodes.loc[nodes["node_id"] == nid].iloc[0]
		try:
			return float(r["end_frame"] - r["start_frame"])
		except Exception:
			return 1e18

	for h in tag_hints:
		label = str(h.get("anchor_key"))
		tid = str(h.get("tracklet_id"))
		fi, _, _ = _infer_frame_span_from_hint(h)
		try:
			fi_i = int(fi) if fi is not None else None
		except Exception:
			fi_i = None
		entry: Dict[str, Any] = {
			"tracklet_id": tid,
			"anchor_key": label,
			"frame_index": fi_i,
			"bound_node_id": None,
			"bound_node_type": None,
			"status": "unbound",
			"n_solo_candidates": 0,
			"n_group_candidates": 0,
			"n_total_candidates": 0,
			"chosen_policy": "min_span_then_node_id",
		}
		if fi_i is None:
			ledger.append(entry)
			continue

		best_solo_id: str | None = None
		cand_solo = by_tid_solo.get(tid)
		if cand_solo is not None and len(cand_solo) > 0:
			c = cand_solo[(cand_solo["start_frame"] <= fi_i) & (cand_solo["end_frame"] >= fi_i)].copy()
			entry["n_solo_candidates"] = int(len(c))
			if len(c) > 0:
				best_solo_id = _best_node_from_df(c)

		best_group_id: str | None = None
		cand_group = by_tid_group.get(tid)
		if cand_group is not None and len(cand_group) > 0:
			c = cand_group[(cand_group["start_frame"] <= fi_i) & (cand_group["end_frame"] >= fi_i)].copy()
			entry["n_group_candidates"] = int(len(c))
			if len(c) > 0:
				best_group_id = _best_node_from_df(c)

		entry["n_total_candidates"] = int(entry.get("n_solo_candidates", 0) + entry.get("n_group_candidates", 0))

		if best_solo_id is None and best_group_id is None:
			ledger.append(entry)
			continue

		# Choose between SOLO and GROUP candidates deterministically:
		# prefer smaller span; if tie, prefer SOLO; then node_id.
		chosen_id: str
		chosen_type: str
		if best_solo_id is not None and best_group_id is None:
			chosen_id, chosen_type = best_solo_id, "NodeType.SINGLE_TRACKLET"
		elif best_solo_id is None and best_group_id is not None:
			chosen_id, chosen_type = best_group_id, "NodeType.GROUP_TRACKLET"
		else:
			span_s = _span_for_node(str(best_solo_id))
			span_g = _span_for_node(str(best_group_id))
			if span_s < span_g:
				chosen_id, chosen_type = str(best_solo_id), "NodeType.SINGLE_TRACKLET"
			elif span_g < span_s:
				chosen_id, chosen_type = str(best_group_id), "NodeType.GROUP_TRACKLET"
			else:
				# tie: prefer SOLO, then node_id
				if str(best_solo_id) <= str(best_group_id):
					chosen_id, chosen_type = str(best_solo_id), "NodeType.SINGLE_TRACKLET"
				else:
					chosen_id, chosen_type = str(best_group_id), "NodeType.GROUP_TRACKLET"

		entry["bound_node_id"] = chosen_id
		entry["bound_node_type"] = chosen_type
		entry["status"] = "bound_solo" if chosen_type == "NodeType.SINGLE_TRACKLET" else "bound_group"
		ledger.append(entry)

		if chosen_type == "NodeType.SINGLE_TRACKLET":
			prev = forced_solo.get(chosen_id)
			if prev is not None and prev != label:
				raise ValueError(f"Conflicting tag pings bound to the same SOLO node: node={chosen_id} {prev} vs {label}")
			forced_solo[chosen_id] = label
		else:
			forced_group.setdefault(chosen_id, set()).add(label)
			if len(forced_group[chosen_id]) > 2:
				raise ValueError(
					f"Group node received >2 distinct tag pings: node={chosen_id} tags={sorted(forced_group[chosen_id])}"
				)

	return (labels_domain, forced_solo, forced_group, ledger)


def _write_solution_ledger_json(
	*,
	layout: ClipOutputLayout,
	compiled: CompiledInputs,
	res: ILPResult,
	checkpoint: str,
	manifest: ClipManifest,
	tag_info: Dict[str, Any] | None = None,
) -> Path:
	"""Write a D3 solution ledger to _debug for full decision transparency.

	This is a dev-only artifact (not an F0 contract). It ties:
	  - selected edges (with flow) -> D2 cost term breakdown + key features
	  - dropped base_tracklet_ids -> penalty applied
	  - objective decomposition summary
	"""
	dbg = _debug_dir(layout)
	dbg.mkdir(parents=True, exist_ok=True)
	out = dbg / "d3_solution_ledger.json"

	# Selected edges dataframe (deterministic ordering)
	edges_sel = compiled.edges_df.copy()
	edges_sel["edge_id"] = edges_sel["edge_id"].astype(str)
	edges_sel = edges_sel[edges_sel["edge_id"].isin(res.selected_edge_ids)].copy()
	if len(edges_sel) > 0:
		edges_sel = edges_sel.sort_values(["edge_id"], kind="mergesort").reset_index(drop=True)
	else:
		edges_sel = edges_sel.iloc[0:0].copy()

	# Join full costs rows for selected edges (1:1 by edge_id)
	costs_sel = compiled.costs_df.copy()
	costs_sel["edge_id"] = costs_sel["edge_id"].astype(str)
	costs_sel = costs_sel[costs_sel["edge_id"].isin(res.selected_edge_ids)].copy()
	if len(costs_sel) > 0:
		costs_sel = costs_sel.sort_values(["edge_id"], kind="mergesort").reset_index(drop=True)

	if len(edges_sel) > 0 and len(costs_sel) > 0:
		edges_sel = edges_sel.merge(costs_sel, on="edge_id", how="left", validate="1:1", suffixes=("", "_cost"))

	# Flow for each selected edge
	edges_sel["flow"] = edges_sel["edge_id"].map(lambda eid: int(res.flow_by_edge_id.get(str(eid), 0)))

	# Objective decomposition from ledger perspective
	sum_edge_costs = 0.0
	sum_edge_costs_w_flow = 0.0
	n_edge_instances = 0
	if len(edges_sel) > 0 and "total_cost" in edges_sel.columns:
		for _, r in edges_sel.iterrows():
			try:
				c = float(r.get("total_cost", 0.0))
			except Exception:
				c = 0.0
			f = int(r.get("flow", 0))
			sum_edge_costs += c
			sum_edge_costs_w_flow += c * float(f)
			n_edge_instances += int(f)

	penalty = res.unexplained_tracklet_penalty
	dropped = sorted([str(x) for x in (res.dropped_tracklet_ids or [])])
	explained = sorted([str(x) for x in (res.explained_tracklet_ids or [])])
	sum_penalties = 0.0
	if penalty is not None and penalty > 0 and len(dropped) > 0:
		sum_penalties = float(penalty) * float(len(dropped))

	# Build selected edge records with stable schema:
	# - terms: all "term_*" columns present
	# - features: fixed allowlist to avoid drift
	term_cols = [c for c in edges_sel.columns if isinstance(c, str) and c.startswith("term_")]
	term_cols = sorted(term_cols)
	feature_allow = [
		"dt_frames",
		"dt_s",
		"dist_m",
		"v_req_mps",
		"dist_norm",
		"contact_rel",
		"endpoint_flagged",
	]
	selected_edges: List[Dict[str, Any]] = []
	for _, r in edges_sel.iterrows():
		rec: Dict[str, Any] = {
			"edge_id": str(r.get("edge_id")),
			"flow": int(r.get("flow", 0)),
			"edge_type": str(r.get("edge_type")),
			"src_node_id": str(r.get("u")),
			"dst_node_id": str(r.get("v")),
		}
		# Capacity diagnostics (logging only)
		cap_raw = r.get("capacity", None)
		cap_eff_df = r.get("capacity_eff", None) if "capacity_eff" in edges_sel.columns else None
		try:
			rec["capacity_raw"] = int(cap_raw) if cap_raw is not None and not pd.isna(cap_raw) else None
		except Exception:
			rec["capacity_raw"] = None
		payload_fields = _payload_fields_for_logging(r.get("payload_json", None))
		desired_cap = payload_fields.get("payload_desired_capacity")
		try:
			rec["payload_desired_capacity"] = int(desired_cap) if desired_cap is not None else None
		except Exception:
			rec["payload_desired_capacity"] = None

		# IMPORTANT: For ledger consistency, compute effective capacity the same way the solver does:
		# capacity_eff := max(capacity_raw, payload_desired_capacity) when payload_desired_capacity is present.
		# Fall back to dataframe capacity_eff only if desired_capacity is missing.
		cap_eff_calc: int | None = None
		try:
			if isinstance(rec.get("capacity_raw"), int) and isinstance(rec.get("payload_desired_capacity"), int):
				cap_eff_calc = max(int(rec["capacity_raw"]), int(rec["payload_desired_capacity"]))
		except Exception:
			cap_eff_calc = None
		if cap_eff_calc is None:
			try:
				cap_eff_calc = int(cap_eff_df) if cap_eff_df is not None and not pd.isna(cap_eff_df) else None
			except Exception:
				cap_eff_calc = None
		if cap_eff_calc is None:
			cap_eff_calc = rec.get("capacity_raw")
		rec["capacity_eff"] = cap_eff_calc
		if isinstance(rec.get("capacity_eff"), int):
			rec["expected_var_domain"] = f"0..{rec.get('capacity_eff')}"
		if isinstance(rec.get("payload_desired_capacity"), int) and isinstance(rec.get("capacity_eff"), int):
			rec["desired_gt_eff"] = bool(rec["payload_desired_capacity"] > rec["capacity_eff"])
		# Total cost
		try:
			rec["total_cost"] = float(r.get("total_cost"))
		except Exception:
			rec["total_cost"] = None

		# Term breakdown
		terms: Dict[str, Any] = {}
		for c in term_cols:
			val = r.get(c, None)
			try:
				terms[str(c)] = float(val) if val is not None and (not pd.isna(val)) else None
			except Exception:
				terms[str(c)] = None
		rec["terms"] = terms

		# Key features (stable allowlist)
		feats: Dict[str, Any] = {}
		for c in feature_allow:
			if c not in edges_sel.columns:
				continue
			val = r.get(c, None)
			if val is None or (isinstance(val, float) and math.isnan(val)):
				feats[c] = None
			else:
				# keep ints as ints, floats as floats
				if c == "dt_frames":
					try:
						feats[c] = int(val)
					except Exception:
						feats[c] = None
				elif c == "endpoint_flagged":
					feats[c] = bool(val)
				else:
					try:
						feats[c] = float(val)
					except Exception:
						feats[c] = None
		rec["features"] = feats
		selected_edges.append(rec)

	capacity_summary = {
		"num_selected_edges_with_flow_gt_1": int(
				 sum(1 for e in selected_edges if int(e.get("flow", 0)) > 1)
		),
		"num_selected_edges_payload_desired_capacity_gt_1": int(
				 sum(
					 1
					 for e in selected_edges
					 if isinstance(e.get("payload_desired_capacity"), int)
					 and e.get("payload_desired_capacity") > 1
				 )
		),
		"num_selected_edges_where_desired_gt_eff": int(
				 sum(1 for e in selected_edges if e.get("desired_gt_eff") is True)
		),
		"num_all_edges_payload_desired_capacity_gt_1": int(
				 sum(
					 1
					 for _, rr in compiled.edges_df.iterrows()
					 if isinstance(
						 _payload_fields_for_logging(rr.get("payload_json")).get("payload_desired_capacity"),
						 (int, float),
					 )
					 and int(
						 _payload_fields_for_logging(rr.get("payload_json")).get("payload_desired_capacity")
					 )
					 > 1
				 )
		),
	}

	ledger: Dict[str, Any] = {
		"artifact_type": "d3_solution_ledger",
		"schema_version": 1,
		"checkpoint": checkpoint,
		"created_at_ms": int(time.time() * 1000),
		"clip_id": manifest.clip_id,
		"camera_id": manifest.camera_id,
		"objective": {
			"status": res.status,
			"objective_scaled": res.objective_scaled,
			"objective_value": res.objective_value,
			"cost_scale": res.cost_scale,
			"sum_selected_edge_costs": float(sum_edge_costs),
			"sum_selected_edge_costs_weighted_by_flow": float(sum_edge_costs_w_flow),
			"sum_unexplained_penalties": float(sum_penalties),
			"n_selected_edges": int(len(res.selected_edge_ids)),
			"n_selected_edge_instances": int(n_edge_instances),
			"n_tracklets_total": int(res.n_tracklets_total),
			"n_tracklets_explained": int(res.n_tracklets_explained),
			"n_tracklets_unexplained": int(res.n_tracklets_unexplained),
			"unexplained_tracklet_penalty": res.unexplained_tracklet_penalty,
		},
		"rounding": {
			"rounding_n_edges": res.rounding_n_edges,
			"rounding_n_edges_nonzero": res.rounding_n_edges_nonzero,
			"rounding_max_abs_scaled_error": res.rounding_max_abs_scaled_error,
			"rounding_max_abs_cost_error": res.rounding_max_abs_cost_error,
		},
		"dropped_tracklets": [
			{"base_tracklet_id": tid, "penalty": float(penalty) if penalty is not None else 0.0} for tid in dropped
		],
		"explained_tracklets": explained,
		"capacity_summary": capacity_summary,
		"selected_edges": selected_edges,
	}
	if tag_info is not None:
		# Dev-only extension for POC_2_TAGS transparency.
		ledger["tags"] = tag_info

	out.write_text(json.dumps(ledger, sort_keys=True, indent=2), encoding="utf-8")
	return out


def _extract_entity_paths_format_a(
	*, nodes_df: pd.DataFrame, edges_df: pd.DataFrame, flow_by_edge_id: Dict[str, int]
) -> List[Dict[str, Any]]:
	"""Decompose selected edge flows into per-entity SOURCE->SINK paths (Format A).

	Format A is intended to be human-auditable and temporally monotone:
	  - Each entity is a single path through the DAG from SOURCE to SINK.
	  - Steps are ordered by traversal; we include node frame ranges to sanity-check monotonicity.

	This is a debug/POC artifact only (not an F0 contract).
	"""
	nodes = nodes_df.copy()
	edges = edges_df.copy()
	nodes["node_id"] = nodes["node_id"].astype(str)
	edges["edge_id"] = edges["edge_id"].astype(str)
	edges["u"] = edges["u"].astype(str)
	edges["v"] = edges["v"].astype(str)

	# Index nodes for metadata lookup
	nodes_ix = nodes.set_index("node_id", drop=False)
	edges_ix = edges.set_index("edge_id", drop=False)

	# Build remaining-capacity adjacency from flow
	remaining: Dict[str, int] = {str(k): int(v) for k, v in flow_by_edge_id.items() if int(v) > 0}
	out_by_u: Dict[str, List[str]] = {}
	edge_row: Dict[str, Any] = {}
	for _, e in edges.sort_values(["edge_id"], kind="mergesort").iterrows():
		eid = str(e["edge_id"])
		edge_row[eid] = e.to_dict()
		if remaining.get(eid, 0) <= 0:
			continue
		u = str(e["u"])
		out_by_u.setdefault(u, []).append(eid)

	def node_meta(nid: str) -> Dict[str, Any]:
		if nid not in nodes_ix.index:
			return {"node_id": nid}
		r = nodes_ix.loc[nid]
		out: Dict[str, Any] = {"node_id": str(r.get("node_id")), "node_type": str(r.get("node_type"))}
		# Optional temporal hints
		for k in ("start_frame", "end_frame", "base_tracklet_id", "carrier_tracklet_id", "disappearing_tracklet_id", "new_tracklet_id"):
			if k in r.index and pd.notna(r[k]):
				out[k] = int(r[k]) if isinstance(r[k], (int, float)) and str(r[k]).isdigit() else str(r[k])
		return out

	# Identify SOURCE/SINK ids (robust to missing columns already validated upstream)
	source_id = _find_unique_node_id(nodes, node_type="NodeType.SOURCE")
	sink_id = _find_unique_node_id(nodes, node_type="NodeType.SINK")

	total_flow = 0
	for eid in out_by_u.get(source_id, []):
		total_flow += remaining.get(eid, 0)

	entities: List[Dict[str, Any]] = []
	for ent_i in range(total_flow):
		cur = source_id
		steps: List[Dict[str, Any]] = []
		visited_guard = 0
		tracklets_in_order: List[str] = []

		while cur != sink_id:
			visited_guard += 1
			if visited_guard > 5000:
				# Defensive guard: should never happen in a valid acyclic graph.
				raise RuntimeError("Path extraction exceeded step limit; possible cycle in selected edges.")

			choices = out_by_u.get(cur, [])
			# pick first edge with remaining flow (deterministic due to sorted edge_id order)
			next_eid = None
			for eid in choices:
				if remaining.get(eid, 0) > 0:
					next_eid = eid
					break
			if next_eid is None:
				raise RuntimeError(f"Failed to extract full path: stuck at node_id={cur}")

			remaining[next_eid] -= 1
			if remaining[next_eid] <= 0:
				remaining.pop(next_eid, None)

			e = edge_row[next_eid]
			u = str(e["u"])
			v = str(e["v"])
			step = {
				"edge_id": str(e.get("edge_id")),
				"edge_type": str(e.get("edge_type")),
				"u": u,
				"v": v,
			}
			for k in ("dt_frames", "merge_end", "split_start", "capacity"):
				if k in e and pd.notna(e[k]):
					step[k] = int(e[k]) if isinstance(e[k], (int, float)) else e[k]
			step["u_node"] = node_meta(u)
			step["v_node"] = node_meta(v)

			# Tracklets (best-effort): use base_tracklet_id if present
			for nid in (u, v):
				if nid in nodes_ix.index and "base_tracklet_id" in nodes_ix.columns:
					bt = nodes_ix.loc[nid].get("base_tracklet_id")
					if pd.notna(bt):
						bt_s = str(bt)
						if (len(tracklets_in_order) == 0) or (tracklets_in_order[-1] != bt_s):
							tracklets_in_order.append(bt_s)

			steps.append(step)
			cur = v

		# Temporal monotonicity check (best-effort)
		start_frames = []
		for s in steps:
			vnode = s.get("v_node", {})
			sf = vnode.get("start_frame")
			if isinstance(sf, int):
				start_frames.append(sf)
		temporal_monotone = start_frames == sorted(start_frames)

		entities.append(
			{
				"entity_id": ent_i + 1,
				"steps": steps,
				"tracklets_in_order": tracklets_in_order,
				"temporal_monotone_by_start_frame": bool(temporal_monotone),
				"april_tag_found_in": None,  # filled by _write_entities_format_a
			}
		)

	return entities


def _write_entities_format_a(
	*, layout: ClipOutputLayout, compiled: CompiledInputs, res: ILPResult, checkpoint: str, manifest: ClipManifest
) -> Path:
	"""Write Format A entity paths to _debug and return output path."""
	dbg = _debug_dir(layout)
	dbg.mkdir(parents=True, exist_ok=True)

	entities = _extract_entity_paths_format_a(
		nodes_df=compiled.nodes_df, edges_df=compiled.edges_df, flow_by_edge_id=res.flow_by_edge_id
	)

	# ---- AprilTag annotation (from D2 normalized constraints) ----
	# We annotate an entity with any tag:<id> anchor_keys whose must-link group intersects
	# any tracklet used in the entity path.
	tag_by_tracklet: Dict[str, Set[str]] = {}
	try:
		groups = compiled.constraints.get("must_link_groups", [])
		if isinstance(groups, list):
			for g in groups:
				if not isinstance(g, dict):
					continue
				anchor = g.get("anchor_key")
				if not (isinstance(anchor, str) and anchor.startswith("tag:")):
					continue
				tids = g.get("tracklet_ids", [])
				if not isinstance(tids, list):
					continue
				for tid in tids:
					if isinstance(tid, str) and tid:
						tag_by_tracklet.setdefault(tid, set()).add(anchor)
	except Exception:
		# Best-effort only; never break debug writing.
		tag_by_tracklet = {}

	for ent in entities:
		tids = ent.get("tracklets_in_order", [])
		found: Set[str] = set()
		if isinstance(tids, list):
			for tid in tids:
				if isinstance(tid, str) and tid in tag_by_tracklet:
					found |= tag_by_tracklet[tid]
		ent["april_tag_found_in"] = sorted(found) if found else None

	out = dbg / "d3_entities_format_a.json"
	payload = {
		"schema_version": 1,
		"checkpoint": checkpoint,
		"clip_id": manifest.clip_id,
		"camera_id": manifest.camera_id,
		"n_entities": len(entities),
		"entities": entities,
	}
	out.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")
	return out


def _cost_scale_for(costs: pd.Series) -> int:
	"""Choose a deterministic integer scale so we can use CP-SAT's integer objective.

	We pick 1000 which preserves 0.001 precision; D2 costs commonly include 0.01 increments.
	We also validate that costs * scale are close to integers.
	"""
	return 1000



def _scaled_costs(costs_df: pd.DataFrame, *, scale: int) -> tuple[Dict[str, int], Dict[str, float | int]]:
	if "edge_id" not in costs_df.columns or "total_cost" not in costs_df.columns:
		raise ValueError("d2_edge_costs missing required columns: edge_id, total_cost")
	out: Dict[str, int] = {}
	n_edges = 0
	n_nonzero = 0
	max_abs_scaled_err = 0.0
	for _, row in costs_df.iterrows():
		n_edges += 1
		edge_id = str(row["edge_id"])
		c = float(row["total_cost"])
		s = c * scale
		rounded = int(round(s))
		err = abs(s - float(rounded))
		if err > 0.0:
			nonlocal_max = max_abs_scaled_err
			if err > nonlocal_max:
				max_abs_scaled_err = err
			n_nonzero += 1
		out[edge_id] = rounded
	stats: Dict[str, float | int] = {
		"rounding_n_edges": int(n_edges),
		"rounding_n_edges_nonzero": int(n_nonzero),
		"rounding_max_abs_scaled_error": float(max_abs_scaled_err),
		"rounding_max_abs_cost_error": float(max_abs_scaled_err) / float(scale) if scale > 0 else float("nan"),
	}
	return out, stats


def _find_unique_node_id(nodes_df: pd.DataFrame, *, node_type: str) -> str:
	if "node_type" not in nodes_df.columns or "node_id" not in nodes_df.columns:
		raise ValueError("d1_graph_nodes missing required columns: node_id, node_type")
	m = nodes_df[nodes_df["node_type"].astype(str) == node_type]
	if len(m) != 1:
		raise ValueError(f"Expected exactly 1 node with node_type={node_type}, found {len(m)}")
	return str(m.iloc[0]["node_id"])


def _emit_d3_ilp_transparency_json(
	*,
	debug_dir: Path,
	nodes: pd.DataFrame,
	edges: pd.DataFrame,
	use_flow_int: bool,
	max_edge_cap: int,
	max_node_cap: int,
	enforce_solo_labels: bool,
	node_cap_eff: Dict[str, int],
	node_incident_max_edge_cap_eff: Dict[str, int],
	node_cap_drivers: Dict[str, List[str]],
) -> Path:
	"""Emit a deterministic, dev-only transparency artifact for D3 ILP.

	This artifact is intended to answer:
	  - Did we compute capacity_eff as intended (e.g., payload_json.desired_capacity)?
	  - Did use_flow_int flip on?
	  - Which edges are expected to be IntVar vs BoolVar?
	  - Basic degree/capacity stats that can reveal structural impossibilities pre-solve.

	This is *not* an F0 contract artifact.
	"""
	debug_dir.mkdir(parents=True, exist_ok=True)
	out = debug_dir / "d3_ilp_transparency.json"

	# Edge summaries (deterministic ordering)
	edges2 = edges.copy()
	if "edge_id" in edges2.columns:
		edges2["edge_id"] = edges2["edge_id"].astype(str)
		edges2 = edges2.sort_values(["edge_id"], kind="mergesort").reset_index(drop=True)

	def _payload_summary(payload_json: Any) -> Dict[str, Any]:
		"""Best-effort parse of payload_json into a compact, stable summary.

		NOTE: This is dev-only transparency. Keep it robust to partial/legacy payloads.
		"""
		out: Dict[str, Any] = {
			"payload_json_parse_ok": False,
			"payload_desired_capacity": None,
			"payload_dest_groupish": None,
			"payload_src_groupish": None,
			"payload_promoted_dest": None,
			"payload_promoted_src": None,
			"payload_reconnect": None,
		}
		if not isinstance(payload_json, str) or not payload_json:
			return out
		try:
			payload = json.loads(payload_json)
		except Exception:
			return out
		out["payload_json_parse_ok"] = True
		if not isinstance(payload, dict):
			return out
		# Common fields we rely on for capacity/continuation semantics
		try:
			if payload.get("desired_capacity", None) is not None:
				out["payload_desired_capacity"] = int(payload.get("desired_capacity"))
		except Exception:
			out["payload_desired_capacity"] = None
		if "dest_groupish" in payload:
			out["payload_dest_groupish"] = bool(payload.get("dest_groupish"))
		if "src_groupish" in payload:
			out["payload_src_groupish"] = bool(payload.get("src_groupish"))
		if "promoted_dest" in payload:
			out["payload_promoted_dest"] = bool(payload.get("promoted_dest"))
		if "promoted_src" in payload:
			out["payload_promoted_src"] = bool(payload.get("promoted_src"))
		if "reconnect" in payload:
			out["payload_reconnect"] = bool(payload.get("reconnect"))
		return out

	def _is_group_continuation_edge(payload_sum: Dict[str, Any]) -> bool:
		"""Domain predicate: edge is intended to carry GROUP flow across occlusion.

		We treat 'group continues as group' as:
		  - payload.dest_groupish == True
		  - payload.desired_capacity == 2
		"""
		return bool(
			payload_sum.get("payload_dest_groupish") is True
			and payload_sum.get("payload_desired_capacity") == 2
		)

	edges_cap_gt_1: List[Dict[str, Any]] = []
	edges_payload_desired_gt_1: List[Dict[str, Any]] = []
	edge_type_counts: Dict[str, int] = {}
	cap_counts: Dict[str, int] = {"cap_eff_eq_1": 0, "cap_eff_gt_1": 0}
	group_continuation_edge_ids: List[str] = []

	for _, r in edges2.iterrows():
		eid = str(r.get("edge_id"))
		etype = str(r.get("edge_type"))
		edge_type_counts[etype] = int(edge_type_counts.get(etype, 0) + 1)
		cap_raw = int(r.get("capacity", 1) or 1)
		cap_eff = int(r.get("capacity_eff", cap_raw) or 1)
		payload_sum = _payload_summary(r.get("payload_json", None))
		if _is_group_continuation_edge(payload_sum):
			group_continuation_edge_ids.append(eid)
		# Diagnostics: edges whose payload *requests* capacity > 1
		desired_cap = payload_sum.get("payload_desired_capacity")
		if isinstance(desired_cap, (int, float)) and int(desired_cap) > 1:
			edges_payload_desired_gt_1.append(
				{
					"edge_id": eid,
					"edge_type": etype,
					"u": str(r.get("u")),
					"v": str(r.get("v")),
					"desired_capacity": int(desired_cap),
					"capacity_raw": cap_raw,
					"capacity_eff": cap_eff,
					"desired_gt_eff": bool(cap_eff < int(desired_cap)),
					"payload_is_group_continuation": _is_group_continuation_edge(payload_sum),
				}
			)
		if cap_eff > 1:
			cap_counts["cap_eff_gt_1"] += 1
			expected_var_kind = "IntVar"
			edges_cap_gt_1.append(
				{
					"edge_id": eid,
					"edge_type": etype,
					"u": str(r.get("u")),
					"v": str(r.get("v")),
					"capacity_raw": cap_raw,
					"capacity_eff": cap_eff,
					**payload_sum,
					"payload_is_group_continuation": _is_group_continuation_edge(payload_sum),
					"expected_var_kind": expected_var_kind,
					"expected_var_domain": f"0..{cap_eff}",
				}
			)
		else:
			cap_counts["cap_eff_eq_1"] += 1

	# Node degree stats
	nodes2 = nodes.copy()
	if "node_id" in nodes2.columns:
		nodes2["node_id"] = nodes2["node_id"].astype(str)
		nodes2 = nodes2.sort_values(["node_id"], kind="mergesort").reset_index(drop=True)
	deg_in: Dict[str, int] = {str(nid): 0 for nid in nodes2["node_id"].astype(str).tolist()}
	deg_out: Dict[str, int] = {str(nid): 0 for nid in nodes2["node_id"].astype(str).tolist()}
	for _, r in edges2.iterrows():
		u = str(r.get("u"))
		v = str(r.get("v"))
		if u in deg_out:
			deg_out[u] += 1
		if v in deg_in:
			deg_in[v] += 1

	node_degrees: List[Dict[str, Any]] = []
	for _, r in nodes2.iterrows():
		nid = str(r.get("node_id"))
		try:
			cap_n = int(r.get("capacity", 1) or 1)
		except Exception:
			cap_n = 1
		node_degrees.append(
			{
				"node_id": nid,
				"node_type": str(r.get("node_type")),
				"capacity": cap_n,
				"capacity_raw": cap_n,
				"capacity_eff": int(node_cap_eff.get(nid, cap_n)),
				"incident_max_edge_cap_eff": int(node_incident_max_edge_cap_eff.get(nid, 1)),
				"cap_eff_driven_by_edges": list(node_cap_drivers.get(nid, [])),
				"deg_in": int(deg_in.get(nid, 0)),
				"deg_out": int(deg_out.get(nid, 0)),
			}
		)

	rec: Dict[str, Any] = {
		"use_flow_int": bool(use_flow_int),
		"max_edge_cap": int(max_edge_cap),
		"max_node_cap": int(max_node_cap),
		"n_nodes": int(len(nodes2)),
		"n_edges": int(len(edges2)),
		"enforce_solo_labels": bool(enforce_solo_labels),
		"edge_type_counts": {k: int(edge_type_counts[k]) for k in sorted(edge_type_counts.keys())},
		"capacity_eff_counts": dict(cap_counts),
		"group_continuation_edge_ids": sorted(group_continuation_edge_ids),
		"edges_capacity_eff_gt_1": edges_cap_gt_1,
		"edges_payload_desired_capacity_gt_1": edges_payload_desired_gt_1,
		"node_degrees": node_degrees,
	}

	with open(out, "w", encoding="utf-8") as f:
		json.dump(rec, f, indent=2, sort_keys=True)
	return out


def _payload_fields_for_logging(payload_json: Any) -> Dict[str, Any]:
	"""Parse a small set of payload_json fields for dev-only diagnostics.

	This helper is intentionally aligned with the transparency writer's logic,
	but centralized so other debug artifacts can reference the same field names.

	No solver behavior should depend on this.
	"""
	out: Dict[str, Any] = {
		"payload_json_parse_ok": False,
		"payload_desired_capacity": None,
		"payload_dest_groupish": None,
		"payload_src_groupish": None,
		"payload_promoted_dest": None,
		"payload_promoted_src": None,
		"payload_reconnect": None,
	}
	if not isinstance(payload_json, str) or not payload_json:
		return out
	try:
		payload = json.loads(payload_json)
	except Exception:
		return out
	out["payload_json_parse_ok"] = True
	if not isinstance(payload, dict):
		return out
	try:
		if payload.get("desired_capacity", None) is not None:
			out["payload_desired_capacity"] = int(payload.get("desired_capacity"))
	except Exception:
		out["payload_desired_capacity"] = None
	if "dest_groupish" in payload:
		out["payload_dest_groupish"] = bool(payload.get("dest_groupish"))
	if "src_groupish" in payload:
		out["payload_src_groupish"] = bool(payload.get("src_groupish"))
	if "promoted_dest" in payload:
		out["payload_promoted_dest"] = bool(payload.get("promoted_dest"))
	if "promoted_src" in payload:
		out["payload_promoted_src"] = bool(payload.get("promoted_src"))
	if "reconnect" in payload:
		out["payload_reconnect"] = bool(payload.get("reconnect"))
	return out


def _emit_d3_ilp_variables_json(
	*,
	debug_dir: Path,
	edges: pd.DataFrame,
	use_flow_int: bool,
	var_by_edge: Dict[str, Any],
	edge_used: Dict[str, Any],
	cost_int: Dict[str, int],
	scale: int,
	costs_df: pd.DataFrame | None = None,
) -> Path:
	"""Emit dev-only variable + objective coefficient diagnostics.

	Goal: make infeasibility debugging possible without guessing.
	This does NOT change solver behavior.
	"""
	debug_dir.mkdir(parents=True, exist_ok=True)
	out = debug_dir / "d3_ilp_variables.json"

	edges2 = edges.copy()
	if "edge_id" in edges2.columns:
		edges2["edge_id"] = edges2["edge_id"].astype(str)
		edges2 = edges2.sort_values(["edge_id"], kind="mergesort").reset_index(drop=True)

	# Optional: indexed view of per-edge cost terms/features from D2.
	costs_index: Dict[str, Dict[str, Any]] = {}
	term_cols: List[str] = []
	feature_cols: List[str] = []
	if costs_df is not None and len(costs_df) > 0 and "edge_id" in costs_df.columns:
		costs = costs_df.copy()
		costs["edge_id"] = costs["edge_id"].astype(str)
		term_cols = [c for c in costs.columns if c.startswith("term_")]
		feature_cols = [
			c
			for c in ["dt_frames", "dt_s", "dist_m", "v_req_mps", "dist_norm", "contact_rel", "endpoint_flagged"]
			if c in costs.columns
		]
		value_cols = [c for c in ["total_cost", *term_cols, *feature_cols] if c in costs.columns]
		# Build a lightweight index: edge_id -> selected scalar fields.
		for _, crow in costs.iterrows():
			eid = str(crow["edge_id"])
			costs_index[eid] = {c: crow.get(c) for c in value_cols}

	def _var_kind(v: Any) -> str:
		# CP-SAT python types are not stable across ortools versions; use domain heuristics.
		try:
			lb = int(getattr(v, "Lb", lambda: 0)())
			ub = int(getattr(v, "Ub", lambda: 0)())
			if lb == 0 and ub == 1:
				return "BoolVar"
			return "IntVar"
		except Exception:
			return "UnknownVar"

	rows: List[Dict[str, Any]] = []
	for _, r in edges2.iterrows():
		eid = str(r.get("edge_id"))
		v = var_by_edge.get(eid)
		vu = edge_used.get(eid)
		cap_raw = int(r.get("capacity", 1) or 1)
		cap_eff = int(r.get("capacity_eff", cap_raw) or cap_raw)
		coef = int(cost_int.get(eid, 0))
		payload_fields = _payload_fields_for_logging(r.get("payload_json"))
		# Attach any available per-edge cost terms/features from costs_df.
		cost_fields: Dict[str, Any] = {}
		if eid in costs_index:
			cinfo = costs_index[eid]
			# Scalar total_cost
			if "total_cost" in cinfo and cinfo["total_cost"] is not None:
				try:
					cost_fields["total_cost"] = float(cinfo["total_cost"])
				except Exception:
					cost_fields["total_cost"] = cinfo["total_cost"]
			# Individual term_* columns
			for col in term_cols:
				val = cinfo.get(col)
				if val is None:
					continue
				try:
					cost_fields[col] = float(val)
				except Exception:
					cost_fields[col] = val
			# Selected feature columns for context
			for col in feature_cols:
				val = cinfo.get(col)
				if val is None:
					continue
				if isinstance(val, (int, float, bool, str)):
					cost_fields[col] = val
				else:
					try:
						cost_fields[col] = float(val)
					except Exception:
						cost_fields[col] = str(val)
		rows.append(
			{
				"edge_id": eid,
				"edge_type": str(r.get("edge_type")),
				"u": str(r.get("u")),
				"v": str(r.get("v")),
				"capacity_raw": int(cap_raw),
				"capacity_eff": int(cap_eff),
				"payload_json": str(r.get("payload_json")) if isinstance(r.get("payload_json"), str) else None,
				**payload_fields,
				**cost_fields,
				"use_flow_int": bool(use_flow_int),
				"var_name": str(getattr(v, "Name", lambda: None)()) if v is not None else None,
				"var_kind": _var_kind(v) if v is not None else None,
				"used_var_name": str(getattr(vu, "Name", lambda: None)()) if vu is not None else None,
				"obj_coef_scaled": int(coef),
				"obj_coef": float(coef) / float(scale) if scale > 0 else None,
			}
		)

	meta = {
		"use_flow_int": bool(use_flow_int),
		"scale": int(scale),
		"n_edges": int(len(edges2)),
		"n_obj_terms": int(len(rows)),
		"n_obj_terms_nonzero": int(sum(1 for x in rows if int(x.get("obj_coef_scaled", 0)) != 0)),
		"n_edges_cap_eff_gt_1": int(sum(1 for x in rows if int(x.get("capacity_eff", 1)) > 1)),
	}

	with open(out, "w", encoding="utf-8") as f:
		json.dump({"meta": meta, "edges": rows}, f, indent=2, sort_keys=True)
	return out


def _emit_d3_ilp_node_equations_json(
	*,
	debug_dir: Path,
	nodes: pd.DataFrame,
	edges: pd.DataFrame,
	in_edges: Dict[str, List[str]],
	out_edges: Dict[str, List[str]],
	node_cap_eff: Dict[str, int],
	node_incident_max_edge_cap_eff: Dict[str, int],
	node_cap_drivers: Dict[str, List[str]],
) -> Path:
	"""Emit dev-only per-node balance/capacity equation ingredients.

	This does not attempt to prove satisfiable/unsat; it provides full visibility into
	which edge-variables participate in each node's constraints.
	"""
	debug_dir.mkdir(parents=True, exist_ok=True)
	out = debug_dir / "d3_ilp_node_equations.json"

	n2 = nodes.copy()
	if "node_id" in n2.columns:
		n2["node_id"] = n2["node_id"].astype(str)
		n2 = n2.sort_values(["node_id"], kind="mergesort").reset_index(drop=True)

	edges2 = edges.copy()
	if "edge_id" in edges2.columns:
		edges2["edge_id"] = edges2["edge_id"].astype(str)
		edges2["u"] = edges2["u"].astype(str)
		edges2["v"] = edges2["v"].astype(str)
		edges2 = edges2.sort_values(["edge_id"], kind="mergesort").reset_index(drop=True)
	etype_by_id = {str(rr["edge_id"]): str(rr.get("edge_type")) for _, rr in edges2.iterrows()}
	# Per-edge metadata for node-level diagnostics (dev-only)
	edge_meta_by_id: Dict[str, Dict[str, Any]] = {}
	for _, rr in edges2.iterrows():
		eid = str(rr.get("edge_id"))
		try:
			cap_raw = int(rr.get("capacity", 1) or 1)
		except Exception:
			cap_raw = 1
		try:
			cap_eff = int(rr.get("capacity_eff", cap_raw) or cap_raw)
		except Exception:
			cap_eff = cap_raw
		meta: Dict[str, Any] = {
			"u": str(rr.get("u")),
			"v": str(rr.get("v")),
			"capacity_raw": int(cap_raw),
			"capacity_eff": int(cap_eff),
			"payload_json": str(rr.get("payload_json")) if isinstance(rr.get("payload_json"), str) else None,
		}
		meta.update(_payload_fields_for_logging(rr.get("payload_json")))
		edge_meta_by_id[eid] = meta

	rows: List[Dict[str, Any]] = []
	for _, r in n2.iterrows():
		nid = str(r.get("node_id"))
		ntype = str(r.get("node_type"))
		try:
			cap = int(r.get("capacity", 1) or 1)
		except Exception:
			cap = 1
		ins = sorted([str(x) for x in in_edges.get(nid, [])])
		outs = sorted([str(x) for x in out_edges.get(nid, [])])
		rows.append(
			{
				"node_id": nid,
				"node_type": ntype,
				"capacity": int(cap),
				"capacity_raw": int(cap),
				"capacity_eff": int(node_cap_eff.get(nid, cap)),
				"incident_max_edge_cap_eff": int(node_incident_max_edge_cap_eff.get(nid, 1)),
				"cap_eff_driven_by_edges": list(node_cap_drivers.get(nid, [])),
				"in_edges": [
					{"edge_id": eid, "edge_type": etype_by_id.get(eid), **edge_meta_by_id.get(eid, {})}
					for eid in ins
				],
				"out_edges": [
					{"edge_id": eid, "edge_type": etype_by_id.get(eid), **edge_meta_by_id.get(eid, {})}
					for eid in outs
				],
				"deg_in": int(len(ins)),
				"deg_out": int(len(outs)),
			}
		)

	meta = {
		"n_nodes": int(len(n2)),
		"n_edges": int(len(edges2)),
		"n_nodes_zero_in_non_source": int(
																				sum(1 for x in rows if x.get("node_type") != "NodeType.SOURCE" and int(x.get("deg_in", 0)) == 0)
		),
		"n_nodes_zero_out_non_sink": int(
																				sum(1 for x in rows if x.get("node_type") != "NodeType.SINK" and int(x.get("deg_out", 0)) == 0)
		),
	}

	with open(out, "w", encoding="utf-8") as f:
		json.dump({"meta": meta, "nodes": rows}, f, indent=2, sort_keys=True)
	return out


def _emit_d3_cp_model_dump(*, debug_dir: Path, model: cp_model.CpModel) -> Dict[str, str]:
	"""Emit a human-readable CP-SAT model dump (dev-only).

	Best-effort: never fail the solve if this cannot be emitted.
	"""
	debug_dir.mkdir(parents=True, exist_ok=True)
	paths: Dict[str, str] = {}
	try:
		proto = model.Proto()
		stats_path = debug_dir / "d3_cp_model_stats.json"
		stats = {
			"n_variables": int(len(getattr(proto, "variables", []))),
			"n_constraints": int(len(getattr(proto, "constraints", []))),
			"has_objective": bool(getattr(proto, "objective", None) is not None),
		}
		stats_path.write_text(json.dumps(stats, indent=2, sort_keys=True), encoding="utf-8")
		paths["d3_cp_model_stats_json"] = str(stats_path)
		if text_format is not None:
			pbtxt_path = debug_dir / "d3_cp_model.pbtxt"
			pbtxt_path.write_text(text_format.MessageToString(proto), encoding="utf-8")
			paths["d3_cp_model_pbtxt"] = str(pbtxt_path)
	except Exception:
		pass
	return paths


def solve_structure_ilp_core(
	*,
	nodes_df: pd.DataFrame,
	edges_df: pd.DataFrame,
	costs_df: pd.DataFrame,
	constraints: Dict[str, Any] | None = None,
	# Debugging (dev-only; emits _debug artifacts)
	debug_dir: Path | None = None,
	emit_transparency: bool = True,
	# POC_2_TAGS: identity label enforcement (SOLO nodes only)
	enforce_solo_labels: bool = False,
	labels_domain: List[str] | None = None,
	forced_solo_node_labels: Dict[str, str] | None = None,
	forced_group_node_labels: Dict[str, Set[str]] | None = None,
	tag_solution_out: Dict[str, Any] | None = None,
	unexplained_tracklet_penalty: float | None = None,
	unexplained_group_ping_penalty: float | None = None,
	unexplained_solo_ping_penalty: float | None = None,
	# Tag fragmentation (time-separated): prefer continuity but allow multiple disjoint fragments
	tag_fragment_start_penalty: float | None = None,
	tag_overlap_enforced: bool = True,
	group_boundary_window_frames: int = 10,
) -> ILPResult:
	"""Pure core solver used by POC_1 (unit-test friendly; no I/O)."""
	start = time.time()

	# Required columns
	for col in ("node_id", "node_type", "capacity"):
		if col not in nodes_df.columns:
			raise ValueError(f"d1_graph_nodes missing required column: {col}")
	for col in ("edge_id", "u", "v", "edge_type", "capacity"):
		if col not in edges_df.columns:
			raise ValueError(f"d1_graph_edges missing required column: {col}")

	# Normalize ids
	nodes = nodes_df.copy()
	edges = edges_df.copy()
	nodes["node_id"] = nodes["node_id"].astype(str)
	edges["edge_id"] = edges["edge_id"].astype(str)
	edges["u"] = edges["u"].astype(str)
	edges["v"] = edges["v"].astype(str)

	if unexplained_tracklet_penalty is not None and unexplained_tracklet_penalty < 0:
		raise ValueError("unexplained_tracklet_penalty must be >= 0 when provided")

	node_ids = set(nodes["node_id"].tolist())
	if not set(edges["u"]).issubset(node_ids) or not set(edges["v"]).issubset(node_ids):
		missing_u = sorted(set(edges["u"]) - node_ids)[:25]
		missing_v = sorted(set(edges["v"]) - node_ids)[:25]
		raise ValueError(f"Edges reference unknown node_id(s): missing_u={missing_u} missing_v={missing_v}")

	source_id = _find_unique_node_id(nodes, node_type="NodeType.SOURCE")
	sink_id = _find_unique_node_id(nodes, node_type="NodeType.SINK")

	# Default unexplained-tracklet penalty (stable, config-overridable).
	# Manager-locked rule: do NOT derive from max edge costs.
	# Use a stable default ≈ 3–5×(birth + death); we pick 4× as midpoint.
	if unexplained_tracklet_penalty is None:
		# Only compute a default when the clip contains track nodes (otherwise keep disabled).
		has_track_nodes_tmp = nodes["node_type"].astype(str).isin(
			["NodeType.SINGLE_TRACKLET", "NodeType.GROUP_TRACKLET"]
		).any()
		if has_track_nodes_tmp:
			birth_costs: List[float] = []
			death_costs: List[float] = []
			if "edge_id" in costs_df.columns and "total_cost" in costs_df.columns:
				for _, r in costs_df.iterrows():
					eid = str(r["edge_id"])
					c = float(r["total_cost"])
					if eid.startswith("E:BIRTH:"):
						birth_costs.append(c)
					elif eid.startswith("E:DEATH:"):
						death_costs.append(c)
			birth_med = float(statistics.median(birth_costs)) if len(birth_costs) > 0 else 0.0
			death_med = float(statistics.median(death_costs)) if len(death_costs) > 0 else 0.0
			base = birth_med + death_med
			if base > 0.0:
				unexplained_tracklet_penalty = 4.0 * base
			else:
				# Conservative fallback if birth/death edges are absent in this graph.
				unexplained_tracklet_penalty = 1000.0

	# Deterministic edge ordering
	edges = edges.sort_values(["edge_id"], kind="mergesort").reset_index(drop=True)

	# Determine variable type: bool if all capacities are 1, else integer flow.
	# NOTE: Some edges may request a higher capacity via payload_json.desired_capacity.
	# We fold that into an effective capacity so use_flow_int flips on when needed.
	def _effective_edge_capacity(e_row: pd.Series) -> int:
		cap = int(e_row.get("capacity", 1) or 1)
		payload_json = e_row.get("payload_json", None)
		if isinstance(payload_json, str) and payload_json:
			try:
				payload = json.loads(payload_json)
			except Exception:
				payload = None
			if isinstance(payload, dict):
				dc = payload.get("desired_capacity", None)
				if dc is not None:
					try:
						cap = max(cap, int(dc))
					except Exception:
						pass
		return cap

	edges = edges.copy()
	edges["capacity_eff"] = edges.apply(_effective_edge_capacity, axis=1)
	max_edge_cap = int(pd.to_numeric(edges["capacity_eff"], errors="raise").max()) if len(edges) else 1
	max_node_cap = int(pd.to_numeric(nodes["capacity"], errors="raise").max()) if ("capacity" in nodes.columns and len(nodes)) else 1
	use_flow_int = (max_edge_cap > 1) or (max_node_cap > 1)

	scale = _cost_scale_for(costs_df["total_cost"] if "total_cost" in costs_df.columns else pd.Series([], dtype=float))
	cost_int, rounding_stats = _scaled_costs(costs_df, scale=scale)

	model = cp_model.CpModel()
	# Decision variables per edge: either BoolVar (0/1) or IntVar (0..cap).
	var_by_edge: Dict[str, Any] = {}
	for e in edges.itertuples(index=False):
		edge_id = str(e.edge_id)
		cap_e = int(getattr(e, "capacity_eff", getattr(e, "capacity", 1)) or 1)
		# If any edge needs cap>1, we should already have flipped use_flow_int=True above.
		if (not use_flow_int) and cap_e != 1:
			raise ValueError(
				f"Binary edge selection requires capacity_eff=1, got {cap_e} for edge_id={edge_id}"
			)
		if use_flow_int or cap_e > 1:
			var_by_edge[edge_id] = model.NewIntVar(0, cap_e, f"f_{edge_id}")
		else:
			var_by_edge[edge_id] = model.NewBoolVar(f"x_{edge_id}")

	# A helper boolean that indicates whether an edge is used at all (flow >= 1).
	# This preserves existing constraint patterns even when flow vars are IntVar(0..cap).
	edge_used: Dict[str, cp_model.BoolVar] = {}
	for _, e in edges.iterrows():
		edge_id = str(e["edge_id"])
		v = var_by_edge[edge_id]
		cap_e = int(e.get("capacity_eff", e.get("capacity", 1)) or 1)
		if use_flow_int or cap_e > 1:
			u = model.NewBoolVar(f"used_{edge_id}")
			edge_used[edge_id] = u
			model.Add(v >= 1).OnlyEnforceIf(u)
			model.Add(v == 0).OnlyEnforceIf(u.Not())
		else:
			edge_used[edge_id] = v  # BoolVar already

	# Build adjacency lists
	in_edges: Dict[str, List[str]] = {nid: [] for nid in nodes["node_id"].astype(str).tolist()}
	out_edges: Dict[str, List[str]] = {nid: [] for nid in nodes["node_id"].astype(str).tolist()}
	for _, e in edges.iterrows():
		u = str(e["u"])
		v = str(e["v"])
		eid = str(e["edge_id"])
		out_edges[u].append(eid)
		in_edges[v].append(eid)

	# Pre-index node rows for caps / tracklet grouping
	nodes_ix = nodes.set_index("node_id", drop=False)
	edges_ix = edges.set_index("edge_id", drop=False)

	# Effective node capacity: allow nodes to carry >1 units of flow when any incident
	# edge has an effective capacity >1 (e.g., promoted/groupish arrivals).
	#
	# This is critical for "promoted" SINGLE_TRACKLET nodes (like T:t11/T:t14) that are
	# still represented as NodeType.SINGLE_TRACKLET in D1 but may need to carry 2 units
	# of flow to support group-to-group continuity.
	node_cap_eff: Dict[str, int] = {}
	for _, n in nodes.iterrows():
		nid = str(n["node_id"])
		cap_base = int(n.get("capacity", 1) or 1)
		inc = 1
		for eid in in_edges.get(nid, []):
			try:
				inc = max(inc, int(edges_ix.loc[eid].get("capacity_eff", edges_ix.loc[eid].get("capacity", 1)) or 1))
			except Exception:
				pass
		for eid in out_edges.get(nid, []):
			try:
				inc = max(inc, int(edges_ix.loc[eid].get("capacity_eff", edges_ix.loc[eid].get("capacity", 1)) or 1))
			except Exception:
				pass
		node_cap_eff[nid] = max(cap_base, inc)

	# For debugging: record which incident edge(s) drive node_cap_eff above the raw node capacity.
	node_incident_max_edge_cap_eff: Dict[str, int] = {}
	node_cap_drivers: Dict[str, List[str]] = {}
	for _, n in nodes.iterrows():
		nid = str(n["node_id"])
		cap_base = int(n.get("capacity", 1) or 1)
		incident_eids = list(in_edges.get(nid, [])) + list(out_edges.get(nid, []))
		inc_max = 1
		for eid in incident_eids:
			try:
				inc_max = max(inc_max, int(edges_ix.loc[eid].get("capacity_eff", edges_ix.loc[eid].get("capacity", 1)) or 1))
			except Exception:
				pass
		node_incident_max_edge_cap_eff[nid] = int(inc_max)
		if inc_max > cap_base:
			drivers: List[str] = []
			for eid in incident_eids:
				try:
					cap_e = int(edges_ix.loc[eid].get("capacity_eff", edges_ix.loc[eid].get("capacity", 1)) or 1)
				except Exception:
					cap_e = 1
				if cap_e == inc_max:
					drivers.append(str(eid))
			node_cap_drivers[nid] = sorted(set(drivers))
		else:
			node_cap_drivers[nid] = []

	# Dev-only transparency artifact for debugging infeasibility / capacity interactions.
	# Emit after node_cap_eff is computed so the artifact can report effective node capacities.
	if debug_dir is not None and emit_transparency:
		_emit_d3_ilp_transparency_json(
			debug_dir=debug_dir,
			nodes=nodes,
			edges=edges,
			use_flow_int=bool(use_flow_int),
			max_edge_cap=int(max_edge_cap),
			max_node_cap=int(max_node_cap),
			enforce_solo_labels=bool(enforce_solo_labels),
			node_cap_eff=node_cap_eff,
			node_incident_max_edge_cap_eff=node_incident_max_edge_cap_eff,
			node_cap_drivers=node_cap_drivers,
		)

	# Tracklet usage vars (base_tracklet_id). Used for coverage + identity constraints.
	use_tid: Dict[str, cp_model.IntVar] = {}
	# Index SINGLE_TRACKLET nodes by base_tracklet_id
	single_nodes: List[str] = []
	if "base_tracklet_id" in nodes.columns and "node_type" in nodes.columns:
		single = nodes[nodes["node_type"].astype(str) == "NodeType.SINGLE_TRACKLET"].copy()
		if "base_tracklet_id" in single.columns:
			single["base_tracklet_id"] = single["base_tracklet_id"].astype(str)
			for tid in sorted(single["base_tracklet_id"].unique().tolist()):
				use_tid[str(tid)] = model.NewBoolVar(f"tid_used_{tid}")
			single_nodes = [str(x) for x in single["node_id"].astype(str).tolist()]

	# Non-terminal nodes: conservation + capacity
	for _, n in nodes.iterrows():
		nid = str(n["node_id"])
		ntype = str(n["node_type"])
		if ntype in ("NodeType.SOURCE", "NodeType.SINK"):
			continue
		cap_n = int(node_cap_eff.get(nid, int(n.get("capacity", 1) or 1)))
		ins = [var_by_edge[eid] for eid in in_edges[nid]]
		outs = [var_by_edge[eid] for eid in out_edges[nid]]
		model.Add(sum(ins) == sum(outs))
		model.Add(sum(ins) <= cap_n)
		model.Add(sum(outs) <= cap_n)

	# Inflow expressions (useful for coverage / identity constraints).
	flow_in_by_node: Dict[str, Any] = {}
	for _, n in nodes.iterrows():
		nid = str(n["node_id"])
		ins = [var_by_edge[eid] for eid in in_edges[nid]]
		flow_in_by_node[nid] = sum(ins) if len(ins) > 0 else 0

	# ---- POC_2_TAGS: SOLO label variables + constraints ----
	# We intentionally label ONLY capacity=1 SINGLE_TRACKLET nodes in v1.
	u_solo: Dict[str, cp_model.IntVar] = {}
	miss_solo_ping: Dict[str, cp_model.IntVar] = {}
	miss_group_ping: Dict[Tuple[str, str], cp_model.IntVar] = {}
	y_solo: Dict[Tuple[str, str], cp_model.IntVar] = {}
	# Tag fragmentation / overlap diagnostics (must exist even when enforcement is off)
	frag_start_vars: List[cp_model.IntVar] = []
	frag_start_vars_by_tag: Dict[str, List[cp_model.IntVar]] = {}
	n_overlap_constraints_added = 0
	if enforce_solo_labels:
		if labels_domain is None:
			raise ValueError("POC_2_TAGS requires labels_domain when enforce_solo_labels=True")
		if forced_solo_node_labels is None:
			forced_solo_node_labels = {}
		# Deterministic label order: caller must provide stable ordering.
		labels = [str(k) for k in labels_domain]
		# Validate all forced labels exist in the domain.
		missing_labels = sorted({str(v) for v in forced_solo_node_labels.values()} - set(labels))
		if len(missing_labels) > 0:
			raise ValueError(f"POC_2_TAGS forced labels not present in labels_domain: {missing_labels[:25]}")

		solo_nodes_df = nodes[nodes["node_type"].astype(str) == "NodeType.SINGLE_TRACKLET"].copy()
		solo_nodes_df["node_id"] = solo_nodes_df["node_id"].astype(str)
		# Create u_n and y_{n,k} for each SOLO node.
		for nid in sorted([str(x) for x in solo_nodes_df["node_id"].tolist()]):
			# capacity=1 is required for SOLO labeling. Use *effective* node capacity so
			# promoted/groupish SINGLE_TRACKLET nodes (cap_eff=2) are excluded.
			cap_n = int(node_cap_eff.get(nid, 1))
			if cap_n != 1:
				continue
			u = model.NewBoolVar(f"u_solo_{nid}")
			u_solo[nid] = u
			# Link used var to inflow expression (0/1 for SOLO segments).
			model.Add(flow_in_by_node[nid] == u)
			# Exactly one label if used, else none.
			row = []
			for k in labels:
				k_safe = str(k).replace(":", "_")
				y = model.NewBoolVar(f"y_{nid}_{k_safe}")
				y_solo[(nid, k)] = y
				row.append(y)
			model.Add(sum(row) == u)
			# Soft ping binding (if any): either explain the ping by using the node
			# with the required label, or pay a miss penalty.
			req = forced_solo_node_labels.get(nid)
			if req is not None:
				model.Add(y_solo[(nid, str(req))] == u)
				# Ping coverage is soft: use if feasible; otherwise allow a miss with penalty.
				nid_safe = str(nid).replace(":", "_")
				miss = model.NewBoolVar(f"miss_solo_ping_{nid_safe}")
				miss_solo_ping[nid] = miss
				# Either use the node (u==1) or pay a miss penalty (miss==1).
				model.Add(u + miss == 1)

		# Continuity across selected SOLO->SOLO CONTINUE edges
		if "edge_type" in edges.columns:
			for _, e in edges.iterrows():
				if str(e.get("edge_type")) != "EdgeType.CONTINUE":
					continue
				u = str(e.get("u"))
				v = str(e.get("v"))
				if u not in u_solo or v not in u_solo:
					continue
				# CONTINUE edges may have a higher effective capacity (payload_json.desired_capacity).
				# This continuity constraint is driven by a *binary* indicator: edge_used (flow >= 1),
				# never the raw flow var (which may be IntVar(0..cap)).
				eid = str(e.get("edge_id"))
				x_e = edge_used[eid]
				for k in labels:
					# |y_u,k - y_v,k| <= 1 - x_e
					model.Add(y_solo[(u, k)] - y_solo[(v, k)] <= 1 - x_e)
					model.Add(y_solo[(v, k)] - y_solo[(u, k)] <= 1 - x_e)

		# ---- POC_2_TAGS: GROUP label variables + constraints ----
		# Group nodes may carry up to two concrete tags (partner identities). UNKNOWN is implicit.
		u_group: Dict[str, cp_model.IntVar] = {}
		y_group: Dict[Tuple[str, str], cp_model.IntVar] = {}
		if forced_group_node_labels is None:
			forced_group_node_labels = {}
		labels_concrete = [str(k) for k in labels if str(k) != "UNKNOWN"]
		# Validate forced group labels are in the domain and within capacity.
		missing_g = sorted(
			{str(v) for reqs in forced_group_node_labels.values() for v in reqs} - set(labels_concrete)
		)
		if len(missing_g) > 0:
			raise ValueError(f"POC_2_TAGS forced group labels not present in labels_domain: {missing_g[:25]}")

		group_nodes_df = nodes[nodes["node_type"].astype(str) == "NodeType.GROUP_TRACKLET"].copy()
		group_nodes_df["node_id"] = group_nodes_df["node_id"].astype(str)
		for gid in sorted([str(x) for x in group_nodes_df["node_id"].tolist()]):
			u = model.NewBoolVar(f"u_group_{gid}")
			u_group[gid] = u
			# Link used var to inflow expression (0 when unused, >=1 when used).
			fin = flow_in_by_node.get(gid, 0)
			model.Add(fin >= 1).OnlyEnforceIf(u)
			model.Add(fin == 0).OnlyEnforceIf(u.Not())
			row = []
			for k in labels_concrete:
				k_safe = str(k).replace(":", "_")
				y = model.NewBoolVar(f"yg_{gid}_{k_safe}")
				y_group[(gid, k)] = y
				row.append(y)
				model.Add(y <= u)
			# Capacity: at most two tags if used.
			model.Add(sum(row) <= 2 * u)

			# Soft ping binding(s) (if any): group must carry each required tag OR pay a miss penalty.
			reqs = forced_group_node_labels.get(gid, set())
			if len(reqs) > 2:
				raise ValueError(f"POC_2_TAGS group node cannot carry >2 forced tags: {gid} tags={sorted(reqs)}")
			for req in sorted({str(x) for x in reqs}, key=_stable_tag_sort_key):
				# Ping coverage is soft: require the tag to be carried OR pay a miss penalty.
				# If miss==0, then y_group==1 which implies u==1 via y<=u.
				gid_safe = str(gid).replace(":", "_")
				req_safe = str(req).replace(":", "_")
				miss = model.NewBoolVar(f"miss_group_ping_{gid_safe}_{req_safe}")
				miss_group_ping[(gid, req)] = miss
				model.Add(y_group[(gid, req)] + miss == 1)

		# Identity propagation across SOLO<->GROUP edges when an edge is used.
		# If a labeled SOLO node connects to a GROUP node via a selected edge, then the GROUP must include that label.
		if "edge_type" in edges.columns and len(labels_concrete) > 0:
			for _, e in edges.iterrows():
				etype = str(e.get("edge_type"))
				if etype not in ("EdgeType.CONTINUE", "EdgeType.MERGE", "EdgeType.SPLIT"):
					continue
				eid = str(e.get("edge_id"))
				u = str(e.get("u"))
				v = str(e.get("v"))
				used_e = edge_used.get(eid)
				if used_e is None:
					continue
				# SOLO -> GROUP
				if u in u_solo and v in u_group:
					for k in labels_concrete:
						model.Add(y_group[(v, k)] >= y_solo[(u, k)] + used_e - 1)
				# GROUP -> SOLO
				if u in u_group and v in u_solo:
					for k in labels_concrete:
						model.Add(y_group[(u, k)] >= y_solo[(v, k)] + used_e - 1)



		# ------------------------------------------------------------
		# POC_2_TAGS: time-separated tag fragmentation
		#
		# Allow the same AprilTag (tag:<id>) to appear in multiple disconnected
		# time windows (identity fragments), but forbid temporal overlap.
		# Also add a soft penalty for starting new fragments to prefer continuity.
		#
		# This prevents infeasibility when the graph cannot bridge occlusions.
		# ------------------------------------------------------------
		# ------------------------------------------------------------
		# Tag fragmentation (time-separated) — allow multiple fragments per tag,
		# but forbid temporal overlap (one person cannot be in two nodes at once).
		# Also add a soft penalty for starting a new fragment to prefer continuity.
		# ------------------------------------------------------------
		if enforce_solo_labels and labels_domain is not None and tag_overlap_enforced:
			# Build a unified view of label-bearing nodes (SOLO + GROUP) with frame spans.
			# Only enforce for real AprilTag labels.
			apriltag_labels = [str(k) for k in labels_domain if str(k).startswith("tag:")]
			# node_id -> (start_frame, end_frame)
			span_by_node: Dict[str, Tuple[int, int]] = {}
			for _, r in nodes.iterrows():
				nt = str(r.get("node_type"))
				if nt not in ("NodeType.SINGLE_TRACKLET", "NodeType.GROUP_TRACKLET"):
					continue
				nid = str(r.get("node_id"))
				try:
					sf = int(r.get("start_frame"))
					ef = int(r.get("end_frame"))
				except Exception:
					continue
				span_by_node[nid] = (sf, ef)

			def _overlaps(a: Tuple[int,int], b: Tuple[int,int]) -> bool:
				# inclusive overlap on frame indices
				return not (a[1] < b[0] or b[1] < a[0])

			# Collect candidate node ids that have a y-var for a given tag.
			def _y_for(nid: str, k: str) -> cp_model.IntVar | None:
				vv = y_solo.get((nid, k))
				if vv is not None:
					return vv
				return y_group.get((nid, k))

			# Hard no-overlap constraints per tag.
			# Use sweep-line for determinism and to avoid full O(N^2) where possible.
			for k in apriltag_labels:
				cand = [nid for nid in span_by_node.keys() if _y_for(nid, k) is not None]
				cand.sort(key=lambda nid: (span_by_node[nid][0], span_by_node[nid][1], nid))
				active: List[str] = []
				for nid in cand:
					sf, ef = span_by_node[nid]
					# evict non-overlapping
					active = [aid for aid in active if span_by_node[aid][1] >= sf]
					for aid in active:
						# they overlap by construction (aid.end >= nid.start) but keep safe
						if not _overlaps(span_by_node[aid], (sf, ef)):
							continue
						va = _y_for(aid, k)
						vb = _y_for(nid, k)
						if va is None or vb is None:
							continue
						model.Add(va + vb <= 1)
						n_overlap_constraints_added += 1
					active.append(nid)

			# Fragment-start penalty: a node starts a fragment for tag k if it carries k
			# and there is no selected inbound edge from a predecessor that also carries k.
			if tag_fragment_start_penalty is not None and float(tag_fragment_start_penalty) > 0:
				# Precompute inbound edge lists with endpoints
				in_eids_by_node = in_edges
				for k in apriltag_labels:
					frag_start_vars_by_tag[k] = []
					for nid in sorted(span_by_node.keys()):
						y_nk = _y_for(nid, k)
						if y_nk is None:
							continue
						# Build continuity aux vars for each inbound edge from a labeled predecessor
						cont_vars: List[cp_model.IntVar] = []
						for eid in in_eids_by_node.get(nid, []):
							try:
								u = str(edges_ix.loc[eid]["u"])
							except Exception:
								u = None
							if u is None or str(u) == source_id:
								continue
							y_uk = _y_for(str(u), k)
							if y_uk is None:
								continue
							used_e = edge_used.get(eid)
							if used_e is None:
								continue
							c = model.NewBoolVar(f"cont_{eid}_{k.replace(':','_')}")
							model.Add(c <= used_e)
							model.Add(c <= y_uk)
							model.Add(c >= used_e + y_uk - 1)
							cont_vars.append(c)
						# has_prev_same_tag
						has_prev = model.NewBoolVar(f"has_prev_{nid}_{k.replace(':','_')}")
						if len(cont_vars) == 0:
							model.Add(has_prev == 0)
						else:
							model.AddMaxEquality(has_prev, cont_vars)
						# frag_start
						fs = model.NewBoolVar(f"frag_start_{nid}_{k.replace(':','_')}")
						model.Add(fs <= y_nk)
						model.Add(fs <= has_prev.Not())
						model.Add(fs >= y_nk - has_prev)
						frag_start_vars.append(fs)
						frag_start_vars_by_tag[k].append(fs)

	# Terminals: balance total flow (let K be decided by costs)
	src_out = [var_by_edge[eid] for eid in out_edges[source_id]]
	snk_in = [var_by_edge[eid] for eid in in_edges[sink_id]]
	model.Add(sum(src_out) == sum(snk_in))
	# Optional: require >=1 only when the graph actually contains track nodes.
	# This avoids forcing infeasible/meaningless solutions on empty clips.
	has_track_nodes = nodes["node_type"].astype(str).isin(
		["NodeType.SINGLE_TRACKLET", "NodeType.GROUP_TRACKLET"]
	).any()
	enforced_min_one_path = bool(has_track_nodes and len(src_out) > 0)
	if enforced_min_one_path:
		model.Add(sum(src_out) >= 1)

	# ------------------------------------------------------------
	# GROUP_TRACKLET semantics (Worker G)
	#
	# Hard constraints that restore semantic meaning of overlap episodes:
	#  - GROUP_TRACKLET usage is 0-or-2 (never 1).
	#  - When a GROUP_TRACKLET is used, the carrier participant must traverse via the
	#    deterministic chain CONT edge(s), and the second participant must traverse via
	#    the group MERGE/SPLIT structure, except for boundary substitutes:
	#      - Start boundary: second participant already present at t=0 (extra BIRTH capacity).
	#      - End boundary: still merged at end (extra DEATH capacity).
	#
	# If required MERGE/SPLIT/BIRTH/DEATH/CONT edges are missing for a group episode,
	# we force group usage to 0 (never make the model infeasible).
	# ------------------------------------------------------------
	if group_boundary_window_frames < 0:
		raise ValueError("group_boundary_window_frames must be >= 0")

	# Clip boundary frames (for boundary substitutes)
	track_nodes = nodes[nodes["node_type"].astype(str).isin(["NodeType.SINGLE_TRACKLET", "NodeType.GROUP_TRACKLET"])].copy()
	clip_first_frame = 0
	clip_last_frame = 0
	if len(track_nodes) > 0 and "end_frame" in track_nodes.columns:
		end_frames = pd.to_numeric(track_nodes["end_frame"], errors="coerce").dropna()
		if len(end_frames) > 0:
			clip_last_frame = int(end_frames.max())

	# Edge-type lookup for adjacency filtering and endpoint access
	edge_type_by_id: Dict[str, str] = {}
	edge_u_by_id: Dict[str, str] = {}
	edge_v_by_id: Dict[str, str] = {}
	for _, e in edges.iterrows():
		eid_str = str(e["edge_id"])
		edge_type_by_id[eid_str] = str(e.get("edge_type"))
		edge_u_by_id[eid_str] = str(e.get("u"))
		edge_v_by_id[eid_str] = str(e.get("v"))

	# Node-type lookup so we can detect GROUP→GROUP carrier-chain continuation.
	node_type_by_id: Dict[str, str] = {}
	if "node_id" in nodes.columns and "node_type" in nodes.columns:
		for _, n in nodes.iterrows():
			node_type_by_id[str(n.get("node_id"))] = str(n.get("node_type"))

	# ------------------------------------------------------------
	# Segment connectivity constraint (Worker I)
	# For structural SINGLE_TRACKLET→SINGLE_TRACKLET edges that connect
	# segments of the SAME base_tracklet_id, enforce directed joint-usage
	# along time: flow_in[v] >= flow_in[u].
	# (Equality is too strong and can over-constrain valid graphs.)
	# ------------------------------------------------------------
	if "base_tracklet_id" in nodes.columns and "node_type" in nodes.columns:
		single_ix = nodes[nodes["node_type"].astype(str) == "NodeType.SINGLE_TRACKLET"].set_index("node_id", drop=False)
		# Map node_id -> base_tracklet_id for SINGLE_TRACKLET nodes
		node_base: Dict[str, str] = {}
		for nid, r in single_ix.iterrows():
			bt = r.get("base_tracklet_id", None)
			if bt is None or (isinstance(bt, float) and math.isnan(bt)):
				continue
			node_base[str(r["node_id"])] = str(bt)
		for _, e in edges.iterrows():
			u = str(e["u"])
			v = str(e["v"])
			etype = str(e.get("edge_type"))
			# "structural (non-group)" constraint applied only on CONTINUE edges
			# where both endpoints are SINGLE_TRACKLET and share base_tracklet_id.
			if etype != "EdgeType.CONTINUE":
				continue
			bu = node_base.get(u, None)
			bv = node_base.get(v, None)
			if bu is None or bv is None:
				continue
			if bu != bv:
				continue
			# Directed implication along time.
			model.Add(flow_in_by_node[v] >= flow_in_by_node[u])

	# ------------------------------------------------------------
	# Cannot-link enforcement (Worker I)
	# Semantics (PM-confirmed): cannot-link means "must not be the same entity".
	# Implemented structurally by forbidding identity-continuation edges (EdgeType.CONTINUE)
	# that stitch across the cannot-link pair.
	# ------------------------------------------------------------
	if constraints is not None:
		cl_pairs = constraints.get("cannot_link_pairs", None)
		if isinstance(cl_pairs, list) and len(cl_pairs) > 0:
			# Precompute SINGLE_TRACKLET node -> base_tracklet_id mapping (if available).
			node_base2: Dict[str, str] = {}
			if "base_tracklet_id" in nodes.columns and "node_type" in nodes.columns:
				single2 = nodes[nodes["node_type"].astype(str) == "NodeType.SINGLE_TRACKLET"].copy()
				if "base_tracklet_id" in single2.columns:
					single2["base_tracklet_id"] = single2["base_tracklet_id"].astype(str)
					for _, r in single2.iterrows():
						node_base2[str(r["node_id"])] = str(r["base_tracklet_id"])
				# Disable only CONTINUE edges crossing the pair.
				for pair in cl_pairs:
					if not (isinstance(pair, (list, tuple)) and len(pair) == 2):
						continue
					a = str(pair[0])
					b = str(pair[1])
					for _, e in edges.iterrows():
						etype = str(e.get("edge_type"))
						if etype != "EdgeType.CONTINUE":
							continue
						u = str(e["u"])
						v = str(e["v"])
						bu = node_base2.get(u, None)
						bv = node_base2.get(v, None)
						if bu is None or bv is None:
							continue
						if (bu == a and bv == b) or (bu == b and bv == a):
							eid = str(e["edge_id"])
							if eid in var_by_edge:
								model.Add(var_by_edge[eid] == 0)

		# ------------------------------------------------------------
		# Precompute CONTINUE edges by base-tracklet pair (SINGLE↔SINGLE only).
		# Used for safe, conditional group-derived tightening without relying on CONTINUE(d,n).
		# ------------------------------------------------------------
		node_base_single: Dict[str, str] = {}
		if "base_tracklet_id" in nodes.columns and "node_type" in nodes.columns:
			single_tmp = nodes[nodes["node_type"].astype(str) == "NodeType.SINGLE_TRACKLET"].copy()
			if len(single_tmp) > 0 and "base_tracklet_id" in single_tmp.columns:
				single_tmp["node_id"] = single_tmp["node_id"].astype(str)
				single_tmp["base_tracklet_id"] = single_tmp["base_tracklet_id"].astype(str)
				for _, r in single_tmp.iterrows():
					node_base_single[str(r["node_id"])] = str(r["base_tracklet_id"])

		def _pair_key(a: str, b: str) -> Tuple[str, str]:
			return (a, b) if a <= b else (b, a)

		continue_edges_by_pair: Dict[Tuple[str, str], List[str]] = {}
		if len(node_base_single) > 0:
			for _, e in edges.iterrows():
				etype = str(e.get("edge_type"))
				if etype != "EdgeType.CONTINUE":
					continue
				u = str(e["u"])
				v = str(e["v"])
				bu = node_base_single.get(u, None)
				bv = node_base_single.get(v, None)
				if bu is None or bv is None:
					continue
				# Skip same-base segment connectivity; handled elsewhere.
				if bu == bv:
					continue
				key = _pair_key(bu, bv)
				eid = str(e["edge_id"])
				if eid in var_by_edge:
					continue_edges_by_pair.setdefault(key, []).append(eid)

		def _continue_eids(a: str, b: str) -> List[str]:
			return continue_edges_by_pair.get(_pair_key(a, b), [])

		# Optional: AprilTag-derived must-link pairs can supersede group-derived cannot-links.
		tag_must_link: Set[Tuple[str, str]] = set()
		if constraints is not None:
			tml = constraints.get("tag_must_link_pairs", None)
			if isinstance(tml, list):
				for pair in tml:
					if isinstance(pair, (list, tuple)) and len(pair) == 2:
						tag_must_link.add(_pair_key(str(pair[0]), str(pair[1])))

	# For each GROUP_TRACKLET node, enforce structural overlap semantics
	if "carrier_tracklet_id" in nodes.columns and "node_type" in nodes.columns:
		groups = nodes[nodes["node_type"].astype(str) == "NodeType.GROUP_TRACKLET"].copy()

		def _edge_payload_summary(eid: str) -> Dict[str, Any]:
			"""Best-effort payload parse for CONTINUE-edge semantics inside constraints."""
			row = edges_ix.loc[str(eid)]
			payload_json = row.get("payload_json", None)
			out: Dict[str, Any] = {"desired_capacity": None, "dest_groupish": None, "parse_ok": False}
			if isinstance(payload_json, str) and payload_json:
				try:
					payload = json.loads(payload_json)
				except Exception:
					payload = None
				if isinstance(payload, dict):
					out["parse_ok"] = True
					try:
						if payload.get("desired_capacity", None) is not None:
							out["desired_capacity"] = int(payload.get("desired_capacity"))
					except Exception:
						out["desired_capacity"] = None
					if "dest_groupish" in payload:
						out["dest_groupish"] = bool(payload.get("dest_groupish"))
			return out

		def _is_group_continuation_edge(eid: str) -> bool:
			ps = _edge_payload_summary(str(eid))
			return bool(ps.get("dest_groupish") is True and ps.get("desired_capacity") == 2)

		for _, gn in groups.iterrows():
			gid = str(gn["node_id"])

			used = model.NewBoolVar(f"g_used_{gid}")

			# Total flow through group is either 0 or 2.
			in_vars = [var_by_edge[eid] for eid in in_edges.get(gid, [])]
			model.Add(sum(in_vars) == 2 * used)

			# Determine boundary eligibility
			gs = gn.get("start_frame", None)
			ge = gn.get("end_frame", None)
			try:
				gs_i = int(gs) if gs is not None and not (isinstance(gs, float) and math.isnan(gs)) else None
			except Exception:
				gs_i = None
			try:
				ge_i = int(ge) if ge is not None and not (isinstance(ge, float) and math.isnan(ge)) else None
			except Exception:
				ge_i = None

			is_start_boundary = bool(gs_i is not None and gs_i <= (clip_first_frame + int(group_boundary_window_frames) - 1))
			is_end_boundary = bool(ge_i is not None and ge_i >= (clip_last_frame - int(group_boundary_window_frames) + 1))

			# These edges represent the *carrier solo* continuing across segments (capacity 1).
			carrier_cont_in = [
				eid
				for eid in in_edges.get(gid, [])
				if str(eid).startswith("E:CONT:") and edge_type_by_id.get(str(eid)) == "EdgeType.CONTINUE"
			]
			carrier_cont_out = [
				eid
				for eid in out_edges.get(gid, [])
				if str(eid).startswith("E:CONT:") and edge_type_by_id.get(str(eid)) == "EdgeType.CONTINUE"
			]
			carrier_cont_in_eid = sorted([str(x) for x in carrier_cont_in])[0] if len(carrier_cont_in) > 0 else None
			carrier_cont_out_eid = sorted([str(x) for x in carrier_cont_out])[0] if len(carrier_cont_out) > 0 else None

			# Group-continuation CONT edges (payload predicate): carry BOTH identities (capacity 2)
			group_cont_in = [
				eid
				for eid in in_edges.get(gid, [])
				if edge_type_by_id.get(str(eid)) == "EdgeType.CONTINUE" and _is_group_continuation_edge(str(eid))
			]
			group_cont_out = [
				eid
				for eid in out_edges.get(gid, [])
				if edge_type_by_id.get(str(eid)) == "EdgeType.CONTINUE" and _is_group_continuation_edge(str(eid))
			]
			group_cont_in_eid = sorted([str(x) for x in group_cont_in])[0] if len(group_cont_in) > 0 else None
			group_cont_out_eid = sorted([str(x) for x in group_cont_out])[0] if len(group_cont_out) > 0 else None

			# Second participant structural edges
			merge_in = [eid for eid in in_edges.get(gid, []) if edge_type_by_id.get(str(eid)) == "EdgeType.MERGE"]
			split_out = [eid for eid in out_edges.get(gid, []) if edge_type_by_id.get(str(eid)) == "EdgeType.SPLIT"]
			birth_in = f"E:BIRTH:{gid}"
			death_out = f"E:DEATH:{gid}"

			# Metadata flags
			disp = gn.get("disappearing_tracklet_id", None)
			new_tid = gn.get("new_tracklet_id", None)
			has_disp = disp is not None and pd.notna(disp) and str(disp) != "none"
			has_new = new_tid is not None and pd.notna(new_tid) and str(new_tid) != "none"

			# ------------------------------------------------------------
			# Safe group-based identity tightening (PM-owned, stable):
			# When a group episode is USED, prevent the carrier identity from being
			# stitched (via CONTINUE) to the other participant tids implied by the group.
			#
			# This does NOT require CONTINUE(d,n) to exist.
			# It only disables CONTINUE edges for (carrier, disappearing) and (carrier, new),
			# gated by group usage. AprilTag-derived must-links (if provided) supersede.
			# ------------------------------------------------------------
			def _norm_tid(x: Any) -> str | None:
				if x is None:
					return None
				if isinstance(x, float) and math.isnan(x):
					return None
				s = str(x)
				if s == "none":
					return None
				return s

			c_tid = _norm_tid(gn.get("carrier_tracklet_id", None))
			d_tid = _norm_tid(gn.get("disappearing_tracklet_id", None))
			n_tid = _norm_tid(gn.get("new_tracklet_id", None))

			# Carrier cannot-link to disappearing/new, conditional on group usage.
			for a, b in ((c_tid, d_tid), (c_tid, n_tid)):
				if a is None or b is None:
					continue
				key = _pair_key(str(a), str(b))
				# If AprilTag evidence says these must-link, do not apply group-derived cannot-link.
				if key in tag_must_link:
					continue
				for eid in _continue_eids(str(a), str(b)):
					model.Add(var_by_edge[eid] == 0).OnlyEnforceIf(used)

			# Carrier-chain CONT edges (capacity 1) are optional structural edges.
			# When they exist and the group node is used, they should be saturated.
			if carrier_cont_in_eid is not None:
				model.Add(var_by_edge[carrier_cont_in_eid] == 1 * used)
			if carrier_cont_out_eid is not None:
				model.Add(var_by_edge[carrier_cont_out_eid] == 1 * used)

			# Group-continuation CONT edges (capacity 2) are *options* the solver may choose.
			# We do NOT force them on/off here; flow conservation + costs decide.
			if group_cont_in_eid is not None:
				model.Add(var_by_edge[group_cont_in_eid] <= 2 * used)
				# Group-continuation edges are 0 or 2 (never 1)
				model.Add(var_by_edge[group_cont_in_eid] != 1)
			if group_cont_out_eid is not None:
				model.Add(var_by_edge[group_cont_out_eid] <= 2 * used)
				model.Add(var_by_edge[group_cont_out_eid] != 1)

			# Second participant must enter via MERGE unless start-boundary substitute applies.
			if has_disp:
				if len(merge_in) == 0:
					model.Add(used == 0)
				else:
					# D1 emits exactly one MERGE edge into this group segment.
					model.Add(var_by_edge[merge_in[0]] == used)
					# Disallow any additional MERGE edges if present.
					for extra in merge_in[1:]:
						model.Add(var_by_edge[extra] == 0)
			else:
				# No disappearing participant: allow (e.g., group persists from before clip).
				# Disallow MERGE edges in this case.
				for eid in merge_in:
					model.Add(var_by_edge[eid] == 0)

			# Second participant must exit via SPLIT unless end-boundary substitute applies.
			if has_new:
				if len(split_out) == 0:
					model.Add(used == 0)
				else:
					model.Add(var_by_edge[split_out[0]] == used)
					# Split ownership (implication): if group is used, the new tracklet must be considered used.
					tid_new = str(new_tid)
					if tid_new in use_tid:
						model.Add(used <= use_tid[tid_new])
					else:
						use_tid[tid_new] = model.NewBoolVar(f"tid_used_{tid_new}")
						model.Add(used <= use_tid[tid_new])
					for extra in split_out[1:]:
						model.Add(var_by_edge[extra] == 0)
			else:
				# No new participant: allow (e.g., group occludes or continues as group).
				for eid in split_out:
					model.Add(var_by_edge[eid] == 0)

				# ------------------------------------------------------------
				# POC_2_TAGS: cross-GROUP label continuity for carrier chain.
				# When a group episode is used and has deterministic CONT-in/out edges,
				# enforce that the carrier's SOLO label before and after the group match.
				# GROUP nodes themselves remain unlabeled.
				# ------------------------------------------------------------
				if enforce_solo_labels and carrier_cont_in_eid is not None and carrier_cont_out_eid is not None:
					u_before = edge_u_by_id.get(str(carrier_cont_in_eid))
					v_after = edge_v_by_id.get(str(carrier_cont_out_eid))
					if u_before in u_solo and v_after in u_solo and labels_domain is not None:
						for k in [str(lbl) for lbl in labels_domain]:
							model.Add(y_solo[(u_before, k)] == y_solo[(v_after, k)]).OnlyEnforceIf(used)

	# ------------------------------------------------------------
	# Coverage policy (Worker I, manager-locked)
	#
	# Soft coverage with strong penalty:
	#  - use_tid[tid] ∈ {0,1} indicates whether base tracklet tid is included in the explanation.
	#  - For each segment node n in tid: flow_in[n] <= use_tid[tid]
	#  - If use_tid[tid]=1, at least one segment is used: sum_n flow_in[n] >= use_tid[tid]
	#  - Penalize dropped tracklets once per base_tracklet_id: penalty * (1 - use_tid[tid])
	# ------------------------------------------------------------
	tracklet_penalty_scaled: int | None = None
	drop_var_by_tid: Dict[str, cp_model.IntVar] = {}

	if unexplained_tracklet_penalty is not None and float(unexplained_tracklet_penalty) > 0:
		tracklet_penalty_scaled = int(round(float(unexplained_tracklet_penalty) * float(scale)))

		if "base_tracklet_id" in nodes.columns and "node_type" in nodes.columns:
			single = nodes[nodes["node_type"].astype(str) == "NodeType.SINGLE_TRACKLET"].copy()
			single["base_tracklet_id"] = single["base_tracklet_id"].astype(str)

			for tid, grp in single.groupby("base_tracklet_id", sort=True):
				tid = str(tid)
				# Ensure var exists even if precomputed block didn't create it (defensive).
				if tid not in use_tid:
					use_tid[tid] = model.NewBoolVar(f"tid_used_{tid}")
				node_list = [str(x) for x in grp["node_id"].astype(str).tolist()]
				# Per-segment: flow_in <= node_cap_eff[nid] * use_tid
				# This preserves the original behavior for capacity_eff == 1 while
				# allowing promoted SINGLE_TRACKLET nodes (e.g., group-continuation
				# arrivals) to legally carry multiple units of flow when their
				# effective capacity has been raised.
				for nid in node_list:
					cap_n = int(node_cap_eff.get(nid, 1))
					model.Add(flow_in_by_node[nid] <= cap_n * use_tid[tid])
				# If used, at least one segment must carry flow.
				model.Add(sum(flow_in_by_node[nid] for nid in node_list) >= use_tid[tid])
				# Drop var and linkage
				drop = model.NewBoolVar(f"tid_drop_{tid}")
				model.Add(drop + use_tid[tid] == 1)
				drop_var_by_tid[tid] = drop

	# Objective
	terms = []
	for _, e in edges.iterrows():
		eid = str(e["edge_id"])
		coef = int(cost_int[eid])
		terms.append(coef * var_by_edge[eid])
	# Add unexplained-tracklet penalties (if enabled)
	if tracklet_penalty_scaled is not None and tracklet_penalty_scaled > 0:
		for tid, drop in drop_var_by_tid.items():
			terms.append(int(tracklet_penalty_scaled) * drop)

	# Add ping-miss penalties (soft coverage for ping-bound nodes)
	if (
		unexplained_solo_ping_penalty is not None
		and float(unexplained_solo_ping_penalty) > 0
		and len(miss_solo_ping) > 0
	):
		pen_scaled = int(round(float(unexplained_solo_ping_penalty) * float(scale)))
		for nid, miss in miss_solo_ping.items():
			terms.append(int(pen_scaled) * miss)
	if (
		unexplained_group_ping_penalty is not None
		and float(unexplained_group_ping_penalty) > 0
		and len(miss_group_ping) > 0
	):
		pen_scaled = int(round(float(unexplained_group_ping_penalty) * float(scale)))
		for (gid, req), miss in miss_group_ping.items():
			terms.append(int(pen_scaled) * miss)

	# Add tag-fragment-start penalties (time-separated fragmentation)
	if tag_fragment_start_penalty is not None and float(tag_fragment_start_penalty) > 0 and len(frag_start_vars) > 0:
		pen_scaled = int(round(float(tag_fragment_start_penalty) * float(scale)))
		for fs in frag_start_vars:
			terms.append(int(pen_scaled) * fs)
	model.Minimize(sum(terms))

	# ------------------------------------------------------------------
	# Dev-only: full under-the-hood visibility for infeasibility debugging
	#
	# These files are intended to make it possible to answer questions like:
	#  - Which edges became IntVar vs BoolVar, and what were their domains?
	#  - What objective coefficients were actually used (after scaling)?
	#  - Which edge vars appear in each node's flow-balance constraints?
	#  - How big is the compiled CP-SAT model (vars/constraints)?
	#
	# IMPORTANT: this is logging only; it must never affect feasibility.
	# ------------------------------------------------------------------
	if debug_dir is not None and emit_transparency:
		try:
			_emit_d3_ilp_variables_json(
				debug_dir=debug_dir,
				edges=edges,
				use_flow_int=bool(use_flow_int),
				var_by_edge=var_by_edge,
				edge_used=edge_used,
				cost_int=cost_int,
				scale=int(scale),
				costs_df=costs_df,
			)
		except Exception:
			# Never fail a solve due to debug output.
			pass
		try:
			_emit_d3_ilp_node_equations_json(
				debug_dir=debug_dir,
				nodes=nodes,
				edges=edges,
				in_edges=in_edges,
				out_edges=out_edges,
			)
		except Exception:
			pass
		try:
			_emit_d3_cp_model_dump(debug_dir=debug_dir, model=model)
		except Exception:
			pass

	solver = cp_model.CpSolver()
	solver.parameters.num_search_workers = 1
	solver.parameters.random_seed = 0
	solver.parameters.max_time_in_seconds = 30.0
	solver.parameters.log_search_progress = False

	status_code = solver.Solve(model)
	status = solver.StatusName(status_code)

	runtime_ms = int(round((time.time() - start) * 1000))

	selected: List[str] = []
	flow_by_edge: Dict[str, int] = {}
	if status in ("OPTIMAL", "FEASIBLE"):
		for _, e in edges.iterrows():
			eid = str(e["edge_id"])
			val = int(solver.Value(var_by_edge[eid]))
			flow_by_edge[eid] = val
			if val > 0:
				selected.append(eid)

	obj_scaled = None
	obj_value = None
	if status in ("OPTIMAL", "FEASIBLE"):
		obj_scaled = int(round(solver.ObjectiveValue()))
		obj_value = float(obj_scaled) / float(scale)

	n_tracklets_total = int(len(drop_var_by_tid))
	n_tracklets_explained = 0
	n_tracklets_unexplained = 0
	dropped_tracklet_ids: List[str] = []
	explained_tracklet_ids: List[str] = []
	if status in ("OPTIMAL", "FEASIBLE") and n_tracklets_total > 0:
		for tid, drop in drop_var_by_tid.items():
			if int(solver.Value(drop)) == 1:
				n_tracklets_unexplained += 1
				dropped_tracklet_ids.append(str(tid))
			else:
				n_tracklets_explained += 1
				explained_tracklet_ids.append(str(tid))

	dropped_tracklet_ids = sorted(dropped_tracklet_ids)
	explained_tracklet_ids = sorted(explained_tracklet_ids)

	# POC_2_TAGS transparency: report label assignments and ping coverage diagnostics for used nodes.
	if enforce_solo_labels and tag_solution_out is not None:
		info: Dict[str, Any] = {
			"labels_domain": [str(k) for k in (labels_domain or [])],
			"forced_solo_node_labels": {str(nid): str(lbl) for nid, lbl in (forced_solo_node_labels or {}).items()},
			"forced_group_node_labels": {str(nid): sorted({str(x) for x in reqs}, key=_stable_tag_sort_key) for nid, reqs in (forced_group_node_labels or {}).items()},
			"solo_node_labels_used": [],
			"group_node_labels_used": [],
			"tag_overlap_enforced": bool(tag_overlap_enforced),
			"n_overlap_constraints_added": int(n_overlap_constraints_added),
			"tag_fragment_start_penalty": float(tag_fragment_start_penalty) if tag_fragment_start_penalty is not None else None,
			"n_fragment_starts_total": None,
			"n_fragment_starts_by_tag": {},
			"n_missed_solo_pings": 0,
			"n_missed_group_pings": 0,
			"missed_solo_pings": [],
			"missed_group_pings": [],
		}
		if status in ("OPTIMAL", "FEASIBLE"):
			rows: List[Dict[str, Any]] = []
			for nid, u in sorted(u_solo.items(), key=lambda kv: kv[0]):
				used = int(solver.Value(u))
				if used != 1:
					continue
				label = None
				for k in (labels_domain or []):
					vv = y_solo.get((nid, str(k)))
					if vv is not None and int(solver.Value(vv)) == 1:
						label = str(k)
						break
				rows.append({"node_id": str(nid), "label": label})
			info["solo_node_labels_used"] = rows
			# Group nodes are allowed up to 2 labels; report which tags were selected.
			grows: List[Dict[str, Any]] = []
			for gid, u in sorted(u_group.items(), key=lambda kv: kv[0]):
				used = int(solver.Value(u))
				if used != 1:
					continue
				labels2: List[str] = []
				for k in [str(lbl) for lbl in (labels_domain or []) if str(lbl) != "UNKNOWN"]:
					vv = y_group.get((gid, str(k)))
					if vv is not None and int(solver.Value(vv)) == 1:
						labels2.append(str(k))
				grows.append({"node_id": str(gid), "labels": sorted(labels2, key=_stable_tag_sort_key)})
			info["group_node_labels_used"] = grows
			# Fragment-start counts (time-separated fragmentation)
			if len(frag_start_vars) > 0 and tag_fragment_start_penalty is not None and float(tag_fragment_start_penalty) > 0:
				total = 0
				by_tag: Dict[str, int] = {}
				for k, lst in sorted(frag_start_vars_by_tag.items(), key=lambda kv: _stable_tag_sort_key(kv[0])):
					c = sum(int(solver.Value(v)) for v in lst)
					by_tag[str(k)] = int(c)
					total += int(c)
				info["n_fragment_starts_total"] = int(total)
				info["n_fragment_starts_by_tag"] = by_tag

			# Missed ping diagnostics (soft coverage)
			ms: List[Dict[str, Any]] = []
			for nid, miss in sorted(miss_solo_ping.items(), key=lambda kv: kv[0]):
				if int(solver.Value(miss)) == 1:
					ms.append({"node_id": str(nid), "required_label": str((forced_solo_node_labels or {}).get(nid))})
			mg: List[Dict[str, Any]] = []
			for (gid, req), miss in sorted(miss_group_ping.items(), key=lambda kv: (kv[0][0], kv[0][1])):
				if int(solver.Value(miss)) == 1:
					mg.append({"node_id": str(gid), "required_label": str(req)})
			info["missed_solo_pings"] = ms
			info["missed_group_pings"] = mg
			info["n_missed_solo_pings"] = int(len(ms))
			info["n_missed_group_pings"] = int(len(mg))

		if tag_solution_out is not None:
			# Preserve any upstream fields already populated by the wrapper (e.g., penalty scaling / ref costs).
			existing = dict(tag_solution_out)
			existing.update(info)
			tag_solution_out.clear()
			tag_solution_out.update(existing)

	return ILPResult(
		status=status,
		objective_scaled=obj_scaled,
		objective_value=obj_value,
		runtime_ms=runtime_ms,
		selected_edge_ids=sorted(selected),
		flow_by_edge_id=flow_by_edge,
		cost_scale=scale,
		enforced_min_one_path=enforced_min_one_path,
		rounding_n_edges=int(rounding_stats.get("rounding_n_edges", 0)),
		rounding_n_edges_nonzero=int(rounding_stats.get("rounding_n_edges_nonzero", 0)),
		rounding_max_abs_scaled_error=float(rounding_stats.get("rounding_max_abs_scaled_error", 0.0)),
		rounding_max_abs_cost_error=float(rounding_stats.get("rounding_max_abs_cost_error", 0.0)),
		unexplained_tracklet_penalty=float(unexplained_tracklet_penalty) if unexplained_tracklet_penalty is not None else None,
		n_tracklets_total=int(n_tracklets_total),
		n_tracklets_explained=int(n_tracklets_explained),
		n_tracklets_unexplained=int(n_tracklets_unexplained),
		dropped_tracklet_ids=dropped_tracklet_ids,
		explained_tracklet_ids=explained_tracklet_ids,
	)



def solve_structure_ilp(
	*,
	compiled: CompiledInputs,
	layout: ClipOutputLayout,
	manifest: ClipManifest,
	checkpoint: str,
	unexplained_tracklet_penalty: float | None = None,
	# Tag fragmentation (time-separated): prefer continuity but allow multiple disjoint fragments
	tag_fragment_start_penalty: float | None = None,
	tag_overlap_enforced: bool = True,
	group_boundary_window_frames: int = 10,
) -> ILPResult:
	"""POC_1 wrapper: solve + write debug outputs + audit summary."""
	dbg = _debug_dir(layout)
	dbg.mkdir(parents=True, exist_ok=True)
	res = solve_structure_ilp_core(
		nodes_df=compiled.nodes_df,
		edges_df=compiled.edges_df,
		costs_df=compiled.costs_df,
		constraints=compiled.constraints,
		debug_dir=dbg,
		emit_transparency=True,
		unexplained_tracklet_penalty=unexplained_tracklet_penalty,
		tag_fragment_start_penalty=tag_fragment_start_penalty,
		tag_overlap_enforced=bool(tag_overlap_enforced),
		group_boundary_window_frames=int(group_boundary_window_frames),
	)

	# Selected edges parquet
	edges = compiled.edges_df.copy()
	edges["edge_id"] = edges["edge_id"].astype(str)
	edges = edges[edges["edge_id"].isin(res.selected_edge_ids)].copy()
	if len(edges) > 0:
		edges = edges.sort_values(["edge_id"], kind="mergesort").reset_index(drop=True)
		# Merge costs (already aligned/1:1 by edge_id), add flow
		costs = compiled.costs_df[["edge_id", "total_cost"]].copy()
		costs["edge_id"] = costs["edge_id"].astype(str)
		edges = edges.merge(costs, on="edge_id", how="left", validate="1:1")
		edges["flow"] = edges["edge_id"].map(lambda eid: int(res.flow_by_edge_id.get(str(eid), 0)))
	else:
		edges["total_cost"] = []
		edges["flow"] = []

	out_sel = dbg / "d3_selected_edges.parquet"
	edges.to_parquet(out_sel, index=False)

	# Entity paths (Format A JSON)
	out_entities = _write_entities_format_a(layout=layout, compiled=compiled, res=res, checkpoint=checkpoint, manifest=manifest)

	# Full transparency ledger (JSON)
	out_ledger = _write_solution_ledger_json(layout=layout, compiled=compiled, res=res, checkpoint=checkpoint, manifest=manifest)

	def rel(p: Path) -> str:
		return str(p.relative_to(layout.clip_root))

	# Audit summary
	edge_type_counts: Dict[str, int] = {}
	if "edge_type" in edges.columns:
		for k, v in edges["edge_type"].astype(str).value_counts().items():
			edge_type_counts[str(k)] = int(v)

	# Total paths (K) inferred from SOURCE outflow (if available)
	k_paths = None
	try:
		source_id = _find_unique_node_id(compiled.nodes_df, node_type="NodeType.SOURCE")
		out_ids = compiled.edges_df[compiled.edges_df["u"].astype(str) == str(source_id)]["edge_id"].astype(str).tolist()
		k_paths = int(sum(res.flow_by_edge_id.get(eid, 0) for eid in out_ids))
	except Exception:
		k_paths = None

	append_audit_event(
		layout=layout,
		event={
			"event_type": "d3_ilp_summary",
			"clip_id": manifest.clip_id,
			"camera_id": manifest.camera_id,
			"checkpoint": checkpoint,
			"status": res.status,
			"objective_value": res.objective_value,
			"objective_scaled": res.objective_scaled,
			"cost_scale": res.cost_scale,
			"enforced_min_one_path": res.enforced_min_one_path,
			"rounding": {
				"n_edges": res.rounding_n_edges,
				"n_edges_nonzero": res.rounding_n_edges_nonzero,
				"max_abs_scaled_error": res.rounding_max_abs_scaled_error,
				"max_abs_cost_error": res.rounding_max_abs_cost_error,
			},
			"runtime_ms": res.runtime_ms,
			"n_selected_edges": len(res.selected_edge_ids),
			"selected_edge_type_counts": dict(sorted(edge_type_counts.items(), key=lambda kv: kv[0])),
			"explain_or_penalize": {
				"unexplained_tracklet_penalty": res.unexplained_tracklet_penalty,
				"n_tracklets_total": res.n_tracklets_total,
				"n_tracklets_explained": res.n_tracklets_explained,
				"n_tracklets_unexplained": res.n_tracklets_unexplained,
			},
			"n_paths": k_paths,
			"debug_outputs": {
				"d3_selected_edges_parquet": rel(out_sel),
				"d3_entities_format_a_json": rel(out_entities),
				"d3_solution_ledger_json": rel(out_ledger),
			},
		},
	)

	return res


def solve_structure_ilp_tags(
	*,
	compiled: CompiledInputs,
	layout: ClipOutputLayout,
	manifest: ClipManifest,
	checkpoint: str,
	unexplained_tracklet_penalty: float | None = None,
	unexplained_group_ping_penalty: float | None = None,
	# Tag fragmentation (time-separated): prefer continuity but allow multiple disjoint fragments
	tag_fragment_start_penalty: float | None = None,
	tag_overlap_enforced: bool = True,
	group_boundary_window_frames: int = 10,
	# Professional penalty scaling (optional; if omitted, fall back to absolute penalties)
	penalty_ref_edge_cost_quantile: float | None = None,
	penalty_ref_edge_cost_min: float | None = None,
	solo_ping_miss_penalty_mult: float | None = None,
	group_ping_miss_penalty_mult: float | None = None,
	solo_ping_miss_penalty_abs: float | None = None,
	group_ping_miss_penalty_abs: float | None = None,
	tag_fragment_start_penalty_mult: float | None = None,
	tag_fragment_start_penalty_abs: float | None = None,
) -> ILPResult:
	"""POC_2_TAGS wrapper: solve with SOLO label enforcement + write debug outputs + audit summary.

	Behavioral contract (v1):
	- Bind tag pings to SINGLE_TRACKLET or GROUP_TRACKLET nodes by time overlap.
	- SINGLE_TRACKLET nodes (capacity=1) have exactly one label when used.
	- GROUP_TRACKLET nodes may carry up to two concrete labels (partner identities); UNKNOWN is implicit.
	- Prefer time-local tag_pings from D2 constraints when available.
	- Fall back to Stage C identity_hints.jsonl when tag_pings are absent.
	- Ping-bound nodes (SOLO or GROUP) are *strongly preferred* to be explained; misses incur penalties.
	- Selected SOLO->SOLO CONTINUE edges enforce label equality.
	- Selected SOLO<->GROUP edges propagate labels into the GROUP node.
	"""
	identity_hints = _read_jsonl(layout.identity_hints_jsonl())

	constraints_raw = compiled.constraints or {}
	tag_pings_raw = constraints_raw.get("tag_pings") if isinstance(constraints_raw, dict) else None
	tag_source = "stage_c_identity_hints"
	n_tag_pings = 0
	# When D2 has emitted time-local tag_pings, prefer them over raw hints.
	if isinstance(tag_pings_raw, list) and len(tag_pings_raw) > 0:
		tag_source = "d2_tag_pings"
		tag_pings = []
		for rec in tag_pings_raw:
			if not isinstance(rec, dict):
				continue
			tracklet_id = rec.get("tracklet_id")
			anchor_key = rec.get("anchor_key")
			fi = rec.get("frame_index")
			if not isinstance(tracklet_id, str) or not isinstance(anchor_key, str) or not isinstance(fi, int):
				continue
			# Construct a minimal identity_hint-shaped record so we can reuse
			# the existing SOLO binding logic without changing its semantics.
			h = {
				"artifact_type": "identity_hint",
				"constraint": "must_link",
				"anchor_key": str(anchor_key),
				"tracklet_id": str(tracklet_id),
				"evidence": {"frame_index": int(fi)},
			}
			if "confidence" in rec:
				try:
					h["confidence"] = float(rec["confidence"]) if rec["confidence"] is not None else None
				except Exception:
					pass
			tag_pings.append(h)
		tag_hints = tag_pings
		n_tag_pings = len(tag_pings)
	else:
		# Legacy path: extract tag must_link hints directly from Stage C.
		tag_hints = _extract_tag_hints(identity_hints)

	labels_domain, forced_solo_by_node, forced_group_by_node, binding_ledger = _bind_tag_hints_to_nodes(
		nodes_df=compiled.nodes_df, tag_hints=tag_hints
	)

	# Only enforce SOLO/GROUP label logic if we actually have concrete AprilTag labels.
	# Otherwise, enabling enforce_solo_labels would create zero-label domains for SOLO nodes,
	# forcing all SOLO nodes unused and making the model infeasible when we require >=1 path.
	labels_domain = [str(k) for k in (labels_domain or [])]
	labels_available = any(str(k).startswith("tag:") for k in labels_domain)
	if not labels_available:
		forced_solo_by_node = {}
		forced_group_by_node = {}

	# Policy: tag pings must be explainable by the graph.
	unbound = [
			r for r in binding_ledger
			if r.get("status") == "unbound"
			and r.get("frame_index") is not None
			and str(r.get("anchor_key", "")).startswith("tag:")
	]

	# Tag-solution container populated by solve_structure_ilp_core when labels are enforced.
	tag_solution: Dict[str, Any] = {}

	# Derive reference edge cost for penalty scaling.
	# Use either an explicit minimum, or a quantile of positive edge costs.
	edge_costs = compiled.costs_df["total_cost"] if "total_cost" in compiled.costs_df.columns else pd.Series([], dtype=float)
	edge_costs = pd.to_numeric(edge_costs, errors="coerce")
	pos_costs = edge_costs[edge_costs > 0].dropna()

	q = float(penalty_ref_edge_cost_quantile) if penalty_ref_edge_cost_quantile is not None else 0.9
	if penalty_ref_edge_cost_min is not None:
		ref_min = float(penalty_ref_edge_cost_min)
	else:
		if len(pos_costs) > 0:
			try:
				ref_min = float(pos_costs.quantile(q))
			except Exception:
				ref_min = float(pos_costs.median())
		else:
			ref_min = 1.0

	ref_edge_cost = float(ref_min)
	if not math.isfinite(ref_edge_cost) or ref_edge_cost <= 0.0:
		ref_edge_cost = 1.0

	# Effective penalties (cost units). Abs overrides take precedence.
	eff_solo_miss = solo_ping_miss_penalty_abs
	if eff_solo_miss is None:
		eff_solo_miss = (
			float(solo_ping_miss_penalty_mult) if solo_ping_miss_penalty_mult is not None else 50.0
		) * ref_edge_cost

	eff_group_miss = group_ping_miss_penalty_abs
	if eff_group_miss is None:
		if group_ping_miss_penalty_mult is not None:
			eff_group_miss = float(group_ping_miss_penalty_mult) * ref_edge_cost
		else:
			# Legacy fallback: preserve previous unexplained_group_ping_penalty behavior when multipliers are not set.
			eff_group_miss = unexplained_group_ping_penalty if unexplained_group_ping_penalty is not None else 5000.0

	eff_frag = tag_fragment_start_penalty_abs
	if eff_frag is None:
		if tag_fragment_start_penalty_mult is not None:
			eff_frag = float(tag_fragment_start_penalty_mult) * ref_edge_cost
		else:
			# Legacy fallback: preserve absolute tag_fragment_start_penalty when multipliers are not set.
			eff_frag = tag_fragment_start_penalty if tag_fragment_start_penalty is not None else 2500.0

	# Dev-only transparency artifact for POC_2_TAGS as well: reuse the clip _debug dir.
	dbg = _debug_dir(layout)
	dbg.mkdir(parents=True, exist_ok=True)

	res = solve_structure_ilp_core(
		nodes_df=compiled.nodes_df,
		edges_df=compiled.edges_df,
		costs_df=compiled.costs_df,
		constraints=compiled.constraints,
		debug_dir=dbg,
		emit_transparency=True,
		unexplained_tracklet_penalty=unexplained_tracklet_penalty,
		unexplained_group_ping_penalty=float(eff_group_miss),
		unexplained_solo_ping_penalty=float(eff_solo_miss),
		tag_fragment_start_penalty=float(eff_frag),
		tag_overlap_enforced=bool(tag_overlap_enforced),
		group_boundary_window_frames=int(group_boundary_window_frames),
		enforce_solo_labels=bool(labels_available),
		labels_domain=labels_domain,
		forced_solo_node_labels=forced_solo_by_node,
		forced_group_node_labels=forced_group_by_node,
		tag_solution_out=tag_solution,
	)

	# Pre/post-solve diagnostics for transparency (does not affect solve).
	# 1) Graph reachability
	reach = _compute_graph_reachability(nodes_df=compiled.nodes_df, edges_df=compiled.edges_df)
	source_id = reach["source_id"]
	sink_id = reach["sink_id"]
	reachable_from_source: Set[str] = reach["reachable_from_source_set"]
	can_reach_sink: Set[str] = reach["can_reach_sink_set"]

	# 2) Forced-tag overlap checks (same-tag overlap is a hard contradiction under no-overlap rule)
	forced_diag = _forced_tag_overlap_and_counts(
		nodes_df=compiled.nodes_df,
		forced_solo_node_labels=forced_solo_by_node,
		forced_group_node_labels=forced_group_by_node,
	)

	# 3) Compute k_paths (existing meaning: SOURCE outflow) for ledger symmetry with audit
	try:
		out_ids_k = compiled.edges_df[compiled.edges_df["u"].astype(str) == str(source_id)]["edge_id"].astype(str).tolist()
		k_paths_inferred = int(sum(res.flow_by_edge_id.get(eid, 0) for eid in out_ids_k))
	except Exception:
		k_paths_inferred = None
	k_metrics = _compute_k_metrics(nodes_df=compiled.nodes_df, edges_df=compiled.edges_df, res=res, k_paths_inferred=k_paths_inferred)

	# 4) Forced-node reachability report
	forced_nodes_reachability: List[Dict[str, Any]] = []
	for nid, lbl in sorted(forced_solo_by_node.items(), key=lambda kv: str(kv[0])):
		nid_s = str(nid)
		forced_nodes_reachability.append(
			{
				"node_id": nid_s,
				"node_type": "SOLO",
				"label": str(lbl),
				"reachable_from_source": bool(nid_s in reachable_from_source),
				"can_reach_sink": bool(nid_s in can_reach_sink),
				"on_some_source_to_sink_path": bool((nid_s in reachable_from_source) and (nid_s in can_reach_sink)),
			}
		)
	for gid, reqs in sorted(forced_group_by_node.items(), key=lambda kv: str(kv[0])):
		gid_s = str(gid)
		forced_nodes_reachability.append(
			{
				"node_id": gid_s,
				"node_type": "GROUP",
				"labels": sorted({str(x) for x in reqs}, key=_stable_tag_sort_key),
				"reachable_from_source": bool(gid_s in reachable_from_source),
				"can_reach_sink": bool(gid_s in can_reach_sink),
				"on_some_source_to_sink_path": bool((gid_s in reachable_from_source) and (gid_s in can_reach_sink)),
			}
		)

	# Prepare tag ledger section (deterministic ordering).
	tag_info: Dict[str, Any] = {
		"identity_hints_path": str(layout.identity_hints_jsonl().relative_to(layout.clip_root)),
		"n_identity_hints": int(len(identity_hints)),
		"n_tag_hints": int(len(tag_hints)),
		"n_tag_pings": int(n_tag_pings),
		"labels_available": bool(labels_available),
		"enforce_solo_labels_effective": bool(labels_available),
		"penalty_ref_edge_cost_quantile": float(q),
		"penalty_ref_edge_cost_min": float(ref_min),
		"ref_edge_cost": float(ref_edge_cost),
		"solo_ping_miss_penalty_cost": float(eff_solo_miss),
		"group_ping_miss_penalty_cost": float(eff_group_miss),
		"tag_fragment_start_penalty_cost": float(eff_frag),
		"tag_source": str(tag_source),
		"binding_ledger": binding_ledger,
		"n_unbound_tag_pings": int(len(unbound)),
		"n_forced_solo_bindings": int(len(forced_solo_by_node)),
		"n_forced_group_bindings": int(len(forced_group_by_node)),
		# New transparency:
		"k_metrics": k_metrics,
		"graph_reachability": {
			"source_id": str(source_id),
			"sink_id": str(sink_id),
			"graph_any_source_to_sink_path": bool(reach["graph_any_source_to_sink_path"]),
		},
		"forced_tag_diagnostics": forced_diag,
		"forced_nodes_reachability": forced_nodes_reachability,
	}
	# tag_solution is populated by the core solver (used SOLO labels) when feasible.
	if len(tag_solution) > 0:
		tag_info.update(tag_solution)

	dbg = _debug_dir(layout)
	dbg.mkdir(parents=True, exist_ok=True)

	# Selected edges parquet (same as POC_1)
	edges = compiled.edges_df.copy()
	edges["edge_id"] = edges["edge_id"].astype(str)
	edges = edges[edges["edge_id"].isin(res.selected_edge_ids)].copy()
	if len(edges) > 0:
		edges = edges.sort_values(["edge_id"], kind="mergesort").reset_index(drop=True)
		costs = compiled.costs_df[["edge_id", "total_cost"]].copy()
		costs["edge_id"] = costs["edge_id"].astype(str)
		edges = edges.merge(costs, on="edge_id", how="left", validate="1:1")
		edges["flow"] = edges["edge_id"].map(lambda eid: int(res.flow_by_edge_id.get(str(eid), 0)))
	else:
		edges["total_cost"] = []
		edges["flow"] = []

	out_sel = dbg / "d3_selected_edges.parquet"
	edges.to_parquet(out_sel, index=False)

	# Entity paths (Format A JSON)
	out_entities = _write_entities_format_a(layout=layout, compiled=compiled, res=res, checkpoint=checkpoint, manifest=manifest)

	# Full transparency ledger (JSON), extended with tag info
	out_ledger = _write_solution_ledger_json(
		layout=layout, compiled=compiled, res=res, checkpoint=checkpoint, manifest=manifest, tag_info=tag_info
	)

	def rel(p: Path) -> str:
		return str(p.relative_to(layout.clip_root))

	# Audit summary (match POC_1 shape; add tag counts)
	edge_type_counts: Dict[str, int] = {}
	if "edge_type" in edges.columns:
		for k, v in edges["edge_type"].astype(str).value_counts().items():
			edge_type_counts[str(k)] = int(v)

	k_paths = None
	try:
		source_id = _find_unique_node_id(compiled.nodes_df, node_type="NodeType.SOURCE")
		out_ids = compiled.edges_df[compiled.edges_df["u"].astype(str) == str(source_id)]["edge_id"].astype(str).tolist()
		k_paths = int(sum(res.flow_by_edge_id.get(eid, 0) for eid in out_ids))
	except Exception:
		k_paths = None

	append_audit_event(
		layout=layout,
		event={
			"event_type": "d3_ilp_summary",
			"clip_id": manifest.clip_id,
			"camera_id": manifest.camera_id,
			"checkpoint": checkpoint,
			"status": res.status,
			"objective_value": res.objective_value,
			"objective_scaled": res.objective_scaled,
			"cost_scale": res.cost_scale,
			"enforced_min_one_path": res.enforced_min_one_path,
			"rounding": {
				"n_edges": res.rounding_n_edges,
				"n_edges_nonzero": res.rounding_n_edges_nonzero,
				"max_abs_scaled_error": res.rounding_max_abs_scaled_error,
				"max_abs_cost_error": res.rounding_max_abs_cost_error,
			},
			"runtime_ms": res.runtime_ms,
			"n_selected_edges": len(res.selected_edge_ids),
			"selected_edge_type_counts": dict(sorted(edge_type_counts.items(), key=lambda kv: kv[0])),
			"explain_or_penalize": {
				"unexplained_tracklet_penalty": res.unexplained_tracklet_penalty,
				"n_tracklets_total": res.n_tracklets_total,
				"n_tracklets_explained": res.n_tracklets_explained,
				"n_tracklets_unexplained": res.n_tracklets_unexplained,
			},
			"n_paths": k_paths,
			"tags": {
				"n_identity_hints": int(len(identity_hints)),
				"n_tag_hints": int(len(tag_hints)),
				"n_tag_pings": int(n_tag_pings),
				"tag_source": str(tag_source),
				"labels_available": bool(labels_available),
				"enforce_solo_labels_effective": bool(labels_available),
				"n_forced_solo_bindings": int(len(forced_solo_by_node)),
				"n_forced_group_bindings": int(len(forced_group_by_node)),
				# Compact diagnostics for quick inspection in audit stream.
				"graph_any_source_to_sink_path": bool(reach["graph_any_source_to_sink_path"]),
				"forced_same_tag_overlaps_n": int(len(forced_diag.get("forced_same_tag_overlaps", []))),
				"forced_nodes_unreachable_n": int(
					sum(1 for r in forced_nodes_reachability if not bool(r.get("on_some_source_to_sink_path")))
				),
				"k_metrics": {
					"k_paths_inferred_from_solution": k_metrics.get("k_paths_inferred_from_solution"),
					"k_min_required_by_constraints": k_metrics.get("k_min_required_by_constraints"),
					"k_max_possible_from_graph_source_cap": k_metrics.get("k_max_possible_from_graph_source_cap"),
					"k_max_possible_from_graph_sink_cap": k_metrics.get("k_max_possible_from_graph_sink_cap"),
				},
			},
			"debug_outputs": {
				"d3_selected_edges_parquet": rel(out_sel),
				"d3_entities_format_a_json": rel(out_entities),
				"d3_solution_ledger_json": rel(out_ledger),
			},
		},
	)

	return res