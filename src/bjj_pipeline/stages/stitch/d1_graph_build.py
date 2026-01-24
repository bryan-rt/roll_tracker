"""Stage D1 — Candidate graph construction.

D1 constructs a solver-agnostic graph over tracklets, including explicit
GROUP_TRACKLET nodes (capacity=2) and merge/split edges to represent
2→1→2 events.

Authoritative spatial source: Stage D0 bank frames (repaired coords with
per-frame fallback to raw coords).
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from bjj_pipeline.stages.stitch.graph import (
	EdgeType,
	GraphEdge,
	GraphNode,
	NodeType,
	TrackletGraph,
)


def _now_ms() -> int:
	return int(time.time() * 1000)


def _cfg_get(cfg: Dict[str, Any], path: str, default: Any = None) -> Any:
	cur: Any = cfg
	for part in path.split("."):
		if not isinstance(cur, dict) or part not in cur:
			return default
		cur = cur[part]
	return cur


def _write_audit_event(audit_path: Path, event: Dict[str, Any]) -> None:
	audit_path.parent.mkdir(parents=True, exist_ok=True)
	with audit_path.open("a", encoding="utf-8") as f:
		f.write(json.dumps(event, sort_keys=True) + "\n")


def _is_finite(x: Any) -> bool:
	try:
		return x is not None and math.isfinite(float(x))
	except Exception:
		return False


def _effective_xy_row(row: pd.Series) -> Optional[Tuple[float, float, bool]]:
	"""Return (x_eff, y_eff, used_raw_fallback) or None if invalid."""
	x_rep = row.get("x_m_repaired", None)
	y_rep = row.get("y_m_repaired", None)
	x_raw = row.get("x_m", None)
	y_raw = row.get("y_m", None)
	use_raw = (not _is_finite(x_rep)) or (not _is_finite(y_rep))
	if use_raw:
		x = x_raw
		y = y_raw
	else:
		x = x_rep
		y = y_rep
	if not (_is_finite(x) and _is_finite(y)):
		return None
	return (float(x), float(y), bool(use_raw))


def _dist_m(a: Tuple[float, float], b: Tuple[float, float]) -> float:
	return float(math.hypot(a[0] - b[0], a[1] - b[1]))


def _parse_json_list(s: Any) -> List[str]:
	if s is None:
		return []
	if isinstance(s, list):
		return [str(x) for x in s]
	if not isinstance(s, str):
		return []
	ss = s.strip()
	if not ss:
		return []
	try:
		obj = json.loads(ss)
		if isinstance(obj, list):
			return [str(x) for x in obj]
	except Exception:
		return []
	return []


def _get_manifest_fields(manifest: Any) -> Tuple[Optional[float], Optional[int], Optional[int]]:
	"""Extract fps, frame_count, duration_ms from manifest (dict or ClipManifest)."""
	fps = None
	frame_count = None
	duration_ms = None
	for key in ("fps", "frame_count", "duration_ms"):
		if hasattr(manifest, key):
			val = getattr(manifest, key)
		elif isinstance(manifest, dict):
			val = manifest.get(key)
		else:
			val = None
		if key == "fps":
			fps = float(val) if val is not None else None
		elif key == "frame_count":
			frame_count = int(val) if val is not None else None
		elif key == "duration_ms":
			duration_ms = int(val) if val is not None else None
	return fps, frame_count, duration_ms


def run_d1(*, cfg: Dict[str, Any], layout: Any, manifest: Any) -> TrackletGraph:
	"""Build the candidate graph for D1.

	Reads D0 bank artifacts from `layout`:
	  - stage_D/tracklet_bank_frames.parquet
	  - stage_D/tracklet_bank_summaries.parquet

	Writes audit events into stage_D/audit.jsonl.
	"""
	# ---- config ----
	d1_cfg = _cfg_get(cfg, "stages.stage_D.d1", {}) or {}
	enable_group_nodes = bool(d1_cfg.get("enable_group_nodes", True))
	max_continue_gap_frames = int(d1_cfg.get("max_continue_gap_frames", 90))
	start_window_frames = int(d1_cfg.get("start_window_frames", 10))
	end_window_frames = int(d1_cfg.get("end_window_frames", 10))
	merge_dist_m = float(d1_cfg.get("merge_dist_m", 0.45))
	merge_end_sync_frames = int(d1_cfg.get("merge_end_sync_frames", 3))
	merge_disappear_gap_frames = int(d1_cfg.get("merge_disappear_gap_frames", 6))
	split_dist_m = float(d1_cfg.get("split_dist_m", 0.60))
	split_search_horizon_frames = int(d1_cfg.get("split_search_horizon_frames", 120))

	fps, frame_count, duration_ms = _get_manifest_fields(manifest)
	last_frame = (frame_count - 1) if frame_count is not None else None

	# ---- paths ----
	frames_path = Path(layout.tracklet_bank_frames_parquet())
	summ_path = Path(layout.tracklet_bank_summaries_parquet())
	audit_path = Path(layout.audit_jsonl("D"))

	# ---- load ----
	tf = pd.read_parquet(frames_path)
	ts = pd.read_parquet(summ_path)

	# required columns
	for col in ("tracklet_id", "frame_index"):
		if col not in tf.columns:
			raise ValueError(f"D1 requires {col} in bank frames: {frames_path}")
	for col in ("x_m_repaired", "y_m_repaired", "x_m", "y_m"):
		if col not in tf.columns:
			raise ValueError(f"D1 requires {col} in bank frames: {frames_path}")
	for col in ("tracklet_id", "start_frame", "end_frame"):
		if col not in ts.columns:
			raise ValueError(f"D1 requires {col} in bank summaries: {summ_path}")

	on_mat_missing = "on_mat" not in tf.columns
	if on_mat_missing:
		tf = tf.copy()
		tf["on_mat"] = True

	# normalize types
	tf = tf.copy()
	tf["tracklet_id"] = tf["tracklet_id"].astype(str)
	tf["frame_index"] = tf["frame_index"].astype(int)
	ts = ts.copy()
	ts["tracklet_id"] = ts["tracklet_id"].astype(str)
	ts["start_frame"] = ts["start_frame"].astype(int)
	ts["end_frame"] = ts["end_frame"].astype(int)

	# Optional identity columns that D0 aggregates for us.
	if "must_link_anchor_key" not in ts.columns:
		ts["must_link_anchor_key"] = None
	if "cannot_link_anchor_keys_json" not in ts.columns:
		ts["cannot_link_anchor_keys_json"] = None

	# pre-index summaries for fast lookup
	ts_by_tid = {r["tracklet_id"]: r for _, r in ts.iterrows()}

	# frames grouped
	tf = tf.sort_values(["tracklet_id", "frame_index"], kind="mergesort")
	frames_by_tid: Dict[str, pd.DataFrame] = {tid: g for tid, g in tf.groupby("tracklet_id", sort=False)}

	# ---- endpoint helpers ----
	used_raw_frames = 0
	used_repaired_frames = 0
	used_raw_start = 0
	used_raw_end = 0
	missing_endpoint_coords = 0

	def endpoint_start(tid: str) -> Optional[Tuple[Tuple[float, float], bool]]:
		nonlocal used_raw_frames, used_repaired_frames, used_raw_start, missing_endpoint_coords
		row = ts_by_tid.get(tid)
		if row is None:
			missing_endpoint_coords += 1
			return None
		sf = int(row["start_frame"])
		ef = sf + start_window_frames
		gdf = frames_by_tid.get(tid)
		if gdf is None:
			missing_endpoint_coords += 1
			return None
		window = gdf[(gdf["frame_index"] >= sf) & (gdf["frame_index"] <= ef)]
		for _, fr in window.iterrows():
			if not bool(fr.get("on_mat", True)):
				continue
			res = _effective_xy_row(fr)
			if res is None:
				continue
			x, y, used_raw = res
			if used_raw:
				used_raw_frames += 1
				used_raw_start += 1
			else:
				used_repaired_frames += 1
			return ((x, y), used_raw)
		missing_endpoint_coords += 1
		return None

	def endpoint_end(tid: str) -> Optional[Tuple[Tuple[float, float], bool]]:
		nonlocal used_raw_frames, used_repaired_frames, used_raw_end, missing_endpoint_coords
		row = ts_by_tid.get(tid)
		if row is None:
			missing_endpoint_coords += 1
			return None
		ef = int(row["end_frame"])
		sf = ef - end_window_frames
		gdf = frames_by_tid.get(tid)
		if gdf is None:
			missing_endpoint_coords += 1
			return None
		window = gdf[(gdf["frame_index"] >= sf) & (gdf["frame_index"] <= ef)]
		for _, fr in window.iloc[::-1].iterrows():
			if not bool(fr.get("on_mat", True)):
				continue
			res = _effective_xy_row(fr)
			if res is None:
				continue
			x, y, used_raw = res
			if used_raw:
				used_raw_frames += 1
				used_raw_end += 1
			else:
				used_repaired_frames += 1
			return ((x, y), used_raw)
		missing_endpoint_coords += 1
		return None

	def carrier_pos_at_frame(tid: str, frame_idx: int) -> Optional[Tuple[Tuple[float, float], bool]]:
		nonlocal used_raw_frames, used_repaired_frames
		gdf = frames_by_tid.get(tid)
		if gdf is None:
			return None
		row = gdf[gdf["frame_index"] == frame_idx]
		if len(row) == 0:
			return None
		fr = row.iloc[0]
		if not bool(fr.get("on_mat", True)):
			return None
		res = _effective_xy_row(fr)
		if res is None:
			return None
		x, y, used_raw = res
		if used_raw:
			used_raw_frames += 1
		else:
			used_repaired_frames += 1
		return ((x, y), used_raw)

	# ---- build base graph ----
	g = TrackletGraph()
	g.add_node(GraphNode(node_id="SOURCE", type=NodeType.SOURCE, capacity=0, start_frame=None, end_frame=None, payload={}))
	g.add_node(GraphNode(node_id="SINK", type=NodeType.SINK, capacity=0, start_frame=None, end_frame=None, payload={}))

	# nodes and birth/death
	for _, r in ts.sort_values(["start_frame", "tracklet_id"], kind="mergesort").iterrows():
		tid = str(r["tracklet_id"])
		node_id = f"T:{tid}"
		payload = {
			"tracklet_id": tid,
			"must_link_anchor_key": r.get("must_link_anchor_key", None),
			"cannot_link_anchor_keys": _parse_json_list(r.get("cannot_link_anchor_keys_json", None)),
		}
		g.add_node(
			GraphNode(
				node_id=node_id,
				type=NodeType.SINGLE_TRACKLET,
				capacity=1,
				start_frame=int(r["start_frame"]),
				end_frame=int(r["end_frame"]),
				payload=payload,
			)
		)
		g.add_edge(GraphEdge(edge_id=f"E:BIRTH:{node_id}", u="SOURCE", v=node_id, type=EdgeType.BIRTH, capacity=1, payload={}))
		g.add_edge(GraphEdge(edge_id=f"E:DEATH:{node_id}", u=node_id, v="SINK", type=EdgeType.DEATH, capacity=1, payload={}))

	# helper for cannot-link pruning using anchor keys
	def cannot_link_for(tid: str) -> List[str]:
		row = ts_by_tid.get(tid)
		if row is None:
			return []
		return _parse_json_list(row.get("cannot_link_anchor_keys_json", None))

	def must_link_key(tid: str) -> Optional[str]:
		row = ts_by_tid.get(tid)
		if row is None:
			return None
		val = row.get("must_link_anchor_key", None)
		return str(val) if val not in (None, "", "null") else None

	# ---- CONTINUE edges (single->single) ----
	continue_pruned_cannot = 0
	singles = ts.sort_values(["end_frame", "tracklet_id"], kind="mergesort")[["tracklet_id", "start_frame", "end_frame"]]
	singles_list = [(str(r.tracklet_id), int(r.start_frame), int(r.end_frame)) for r in singles.itertuples(index=False)]

	# Pre-sort potential successors by start_frame
	succ = ts.sort_values(["start_frame", "tracklet_id"], kind="mergesort")[["tracklet_id", "start_frame", "end_frame"]]
	succ_list = [(str(r.tracklet_id), int(r.start_frame), int(r.end_frame)) for r in succ.itertuples(index=False)]

	for tid_i, _si, ei in singles_list:
		for tid_j, sj, _ej in succ_list:
			if sj <= ei:
				continue
			dt = sj - ei
			if dt > max_continue_gap_frames:
				break

			# cannot-link pruning based on anchor keys
			mi = must_link_key(tid_i)
			mj = must_link_key(tid_j)
			if mi is not None and mj is not None:
				# different anchors are fine; cannot-link lists apply
				pass
			if mj is not None and mj in set(cannot_link_for(tid_i)):
				continue_pruned_cannot += 1
				continue
			if mi is not None and mi in set(cannot_link_for(tid_j)):
				continue_pruned_cannot += 1
				continue

			u = f"T:{tid_i}"
			v = f"T:{tid_j}"
			g.add_edge(
				GraphEdge(
					edge_id=f"E:CONT:{u}->{v}",
					u=u,
					v=v,
					type=EdgeType.CONTINUE,
					capacity=1,
					payload={"dt_frames": dt},
				)
			)

	# ---- GROUP inference ----
	group_nodes_created = 0
	group_open_ended = 0
	group_from_start = 0
	merge_split_groups = 0
	suppressed_continue_edges = 0

	# Precompute endpoints for efficiency (deterministic)
	endpoints_end: Dict[str, Optional[Tuple[Tuple[float, float], bool]]] = {}
	endpoints_start: Dict[str, Optional[Tuple[Tuple[float, float], bool]]] = {}
	for tid, _, _ in singles_list:
		endpoints_end[tid] = endpoint_end(tid)
		endpoints_start[tid] = endpoint_start(tid)

	# candidate merges: pairs close in time and space at end
	candidates: List[Tuple[int, str, str, float]] = []  # (disappear_end, disappear_tid, carrier_tid, dist_m)

	def _carrier_pos_near_frame(tid: str, target_frame: int, max_offset: int) -> Optional[Tuple[Tuple[float, float], bool]]:
		# Prefer exact frame; then nearest offsets.
		for off in range(0, max_offset + 1):
			for f in ((target_frame - off), (target_frame + off)) if off > 0 else (target_frame,):
				res = carrier_pos_at_frame(tid, int(f))
				if res is not None:
					return res
		return None

	for tid_d, sd, ed in singles_list:
		disp_end = int(ed)
		p_disp = endpoints_end.get(tid_d)
		if p_disp is None:
			continue
		for tid_s, ss, es in singles_list:
			if tid_s == tid_d:
				continue
			# carrier must exist at the disappearance frame
			if int(ss) > disp_end:
				continue
			if int(es) < disp_end:
				continue
			p_car = _carrier_pos_near_frame(tid_s, disp_end, merge_end_sync_frames)
			if p_car is None:
				continue
			d = _dist_m(p_disp[0], p_car[0])
			if d > merge_dist_m:
				continue
			candidates.append((disp_end, tid_d, tid_s, float(d)))

	# Deterministic greedy selection: one carrier per disappearing tracklet.
	candidates.sort(key=lambda t: (t[0], t[1], t[3], t[2]))

	used_in_merge: Dict[str, int] = {}
	groups: List[Dict[str, Any]] = []

	def _tracklet_span(tid: str) -> Tuple[int, int]:
		row = ts_by_tid[tid]
		return int(row["start_frame"]), int(row["end_frame"])

	for disp_end, disappear, survivor, _dist in candidates:
		if used_in_merge.get(disappear) == disp_end or used_in_merge.get(survivor) == disp_end:
			continue
		used_in_merge[disappear] = disp_end
		used_in_merge[survivor] = disp_end

		group_start = disp_end + 1
		_surv_start, surv_end = _tracklet_span(survivor)
		if surv_end < group_start:
			continue

		best_split: Optional[Tuple[int, float, str]] = None  # (start_frame, dist, tid)
		for tid_n, sn, _en in succ_list:
			if tid_n == survivor:
				continue
			if sn < group_start:
				continue
			if sn > group_start + split_search_horizon_frames:
				break
			psn = endpoints_start.get(tid_n)
			if psn is None:
				continue
			pcar = carrier_pos_at_frame(survivor, sn)
			if pcar is None:
				pcar = endpoints_end.get(survivor)
			if pcar is None:
				continue
			d = _dist_m(psn[0], pcar[0])
			if d > split_dist_m:
				continue
			cand = (sn, d, tid_n)
			if best_split is None or cand < best_split:
				best_split = cand

		if best_split is not None:
			split_start, split_dist, tid_new = best_split
			group_end = split_start - 1
			if group_end < group_start:
				continue
			groups.append(
				{
					"kind": "merge_split",
					"carrier": survivor,
					"disappear": disappear,
					"new": tid_new,
					"group_start": group_start,
					"group_end": group_end,
					"split_start": split_start,
					"split_dist_m": split_dist,
					"merge_end": disp_end,
				}
			)
			merge_split_groups += 1
		else:
			if last_frame is not None:
				group_end = last_frame
				if group_end >= group_start:
					groups.append(
						{
							"kind": "merge_open_end",
							"carrier": survivor,
							"disappear": disappear,
							"new": None,
							"group_start": group_start,
							"group_end": group_end,
							"merge_end": disp_end,
						}
					)
					group_open_ended += 1

	# Clip-start merged inference: split-like event without explicit merge
	if enable_group_nodes:
		for tid_s, ss, _es in succ_list:
			if ss > 2:
				break
			best_split2: Optional[Tuple[int, float, str]] = None
			for tid_n, sn, _en in succ_list:
				if tid_n == tid_s:
					continue
				if sn <= ss:
					continue
				if sn > ss + split_search_horizon_frames:
					break
				psn = endpoints_start.get(tid_n)
				if psn is None:
					continue
				# Require direct evidence the carrier exists at the split start frame.
				pcar = carrier_pos_at_frame(tid_s, sn)
				if pcar is None:
					continue
				d = _dist_m(psn[0], pcar[0])
				if d > split_dist_m:
					continue
				cand = (sn, d, tid_n)
				if best_split2 is None or cand < best_split2:
					best_split2 = cand
			if best_split2 is None:
				continue
			sn, dist2, tid_n = best_split2
			group_start = 0
			group_end = sn - 1
			if group_end < group_start:
				continue
			# Avoid duplicating a (carrier,new) group span already inferred via merge/split.
			already_pair = any(
				(gr.get("carrier") == tid_s and gr.get("new") == tid_n and int(gr.get("group_end", -1)) == group_end)
				for gr in groups
			)
			if already_pair:
				continue
			already = any(
				(gr["group_start"] == 0 and gr["group_end"] == group_end and gr.get("carrier") == tid_s) for gr in groups
			)
			if already:
				continue
			groups.append(
				{
					"kind": "start_split",
					"carrier": tid_s,
					"disappear": None,
					"new": tid_n,
					"group_start": group_start,
					"group_end": group_end,
					"split_start": sn,
					"split_dist_m": dist2,
				}
			)
			group_from_start += 1
			break

	# ---- materialize groups into graph ----
	if enable_group_nodes:
		for gr in sorted(groups, key=lambda x: (x["group_start"], x["group_end"], str(x.get("carrier")), str(x.get("new")))):
			gs = int(gr["group_start"])
			ge = int(gr["group_end"])
			if ge < gs:
				continue
			carrier = str(gr["carrier"])
			disappear = gr.get("disappear", None)
			new_tid = gr.get("new", None)
			gn_id = f"G:{gs}-{ge}:carrier={carrier}:d={disappear or 'none'}:n={new_tid or 'none'}"
			g.add_node(
				GraphNode(
					node_id=gn_id,
					type=NodeType.GROUP_TRACKLET,
					capacity=2,
					start_frame=gs,
					end_frame=ge,
					payload={
						"carrier_tracklet_id": carrier,
						"disappearing_tracklet_id": disappear,
						"new_tracklet_id": new_tid,
						"kind": gr["kind"],
					},
				)
			)
			group_nodes_created += 1

			# group-at-start: birth cap=2
			if gs == 0:
				g.add_edge(GraphEdge(edge_id=f"E:BIRTH:{gn_id}", u="SOURCE", v=gn_id, type=EdgeType.BIRTH, capacity=2, payload={}))

			# group-at-end: death cap=2
			if last_frame is not None and ge == last_frame:
				g.add_edge(GraphEdge(edge_id=f"E:DEATH:{gn_id}", u=gn_id, v="SINK", type=EdgeType.DEATH, capacity=2, payload={}))

			# merge edges (two singles into group) when we have disappear+carrier
			if disappear is not None:
				u1 = f"T:{disappear}"
				u2 = f"T:{carrier}"
				g.add_edge(
					GraphEdge(
						edge_id=f"E:MERGE:{u1}->{gn_id}",
						u=u1,
						v=gn_id,
						type=EdgeType.MERGE,
						capacity=1,
						payload={"merge_end": int(gr.get("merge_end", -1))},
					)
				)
				g.add_edge(
					GraphEdge(
						edge_id=f"E:MERGE:{u2}->{gn_id}",
						u=u2,
						v=gn_id,
						type=EdgeType.MERGE,
						capacity=1,
						payload={"merge_end": int(gr.get("merge_end", -1))},
					)
				)

			# split edges: group -> carrier and group -> new
			if new_tid is not None:
				v1 = f"T:{carrier}"
				v2 = f"T:{new_tid}"
				g.add_edge(
					GraphEdge(
						edge_id=f"E:SPLIT:{gn_id}->{v1}",
						u=gn_id,
						v=v1,
						type=EdgeType.SPLIT,
						capacity=1,
						payload={"split_start": int(gr.get("split_start", -1))},
					)
				)
				g.add_edge(
					GraphEdge(
						edge_id=f"E:SPLIT:{gn_id}->{v2}",
						u=gn_id,
						v=v2,
						type=EdgeType.SPLIT,
						capacity=1,
						payload={"split_start": int(gr.get("split_start", -1))},
					)
				)

	# ---- suppress conflicting CONTINUE edges around group spans ----
	if enable_group_nodes and group_nodes_created > 0:
		group_intervals: List[Tuple[int, int, str]] = []
		for node in g.nodes.values():
			if node.type == NodeType.GROUP_TRACKLET:
				group_intervals.append((int(node.start_frame), int(node.end_frame), node.node_id))
		group_intervals.sort()

		def _in_any_group(frame_idx: int) -> bool:
			for a, b, _ in group_intervals:
				if a <= frame_idx <= b:
					return True
			return False

		edges_to_remove: List[str] = []
		for e in g.edges.values():
			if e.type != EdgeType.CONTINUE:
				continue
			u_tid = e.u.split("T:", 1)[-1] if e.u.startswith("T:") else None
			v_tid = e.v.split("T:", 1)[-1] if e.v.startswith("T:") else None
			if u_tid is None or v_tid is None:
				continue
			ei = int(ts_by_tid[u_tid]["end_frame"])
			sj = int(ts_by_tid[v_tid]["start_frame"])
			if _in_any_group(ei + 1) or _in_any_group(sj - 1):
				edges_to_remove.append(e.edge_id)
		for eid in edges_to_remove:
			g.edges.pop(eid, None)
		suppressed_continue_edges = len(edges_to_remove)

	# ---- final validate ----
	g.validate()

	# ---- audit ----
	audit_evt = {
		"ts_ms": _now_ms(),
		"stage": "D1",
		"event": "d1_graph_built",
		"inputs": {
			"bank_frames": str(frames_path),
			"bank_summaries": str(summ_path),
		},
		"manifest": {
			"fps": fps,
			"frame_count": frame_count,
			"duration_ms": duration_ms,
		},
		"coords": {
			"primary": ["x_m_repaired", "y_m_repaired"],
			"fallback": ["x_m", "y_m"],
			"on_mat_missing": on_mat_missing,
			"frames_used_repaired": used_repaired_frames,
			"frames_used_raw_fallback": used_raw_frames,
			"start_endpoints_used_raw": used_raw_start,
			"end_endpoints_used_raw": used_raw_end,
		},
		"graph": {
			"n_nodes": len(g.nodes),
			"n_edges": len(g.edges),
			"n_group_nodes": group_nodes_created,
			"n_groups_from_start": group_from_start,
			"n_groups_open_ended": group_open_ended,
			"n_groups_merge_split": merge_split_groups,
			"continue_pruned_cannot": continue_pruned_cannot,
			"suppressed_continue_edges": suppressed_continue_edges,
			"missing_endpoint_coords": missing_endpoint_coords,
		},
	}
	_write_audit_event(audit_path, audit_evt)
	return g
