"""Stage D graph primitives.

D1 constructs a solver-agnostic graph over tracklets (and inferred group spans).
D2 attaches costs; D3 solves (MCF initially; ILP possible later).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class NodeType(str, Enum):
	SOURCE = "SOURCE"
	SINK = "SINK"
	SINGLE_TRACKLET = "SINGLE_TRACKLET"
	GROUP_TRACKLET = "GROUP_TRACKLET"


class EdgeType(str, Enum):
	BIRTH = "BIRTH"
	DEATH = "DEATH"
	CONTINUE = "CONTINUE"
	MERGE = "MERGE"
	SPLIT = "SPLIT"


@dataclass(frozen=True)
class GraphNode:
	node_id: str
	type: NodeType
	capacity: int
	start_frame: Optional[int]
	end_frame: Optional[int]
	payload: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GraphEdge:
	edge_id: str
	u: str
	v: str
	type: EdgeType
	capacity: int
	payload: Dict[str, Any] = field(default_factory=dict)


class TrackletGraph:
	"""Solver-agnostic graph for D1/D2/D3."""

	def __init__(self) -> None:
		self.nodes: Dict[str, GraphNode] = {}
		self.edges: Dict[str, GraphEdge] = {}

	def add_node(self, node: GraphNode) -> None:
		if node.node_id in self.nodes:
			return
		self.nodes[node.node_id] = node

	def add_edge(self, edge: GraphEdge) -> None:
		if edge.edge_id in self.edges:
			return
		self.edges[edge.edge_id] = edge

	def validate(self) -> None:
		# Node validation
		for nid, n in self.nodes.items():
			if n.capacity is None:
				raise ValueError(f"Node {nid} has null capacity")
			if n.type == NodeType.SINGLE_TRACKLET and n.capacity != 1:
				raise ValueError(f"Single node {nid} must have capacity=1")
			if n.type == NodeType.GROUP_TRACKLET and n.capacity != 2:
				raise ValueError(f"Group node {nid} must have capacity=2")
			if n.start_frame is not None and n.end_frame is not None and n.start_frame > n.end_frame:
				raise ValueError(f"Node {nid} has invalid span {n.start_frame}>{n.end_frame}")

		# Edge validation
		for eid, e in self.edges.items():
			if e.capacity is None or e.capacity <= 0:
				raise ValueError(f"Edge {eid} has invalid capacity {e.capacity}")
			if e.u not in self.nodes:
				raise ValueError(f"Edge {eid} refers to missing u node {e.u}")
			if e.v not in self.nodes:
				raise ValueError(f"Edge {eid} refers to missing v node {e.v}")

	@staticmethod
	def _node_type_order(t: NodeType) -> int:
		return {
			NodeType.SOURCE: 0,
			NodeType.SINK: 1,
			NodeType.GROUP_TRACKLET: 2,
			NodeType.SINGLE_TRACKLET: 3,
		}[t]

	@staticmethod
	def _edge_type_order(t: EdgeType) -> int:
		return {
			EdgeType.BIRTH: 0,
			EdgeType.DEATH: 1,
			EdgeType.MERGE: 2,
			EdgeType.SPLIT: 3,
			EdgeType.CONTINUE: 4,
		}[t]

	def sorted_nodes(self):
		return sorted(
			self.nodes.values(),
			key=lambda n: (
				self._node_type_order(n.type),
				n.start_frame if n.start_frame is not None else -1,
				n.end_frame if n.end_frame is not None else -1,
				n.node_id,
			),
		)

	def sorted_edges(self):
		return sorted(
			self.edges.values(),
			key=lambda e: (
				self._edge_type_order(e.type),
				e.u,
				e.v,
				e.edge_id,
			),
		)
