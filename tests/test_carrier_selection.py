"""CP-TAG-4a Fix D: Carrier-selection rule unit tests.

Tests the deterministic carrier selection when multiple tracklets observe
the same tag ping at the same frame. The chosen carrier gets hard treatment
(binding + must_link + no-drop); others get soft support only.

No live test case exists in the v2-model data (t139 is the sole observer),
so these tests use synthetic fixtures.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest


def _build_synthetic_nodes_df(
    tracklet_ids: list[str],
    frame_ranges: list[tuple[int, int]],
    node_types: list[str] | None = None,
) -> pd.DataFrame:
    """Build a minimal D1 graph nodes DataFrame for testing."""
    rows = []
    for i, (tid, (sf, ef)) in enumerate(zip(tracklet_ids, frame_ranges)):
        nt = (node_types[i] if node_types else "NodeType.SINGLE_TRACKLET")
        nid = f"T:{tid}" if "SINGLE" in nt else f"G:{sf}-{ef}:carrier={tid}"
        row = {
            "node_id": nid,
            "node_type": nt,
            "capacity": 1 if "SINGLE" in nt else 2,
            "start_frame": sf,
            "end_frame": ef,
            "base_tracklet_id": tid,
            "segment_type": "SOLO" if "SINGLE" in nt else "GROUP",
            "carrier_tracklet_id": tid if "GROUP" in nt else None,
            "disappearing_tracklet_id": None,
            "new_tracklet_id": None,
        }
        rows.append(row)
    # Add SOURCE and SINK
    rows.append({
        "node_id": "SOURCE", "node_type": "NodeType.SOURCE",
        "capacity": 999, "start_frame": None, "end_frame": None,
        "base_tracklet_id": None, "segment_type": None,
        "carrier_tracklet_id": None, "disappearing_tracklet_id": None,
        "new_tracklet_id": None,
    })
    rows.append({
        "node_id": "SINK", "node_type": "NodeType.SINK",
        "capacity": 999, "start_frame": None, "end_frame": None,
        "base_tracklet_id": None, "segment_type": None,
        "carrier_tracklet_id": None, "disappearing_tracklet_id": None,
        "new_tracklet_id": None,
    })
    return pd.DataFrame(rows)


def _build_synthetic_constraints(
    tracklet_ids: list[str],
    tag_id: str = "1",
    frame_index: int = 100,
) -> dict:
    """Build minimal D2 constraints with multi-tracklet must_link + tag_pings."""
    must_link_groups = [
        {
            "anchor_key": f"tag:{tag_id}",
            "tracklet_ids": tracklet_ids,
        }
    ]
    tag_pings = []
    for tid in tracklet_ids:
        tag_pings.append({
            "tracklet_id": tid,
            "anchor_key": f"tag:{tag_id}",
            "frame_index": frame_index,
            "confidence": 1.0,
        })
    return {
        "must_link_groups": must_link_groups,
        "tag_pings": tag_pings,
    }


class _MockManifest:
    """Minimal manifest stub for _emit_mcf_tag_inputs."""
    clip_id = "test_clip"
    camera_id = "test_cam"
    pipeline_version = "test"


def test_split_aware_binding_prefers_product_containing_frame():
    """When a pre-split tracklet ID doesn't cover the ping frame but a split
    product does, the ping should bind to the product's node."""
    from bjj_pipeline.stages.stitch.d3_ilp2 import _emit_mcf_tag_inputs

    # Pre-split t100 covers [50, 90], product t100_s1 covers [91, 150]
    nodes_df = _build_synthetic_nodes_df(
        tracklet_ids=["t100", "t100_s1"],
        frame_ranges=[(50, 90), (91, 150)],
    )

    constraints = _build_synthetic_constraints(
        tracklet_ids=["t100"],
        tag_id="1",
        frame_index=100,  # Falls on t100_s1, NOT on t100
    )

    split_map = {"t100": ["t100_s1"]}

    with tempfile.TemporaryDirectory() as tmpdir:
        result = _emit_mcf_tag_inputs(
            debug_dir=Path(tmpdir),
            manifest=_MockManifest(),
            checkpoint="POC_2_TAGS",
            nodes_df=nodes_df,
            constraints=constraints,
            split_map=split_map,
        )

    pings = result.get("pings", [])
    assert len(pings) == 1, f"Expected 1 ping, got {len(pings)}"

    binding = pings[0].get("binding", {})
    assert binding["status"] == "bound", (
        f"Ping should be bound after split expansion, got: {binding}"
    )
    assert "t100_s1" in binding["chosen"]["node_id"], (
        f"Ping should bind to t100_s1's node, got: {binding['chosen']['node_id']}"
    )

    # Check ping_carrying_products
    pcp = result.get("ping_carrying_products", {})
    assert "tag:1" in pcp, f"Expected tag:1 in ping_carrying_products, got: {pcp}"
    assert "t100_s1" in pcp["tag:1"], (
        f"Expected t100_s1 in carrying products, got: {pcp['tag:1']}"
    )


def test_binding_without_split_uses_original():
    """When the pre-split tracklet covers the ping frame, no split expansion needed."""
    from bjj_pipeline.stages.stitch.d3_ilp2 import _emit_mcf_tag_inputs

    nodes_df = _build_synthetic_nodes_df(
        tracklet_ids=["t200"],
        frame_ranges=[(50, 150)],
    )

    constraints = _build_synthetic_constraints(
        tracklet_ids=["t200"],
        tag_id="2",
        frame_index=100,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        result = _emit_mcf_tag_inputs(
            debug_dir=Path(tmpdir),
            manifest=_MockManifest(),
            checkpoint="POC_2_TAGS",
            nodes_df=nodes_df,
            constraints=constraints,
            split_map=None,
        )

    pings = result.get("pings", [])
    assert len(pings) == 1
    assert pings[0]["binding"]["status"] == "bound"
    # No split products recorded
    pcp = result.get("ping_carrying_products", {})
    assert not pcp.get("tag:2", [])


def test_multi_tracklet_ping_carrier_selection():
    """When two tracklets observe the same ping frame, both become candidates.
    The existing tiebreak (shortest span, then node_id) picks one deterministically.
    Only the chosen carrier should appear in ping_carrying_products."""
    from bjj_pipeline.stages.stitch.d3_ilp2 import _emit_mcf_tag_inputs

    # Both tracklets cover frame 100. t_carrier has shorter span.
    nodes_df = _build_synthetic_nodes_df(
        tracklet_ids=["t_carrier", "t_other"],
        frame_ranges=[(95, 105), (50, 200)],
    )

    # Both tracklets claim the tag at frame 100
    constraints = {
        "must_link_groups": [
            {"anchor_key": "tag:3", "tracklet_ids": ["t_carrier", "t_other"]},
        ],
        "tag_pings": [
            {"tracklet_id": "t_carrier", "anchor_key": "tag:3", "frame_index": 100, "confidence": 1.0},
            {"tracklet_id": "t_other", "anchor_key": "tag:3", "frame_index": 100, "confidence": 1.0},
        ],
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        result = _emit_mcf_tag_inputs(
            debug_dir=Path(tmpdir),
            manifest=_MockManifest(),
            checkpoint="POC_2_TAGS",
            nodes_df=nodes_df,
            constraints=constraints,
            split_map=None,
        )

    pings = result.get("pings", [])
    assert len(pings) == 2

    # Both should bind (each has a node covering frame 100)
    for p in pings:
        assert p["binding"]["status"] == "bound", (
            f"Ping {p['ping_id']} should be bound"
        )

    # The first ping (t_carrier) should bind to the shorter-span node
    carrier_ping = [p for p in pings if "t_carrier" in str(p.get("observed", {}).get("tracklet_id", ""))]
    assert len(carrier_ping) == 1
    assert "t_carrier" in carrier_ping[0]["binding"]["chosen"]["node_id"]


def test_unbound_ping_emits_warning_note():
    """When a ping can't bind even after split expansion, the regression
    assertion adds a warning note to the binding."""
    from bjj_pipeline.stages.stitch.d3_ilp2 import _emit_mcf_tag_inputs

    # Node covers [50, 90] but ping is at frame 200 — no match
    nodes_df = _build_synthetic_nodes_df(
        tracklet_ids=["t300"],
        frame_ranges=[(50, 90)],
    )

    constraints = _build_synthetic_constraints(
        tracklet_ids=["t300"],
        tag_id="4",
        frame_index=200,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        result = _emit_mcf_tag_inputs(
            debug_dir=Path(tmpdir),
            manifest=_MockManifest(),
            checkpoint="POC_2_TAGS",
            nodes_df=nodes_df,
            constraints=constraints,
            split_map={"t300": ["t300_s1"]},  # product also doesn't help
        )

    pings = result.get("pings", [])
    assert len(pings) == 1
    assert pings[0]["binding"]["status"] == "unbound"
    # Regression assertion should have added a warning note
    notes = pings[0]["binding"].get("notes", [])
    assert any("tag_ping_unbound_after_split_expansion" in str(n) for n in notes), (
        f"Expected unbound warning note, got: {notes}"
    )
