"""Report writer for swap pattern characterization (CP-SWAP-2).

Produces per-camera detail JSON files, a structured _characterization.json,
and a _characterization.md aggregate report.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from pipeline_validation.tracker_swap.characterize import CharacterizationResult

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
OUTPUT_BASE = REPO_ROOT / "outputs" / "_eval" / "tracker_swap"


def _safe_json(obj):
    """Convert numpy/pandas types for JSON serialization."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        if np.isnan(obj):
            return None
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    raise TypeError(f"Cannot serialize {type(obj)}")


def write_characterization_reports(
    model_id: str, results: list[CharacterizationResult]
) -> Path:
    """Write all per-camera and aggregate reports. Returns aggregate md path."""
    model_dir = OUTPUT_BASE / model_id
    model_dir.mkdir(parents=True, exist_ok=True)

    all_data: dict = {}

    for res in results:
        cam_dir = model_dir / res.camera_id
        cam_dir.mkdir(parents=True, exist_ok=True)

        # Per-camera detail files
        for name, data in [
            ("topology.json", res.topology),
            ("persistence.json", res.persistence),
            ("spatial_context.json", res.spatial_context),
            ("spike_profiles.json", res.spike_profiles),
        ]:
            path = cam_dir / name
            with open(path, "w") as f:
                json.dump(data, f, indent=2, default=_safe_json)
            logger.info("Wrote %s (%d entries)", path, len(data))

        # Segments file
        seg_path = cam_dir / "persistence_segments.json"
        with open(seg_path, "w") as f:
            json.dump(res.persistence_segments, f, indent=2, default=_safe_json)

        all_data[res.camera_id] = {
            "topology": res.topology,
            "persistence": res.persistence,
            "persistence_segments": res.persistence_segments,
            "spatial_context": res.spatial_context,
            "spike_profiles": res.spike_profiles,
            "stride": res.stride,
        }

    # Structured JSON
    json_path = model_dir / "_characterization.json"
    with open(json_path, "w") as f:
        json.dump(all_data, f, indent=2, default=_safe_json)

    # Markdown report
    md_path = model_dir / "_characterization.md"
    _write_md(md_path, model_id, results)
    logger.info("Wrote characterization report to %s", md_path)
    return md_path


def _write_md(
    path: Path, model_id: str, results: list[CharacterizationResult]
) -> None:
    L: list[str] = []
    L.append(f"# Swap Pattern Characterization: {model_id}\n")

    # ---------------------------------------------------------------
    # 1. Topology
    # ---------------------------------------------------------------
    L.append("## 1. Swap Topology\n")
    topo_classes = ["exchange", "hop_into_occupied", "hop_into_unoccupied", "cascade"]
    L.append("| Camera | " + " | ".join(topo_classes) + " | total |")
    L.append("|--------" + "|--------" * (len(topo_classes) + 1) + "|")

    for res in results:
        counts = {c: 0 for c in topo_classes}
        for t in res.topology:
            tc = t["topology_class"]
            counts[tc] = counts.get(tc, 0) + 1
        total = sum(counts.values())
        cells = []
        for c in topo_classes:
            n = counts[c]
            pct = f"{100 * n / total:.0f}%" if total > 0 else "—"
            cells.append(f"{n} ({pct})")
        L.append(f"| {res.camera_id} | " + " | ".join(cells) + f" | {total} |")

    # Interpretation
    all_topo = []
    for r in results:
        all_topo.extend(r.topology)
    if all_topo:
        total_all = len(all_topo)
        exchange_pct = 100 * sum(1 for t in all_topo if t["topology_class"] == "exchange") / total_all
        cascade_pct = 100 * sum(1 for t in all_topo if t["topology_class"] == "cascade") / total_all
        L.append(f"\nExchanges account for {exchange_pct:.0f}% of swaps across all cameras. "
                 f"Cascades (3+ tracklets rearranging) account for {cascade_pct:.0f}%.")
        L.append(f"Note: ±{3} GT-frame cascade window = ±3 raw frames on stride-1, "
                 f"±30 raw frames on stride-10.\n")

    # ---------------------------------------------------------------
    # 2. Persistence
    # ---------------------------------------------------------------
    L.append("## 2. Swap Persistence\n")
    pers_classes = ["sustained", "transient_return", "transient_onward"]
    L.append("| Camera | " + " | ".join(pers_classes) + " | total |")
    L.append("|--------" + "|--------" * (len(pers_classes) + 1) + "|")

    all_transient_dwells: list[int] = []
    for res in results:
        counts = {c: 0 for c in pers_classes}
        for p in res.persistence:
            pc = p["persistence_class"]
            counts[pc] = counts.get(pc, 0) + 1
            if pc.startswith("transient"):
                all_transient_dwells.append(p["dwell_gt_frames"])
        total = sum(counts.values())
        cells = []
        for c in pers_classes:
            n = counts[c]
            pct = f"{100 * n / total:.0f}%" if total > 0 else "—"
            cells.append(f"{n} ({pct})")
        L.append(f"| {res.camera_id} | " + " | ".join(cells) + f" | {total} |")

    # Transient dwell histogram
    if all_transient_dwells:
        L.append("\n**Transient dwell time distribution (GT-annotated frames):**\n")
        bins = [(1, 1, "1 frame"), (2, 3, "2-3 frames"), (4, 9, "4-9 frames")]
        for lo, hi, label in bins:
            n = sum(1 for d in all_transient_dwells if lo <= d <= hi)
            L.append(f"- {label}: {n} ({100 * n / len(all_transient_dwells):.0f}%)")

    L.append(f"\nSustained threshold: {10} GT-annotated frames "
             "(= 10 raw frames on stride-1, 100 raw frames on stride-10).\n")

    # ---------------------------------------------------------------
    # 3. Spatial Context
    # ---------------------------------------------------------------
    L.append("## 3. Spatial Context\n")
    L.append("| Camera | Mean swap dist (m) | Median swap dist (m) | dest_occupied % | source_persists % | both % |")
    L.append("|--------|-------------------|---------------------|----------------|------------------|-------|")

    all_dists_m: list[float] = []
    for res in results:
        dists = [s["swap_distance_m"] for s in res.spatial_context if s.get("swap_distance_m") is not None]
        mean_d = f"{np.mean(dists):.3f}" if dists else "—"
        med_d = f"{np.median(dists):.3f}" if dists else "—"
        all_dists_m.extend(dists)

        n_total = len(res.spatial_context)
        if n_total > 0:
            dest_occ = sum(1 for s in res.spatial_context if s.get("dest_was_occupied"))
            src_per = sum(1 for s in res.spatial_context if s.get("source_persists"))
            both = sum(1 for s in res.spatial_context
                      if s.get("dest_was_occupied") and s.get("source_persists"))
            dest_pct = f"{100 * dest_occ / n_total:.0f}%"
            src_pct = f"{100 * src_per / n_total:.0f}%"
            both_pct = f"{100 * both / n_total:.0f}%"
        else:
            dest_pct = src_pct = both_pct = "—"

        L.append(f"| {res.camera_id} | {mean_d} | {med_d} | {dest_pct} | {src_pct} | {both_pct} |")

    # Distance histogram
    if all_dists_m:
        L.append("\n**Swap distance distribution (meters):**\n")
        bins = [(0, 0.3), (0.3, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, float("inf"))]
        labels = ["<0.3m", "0.3–0.5m", "0.5–1.0m", "1.0–2.0m", ">2.0m"]
        for (lo, hi), label in zip(bins, labels):
            n = sum(1 for d in all_dists_m if lo <= d < hi)
            L.append(f"- {label}: {n} ({100 * n / len(all_dists_m):.0f}%)")
    L.append("")

    # ---------------------------------------------------------------
    # 4. Kinematic Spike Profiles
    # ---------------------------------------------------------------
    L.append("## 4. Kinematic Spike Profiles\n")

    # Stride-1 cameras
    stride1_results = [r for r in results if r.stride == 1]
    if stride1_results:
        L.append("### Stride-1 (frame-level profiles)\n")
        spike_classes = ["impulse", "ramp_sustain", "oscillating", "no_spike"]
        for res in stride1_results:
            L.append(f"**{res.camera_id}:**\n")
            L.append("| Spike class | Count | % |")
            L.append("|------------|-------|---|")
            counts = {c: 0 for c in spike_classes}
            for sp in res.spike_profiles:
                sc = sp.get("spike_class")
                if sc:
                    counts[sc] = counts.get(sc, 0) + 1
            total = sum(counts.values())
            for c in spike_classes:
                n = counts[c]
                pct = f"{100 * n / total:.0f}%" if total > 0 else "—"
                L.append(f"| {c} | {n} | {pct} |")
            L.append("")

    # Stride-10 cameras
    stride10_results = [r for r in results if r.stride > 1]
    if stride10_results:
        L.append("### Stride-10 (gap summary)\n")
        L.append("| Camera | Swaps with gap data | Max speed > 2x median | Mean max_speed_in_gap |")
        L.append("|--------|--------------------|-----------------------|----------------------|")
        for res in stride10_results:
            with_data = [sp for sp in res.spike_profiles if sp.get("max_speed_in_gap") is not None]
            exceeds = sum(1 for sp in with_data if sp.get("max_exceeds_2x_median"))
            n_data = len(with_data)
            mean_max = (
                f"{np.mean([sp['max_speed_in_gap'] for sp in with_data]):.2f}"
                if with_data else "—"
            )
            exc_pct = f"{exceeds}/{n_data} ({100 * exceeds / n_data:.0f}%)" if n_data > 0 else "—"
            L.append(f"| {res.camera_id} | {n_data} | {exc_pct} | {mean_max} |")
        L.append("")

    # ---------------------------------------------------------------
    # Design Implications
    # ---------------------------------------------------------------
    L.append("## Design Implications for Splitter\n")
    L.append("*Observations from the data — not prescriptions. "
             "Splitter design is a separate brief.*\n")

    if all_topo:
        total_all = len(all_topo)
        exchange_n = sum(1 for t in all_topo if t["topology_class"] == "exchange")
        cascade_n = sum(1 for t in all_topo if t["topology_class"] == "cascade")
        hop_occ_n = sum(1 for t in all_topo if t["topology_class"] == "hop_into_occupied")

        if exchange_n / total_all > 0.8:
            L.append("- **Two-body exchange dominates** (>80%): a pairwise swap model "
                     "would cover the vast majority of cases.")
        elif exchange_n / total_all > 0.5:
            L.append(f"- **Exchanges are the plurality** ({100 * exchange_n / total_all:.0f}%) "
                     "but not dominant. The splitter should handle exchanges well but "
                     "also account for one-sided hops.")
        else:
            L.append(f"- **Exchanges are a minority** ({100 * exchange_n / total_all:.0f}%). "
                     "A two-body exchange model alone would miss most swap patterns.")

        if cascade_n / total_all > 0.2:
            L.append(f"- **Significant cascade rate** ({100 * cascade_n / total_all:.0f}%): "
                     "the splitter needs multi-tracklet reasoning, not just pairwise.")
        if hop_occ_n / total_all > 0.3:
            L.append(f"- **Hop-into-occupied is common** ({100 * hop_occ_n / total_all:.0f}%): "
                     "the splitter should handle cases where a tracklet jumps onto an "
                     "already-tracked person without a reciprocal swap.")

    # Persistence implications
    all_pers = []
    for r in results:
        all_pers.extend(r.persistence)
    if all_pers:
        transient_n = sum(1 for p in all_pers if p["persistence_class"].startswith("transient"))
        if transient_n / len(all_pers) > 0.3:
            L.append(f"- **Significant transient rate** ({100 * transient_n / len(all_pers):.0f}%): "
                     "the splitter needs a minimum-dwell filter to avoid splitting "
                     "on flickers that self-correct.")
        elif transient_n > 0:
            L.append(f"- **Some transients exist** ({transient_n}/{len(all_pers)}): "
                     "a minimum-dwell filter may help but is not critical.")

    # Spatial implications
    if all_dists_m:
        close_swaps = sum(1 for d in all_dists_m if d < 0.5)
        if close_swaps / len(all_dists_m) > 0.7:
            L.append(f"- **Most swaps are close-range** ({100 * close_swaps / len(all_dists_m):.0f}% "
                     "< 0.5m): consistent with grappling-proximity swap mechanism.")

    # Spike implications
    for res in stride1_results:
        impulse_n = sum(1 for sp in res.spike_profiles if sp.get("spike_class") == "impulse")
        no_spike_n = sum(1 for sp in res.spike_profiles if sp.get("spike_class") == "no_spike")
        total_sp = len(res.spike_profiles)
        if total_sp > 0:
            if impulse_n / total_sp > 0.5:
                L.append(f"- **Impulse spikes dominate on {res.camera_id}** "
                         f"({100 * impulse_n / total_sp:.0f}%): "
                         "a spike-isolation-ratio detector is viable.")
            if no_spike_n / total_sp > 0.3:
                L.append(f"- **Significant no-spike fraction on {res.camera_id}** "
                         f"({100 * no_spike_n / total_sp:.0f}%): "
                         "kinematic signals alone will miss some swaps.")

    for res in stride10_results:
        with_data = [sp for sp in res.spike_profiles if sp.get("max_speed_in_gap") is not None]
        exceeds = sum(1 for sp in with_data if sp.get("max_exceeds_2x_median"))
        if with_data:
            exc_pct = 100 * exceeds / len(with_data)
            L.append(f"- **{res.camera_id} stride-10**: {exc_pct:.0f}% of swaps show "
                     f"speed spike > 2x median within the GT gap, confirming kinematic "
                     "signal is detectable despite GT timing uncertainty.")

    L.append("\n---")
    L.append("*GT assignment uses IoU >= 0.3. Cascade window: ±3 GT-annotated frames. "
             "Sustained threshold: 10 GT-annotated frames.*")

    path.write_text("\n".join(L) + "\n")
