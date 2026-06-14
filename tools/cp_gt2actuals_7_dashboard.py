#!/usr/bin/env python3
"""CP-GT2ACTUALS-7: Interactive 3D dashboard for error-map review.

READ-ONLY viewer of the validated dense GT-to-actuals artifact.
Emits a self-contained HTML file with plotly 3D scatter + JS controls.

Usage:
    PYTHONPATH=src python tools/cp_gt2actuals_7_dashboard.py [--clip vid1|vid2] [--out path.html]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
OUTPUTS = REPO / "outputs"

CLIPS = {
    "vid2": {
        "gt2a": OUTPUTS / "_eval/gt2actuals/J_EDEw/J_EDEw-20260318-200246",
        "stage_d": OUTPUTS / "_eval_gt/J_EDEw/2026-03-18/20/J_EDEw-20260318-200246/stage_D",
        "label": "J_EDEw vid2 (0-4490, authoritative)",
    },
    "vid1": {
        "gt2a": OUTPUTS / "_eval/gt2actuals/J_EDEw/J_EDEw-20260318-200015",
        "stage_d": OUTPUTS / "_eval_gt/J_EDEw/2026-03-18/20/J_EDEw-20260318-200015/stage_D",
        "label": "J_EDEw vid1 (0-3000, corroboration)",
    },
}

def _json(obj):
    """JSON dumps with numpy type handling."""
    return json.dumps(obj, default=lambda x: int(x) if hasattr(x, 'item') else str(x))


JUMP_MARKERS = {
    "tracklet_drift": {"symbol": "star", "label": "Stage A: tracklet_drift"},
    "ilp_misstitch": {"symbol": "square", "label": "Solver: ilp_misstitch"},
    "false_split": {"symbol": "triangle-up", "label": "D0.5: false_split"},
    "group_boundary_jump": {"symbol": "diamond", "label": "Group: boundary_jump"},
    "group_membership_drift": {"symbol": "x", "label": "Group: membership_drift"},
}


def load_mat_blueprint() -> list[dict]:
    path = REPO / "configs" / "mat_blueprint.json"
    if not path.exists():
        return []
    return json.load(open(path))


def build_clip_data(clip_key: str) -> dict:
    """Build all data needed for one clip's dashboard."""
    cfg = CLIPS[clip_key]
    df = pd.read_parquet(cfg["gt2a"] / "gt2actuals_dense.parquet")
    pt = pd.read_parquet(cfg["stage_d"] / "person_tracks.parquet")

    gt_ids = sorted(df.gt_track_id.unique())

    # Find highest-jump GT person for default selection
    jumps = df[df.jump_type.notna()]
    jc = jumps.groupby("gt_track_id").size().sort_values(ascending=False)
    default_gt = int(jc.index[0]) if len(jc) > 0 else gt_ids[0]

    # Subsample factor for line traces (keep all markers)
    n_frames = df.frame_index.nunique()
    subsample = max(1, n_frames // 1500)

    # Build GT worms: per gt_track_id, sorted by frame
    gt_worms = {}
    for gt_id in gt_ids:
        g = df[df.gt_track_id == gt_id].sort_values("frame_index")
        mask = g.x_m_eff.notna() & g.y_m_eff.notna()
        g_valid = g[mask]
        if len(g_valid) == 0:
            continue

        # Subsample line but keep jump frames
        is_jump = g_valid.jump_type.notna()
        line_idx = np.zeros(len(g_valid), dtype=bool)
        line_idx[::subsample] = True
        line_idx |= is_jump.values
        line_idx[-1] = True  # always keep last
        g_sub = g_valid[line_idx]

        gt_worms[int(gt_id)] = {
            "x": g_sub.x_m_eff.tolist(),
            "y": g_sub.y_m_eff.tolist(),
            "t": g_sub.frame_index.tolist(),
            "state": g_sub.state.tolist(),
            "speed": [float(v) if pd.notna(v) else None for v in g_sub.speed_mps_k],
            "isolated": [bool(v) if pd.notna(v) else None for v in g_sub.is_isolated],
        }

    # Build pipeline worms: per person_id trajectory from person_tracks
    pid_worms = {}
    all_pids = set()
    for pids_json in df.person_ids.dropna():
        all_pids.update(json.loads(pids_json))

    for pid in sorted(all_pids):
        p = pt[pt.person_id == pid].sort_values("frame_index")
        mask = p.x_m.notna() & p.y_m.notna()
        p_valid = p[mask]
        if len(p_valid) == 0:
            continue
        # Subsample
        p_sub = p_valid.iloc[::subsample]
        if len(p_valid) > 0 and p_valid.index[-1] not in p_sub.index:
            p_sub = pd.concat([p_sub, p_valid.iloc[[-1]]])

        pid_worms[pid] = {
            "x": p_sub.x_m.tolist(),
            "y": p_sub.y_m.tolist(),
            "t": p_sub.frame_index.tolist(),
        }

    # Build GT-to-PID mapping: for each GT person, which person_ids are assigned?
    gt_to_pids: dict[int, set[str]] = {}
    for gt_id in gt_ids:
        g = df[df.gt_track_id == gt_id]
        pids = set()
        for pids_json in g.person_ids.dropna():
            pids.update(json.loads(pids_json))
        gt_to_pids[int(gt_id)] = pids

    # Build jump markers
    jump_data: dict[str, list[dict]] = {jt: [] for jt in JUMP_MARKERS}
    for _, r in jumps.iterrows():
        if pd.isna(r.x_m_eff) or pd.isna(r.y_m_eff):
            continue
        jt = r.jump_type
        if jt not in jump_data:
            continue
        jump_data[jt].append({
            "x": float(r.x_m_eff),
            "y": float(r.y_m_eff),
            "t": int(r.frame_index),
            "gt": int(r.gt_track_id),
            "state": r.state,
            "prev": r.jump_from_person_ids if pd.notna(r.jump_from_person_ids) else "[]",
            "curr": r.person_ids if pd.notna(r.person_ids) else "[]",
        })

    return {
        "clip_key": clip_key,
        "label": cfg["label"],
        "gt_ids": gt_ids,
        "default_gt": default_gt,
        "subsample": subsample,
        "gt_worms": gt_worms,
        "pid_worms": pid_worms,
        "gt_to_pids": {str(k): sorted(v) for k, v in gt_to_pids.items()},
        "jump_data": jump_data,
        "n_frames": n_frames,
        "n_rows": len(df),
    }


def build_html(clip_data: dict, mat_blueprint: list[dict]) -> str:
    """Build self-contained HTML dashboard."""

    # Pre-compute color palette for GT IDs
    n_gt = len(clip_data["gt_ids"])
    gt_colors = {}
    for i, gt_id in enumerate(clip_data["gt_ids"]):
        hue = int(i * 360 / max(n_gt, 1))
        gt_colors[int(gt_id)] = f"hsl({hue}, 70%, 50%)"

    # Person ID colors (muted, lower saturation)
    all_pids = sorted(clip_data["pid_worms"].keys())
    pid_colors = {}
    for i, pid in enumerate(all_pids):
        hue = int(i * 360 / max(len(all_pids), 1))
        pid_colors[pid] = f"hsl({hue}, 40%, 60%)"

    # Pre-build control HTML fragments (avoids f-string escaping issues)
    gt_options = "".join(
        f'<option value="{gt}" {"selected" if gt == clip_data["default_gt"] else ""}>'
        f'GT {gt}</option>'
        for gt in clip_data["gt_ids"]
    )
    jump_buttons = "".join(
        f'<button class="btn active" id="btn-{jt}" onclick="toggleJump(\'{jt}\')">'
        f'{JUMP_MARKERS[jt]["label"].split(": ")[1]}</button>'
        for jt in JUMP_MARKERS
    )

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>CP-GT2ACTUALS-7: Error Map Dashboard — {clip_data['label']}</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 0; padding: 10px; background: #1a1a2e; color: #eee; }}
h1 {{ font-size: 16px; margin: 5px 0; }}
.controls {{ display: flex; gap: 15px; flex-wrap: wrap; margin-bottom: 10px; font-size: 12px; }}
.control-group {{ background: #16213e; padding: 8px 12px; border-radius: 6px; }}
.control-group label {{ display: block; font-weight: bold; margin-bottom: 4px; color: #a8d8ea; }}
select, button {{ font-size: 11px; padding: 3px 6px; }}
select[multiple] {{ height: 100px; }}
.stats {{ font-size: 11px; color: #888; margin-top: 5px; }}
#plot {{ width: 100%; height: calc(100vh - 180px); }}
.btn {{ cursor: pointer; padding: 4px 10px; border: 1px solid #555; background: #2a2a4a; color: #eee; border-radius: 3px; margin: 2px; }}
.btn.active {{ background: #4a4a8a; border-color: #88f; }}
.legend {{ font-size: 10px; margin-top: 5px; }}
.legend span {{ margin-right: 10px; }}
</style>
</head>
<body>
<h1>CP-GT2ACTUALS-7: {clip_data['label']} &mdash; {clip_data['n_rows']} rows, subsample={clip_data['subsample']}x</h1>
<div class="controls">
  <div class="control-group">
    <label>GT Person</label>
    <select id="gt-select" multiple>
      {gt_options}
    </select>
  </div>
  <div class="control-group">
    <label>Show</label>
    <button class="btn active" id="btn-gt" onclick="toggleLayer('gt')">GT Path</button>
    <button class="btn active" id="btn-pipe" onclick="toggleLayer('pipe')">Pipeline Path</button>
    <button class="btn active" id="btn-markers" onclick="toggleLayer('markers')">Jump Markers</button>
    <br>
    <label style="margin-top:6px">Jump Types</label>
    {jump_buttons}
  </div>
  <div class="control-group">
    <label>Color By</label>
    <button class="btn active" id="btn-color-id" onclick="setColor('id')">Identity</button>
    <button class="btn" id="btn-color-speed" onclick="setColor('speed')">Speed</button>
    <button class="btn" id="btn-color-iso" onclick="setColor('isolated')">Isolated</button>
  </div>
</div>
<div id="plot"></div>

<script>
// === Embedded data ===
const GT_WORMS = {_json(clip_data["gt_worms"])};
const PID_WORMS = {_json(clip_data["pid_worms"])};
const GT_TO_PIDS = {_json(clip_data["gt_to_pids"])};
const JUMP_DATA = {_json(clip_data["jump_data"])};
const GT_IDS = {_json(clip_data["gt_ids"])};
const DEFAULT_GT = {int(clip_data["default_gt"])};
const MAT = {_json(mat_blueprint)};
const GT_COLORS = {_json(gt_colors)};
const PID_COLORS = {_json(pid_colors)};

const JUMP_SYMBOLS = {{
  tracklet_drift: "star",
  ilp_misstitch: "square",
  false_split: "triangle-up",
  group_boundary_jump: "diamond",
  group_membership_drift: "x"
}};

let showGT = true, showPipe = true, showMarkers = true;
let activeJumps = new Set(Object.keys(JUMP_SYMBOLS));
let colorMode = "id";
let selectedGTs = new Set([DEFAULT_GT]);

function toggleLayer(layer) {{
  if (layer === 'gt') showGT = !showGT;
  if (layer === 'pipe') showPipe = !showPipe;
  if (layer === 'markers') showMarkers = !showMarkers;
  document.getElementById('btn-' + layer).classList.toggle('active');
  render();
}}

function toggleJump(jt) {{
  if (activeJumps.has(jt)) activeJumps.delete(jt);
  else activeJumps.add(jt);
  document.getElementById('btn-' + jt).classList.toggle('active');
  render();
}}

function setColor(mode) {{
  colorMode = mode;
  ['id','speed','isolated'].forEach(m => {{
    document.getElementById('btn-color-' + m).classList.toggle('active', m === mode);
  }});
  render();
}}

document.getElementById('gt-select').addEventListener('change', function() {{
  selectedGTs = new Set(Array.from(this.selectedOptions).map(o => parseInt(o.value)));
  render();
}});

function render() {{
  const traces = [];

  // Mat blueprint on z=0 plane
  MAT.forEach(m => {{
    traces.push({{
      type: 'mesh3d',
      x: [m.x, m.x + m.width, m.x + m.width, m.x],
      y: [m.y, m.y, m.y + m.height, m.y + m.height],
      z: [0, 0, 0, 0],
      i: [0, 0], j: [1, 2], k: [2, 3],
      color: 'rgba(100,150,100,0.15)',
      hoverinfo: 'skip',
      showlegend: false,
    }});
  }});

  // GT worms
  if (showGT) {{
    selectedGTs.forEach(gtId => {{
      const w = GT_WORMS[gtId];
      if (!w) return;
      let lineColor;
      if (colorMode === 'id') {{
        lineColor = GT_COLORS[gtId] || 'white';
      }} else if (colorMode === 'speed') {{
        lineColor = w.speed.map(v => v != null ? v : 0);
      }} else {{
        lineColor = w.isolated.map(v => v === true ? 1 : v === false ? 0 : 0.5);
      }}
      traces.push({{
        type: 'scatter3d',
        mode: 'lines',
        x: w.x, y: w.y, z: w.t,
        line: colorMode === 'id'
          ? {{ color: lineColor, width: 3 }}
          : {{ color: lineColor, width: 3, colorscale: colorMode === 'speed' ? 'Hot' : 'RdYlGn', cmin: 0, cmax: colorMode === 'speed' ? 5 : 1 }},
        name: 'GT ' + gtId,
        legendgroup: 'gt' + gtId,
        hovertemplate: 'GT ' + gtId + '<br>frame=%{{z}}<br>x=%{{x:.2f}} y=%{{y:.2f}}<extra></extra>',
      }});
    }});
  }}

  // Pipeline worms
  if (showPipe) {{
    selectedGTs.forEach(gtId => {{
      const pids = GT_TO_PIDS[String(gtId)] || [];
      pids.forEach(pid => {{
        const w = PID_WORMS[pid];
        if (!w) return;
        traces.push({{
          type: 'scatter3d',
          mode: 'lines',
          x: w.x, y: w.y, z: w.t,
          line: {{ color: PID_COLORS[pid] || 'gray', width: 1.5, dash: 'dot' }},
          name: pid,
          legendgroup: 'pid_' + pid,
          opacity: 0.6,
          hovertemplate: pid + '<br>frame=%{{z}}<br>x=%{{x:.2f}} y=%{{y:.2f}}<extra></extra>',
        }});
      }});
    }});
  }}

  // Jump markers
  if (showMarkers) {{
    Object.entries(JUMP_DATA).forEach(([jt, events]) => {{
      if (!activeJumps.has(jt)) return;
      const filtered = events.filter(e => selectedGTs.has(e.gt));
      if (filtered.length === 0) return;
      traces.push({{
        type: 'scatter3d',
        mode: 'markers',
        x: filtered.map(e => e.x),
        y: filtered.map(e => e.y),
        z: filtered.map(e => e.t),
        marker: {{
          symbol: JUMP_SYMBOLS[jt],
          size: 6,
          color: 'yellow',
          line: {{ color: 'red', width: 1 }},
        }},
        name: jt,
        hovertemplate: filtered.map(e =>
          jt + '<br>GT ' + e.gt + ' frame=' + e.t +
          '<br>state=' + e.state +
          '<br>prev=' + e.prev +
          '<br>curr=' + e.curr + '<extra></extra>'
        ),
      }});
    }});
  }}

  const layout = {{
    scene: {{
      xaxis: {{ title: 'X (m)', range: [42, 60] }},
      yaxis: {{ title: 'Y (m)', range: [30, 60] }},
      zaxis: {{ title: 'Frame' }},
      camera: {{ eye: {{ x: 1.5, y: 1.5, z: 0.8 }} }},
      aspectmode: 'manual',
      aspectratio: {{ x: 1, y: 1, z: 2 }},
    }},
    paper_bgcolor: '#1a1a2e',
    plot_bgcolor: '#1a1a2e',
    font: {{ color: '#eee', size: 10 }},
    legend: {{ x: 1.02, y: 1, font: {{ size: 9 }} }},
    margin: {{ l: 0, r: 0, t: 0, b: 0 }},
  }};

  Plotly.react('plot', traces, layout);
}}

render();
</script>
</body>
</html>"""
    return html


def main():
    parser = argparse.ArgumentParser(description="CP-GT2ACTUALS-7 dashboard")
    parser.add_argument("--clip", default="vid2", choices=["vid1", "vid2"])
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    clip_data = build_clip_data(args.clip)
    mat = load_mat_blueprint()

    html = build_html(clip_data, mat)

    out_path = args.out or f"outputs/_eval/gt2actuals/dashboard_{args.clip}.html"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(html)
    print(f"Dashboard written to {out_path}")
    print(f"  Clip: {clip_data['label']}")
    print(f"  GT persons: {len(clip_data['gt_ids'])}, default: gt={clip_data['default_gt']}")
    print(f"  Subsample: {clip_data['subsample']}x (markers never dropped)")
    print(f"  Open in browser: file://{Path(out_path).resolve()}")


if __name__ == "__main__":
    main()
