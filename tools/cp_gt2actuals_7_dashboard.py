#!/usr/bin/env python3
"""CP-GT2ACTUALS-7-REV: Interactive 3D dashboard for error-map review.

READ-ONLY viewer of the validated dense GT-to-actuals artifact.
Single GT worm per person, colored per-frame by switchable channel:
  - person_id (Stage D identity, default) — DOMINANT person_id at frame
  - tracklet_id (Stage A audit) — resolved tracklet through split lineage
  - HSV (appearance) — dominant JOINT (H,V) bin from hist_* columns
  - velocity (kinematic) — speed_mps_k of pipeline-assigned detection

Usage:
    PYTHONPATH=src python tools/cp_gt2actuals_7_dashboard.py [--clip vid1|vid2] [--out path.html]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
OUTPUTS = REPO / "outputs"

CLIPS = {
    "vid2": {
        "gt2a": OUTPUTS / "_eval/gt2actuals/J_EDEw/J_EDEw-20260318-200246",
        "label": "J_EDEw vid2 (0-4490, authoritative)",
    },
    "vid1": {
        "gt2a": OUTPUTS / "_eval/gt2actuals/J_EDEw/J_EDEw-20260318-200015",
        "label": "J_EDEw vid1 (0-3000, corroboration)",
    },
}

JUMP_MARKERS = {
    "tracklet_drift": {"symbol": "star", "label": "Stage A: tracklet_drift"},
    "ilp_misstitch": {"symbol": "square", "label": "Solver: ilp_misstitch"},
    "false_split": {"symbol": "triangle-up", "label": "D0.5: false_split"},
    "group_boundary_jump": {"symbol": "diamond", "label": "Group: boundary_jump"},
    "group_membership_drift": {"symbol": "x", "label": "Group: membership_drift"},
}

# HSV histogram layout: 18 H bins x 8 S bins x 6 V bins = 864
HIST_H = 18
HIST_S = 8
HIST_V = 6


def _json(obj):
    return json.dumps(obj, default=lambda x: int(x) if hasattr(x, 'item') else str(x))


def _dominant_pid(person_ids_json: str | None) -> str | None:
    """Return DOMINANT person_id (most frequent across the frame's assignments).

    For a single frame the JSON is typically a short list from GROUP nodes.
    We pick the first element after sorting — which is the dominant by the
    artifact's convention (person_ids are sorted). For multi-element lists
    (GROUP), we take the first (lowest lexicographic = earliest assigned).
    """
    if not person_ids_json or person_ids_json == "[]":
        return None
    pids = json.loads(person_ids_json)
    return pids[0] if pids else None


def _hsv_color_from_hist(hist_vals: np.ndarray) -> str | None:
    """Derive CSS hsl() from dominant JOINT (H,V) bin.

    Reshapes the 864-dim flat histogram to (18,8,6), marginalizes over S to
    get a (18,6) H×V joint distribution, finds the argmax, then maps:
      H bin -> CSS hue: bin * 20 (OpenCV 0-180 -> CSS 0-360)
      V bin -> CSS lightness: bin * 14 + 15 (maps 6 bins to 15-99%)
    Saturation fixed at 70%.
    """
    if len(hist_vals) != HIST_H * HIST_S * HIST_V:
        return None
    if np.isnan(hist_vals).all():
        return None
    h3d = hist_vals.reshape(HIST_H, HIST_S, HIST_V)
    hv_joint = h3d.sum(axis=1)  # (18, 6) — marginalize over S
    flat_idx = int(np.nanargmax(hv_joint))
    dom_h = flat_idx // HIST_V
    dom_v = flat_idx % HIST_V
    css_hue = dom_h * 20
    css_light = dom_v * 14 + 15
    return f"hsl({css_hue}, 70%, {css_light}%)"


def load_mat_blueprint() -> list[dict]:
    path = REPO / "configs" / "mat_blueprint.json"
    if not path.exists():
        return []
    return json.load(open(path))


def build_clip_data(clip_key: str) -> dict:
    cfg = CLIPS[clip_key]
    df = pd.read_parquet(cfg["gt2a"] / "gt2actuals_dense.parquet")

    gt_ids = sorted(df.gt_track_id.unique())
    hist_cols = [c for c in df.columns if c.startswith("hist_")]

    # Highest-jump GT person
    jumps = df[df.jump_type.notna()]
    jc = jumps.groupby("gt_track_id").size().sort_values(ascending=False)
    default_gt = int(jc.index[0]) if len(jc) > 0 else int(gt_ids[0])

    n_frames = df.frame_index.nunique()
    subsample = max(1, n_frames // 1500)

    # Stable person_id palette
    all_pids = set()
    for pids_json in df.person_ids.dropna():
        all_pids.update(json.loads(pids_json))
    pid_list = sorted(all_pids)
    pid_color_map = {}
    for i, pid in enumerate(pid_list):
        hue = int(i * 360 / max(len(pid_list), 1))
        pid_color_map[pid] = f"hsl({hue}, 80%, 55%)"

    # Stable tracklet_id palette
    all_tids = set(df.resolved_tracklet_id.dropna().unique())
    tid_list = sorted(all_tids)
    tid_color_map = {}
    for i, tid in enumerate(tid_list):
        hue = int(i * 360 / max(len(tid_list), 1))
        tid_color_map[tid] = f"hsl({hue}, 65%, 50%)"

    # Build per-GT worm data
    gt_worms = {}
    for gt_id in gt_ids:
        g = df[df.gt_track_id == gt_id].sort_values("frame_index")
        mask = g.x_m_eff.notna() & g.y_m_eff.notna()
        g_valid = g[mask]
        if len(g_valid) == 0:
            continue

        # Subsample but keep jump frames
        is_jump = g_valid.jump_type.notna()
        keep = np.zeros(len(g_valid), dtype=bool)
        keep[::subsample] = True
        keep |= is_jump.values
        keep[-1] = True
        g_sub = g_valid[keep]

        # Per-frame coloring data
        pid_colors = []
        tid_colors = []
        hsv_colors = []
        speeds = []

        for _, r in g_sub.iterrows():
            # person_id color (dominant)
            dpid = _dominant_pid(r.person_ids)
            pid_colors.append(pid_color_map.get(dpid, "#666") if dpid else "#666")

            # tracklet_id color
            rtid = r.resolved_tracklet_id if pd.notna(r.resolved_tracklet_id) else None
            tid_colors.append(tid_color_map.get(rtid, "#666") if rtid else "#666")

            # HSV color
            if hist_cols and pd.notna(r.get("is_isolated")) and r.is_isolated:
                hvals = np.array([r[c] if pd.notna(r[c]) else np.nan for c in hist_cols])
                hsv_colors.append(_hsv_color_from_hist(hvals) or "__null__")
            else:
                hsv_colors.append("__null__")

            # Velocity (pipeline detection speed)
            speeds.append(float(r.speed_mps_k) if pd.notna(r.speed_mps_k) else None)

        gt_worms[int(gt_id)] = {
            "x": g_sub.x_m_eff.tolist(),
            "y": g_sub.y_m_eff.tolist(),
            "t": [int(f) for f in g_sub.frame_index],
            "pid_colors": pid_colors,
            "tid_colors": tid_colors,
            "hsv_colors": hsv_colors,
            "speeds": speeds,
            "states": g_sub.state.tolist(),
            "pids": [_dominant_pid(r.person_ids) or "" for _, r in g_sub.iterrows()],
            "tids": [str(r.resolved_tracklet_id) if pd.notna(r.resolved_tracklet_id) else "" for _, r in g_sub.iterrows()],
        }

    # Jump markers
    jump_data = {jt: [] for jt in JUMP_MARKERS}
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
        "gt_ids": [int(g) for g in gt_ids],
        "default_gt": default_gt,
        "subsample": subsample,
        "gt_worms": gt_worms,
        "jump_data": jump_data,
        "n_frames": n_frames,
        "n_rows": len(df),
        "pid_color_map": pid_color_map,
        "tid_color_map": tid_color_map,
    }


def build_html(clip_data: dict, mat_blueprint: list[dict]) -> str:
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
<title>GT2ACTUALS-7: {clip_data['label']}</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 0; padding: 10px; background: #1a1a2e; color: #eee; }}
h1 {{ font-size: 15px; margin: 5px 0; }}
.controls {{ display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 8px; font-size: 12px; }}
.cg {{ background: #16213e; padding: 8px 12px; border-radius: 6px; }}
.cg label {{ display: block; font-weight: bold; margin-bottom: 4px; color: #a8d8ea; font-size: 11px; }}
select {{ font-size: 11px; padding: 3px 6px; }}
select[multiple] {{ height: 110px; }}
#plot {{ width: 100%; height: calc(100vh - 170px); }}
.btn {{ cursor: pointer; padding: 4px 8px; border: 1px solid #555; background: #2a2a4a; color: #eee; border-radius: 3px; margin: 2px; font-size: 11px; }}
.btn.active {{ background: #4a4a8a; border-color: #88f; }}
.note {{ font-size: 10px; color: #777; margin-top: 4px; }}
</style>
</head>
<body>
<h1>GT2ACTUALS-7: {clip_data['label']} | {clip_data['n_rows']} rows | subsample {clip_data['subsample']}x (markers never dropped)</h1>
<div class="controls">
  <div class="cg">
    <label>GT Person</label>
    <select id="gt-select" multiple>{gt_options}</select>
  </div>
  <div class="cg">
    <label>Color By</label>
    <button class="btn active" id="btn-c-pid" onclick="setColor('pid')">Person ID</button>
    <button class="btn" id="btn-c-tid" onclick="setColor('tid')">Tracklet ID</button>
    <button class="btn" id="btn-c-hsv" onclick="setColor('hsv')">HSV Appearance</button>
    <button class="btn" id="btn-c-vel" onclick="setColor('vel')">Velocity</button>
    <div class="note" id="color-note">Segments colored by assigned person_id. Gray = no_id.</div>
  </div>
  <div class="cg">
    <label>Jump Markers</label>
    <button class="btn active" id="btn-markers" onclick="toggleMarkers()">Show/Hide All</button>
    <br>
    {jump_buttons}
    <div class="note">Star at tracklet-color bridge between worms = Stage A drift confirmed.</div>
  </div>
</div>
<div id="plot"></div>

<script>
const W = {_json(clip_data["gt_worms"])};
const JUMP = {_json(clip_data["jump_data"])};
const GT_IDS = {_json(clip_data["gt_ids"])};
const MAT = {_json(mat_blueprint)};
const JUMP_SYM = {{tracklet_drift:"star",ilp_misstitch:"square",false_split:"triangle-up",group_boundary_jump:"diamond",group_membership_drift:"x"}};
const COLOR_NOTES = {{
  pid: "Segments colored by assigned person_id. Gray = no_id (unassigned).",
  tid: "Segments colored by tracklet_id. Same color on two GT worms = tracklet drift.",
  hsv: "Segments colored by dominant HSV appearance. Dashed white = no histogram (entangled/miss).",
  vel: "Segments colored by speed (m/s). Blue=still, red=fast. Gap = NaN.",
}};

let colorMode = "pid";
let showMarkers = true;
let activeJumps = new Set(Object.keys(JUMP_SYM));
let selectedGTs = new Set([{int(clip_data["default_gt"])}]);

function setColor(m) {{
  colorMode = m;
  ["pid","tid","hsv","vel"].forEach(c => document.getElementById("btn-c-"+c).classList.toggle("active", c===m));
  document.getElementById("color-note").textContent = COLOR_NOTES[m];
  render();
}}
function toggleMarkers() {{
  showMarkers = !showMarkers;
  document.getElementById("btn-markers").classList.toggle("active");
  render();
}}
function toggleJump(jt) {{
  if (activeJumps.has(jt)) activeJumps.delete(jt); else activeJumps.add(jt);
  document.getElementById("btn-"+jt).classList.toggle("active");
  render();
}}
document.getElementById("gt-select").addEventListener("change", function() {{
  selectedGTs = new Set(Array.from(this.selectedOptions).map(o=>parseInt(o.value)));
  render();
}});

function render() {{
  const traces = [];

  // Mat blueprint
  MAT.forEach(m => {{
    traces.push({{type:"mesh3d",x:[m.x,m.x+m.width,m.x+m.width,m.x],y:[m.y,m.y,m.y+m.height,m.y+m.height],z:[0,0,0,0],i:[0,0],j:[1,2],k:[2,3],color:"rgba(100,150,100,0.15)",hoverinfo:"skip",showlegend:false}});
  }});

  // GT worms — one per selected GT, colored by segments
  selectedGTs.forEach(gtId => {{
    const w = W[gtId];
    if (!w) return;
    const n = w.x.length;
    if (n < 2) return;

    if (colorMode === "vel") {{
      // Velocity: continuous colorscale via scatter3d line.color array
      const colors = w.speeds.map(v => v != null ? v : NaN);
      traces.push({{
        type:"scatter3d", mode:"lines",
        x: w.x, y: w.y, z: w.t,
        line: {{color:colors, width:3, colorscale:"Portland", cmin:0, cmax:5, colorbar:{{title:"m/s",len:0.5,x:1.05}}}},
        name: "GT "+gtId,
        hovertemplate: w.t.map((t,i) => "GT "+gtId+" f="+t+"<br>speed="+(w.speeds[i]!=null?w.speeds[i].toFixed(2):"NaN")+"<br>pid="+w.pids[i]+"<extra></extra>"),
      }});
    }} else {{
      // Segment-grouped coloring: split worm into runs of same color
      let colorArr;
      if (colorMode === "pid") colorArr = w.pid_colors;
      else if (colorMode === "tid") colorArr = w.tid_colors;
      else colorArr = w.hsv_colors;

      let seg_start = 0;
      while (seg_start < n - 1) {{
        const c = colorArr[seg_start];
        let seg_end = seg_start + 1;
        while (seg_end < n && colorArr[seg_end] === c) seg_end++;
        // Draw segment [seg_start, min(seg_end, n-1)] inclusive
        const end = Math.min(seg_end, n - 1);
        const sx = w.x.slice(seg_start, end + 1);
        const sy = w.y.slice(seg_start, end + 1);
        const st = w.t.slice(seg_start, end + 1);

        const isNull = (c === "__null__");
        const isGray = (c === "#666");

        let lineStyle;
        if (isNull) {{
          lineStyle = {{color:"rgba(255,255,255,0.3)", width:1.5, dash:"dash"}};
        }} else if (isGray) {{
          lineStyle = {{color:"#666", width:2}};
        }} else {{
          lineStyle = {{color:c, width:3}};
        }}

        const hoverLabel = colorMode === "pid" ? w.pids : (colorMode === "tid" ? w.tids : w.states);
        traces.push({{
          type:"scatter3d", mode:"lines",
          x: sx, y: sy, z: st,
          line: lineStyle,
          name: "GT "+gtId+(isNull?" (no data)":(isGray?" (no_id)":"")),
          legendgroup: "gt"+gtId,
          showlegend: seg_start === 0,
          hovertemplate: st.map((t,i) => "GT "+gtId+" f="+t+"<br>"+colorMode+"="+hoverLabel[seg_start+i]+"<extra></extra>"),
        }});
        seg_start = seg_end > seg_start + 1 ? seg_end - 1 : seg_end;
      }}
    }}
  }});

  // Jump markers
  if (showMarkers) {{
    Object.entries(JUMP).forEach(([jt, events]) => {{
      if (!activeJumps.has(jt)) return;
      const filtered = events.filter(e => selectedGTs.has(e.gt));
      if (!filtered.length) return;
      traces.push({{
        type:"scatter3d", mode:"markers",
        x: filtered.map(e=>e.x), y: filtered.map(e=>e.y), z: filtered.map(e=>e.t),
        marker: {{symbol:JUMP_SYM[jt], size:6, color:"yellow", line:{{color:"red",width:1}}}},
        name: jt,
        hovertemplate: filtered.map(e => jt+"<br>GT "+e.gt+" f="+e.t+"<br>"+e.state+"<br>prev="+e.prev+"<br>curr="+e.curr+"<extra></extra>"),
      }});
    }});
  }}

  Plotly.react("plot", traces, {{
    scene: {{
      xaxis:{{title:"X (m)",range:[42,60]}},
      yaxis:{{title:"Y (m)",range:[30,60]}},
      zaxis:{{title:"Frame"}},
      camera:{{eye:{{x:1.5,y:1.5,z:0.8}}}},
      aspectmode:"manual", aspectratio:{{x:1,y:1,z:2}},
    }},
    paper_bgcolor:"#1a1a2e", plot_bgcolor:"#1a1a2e",
    font:{{color:"#eee",size:10}},
    legend:{{x:1.02,y:1,font:{{size:9}}}},
    margin:{{l:0,r:0,t:0,b:0}},
  }});
}}
render();
</script>
</body>
</html>"""
    return html


def main():
    parser = argparse.ArgumentParser(description="CP-GT2ACTUALS-7-REV dashboard")
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
    print(f"  Colorings: person_id (default), tracklet_id, HSV, velocity")
    print(f"  Open: file://{Path(out_path).resolve()}")


if __name__ == "__main__":
    main()
