# CP4 — Flow Topology Investigation

**Date:** 2026-05-14
**Branch:** `services_uploader`
**Scope:** Read-only analysis of post-CP3b artifacts.

---

## Verdict: Structural — Parallel Carrier Displacement

The binding constraint is not cost, not capacity, not penalty, and not the solver.
It is a **D1 graph construction pattern**: when two tracklets are simultaneously
present during a merge event, D1 creates GROUP nodes for both carriers. The solver
picks one carrier's GROUP path and the other carrier's GROUP nodes receive zero flow.
This orphans the losing carrier's entire SINGLE↔GROUP chain, making it unexplainable
regardless of penalty magnitude.

This explains every observation across CP2-CP3b:
- J_EDEw t1 (4,427 frames) is displaced by t2 (768 frames) for shared merge events
- PPDmUg t1 (2,926 frames) is displaced by t2 (same merge events, different GROUP nodes)
- FP7oJQ t62 (3,702 frames) is displaced by t75/t70/t93 at parallel merge events
- Penalty has no effect because t1's penalty is paid once for the whole tracklet,
  while its GROUP nodes are structurally unreachable once the parallel carrier wins

**Recommended next step: CP5 — Fix parallel carrier displacement in D1.** Options:
(a) Merge parallel GROUP nodes so both carriers share a single capacity-2 node,
(b) Add cross-carrier CONTINUE edges so the solver can switch carriers mid-chain,
or (c) Extend the explain-or-penalize mechanism to consider GROUP nodes (not just
SINGLE_TRACKLET nodes).

---

## Question 1 — Capacity Audit

### BIRTH/DEATH capacity (per camera)

| Camera | BIRTH total | Allowed | Allowed cap | Used flow | Slack |
|--------|------------|---------|-------------|-----------|-------|
| FP7oJQ | 251 | 251 | 276 | 17 | **259** |
| J_EDEw | 236 | 236 | 254 | 17 | **237** |
| PPDmUg | 73 | 73 | 86 | 8 | **78** |

| Camera | DEATH total | Allowed | Allowed cap | Used flow | Slack |
|--------|------------|---------|-------------|-----------|-------|
| FP7oJQ | 251 | 251 | 293 | 17 | **276** |
| J_EDEw | 236 | 236 | 278 | 17 | **261** |
| PPDmUg | 73 | 73 | 83 | 8 | **75** |

**Capacity is NOT the bottleneck.** All BIRTH/DEATH edges are allowed. FP7oJQ could
create up to 276 paths; it uses 17. Slack is 237-276 across cameras. The solver
voluntarily uses few paths because the cost of additional paths (BIRTH+DEATH per path)
exceeds the penalty savings from explaining more tracklets.

---

## Question 2 — Marginal Cost of Named Dropped Tracklets

### J_EDEw t1 (4,427 frames, full-clip carrier)

| Metric | Value |
|--------|-------|
| Nodes | 15 (8 SINGLE, 7 GROUP) |
| Solo route (B + internal C + D) | **4.21** (2.01 + 0.19 + 2.01) |
| CP3b penalty | **76.6** (766 SINGLE-node frames × 0.1) |
| Coupled tracklets | 8 (t25, t40, t52, t74, t106, t118, t177, t308) |
| Coupled status | **All 8 explained** (0 dropped) |
| Disallowed edges | **0** |
| Selected edges touching t1 nodes | **0** (zero flow on all 15 nodes) |

**The puzzle:** Solo route costs 4.21, penalty is 76.6. Keeping t1 would save 76.6 - 4.21
= 72.4. All coupled tracklets are already explained. No edges are disallowed. Capacity
is available. The solver reports OPTIMAL. Why is t1 dropped?

**Root cause: parallel carrier displacement.** D1 creates GROUP nodes for both carriers
when two tracklets overlap during a merge event:

| Frame range | t1 GROUP node | t2 GROUP node | Disappearing | New |
|-------------|--------------|--------------|-------------|-----|
| [146, 291] | `G:...:carrier=t1:d=t25:n=t40` | `G:...:carrier=t2:d=t25:n=t40` | t25 | t40 |
| [298, 458] | `G:...:carrier=t1:d=t40:n=t52` | `G:...:carrier=t2:d=t40:n=t52` | t40 | t52 |
| [466, 653] | `G:...:carrier=t1:d=t52:n=t74` | `G:...:carrier=t2:d=t52:n=t74` | t52 | t74 |
| [659, 923] | `G:...:carrier=t1:d=t74:n=t106` | `G:...:carrier=t2:d=t74:n=None` | t74 | — |

t2 (768 frames, 0-851) is the shorter tracklet but wins the GROUP node competition.
Entity 11 routes t40→t52→t74→t106→t118→t177 through t2's GROUP nodes. t1's GROUP nodes
get zero flow. Since t1's CONTINUE edges require passing through its GROUP nodes
(SINGLE→GROUP→SINGLE→GROUP→...), and those GROUP nodes are orphaned, t1's entire chain
becomes structurally disconnected.

The solver IS optimal: once it commits to t2's GROUP nodes for the coupled tracklets,
routing through t1's GROUP nodes would require routing the coupled tracklets TWICE
(once through t2's GROUP nodes for their current paths, and again through t1's GROUP
nodes just for t1). This doubles the MERGE/SPLIT edge costs for the coupled tracklets,
which exceeds t1's penalty savings.

### PPDmUg t1 (2,926 frames, full-clip carrier)

| Metric | Value |
|--------|-------|
| Nodes | 19 (10 SINGLE, 9 GROUP) |
| Solo route | **6.01** (2.01 + 1.99 + 2.01) |
| CP3b penalty | **113.7** (1,137 SINGLE-node frames × 0.1) |
| Coupled tracklets | 11 (all explained, 0 dropped) |
| Disallowed edges | **0** |
| Parallel carrier | t2 (9 GROUP nodes, selected by solver) |

Same pattern as J_EDEw. t2 wins the GROUP competition for all 9 merge events. t1's
GROUP chain becomes orphaned.

### J_EDEw t201 (537 frames, tag-bearing — tag:1)

| Metric | Value |
|--------|-------|
| Nodes | 8 (3 SINGLE, 5 GROUP) |
| Solo route | **20.82** (10.01 + 0.80 + 10.01) |
| CP3b penalty | **53.7** (537 frames × 0.1) |
| Coupled tracklets | 5 (3 explained, 2 dropped: t162, t226) |
| Disallowed edges | **2** (shadowed_by_group_chain) |
| Sum penalties (t201 + coupled dropped) | **137.0** |

t201 has a different profile: it's a mid-length interior tracklet (both BIRTH and DEATH
are interior at 10.01 each). Its solo route cost (20.82) is close to its penalty (53.7),
so the savings from keeping it are modest. Additionally, 2 of its coupled tracklets
are also dropped, and 2 edges are disallowed by `shadowed_by_group_chain`. This
tracklet's case is cost-bound, not carrier-displacement-bound.

### FP7oJQ t62 (3,702 frames)

| Metric | Value |
|--------|-------|
| Nodes | 6 (3 SINGLE, 3 GROUP) |
| Solo route | **12.59** (10.01 + 0.57 + 2.01) |
| CP3b penalty | Not computed in this run (below breakeven?) |
| Parallel carriers | t75, t70, t93 at various merge events |

Same parallel-carrier pattern: t62's GROUP nodes have parallel alternatives and the
solver picks the other carriers.

---

## Question 3 — Constraint Barriers

### Disallowed edges per named tracklet

| Tracklet | Disallowed edges | Reasons |
|----------|-----------------|---------|
| J_EDEw t1 | **0** | — |
| PPDmUg t1 | **0** | — |
| J_EDEw t201 | **2** | `shadowed_by_group_chain` |
| FP7oJQ t62 | **0** (on its own nodes; parallel carriers may have disallowed edges) |

The full-clip carriers (t1 on both cameras) have **zero disallowed edges**. Every
routing option is available. The solver's failure to use them is purely a global
optimization choice driven by the parallel-carrier structure.

t201 has 2 edges disallowed by `shadowed_by_group_chain`, which removes CONTINUE
reconnect edges that are shadowed by existing MERGE/SPLIT explanations. This could
be contributing to t201's drop, but the primary factor is its high interior routing
cost (20.82).

---

## Question 4 — Classification and Recommendation

### Per-camera classification

| Camera | Primary binding constraint | Secondary |
|--------|--------------------------|-----------|
| FP7oJQ | **Structural: parallel carrier displacement** | Cost (interior BIRTH+DEATH for non-carrier tracklets) |
| J_EDEw | **Structural: parallel carrier displacement** | Cost (t201: interior BIRTH+DEATH + shadowed edges) |
| PPDmUg | **Structural: parallel carrier displacement** | Cost (same pattern) |

All three cameras are dominated by the same structural issue. The solver is genuinely
OPTIMAL — it's just that the D1 graph creates an either-or structure that forces the
solver to abandon one carrier per merge event.

### Why penalty can never fix this

When the solver commits to carrier=t2 for a merge event, routing through carrier=t1's
GROUP node for the same event would require the coupled tracklet (e.g., t25) to flow
through BOTH GROUP nodes: once on t2's path (where it's already committed) and once
on t1's path. This requires:
- An additional MERGE edge from t25 into t1's GROUP node
- t1's GROUP node to carry capacity for t25's flow on t1's path AND t25's existing
  flow on t2's path

The ILP's flow conservation makes this expensive: t25 can only enter each GROUP node
once per unit of flow. To route t25 through both t1's and t2's GROUP nodes, t25 would
need flow=2 — doubling its routing cost. At that point, the savings from t1's penalty
(76.6) are consumed by the doubled routing cost of ALL 8 coupled tracklets.

### Recommended next step: CP5 — Fix parallel carrier displacement

The cleanest fix is at D1 graph construction. Three options (pick one):

**(a) Merge parallel GROUP nodes (recommended).** When D1 detects that two carriers (t1, t2)
produce GROUP nodes for the same merge event (same disappearing/new tracklets, overlapping
frame ranges), merge them into a single GROUP node with both carriers listed. This
eliminates the either-or choice. The solver can then route both carriers through the
same GROUP node at capacity=2.

**(b) Add cross-carrier CONTINUE edges.** Add a CONTINUE edge from `G:...:carrier=t2:d=t25:n=t40`
to `T:t1:s2:292-297` (and vice versa). This lets the solver route through either carrier's
GROUP nodes for t1's chain continuation. More edges = more flexibility, but also more
solver complexity.

**(c) Extend explain-or-penalize to GROUP nodes.** Currently, only SINGLE_TRACKLET nodes
count for the explained/unexplained determination. If GROUP nodes where `carrier=t1` also
counted, t1's penalty would be higher and would consider more incident edges. This doesn't
fix the structural issue but makes the penalty more reflective of t1's actual graph presence.

Option (a) is recommended because it addresses the root cause at the graph construction
level, is the most principled fix, and doesn't increase solver complexity.

---

## Data Sources

```
outputs/_eval_gt/{cam}/.../stage_D/d1_graph_nodes.parquet
outputs/_eval_gt/{cam}/.../stage_D/d1_graph_edges.parquet
outputs/_eval_gt/{cam}/.../stage_D/d2_edge_costs.parquet
outputs/_eval_gt/{cam}/.../_debug/d3_selected_edges.parquet
outputs/_eval_gt/{cam}/.../_debug/d3_entities_format_a.json
outputs/_eval_gt/{cam}/.../_debug/d3_solution_ledger.json
outputs/_eval_gt/{cam}/.../_debug/d3_ilp2_explain_or_penalize.json
```
