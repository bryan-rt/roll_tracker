import pandas as pd
from pathlib import Path

root = Path('outputs/cam03-20260103-124000_0-30s')
edges = pd.read_parquet(root / 'stage_D' / 'd1_graph_edges.parquet')
costs = pd.read_parquet(root / '_debug' / 'd3_compiled_costs_pruned.parquet')
sel = pd.read_parquet(root / '_debug' / 'd3_selected_edges.parquet')

merged = costs.merge(edges[['edge_id','edge_type','u','v','payload_json']], on='edge_id', how='left')
merged['is_selected'] = merged['edge_id'].isin(sel['edge_id'])

terms = ['t1','t2','t3','t4','t5','t6','t7','t8','t9','t10','t11','t13','t14','t15','t16']

out_path = root / '_debug' / 'd2_costs_manual_focus.txt'
out_path.parent.mkdir(parents=True, exist_ok=True)
with out_path.open('w') as f:
    for t in terms:
        m1 = merged['edge_id'].astype(str).str.contains(t)
        m2 = merged['u'].astype(str).str.contains(t)
        m3 = merged['v'].astype(str).str.contains(t)
        m4 = merged['payload_json'].astype(str).str.contains(t)
        mask = m1 | m2 | m3 | m4
        sub = merged[mask].copy()
        if sub.empty:
            continue
        f.write(f"==== COSTS FOR EDGES MENTIONING {t} (n={len(sub)}) ====\n")
        cols = ['edge_id','edge_type','u','v','is_allowed','total_cost','term_env','term_time','term_vreq','term_group_coherence','term_birth_prior','term_death_prior','term_merge_prior','term_split_prior','is_selected']
        cols = [c for c in cols if c in sub.columns]
        f.write(sub[cols].sort_values('total_cost').to_string(index=False))
        f.write("\n\n")

print('wrote', out_path)
