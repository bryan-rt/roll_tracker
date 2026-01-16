"""Stage B runner: Masks + contact points (Slice 2 placeholder).

Writes schema-correct empty artifacts so orchestration can run end-to-end:
- stage_B/contact_points_refined.parquet (empty table with canonical schema)
- stage_B/masks/*.npz (at least one minimal file so existence checks pass)
- stage_B/audit.jsonl (minimal line)
"""

from __future__ import annotations

import json
from typing import Any, Dict
from pathlib import Path

import numpy as np
import pandas as pd

from bjj_pipeline.contracts.f0_paths import ClipOutputLayout
from bjj_pipeline.contracts import f0_parquet as pq


_FAMILY_TO_DTYPE = {
	"string": "object",  # accept object/StringDtype
	"int": "Int64",  # nullable integer
	"float": "Float64",
	"bool": "boolean",
}


def _empty_df_for_schema_key(key: str) -> pd.DataFrame:
	"""Create an empty pandas DataFrame that passes f0_parquet schema checks."""
	specs = pq.PARQUET_SCHEMAS[key]
	data = {spec.name: pd.Series([], dtype=_FAMILY_TO_DTYPE[spec.family]) for spec in specs}
	return pd.DataFrame(data)


def run(config: Dict[str, Any], inputs: Dict[str, Any]) -> Dict[str, Any]:
	"""Write empty but schema-correct outputs for Stage B."""
	layout: ClipOutputLayout = inputs["layout"]
	layout.ensure_dirs_for_stage("B")
	layout.ensure_mask_dirs()

	# Empty contact_points parquet (no rows). Validators accept empty tables *with schema*.
	_empty_df_for_schema_key("contact_points").to_parquet(layout.contact_points_parquet(), index=False)

	# Minimal mask npz to satisfy existence checks (glob stage_B/masks/*.npz)
	np.savez_compressed(layout.mask_npz_path(frame_index=0, detection_id="placeholder"), mask=np.zeros((1,1), dtype=np.uint8))

	# Minimal audit entry
	Path(layout.audit_jsonl("B")).write_text(json.dumps({"event":"stage_B_placeholder"})+"\n", encoding="utf-8")

	return {
		"contact_points_parquet": layout.rel_to_clip_root(layout.contact_points_parquet()),
		"masks_dir": layout.rel_to_clip_root(layout.masks_dir()),
		"audit_jsonl": layout.rel_to_clip_root(layout.audit_jsonl("B")),
	}


def main() -> None:
	raise SystemExit("Run via roll-tracker CLI; this is a placeholder stage.")
