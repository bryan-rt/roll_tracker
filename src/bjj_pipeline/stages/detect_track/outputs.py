"""Stage A (detect_track) — output helpers.

Slice 2 goal: stop using a hard NotImplemented stub so the orchestrator can
produce schema-correct, validator-passing artifacts even before the real
detector/tracker is wired in.

The actual compute logic will evolve, but the output schema and canonical
paths must remain stable.
"""

from __future__ import annotations

from typing import Dict

import pandas as pd

from bjj_pipeline.contracts import f0_parquet as pq


_FAMILY_TO_DTYPE: Dict[str, str] = {
	"string": "object",     # accept object/StringDtype
	"int": "Int64",         # nullable integer
	"float": "float64",
	"bool": "boolean",
}


def empty_df_for_schema_key(key: str) -> pd.DataFrame:
	"""Create an empty pandas DataFrame that passes f0_parquet schema checks."""
	specs = pq.PARQUET_SCHEMAS[key]
	data = {spec.name: pd.Series([], dtype=_FAMILY_TO_DTYPE[spec.family]) for spec in specs}
	return pd.DataFrame(data)
