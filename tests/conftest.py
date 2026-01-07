"""Test configuration: ensure repository src/ is importable.
Deterministic and minimal sys.path setup for pytest runs from repo root.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Compute repository root: tests/ is one level under repo root
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"

# Insert src path at the front for predictable import resolution
src_str = str(SRC_PATH)
if src_str not in sys.path:
    sys.path.insert(0, src_str)
