from __future__ import annotations

from pathlib import Path
from typing import Dict

import json
import pytest

from bjj_pipeline.config import (
    load_config,
    config_hash,
)
from bjj_pipeline.config.loader import deep_merge
from bjj_pipeline.config.models import PipelineConfig
from bjj_pipeline.config.paths import (
    get_cache_dir,
    cache_path,
)


def write_yaml(path: Path, data: Dict) -> None:
    import yaml
    text = yaml.safe_dump(data, sort_keys=True)
    path.write_text(text, encoding="utf-8")


def test_deep_merge_precedence(tmp_path: Path):
    # Prepare files
    default = tmp_path / "default.yaml"
    camera = tmp_path / "cam.yaml"
    overlay = tmp_path / "overlay.yaml"

    write_yaml(default, {
        "paths": {"cache_dir_name": "_cache"},
        "compute": {"device": "cpu", "num_workers": 0},
        "camera": {"camera_id": "cam01", "fps": 24},
        "stages": {"stage_A": {"frame_stride": 1}},
    })
    write_yaml(camera, {
        "camera": {"camera_id": "cam01", "fps": 30},
        "stages": {"stage_A": {"frame_stride": 2}},
    })
    write_yaml(overlay, {
        "compute": {"num_workers": 4},
        "stages": {"stage_A": {"frame_stride": 3}},
    })

    typed, resolved, h, sources = load_config(default, camera, overlay)
    assert sources == [str(default), str(camera), str(overlay)]
    assert resolved["camera"]["fps"] == 30  # camera overrides default
    assert resolved["compute"]["num_workers"] == 4  # overlay overrides camera/default
    assert resolved["stages"]["stage_A"]["frame_stride"] == 3  # overlay last


def test_forbid_unknown_keys(tmp_path: Path):
    # Unknown top-level key should be rejected by PipelineConfig
    cfg = {
        "paths": {"cache_dir_name": "_cache"},
        "compute": {"device": "cpu"},
        "camera": {"camera_id": "cam01"},
        "unknown": {"foo": 1},
    }
    with pytest.raises(Exception):
        PipelineConfig.model_validate(cfg)


def test_stage_f_config_accepts_export_settings() -> None:
    cfg = {
        "paths": {"cache_dir_name": "_cache"},
        "compute": {"device": "cpu"},
        "camera": {"camera_id": "cam01"},
        "stage_F": {
            "enabled": True,
            "padding_px": 80,
            "low_quantile": 0.05,
            "high_quantile": 0.95,
            "min_crop_width": 160,
            "min_crop_height": 160,
            "consolidate_sessions": False,
            "consolidate_max_gap_frames": 120,
            "consolidate_buffer_sec": 5.0,
            "consolidate_require_nonconflicting_tags": True,
        },
    }

    typed = PipelineConfig.model_validate(cfg)

    assert typed.stage_F is not None
    assert typed.stage_F.padding_px == 80
    assert typed.stage_F.low_quantile == pytest.approx(0.05)
    assert typed.stage_F.high_quantile == pytest.approx(0.95)
    assert typed.stage_F.min_crop_width == 160
    assert typed.stage_F.min_crop_height == 160
    assert typed.stage_F.consolidate_sessions is False
    assert typed.stage_F.consolidate_max_gap_frames == 120
    assert typed.stage_F.consolidate_buffer_sec == pytest.approx(5.0)
    assert typed.stage_F.consolidate_require_nonconflicting_tags is True


def test_config_hash_stable():
    d = {"a": 1, "b": {"x": 2, "y": [3, 4]}}
    h1 = config_hash(d)
    h2 = config_hash(json.loads(json.dumps(d)))
    assert h1 == h2


def test_cache_paths_contained(tmp_path: Path):
    outputs_root = tmp_path / "outputs"
    clip_id = "cam01-foo"
    base = get_cache_dir(outputs_root, clip_id)
    p = cache_path(outputs_root, clip_id, ["tool", "sub", "file.bin"])
    # Ensure path is under base
    assert str(p).startswith(str(base))
    # Traversal attempt should raise
    with pytest.raises(ValueError):
        cache_path(outputs_root, clip_id, ["..", "escape"])  # disallowed
