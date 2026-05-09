"""GT filename to source-video frame index resolver.

Re-exports frame_index_from_filename from gt_loader for API clarity.
"""
from pipeline_validation.common.gt_loader import frame_index_from_filename

__all__ = ["frame_index_from_filename"]
