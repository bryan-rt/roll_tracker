"""CVAT zip extraction and YOLO label parsing.

Extracts GT annotation zips to temp directories, parses 6-field YOLO
track-detection format (class cx cy w h track_id), remaps class 1->0.

CORRECTNESS CONTRACT (non-negotiable for TB-EVAL-1):
    The GT loader must ONLY load labels for frames defined by the model
    manifest's annotated_range x split. Specifically:
    - Iterate frames in annotated_range (start, stop, stride) intersected
      with the requested split (train or val).
    - Load GT only from those exact frame indices.
    - NEVER load GT from a frame outside annotated_range, even if a
      non-empty label file exists for it in the zip. CVAT auto-interpolates
      annotations on non-hand-labeled frames; these are NOT trusted GT.
    - annotated_range (from the manifest) is authoritative. Zip contents
      are advisory only.

# TODO TB-EVAL-1: implement GT zip extraction + label parser
"""
