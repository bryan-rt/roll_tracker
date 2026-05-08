"""Stage A detection recall/precision evaluation.

Compares Stage A detections.parquet against CVAT GT annotations using
Hungarian IoU matching. Reports per-camera and aggregate metrics, split
by in-distribution (train) and held-out (val) frames.

# TODO TB-EVAL-1: implement detection evaluation
"""
