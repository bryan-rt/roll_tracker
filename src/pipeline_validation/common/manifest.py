"""Training-data manifest read/write.

Loads per-model YAML manifests from configs/models/{model_id}.yaml.
Provides helpers to enumerate in-distribution vs held-out (val) frames
for a given camera and annotated range.

# TODO TB-EVAL-1: implement manifest loader + frame enumeration
"""
