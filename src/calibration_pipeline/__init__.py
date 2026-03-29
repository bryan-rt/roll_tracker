"""Calibration pipeline — gym setup and maintenance tools.

Sits alongside bjj_pipeline as a separate top-level module. Outputs
(K + distortion coefficients, refined homographies, affine corrections)
feed into bjj_pipeline via shared config files and Supabase.

These are one-time and periodic gym initialization workflows, NOT
per-session pipeline stages.

Modules
-------
lens_calibration    — Estimate K + distortion from mat edge clicks (CP16b)
blueprint_geometry  — Mat blueprint parsing + geometric queries (CP18)
tracklet_classifier — Tracklet classification + feature extraction (CP18)
mat_walk            — Layer 1: single-camera homography refinement via RANSAC (CP18)
inter_camera_sync   — Layer 2: cross-camera affine alignment (CP18)
calibrate           — Full calibration orchestrator (CP18)
drift_detection     — Daily baseline comparison for camera drift (future)
"""
