"""Calibration pipeline — gym setup and maintenance tools.

Sits alongside bjj_pipeline as a separate top-level module. Outputs
(K + distortion coefficients, refined homographies) feed into bjj_pipeline
via shared config files (configs/cameras/{cam_id}/homography.json) and
Supabase.

These are one-time and periodic gym initialization workflows, NOT
per-session pipeline stages.

Modules
-------
lens_calibration   — Estimate K + distortion from mat edge clicks (CP16b)
mat_walk           — Grid pattern detection from tagged person walk (CP18)
drift_detection    — Daily baseline comparison for camera drift (CP18)
inter_camera_sync  — Cross-camera affine alignment via mat walk (CP18)
"""
