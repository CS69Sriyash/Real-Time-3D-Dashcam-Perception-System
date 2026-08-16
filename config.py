"""
config.py — Central configuration for the Real-Time 3D Dashcam Perception System.

All tunable parameters live here so nothing is hardcoded in the logic
modules. Paths are read from environment variables with safe placeholder
fallbacks — set them for your machine before running.
"""

import os

# ── Paths (override via env vars — never hardcode real paths here) ─────
YOLO_MODEL_PATH = os.environ.get("YOLO_MODEL_PATH", "[PATH_TO_MODEL]/yolov8n.pt")

# OUTPUT_DIR defaults to a real, relative path — not a placeholder — so a
# forgotten env var/flag can't silently create a directory literally named
# "[PATH_TO_OUTPUT_DIR]". Override via --output-dir or the env var when
# you want output written elsewhere.
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "output")

# Loguru sink level — INFO by default, DEBUG for verbose diagnostics
# (device resolution, working resolution, encoder probing). Overridden by
# --verbose on main.py/smoke_test.py without needing the env var.
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")

# Depth Anything V2 is pulled from the Hugging Face Hub, not a local path.
DEPTH_MODEL_NAME = "depth-anything/Depth-Anything-V2-Small-hf"

# ── Detection ────────────────────────────────────────────────────────
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck", 1: "bicycle"}
DEFAULT_CONF_THRESHOLD = 0.35

# ── Risk-level visualization colors (BGR) ───────────────────────────────
RISK_COLORS = {
    "SAFE": (0, 255, 80),
    "CAUTION": (0, 220, 255),
    "WARNING": (0, 140, 255),
    "DANGER": (0, 0, 255),
}

# ── Output canvas layout ────────────────────────────────────────────────
OUTPUT_W = 1920
OUTPUT_H = 540
PANEL_W = OUTPUT_W // 2
PANEL_H = OUTPUT_H

# ── Distance estimation / collision-risk model constants ───────────────
FOCAL_LENGTH_PX = 700.0
REAL_CAR_WIDTH_M = 2.0
REAL_CAR_LENGTH_M = 4.5
BEV_MAX_DEPTH_M = 60.0

# How far left/right of the ego vehicle the BEV panel shows (meters).
# Orthographic (true top-down) scale — not a perspective fan.
BEV_LATERAL_RANGE_M = 20.0

# Risk thresholds (meters), used by utils.get_risk
RISK_DANGER_DIST_M = 8.0
RISK_DANGER_LATERAL_FRAC = 0.4
RISK_WARNING_DIST_M = 15.0
RISK_CAUTION_DIST_M = 25.0

# ── Inference-time working resolution (speed vs. accuracy trade-off) ───
MAX_WORK_WIDTH = 640
DEPTH_INFER_SIZE = (320, 192)  # (width, height)

# ── Depth temporal smoothing ────────────────────────────────────────────
DEPTH_SMOOTHING_ALPHA = 0.6

BANNER = """
+======================================================================+
|          Real-Time 3D Dashcam Perception System                     |
|          YOLO . DepthAnything V2 . BEV . Collision Risk              |
+======================================================================+
"""
