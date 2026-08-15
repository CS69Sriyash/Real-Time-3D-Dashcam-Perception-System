"""
utils.py — Stateless helper functions shared across the pipeline.

Nothing here loads a model or touches disk (besides frame arrays already
in memory) — keeps this module easy to unit test in isolation.
"""

import cv2
import numpy as np

from config import (
    BEV_LATERAL_RANGE_M,
    BEV_MAX_DEPTH_M,
    FOCAL_LENGTH_PX,
    PANEL_H,
    PANEL_W,
    REAL_CAR_LENGTH_M,
    REAL_CAR_WIDTH_M,
    RISK_CAUTION_DIST_M,
    RISK_DANGER_DIST_M,
    RISK_DANGER_LATERAL_FRAC,
    RISK_WARNING_DIST_M,
)


# ── Colormap ─────────────────────────────────────────────────────────────
def build_turbo_lut() -> np.ndarray:
    """Precompute a 256-entry BGR turbo-style lookup table."""
    lut = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        t = i / 255.0
        if t < 0.25:
            r, g, b = 0, int(t * 4 * 255), 255
        elif t < 0.5:
            r, g, b = 0, 255, int((1 - (t - 0.25) * 4) * 255)
        elif t < 0.75:
            r, g, b = int((t - 0.5) * 4 * 255), 255, 0
        else:
            r, g, b = 255, int((1 - (t - 0.75) * 4) * 255), 0
        lut[i] = [b, g, r]  # BGR
    return lut


_TURBO_LUT = build_turbo_lut()


def depth_to_color(depth_norm: np.ndarray) -> np.ndarray:
    idx = (np.clip(depth_norm, 0, 1) * 255).astype(np.uint8)
    return _TURBO_LUT[idx]


# ── Distance / risk model ───────────────────────────────────────────────
def depth_norm_to_meters(d) -> "float | np.ndarray":
    """Convert normalized depth (0-1, HIGHER = CLOSER — Depth Anything V2's
    relative-depth convention) into an approximate distance in meters.

    Shared by estimate_distance() and the BEV point-cloud/box projection
    so both stay consistent with each other. Accepts a scalar or an
    ndarray (vectorized).
    """
    return 1.0 + (1.0 - d) * 79.0


def estimate_distance(depth_map: np.ndarray, box, fw: int, fh: int) -> float:
    """Fuse monocular-depth and projective-geometry distance estimates."""
    x1, y1, x2, y2 = map(int, box)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(fw - 1, x2), min(fh - 1, y2)
    if x2 <= x1 or y2 <= y1:
        return 50.0
    patch = depth_map[y1:y2, x1:x2]
    med = float(np.median(patch))
    d_depth = depth_norm_to_meters(med)
    px_w = max(x2 - x1, 1)
    d_proj = (FOCAL_LENGTH_PX * REAL_CAR_WIDTH_M) / px_w
    return round(float(np.clip(0.5 * d_depth + 0.5 * d_proj, 1.0, 100.0)), 1)


def get_risk(dist: float, box, fw: int) -> str:
    cx = (box[0] + box[2]) / 2
    lat_off = abs(cx - fw / 2) / (fw / 2)
    if dist < RISK_DANGER_DIST_M and lat_off < RISK_DANGER_LATERAL_FRAC:
        return "DANGER"
    if dist < RISK_WARNING_DIST_M:
        return "WARNING"
    if dist < RISK_CAUTION_DIST_M:
        return "CAUTION"
    return "SAFE"


# ── Drawing primitives (2D overlay) ─────────────────────────────────────
def draw_lane(frame: np.ndarray) -> None:
    h, w = frame.shape[:2]
    pts = np.array(
        [
            [int(w * 0.35), int(h * 0.55)],
            [int(w * 0.65), int(h * 0.55)],
            [int(w * 0.85), h - 1],
            [int(w * 0.15), h - 1],
        ],
        dtype=np.int32,
    )
    ov = frame.copy()
    cv2.fillPoly(ov, [pts], (0, 80, 0))
    cv2.addWeighted(ov, 0.18, frame, 0.82, 0, frame)
    cv2.polylines(frame, [pts], True, (0, 255, 60), 1, cv2.LINE_AA)


def glow_box(frame, x1, y1, x2, y2, color, thickness=2) -> None:
    for t, a in [(thickness + 4, 0.15), (thickness + 2, 0.30), (thickness, 1.0)]:
        c = tuple(int(v * a) for v in color)
        cv2.rectangle(frame, (x1, y1), (x2, y2), c, t, cv2.LINE_AA)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)


# ── Bird's-eye-view (BEV) primitives ────────────────────────────────────
def unproject_x(px, depth_m, frame_w: int):
    """Pinhole-camera back-projection: pixel column + estimated depth ->
    real-world lateral offset (meters) from the camera's optical axis.
    X = (px - cx) * Z / f, the standard inverse of the pinhole projection
    equation. Accepts a scalar or ndarray for px/depth_m."""
    return (px - frame_w / 2.0) * depth_m / FOCAL_LENGTH_PX


def world_to_bev_px(x_m, z_m, pw: int, ph: int):
    """Orthographic (constant-scale) top-down projection. A true bird's-
    eye view has NO vanishing point — unlike a perspective camera image,
    scale here doesn't change with distance. z=0 (the ego vehicle) maps
    to the bottom of the canvas; z=BEV_MAX_DEPTH_M maps to the top."""
    z_frac = np.clip(z_m / BEV_MAX_DEPTH_M, 0.0, 1.0)
    y = ph * (1.0 - z_frac)
    x = pw * 0.5 + (x_m / BEV_LATERAL_RANGE_M) * (pw * 0.5)
    return x, y


def draw_grid(canvas: np.ndarray, pw: int, ph: int) -> None:
    """Range rings every 10m + lane-width reference lines + an ego-vehicle
    marker at the origin. Pure orthographic grid — no perspective fan."""
    for z in range(10, int(BEV_MAX_DEPTH_M) + 1, 10):
        _, y = world_to_bev_px(0.0, float(z), pw, ph)
        y = int(y)
        cv2.line(canvas, (0, y), (pw, y), (60, 40, 20), 1, cv2.LINE_AA)
        cv2.putText(
            canvas,
            f"{z}m",
            (6, max(y - 4, 12)),
            cv2.FONT_HERSHEY_DUPLEX,
            0.35,
            (140, 110, 60),
            1,
            cv2.LINE_AA,
        )

    for x_m in (-3.5, 3.5, -7.0, 7.0):
        x, _ = world_to_bev_px(x_m, 0.0, pw, ph)
        x = int(x)
        if 0 <= x < pw:
            cv2.line(canvas, (x, 0), (x, ph), (30, 60, 30), 1, cv2.LINE_AA)

    ex, ey = world_to_bev_px(0.0, 0.0, pw, ph)
    ex, ey = int(ex), int(ey)
    pts = np.array([[ex, ey - 12], [ex - 8, ey + 6], [ex + 8, ey + 6]], dtype=np.int32)
    cv2.fillPoly(canvas, [pts], (0, 255, 120))


def project_pointcloud(
    depth_norm: np.ndarray, canvas: np.ndarray, step: int = 3
) -> None:
    """Back-project each depth pixel to real-world (X, Z) via the pinhole
    model, then plot it on the canvas at true orthographic scale.

    depth_norm must be at the ORIGINAL working-frame resolution (the same
    resolution FOCAL_LENGTH_PX was calibrated against) — not pre-resized
    to the canvas size, or the pinhole math is measuring pixels in the
    wrong scale.
    """
    h, w = depth_norm.shape
    ch, cw = canvas.shape[:2]
    ys = np.arange(h // 2, h, step)
    xs = np.arange(0, w, step)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    d = depth_norm[yy, xx].astype(np.float32)

    z_m = depth_norm_to_meters(d)
    x_m = unproject_x(xx.astype(np.float32), z_m, w)
    bev_x, bev_y = world_to_bev_px(x_m, z_m, cw, ch)
    bev_x = bev_x.astype(np.int32)
    bev_y = bev_y.astype(np.int32)

    mask = (bev_x >= 0) & (bev_x < cw) & (bev_y >= 0) & (bev_y < ch)
    canvas[bev_y[mask].ravel(), bev_x[mask].ravel()] = depth_to_color(d[mask].ravel())


def draw_bev_box(
    canvas,
    cx_px: float,
    dist_m: float,
    label: str,
    color,
    pw: int,
    ph: int,
    frame_w: int,
) -> None:
    """Draw a vehicle marker at its true unprojected (X, Z) position, sized
    to its real-world footprint (REAL_CAR_WIDTH_M x REAL_CAR_LENGTH_M) at
    constant orthographic scale — same size on screen regardless of
    distance, matching how a real top-down map would render it."""
    x_m = unproject_x(cx_px, dist_m, frame_w)
    bev_x, bev_y = world_to_bev_px(x_m, dist_m, pw, ph)
    bev_x, bev_y = int(bev_x), int(bev_y)

    px_per_m_x = (pw * 0.5) / BEV_LATERAL_RANGE_M
    px_per_m_z = ph / BEV_MAX_DEPTH_M
    box_w = max(int(REAL_CAR_WIDTH_M * px_per_m_x), 4)
    box_h = max(int(REAL_CAR_LENGTH_M * px_per_m_z), 4)

    x1, y1 = bev_x - box_w // 2, bev_y - box_h // 2
    x2, y2 = bev_x + box_w // 2, bev_y + box_h // 2

    overlay = canvas.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, 0.35, canvas, 0.65, 0, canvas)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)

    fs = 0.38
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, fs, 1)
    tx, ty = bev_x - tw // 2, y1 - 6
    if 0 < ty < ph:
        cv2.putText(
            canvas, label, (tx, ty), cv2.FONT_HERSHEY_DUPLEX, fs, color, 1, cv2.LINE_AA
        )


# ── Frame composition / HUD ─────────────────────────────────────────────
def compose(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    l = cv2.resize(left, (PANEL_W, PANEL_H))
    r = cv2.resize(right, (PANEL_W, PANEL_H))
    div = np.full((PANEL_H, 4, 3), (0, 255, 80), dtype=np.uint8)
    return np.hstack([l, div, r])


def add_hud(
    frame: np.ndarray, fps: float, frame_no: int, total: int, device_label: str
) -> None:
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, h - 28), (w, h), (5, 5, 5), -1)
    pct = frame_no / max(total, 1) * 100
    txt = (
        f"  FPS: {fps:5.1f}  |  Frame: {frame_no}/{total} ({pct:.1f}%)  "
        f"|  Device: {device_label}"
    )
    cv2.putText(
        frame,
        txt,
        (10, h - 8),
        cv2.FONT_HERSHEY_DUPLEX,
        0.42,
        (0, 220, 80),
        1,
        cv2.LINE_AA,
    )
