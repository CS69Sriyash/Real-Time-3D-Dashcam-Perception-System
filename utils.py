"""
utils.py — Stateless helper functions shared across the pipeline.

Nothing here loads a model or touches disk (besides frame arrays already
in memory) — keeps this module easy to unit test in isolation.
"""

import cv2
import numpy as np

from config import (
    BEV_MAX_DEPTH_M,
    FOCAL_LENGTH_PX,
    PANEL_H,
    PANEL_W,
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
def estimate_distance(depth_map: np.ndarray, box, fw: int, fh: int) -> float:
    """Fuse monocular-depth and projective-geometry distance estimates."""
    x1, y1, x2, y2 = map(int, box)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(fw - 1, x2), min(fh - 1, y2)
    if x2 <= x1 or y2 <= y1:
        return 50.0
    patch = depth_map[y1:y2, x1:x2]
    med = float(np.median(patch))
    # depth_norm convention: HIGHER value = CLOSER (Depth Anything V2's
    # relative-depth output, disparity-style). Invert to get distance.
    d_depth = 1.0 + (1.0 - med) * 79.0
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
def draw_grid(canvas: np.ndarray, pw: int, ph: int) -> None:
    for frac in np.linspace(0.3, 1.0, 10):
        y = int(frac * ph)
        xl = int(pw * 0.5 - pw * 0.4 * frac)
        xr = int(pw * 0.5 + pw * 0.4 * frac)
        alpha = int(80 + 120 * frac)
        cv2.line(canvas, (xl, y), (xr, y), (alpha, 0, alpha // 2), 1, cv2.LINE_AA)
    vp_x, vp_y = pw // 2, int(ph * 0.28)
    for k in np.linspace(-0.42, 0.42, 16):
        bx = int(pw * 0.5 + k * pw)
        alpha = int(60 + 80 * (1 - abs(k) / 0.45))
        cv2.line(canvas, (vp_x, vp_y), (bx, ph), (0, alpha, alpha // 2), 1, cv2.LINE_AA)


def project_pointcloud(
    depth_norm: np.ndarray, canvas: np.ndarray, step: int = 3
) -> None:
    h, w = depth_norm.shape
    ch, cw = canvas.shape[:2]
    ys = np.arange(h // 2, h, step)
    xs = np.arange(0, w, step)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    d = depth_norm[yy, xx].astype(np.float32)
    u = xx.astype(np.float32) / w
    # d: HIGHER = CLOSER. Near points -> bottom of canvas (large bev_y);
    # far points -> near the vanishing point at the top (small bev_y).
    bev_y = (ch * 0.28 + d * ch * 0.70).astype(np.int32)
    spread = 0.38 + 0.52 * d
    bev_x = (cw * 0.5 + (u - 0.5) * cw * spread).astype(np.int32)
    mask = (bev_x >= 0) & (bev_x < cw) & (bev_y >= 0) & (bev_y < ch)
    canvas[bev_y[mask].ravel(), bev_x[mask].ravel()] = depth_to_color(d[mask].ravel())


def draw_bev_box(canvas, cx_norm, dist_m, label, color, pw, ph) -> None:
    d_norm = float(np.clip(1.0 - dist_m / BEV_MAX_DEPTH_M, 0, 1))
    bev_y = int(ph * 0.28 + (1.0 - d_norm) * ph * 0.70)
    spread = 0.38 + 0.52 * d_norm
    bev_x = int(pw * 0.5 + (cx_norm - 0.5) * pw * spread)
    scale = 0.5 + 1.5 * (1.0 - d_norm)
    bw, bh, bd = int(55 * scale), int(35 * scale), int(20 * scale)
    fl, fr = bev_x - bw // 2, bev_x + bw // 2
    ft, fb = bev_y - bh // 2, bev_y + bh // 2
    bl, br = fl + bd, fr + bd
    bt, bb = ft - bd // 2, fb - bd // 2

    def gl(p1, p2):
        for t, a in [(3, 0.2), (2, 0.5), (1, 1.0)]:
            cv2.line(canvas, p1, p2, tuple(int(v * a) for v in color), t, cv2.LINE_AA)

    gl((fl, ft), (fr, ft))
    gl((fr, ft), (fr, fb))
    gl((fr, fb), (fl, fb))
    gl((fl, fb), (fl, ft))
    gl((bl, bt), (br, bt))
    gl((br, bt), (br, bb))
    gl((br, bb), (bl, bb))
    gl((bl, bb), (bl, bt))
    for p1, p2 in [
        ((fl, ft), (bl, bt)),
        ((fr, ft), (br, bt)),
        ((fr, fb), (br, bb)),
        ((fl, fb), (bl, bb)),
    ]:
        gl(p1, p2)

    fs = 0.38 * (0.6 + 0.8 * scale)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, fs, 1)
    tx, ty = bev_x - tw // 2, ft - 6
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
