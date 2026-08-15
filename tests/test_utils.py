"""
Tests for utils.py's pure functions: distance/risk math, pinhole
unprojection, and BEV placement. No model loading, no video I/O — these
run in well under a second and don't need a GPU.

Run with: pytest tests/ -v
"""

import numpy as np
import pytest

from config import (
    BEV_LATERAL_RANGE_M,
    BEV_MAX_DEPTH_M,
    RISK_CAUTION_DIST_M,
    RISK_DANGER_DIST_M,
    RISK_WARNING_DIST_M,
)
from utils import (
    build_turbo_lut,
    depth_norm_to_meters,
    depth_to_color,
    draw_bev_box,
    draw_grid,
    estimate_distance,
    get_risk,
    project_pointcloud,
    unproject_x,
    world_to_bev_px,
)


# ── depth_norm_to_meters ─────────────────────────────────────────────────
def test_depth_near_is_smaller_distance_than_far():
    # Regression test for the near/far inversion bug: depth_norm convention
    # is HIGHER = CLOSER, so a high value must map to a SMALL distance.
    near_m = depth_norm_to_meters(1.0)
    far_m = depth_norm_to_meters(0.0)
    assert near_m < far_m


def test_depth_norm_to_meters_is_monotonic_decreasing():
    values = np.linspace(0.0, 1.0, 11)
    meters = [depth_norm_to_meters(v) for v in values]
    assert all(meters[i] > meters[i + 1] for i in range(len(meters) - 1))


def test_depth_norm_to_meters_vectorized_matches_scalar():
    arr = np.array([0.0, 0.5, 1.0], dtype=np.float32)
    vec_result = depth_norm_to_meters(arr)
    scalar_result = [depth_norm_to_meters(float(v)) for v in arr]
    np.testing.assert_allclose(vec_result, scalar_result, rtol=1e-5)


# ── estimate_distance ────────────────────────────────────────────────────
def test_estimate_distance_near_patch_closer_than_far_patch():
    h, w = 200, 200
    box = [50, 50, 100, 100]
    near_patch = np.full((h, w), 0.95, dtype=np.float32)
    far_patch = np.full((h, w), 0.05, dtype=np.float32)
    d_near = estimate_distance(near_patch, box, w, h)
    d_far = estimate_distance(far_patch, box, w, h)
    assert d_near < d_far


def test_estimate_distance_stays_within_clip_bounds():
    h, w = 100, 100
    box = [0, 0, 100, 100]
    depth = np.zeros((h, w), dtype=np.float32)
    dist = estimate_distance(depth, box, w, h)
    assert 1.0 <= dist <= 100.0


def test_estimate_distance_degenerate_box_returns_fallback():
    depth = np.zeros((50, 50), dtype=np.float32)
    dist = estimate_distance(depth, [10, 10, 10, 10], 50, 50)
    assert dist == 50.0


# ── get_risk ─────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "dist,box,expected",
    [
        # centered, well inside danger distance -> DANGER
        (RISK_DANGER_DIST_M - 1, [290, 0, 310, 50], "DANGER"),
        # same distance, but far off to the side -> not DANGER
        (RISK_DANGER_DIST_M - 1, [0, 0, 20, 50], "WARNING"),
        (RISK_WARNING_DIST_M - 1, [290, 0, 310, 50], "WARNING"),
        (RISK_CAUTION_DIST_M - 1, [290, 0, 310, 50], "CAUTION"),
        (RISK_CAUTION_DIST_M + 5, [290, 0, 310, 50], "SAFE"),
    ],
)
def test_get_risk_thresholds(dist, box, expected):
    assert get_risk(dist, box, fw=600) == expected


# ── unproject_x (pinhole model) ─────────────────────────────────────────
def test_unproject_x_center_pixel_is_zero_offset():
    frame_w = 640
    center_px = frame_w / 2.0
    x_m = unproject_x(center_px, depth_m=20.0, frame_w=frame_w)
    assert x_m == pytest.approx(0.0)


def test_unproject_x_scales_with_depth():
    # Same pixel offset from center, farther away -> larger real-world
    # lateral offset (a fixed angular offset covers more ground at range).
    frame_w = 640
    px = frame_w * 0.75  # right of center
    near_x = unproject_x(px, depth_m=5.0, frame_w=frame_w)
    far_x = unproject_x(px, depth_m=50.0, frame_w=frame_w)
    assert far_x > near_x > 0


# ── world_to_bev_px (orthographic top-down mapping) ─────────────────────
def test_world_to_bev_px_origin_is_bottom_center():
    pw, ph = 960, 540
    x, y = world_to_bev_px(0.0, 0.0, pw, ph)
    assert x == pytest.approx(pw / 2)
    assert y == pytest.approx(ph)


def test_world_to_bev_px_max_depth_is_top():
    pw, ph = 960, 540
    x, y = world_to_bev_px(0.0, BEV_MAX_DEPTH_M, pw, ph)
    assert y == pytest.approx(0.0)


def test_world_to_bev_px_no_perspective_scaling():
    # True orthographic BEV: the same lateral offset maps to the same
    # canvas X regardless of depth (no vanishing-point convergence).
    pw, ph = 960, 540
    x_near, _ = world_to_bev_px(5.0, 5.0, pw, ph)
    x_far, _ = world_to_bev_px(5.0, 50.0, pw, ph)
    assert x_near == pytest.approx(x_far)


def test_world_to_bev_px_lateral_range_maps_to_edges():
    pw, ph = 960, 540
    x_right, _ = world_to_bev_px(BEV_LATERAL_RANGE_M, 0.0, pw, ph)
    assert x_right == pytest.approx(pw)


# ── Rendering smoke tests (no crash, output actually changes the canvas) ─
def test_project_pointcloud_draws_pixels():
    h, w = 180, 320
    depth_norm = np.random.default_rng(0).random((h, w)).astype(np.float32)
    canvas = np.zeros((540, 960, 3), dtype=np.uint8)
    project_pointcloud(depth_norm, canvas)
    assert (canvas.sum(axis=2) > 0).sum() > 0


def test_draw_grid_does_not_crash():
    canvas = np.zeros((540, 960, 3), dtype=np.uint8)
    draw_grid(canvas, 960, 540)  # just must not raise


def test_draw_bev_box_renders_near_expected_position():
    pw, ph = 960, 540
    frame_w = 640
    canvas = np.zeros((ph, pw, 3), dtype=np.uint8)
    draw_bev_box(
        canvas,
        cx_px=frame_w / 2,
        dist_m=10.0,
        label="car 10m",
        color=(0, 255, 80),
        pw=pw,
        ph=ph,
        frame_w=frame_w,
    )
    ex, ey = world_to_bev_px(0.0, 10.0, pw, ph)
    region = canvas[
        max(0, int(ey) - 15) : int(ey) + 15, max(0, int(ex) - 15) : int(ex) + 15
    ]
    assert (region.sum(axis=2) > 0).sum() > 0


# ── Colormap ──────────────────────────────────────────────────────────
def test_turbo_lut_shape_and_dtype():
    lut = build_turbo_lut()
    assert lut.shape == (256, 3)
    assert lut.dtype == np.uint8


def test_depth_to_color_near_and_far_are_different_colors():
    near = depth_to_color(np.array([1.0]))
    far = depth_to_color(np.array([0.0]))
    assert not np.array_equal(near, far)
