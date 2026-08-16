"""
inference.py — Main execution logic: per-frame rendering (2D + BEV panels)
and the end-to-end video processing pipeline.
"""

import os
import subprocess
import sys
import time

import cv2
import numpy as np
import supervision as sv
from loguru import logger
from tqdm import tqdm

from config import (
    BANNER,
    DEFAULT_CONF_THRESHOLD,
    DEPTH_INFER_SIZE,
    MAX_WORK_WIDTH,
    OUTPUT_DIR,
    OUTPUT_H,
    OUTPUT_W,
    PANEL_H,
    PANEL_W,
    RISK_COLORS,
    VEHICLE_CLASSES,
)
from model_loader import DepthEstimator, ObjectTracker, load_yolo, select_device
from utils import (
    add_hud,
    compose,
    draw_bev_box,
    draw_grid,
    draw_lane,
    estimate_distance,
    get_risk,
    glow_box,
    project_pointcloud,
)


def draw_left_panel(frame, tracked, depth_map, tracker: ObjectTracker):
    out = frame.copy()
    draw_lane(out)
    h, w = out.shape[:2]

    if tracked is not None and len(tracked) > 0:
        for i in range(len(tracked.xyxy)):
            box = tracked.xyxy[i].astype(int)
            cls = int(tracked.class_id[i]) if tracked.class_id is not None else 2
            tid = int(tracked.tracker_id[i]) if tracked.tracker_id is not None else 0
            label = VEHICLE_CLASSES.get(cls, "vehicle")
            dist = estimate_distance(depth_map, box, w, h)
            risk = get_risk(dist, box, w)
            color = RISK_COLORS[risk]

            glow_box(out, box[0], box[1], box[2], box[3], color)

            txt = f"{label} {dist:.0f}m {risk}"
            fs = 0.48
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_DUPLEX, fs, 1)
            lx, ly = box[0], max(box[1] - 6, th + 4)
            cv2.rectangle(
                out, (lx - 2, ly - th - 4), (lx + tw + 4, ly + 2), (0, 0, 0), -1
            )
            cv2.putText(
                out, txt, (lx, ly), cv2.FONT_HERSHEY_DUPLEX, fs, color, 1, cv2.LINE_AA
            )

            traj = list(tracker.trajectories[tid])
            for k in range(1, len(traj)):
                cv2.line(out, traj[k - 1], traj[k], color, 1, cv2.LINE_AA)

            vx, vy = tracker.velocities.get(tid, (0, 0))
            cx_b, cy_b = (box[0] + box[2]) // 2, (box[1] + box[3]) // 2
            cv2.arrowedLine(
                out,
                (cx_b, cy_b),
                (cx_b + int(vx * 8), cy_b + int(vy * 8)),
                (255, 255, 0),
                1,
                cv2.LINE_AA,
                tipLength=0.4,
            )

    cv2.putText(
        out,
        "DASHCAM - ADAS PERCEPTION",
        (10, 22),
        cv2.FONT_HERSHEY_DUPLEX,
        0.55,
        (0, 255, 80),
        1,
        cv2.LINE_AA,
    )
    return out


def draw_right_panel(depth_norm, tracked, frame_w, frame_h):
    canvas = np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)
    # depth_norm is already at (frame_w, frame_h) — the same working
    # resolution FOCAL_LENGTH_PX is calibrated against. Feed it to
    # project_pointcloud as-is; resizing it to the canvas size first (as
    # the old code did) would break the pinhole math, which measures
    # pixel columns in the source frame's scale, not the canvas's.
    project_pointcloud(depth_norm, canvas)

    horizon = int(PANEL_H * 0.3)
    grad = np.linspace(200, 0, horizon, dtype=np.uint8)
    canvas[:horizon, :, 0] = np.minimum(
        canvas[:horizon, :, 0].astype(np.int16) + grad[:, None] // 4, 255
    ).astype(np.uint8)

    draw_grid(canvas, PANEL_W, PANEL_H)

    if tracked is not None and len(tracked) > 0 and tracked.tracker_id is not None:
        for i in range(len(tracked.xyxy)):
            box = tracked.xyxy[i]
            cls = int(tracked.class_id[i]) if tracked.class_id is not None else 2
            name = VEHICLE_CLASSES.get(cls, "vehicle")
            dist = estimate_distance(depth_norm, box, frame_w, frame_h)
            risk = get_risk(dist, box, frame_w)
            color = RISK_COLORS[risk]
            cx_px = (box[0] + box[2]) / 2
            draw_bev_box(
                canvas,
                cx_px,
                dist,
                f"{name} {dist:.0f}m {risk}",
                color,
                PANEL_W,
                PANEL_H,
                frame_w,
            )

    cv2.putText(
        canvas,
        "3D BEV - DEPTH PERCEPTION",
        (10, 22),
        cv2.FONT_HERSHEY_DUPLEX,
        0.55,
        (0, 220, 255),
        1,
        cv2.LINE_AA,
    )
    return canvas


def _ffmpeg_encoder() -> tuple[str, list[str]] | None:
    """Probe which H.264-capable encoder this ffmpeg build actually has —
    a hardcoded libx264 breaks on ffmpeg builds compiled without it
    (common on some distro packages for licensing reasons)."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True,
            check=True,
            text=True,
        )
    except Exception:
        return None

    encoders = result.stdout
    if "libx264" in encoders:
        logger.debug("ffmpeg encoder probe: libx264 available")
        return "libx264", ["-c:v", "libx264", "-preset", "fast", "-crf", "23"]
    if "libopenh264" in encoders:
        logger.debug("ffmpeg encoder probe: libx264 unavailable, using libopenh264")
        return "libopenh264", ["-c:v", "libopenh264", "-b:v", "8M"]
    if "h264_nvenc" in encoders:
        logger.debug("ffmpeg encoder probe: using h264_nvenc")
        return "h264_nvenc", ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "23"]
    if "mpeg4" in encoders:
        logger.debug(
            "ffmpeg encoder probe: no H.264 encoder found, falling back to mpeg4"
        )
        return "mpeg4", ["-c:v", "mpeg4", "-q:v", "4"]
    logger.debug("ffmpeg encoder probe: no usable encoder found")
    return None


def _encode_with_ffmpeg(tmp_out: str, final_out: str) -> bool:
    """Compress the raw output via ffmpeg. Returns True on success."""
    encoder = _ffmpeg_encoder()
    if encoder is None:
        return False

    encoder_name, encoder_args = encoder
    logger.info(f"Compressing with {encoder_name} -> {final_out} ...")
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        tmp_out,
        *encoder_args,
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        final_out,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        if os.path.exists(tmp_out):
            os.remove(tmp_out)
        return True
    except subprocess.CalledProcessError as e:
        # ffmpeg puts the real failure near the *end* of stderr, after the
        # version/config banner — tail-slice, not head-slice, or you just
        # see boilerplate instead of the actual error.
        stderr = e.stderr.decode(errors="ignore")
        logger.error(f"FFmpeg ({encoder_name}) failed:\n{stderr[-1500:]}")
        logger.warning(
            f"Raw frames kept at: {tmp_out} — rerun the command above manually to debug."
        )
        return False


def process_video(input_path: str, args, model_path: str = None) -> None:
    logger.opt(raw=True).info(
        BANNER + "\n"
    )  # raw: decorative ASCII banner, no timestamp/level noise

    device, dev_label = select_device(args.device)
    logger.info(f"Device: {dev_label}")

    output_dir = args.output_dir or OUTPUT_DIR
    if output_dir.startswith("[") and output_dir.endswith("]"):
        logger.error(f"OUTPUT_DIR is still a placeholder: {output_dir!r}")
        logger.error("Set --output-dir or the OUTPUT_DIR env var to a real path.")
        sys.exit(1)
    os.makedirs(output_dir, exist_ok=True)
    tmp_out = os.path.join(output_dir, "_tmp_raw.mp4")
    final_out = os.path.join(output_dir, "final_output.mp4")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        logger.error(f"Cannot open: {input_path}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info(
        f"Input:  {input_path}  {src_w}x{src_h}  {src_fps:.1f}fps  {total_frames} frames"
    )
    logger.info(f"Output: {final_out}")

    work_w_preview = min(src_w, MAX_WORK_WIDTH)
    logger.debug(
        f"Working resolution: {work_w_preview}px wide (source downscaled for inference speed)"
    )

    yolo = load_yolo(model_path or args.model, device=device)
    depth_est = DepthEstimator(device)
    depth_est.load()
    tracker = ObjectTracker()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_out, fourcc, src_fps, (OUTPUT_W + 4, OUTPUT_H))

    pbar = tqdm(total=total_frames, desc="Processing", unit="fr")
    t0 = time.time()
    frame_no = 0
    fps_avg = 0.0

    while True:
        ret, bgr = cap.read()
        if not ret:
            break
        frame_no += 1

        work_w = min(src_w, MAX_WORK_WIDTH)
        work_h = int(src_h * (work_w / src_w))
        work = cv2.resize(bgr, (work_w, work_h))

        results = yolo(
            work,
            conf=args.conf,
            classes=list(VEHICLE_CLASSES.keys()),
            verbose=False,
            device=device,
        )
        sv_det = sv.Detections.from_ultralytics(results[0])
        tracked = tracker.update(sv_det)

        depth_input = cv2.resize(work, DEPTH_INFER_SIZE)
        depth_norm = depth_est.estimate(depth_input)
        depth_norm = cv2.resize(depth_norm, (work_w, work_h))

        left_panel = draw_left_panel(work, tracked, depth_norm, tracker)
        right_panel = draw_right_panel(depth_norm, tracked, work_w, work_h)
        composed = compose(left_panel, right_panel)

        elapsed = time.time() - t0
        fps_now = frame_no / max(elapsed, 1e-6)
        fps_avg = 0.9 * fps_avg + 0.1 * fps_now if fps_avg > 0 else fps_now

        add_hud(composed, fps_avg, frame_no, total_frames, dev_label)
        writer.write(composed)

        if args.show:
            preview = cv2.resize(composed, (1280, int(PANEL_H * 1280 / OUTPUT_W)))
            cv2.imshow("3D Dashcam Perception", preview)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                logger.info("User quit.")
                break

        pbar.update(1)

    pbar.close()
    cap.release()
    writer.release()
    if args.show:
        cv2.destroyAllWindows()

    total_elapsed = time.time() - t0
    logger.info(
        f"Done: {frame_no} frames in {total_elapsed:.1f}s "
        f"({frame_no / max(total_elapsed, 1e-6):.1f} fps avg)"
    )

    if _encode_with_ffmpeg(tmp_out, final_out):
        size_mb = os.path.getsize(final_out) / (1024**2)
        logger.success(f"OUTPUT SAVED: {final_out}  ({size_mb:.1f} MB)")
        return

    import shutil

    shutil.move(tmp_out, final_out)
    size_mb = os.path.getsize(final_out) / (1024**2)
    logger.success(f"OUTPUT SAVED (uncompressed): {final_out}  ({size_mb:.1f} MB)")
    logger.info(
        "Install an FFmpeg build with libx264/libopenh264 for smaller files: https://ffmpeg.org/download.html"
    )
