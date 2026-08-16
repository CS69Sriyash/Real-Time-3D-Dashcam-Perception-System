#!/usr/bin/env python3
"""
benchmark.py — Measures real end-to-end per-frame throughput (detection +
depth estimation + tracking + rendering — the full pipeline cost, not just
raw model inference) across a matrix of working resolutions and devices,
and writes the results to BENCHMARKS.md.

This reuses the exact same functions inference.py's process_video() calls
per frame (draw_left_panel, draw_right_panel, compose, ObjectTracker,
DepthEstimator) rather than a separate lightweight loop, so the numbers
reported here match what you'd actually see running main.py — not an
optimistic model-only number that ignores rendering/tracking overhead.

Usage:
    python benchmark.py --model yolov8n.pt
    python benchmark.py --model yolov8n.pt --video sample.mp4 --devices cuda,cpu
    python benchmark.py --model yolov8n.pt --resolutions 320,640,960 --frames 60

With no --video, synthetic random-noise frames are used instead. That
still exercises the real compute path (model forward passes, tracker
updates, rendering) so the FPS numbers are meaningful, but detections on
noise are garbage by definition — use a real clip if you want numbers
you're citing anywhere (a README, a resume, an interview) to reflect
realistic content, not just raw throughput.
"""

import argparse
import platform
import statistics
import time
from datetime import datetime, timezone

import cv2
import numpy as np
import supervision as sv
import torch
from loguru import logger

from config import DEFAULT_CONF_THRESHOLD, DEPTH_INFER_SIZE, VEHICLE_CLASSES
from inference import draw_left_panel, draw_right_panel
from logging_config import configure_logging
from model_loader import DepthEstimator, ObjectTracker, load_yolo, select_device
from utils import compose


def _frame_source(video_path: str | None, width: int, height: int, count: int):
    """Yield `count` BGR frames at (width, height). Reads and loops a real
    video if given, otherwise generates synthetic random-noise frames."""
    if video_path:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")
        yielded = 0
        while yielded < count:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop back to start
                continue
            yield cv2.resize(frame, (width, height))
            yielded += 1
        cap.release()
    else:
        rng = np.random.default_rng(0)
        for _ in range(count):
            yield rng.integers(0, 255, (height, width, 3), dtype=np.uint8)


def _run_one_config(
    model_path: str,
    device: str,
    width: int,
    video_path,
    num_frames: int,
    warmup_frames: int,
) -> dict:
    """Run the real per-frame pipeline at one (device, resolution) config
    and return timing stats. Model/depth-estimator are loaded fresh per
    config — simpler and more isolated than trying to reuse across
    devices, at the cost of some redundant load time (not counted in the
    timed FPS window)."""
    height = int(width * 9 / 16)  # assume 16:9 for synthetic frames / resize target

    yolo = load_yolo(model_path, device=device)
    depth_est = DepthEstimator(device)
    depth_est.load()
    tracker = ObjectTracker()

    times = []
    total = num_frames + warmup_frames
    frames = _frame_source(video_path, width, height, total)

    for i, frame in enumerate(frames):
        t0 = time.perf_counter()

        results = yolo(
            frame,
            conf=DEFAULT_CONF_THRESHOLD,
            classes=list(VEHICLE_CLASSES.keys()),
            verbose=False,
            device=device,
        )
        sv_det = sv.Detections.from_ultralytics(results[0])
        tracked = tracker.update(sv_det)

        depth_input = cv2.resize(frame, DEPTH_INFER_SIZE)
        depth_norm = depth_est.estimate(depth_input)
        depth_norm = cv2.resize(depth_norm, (width, height))

        left = draw_left_panel(frame, tracked, depth_norm, tracker)
        right = draw_right_panel(depth_norm, tracked, width, height)
        compose(left, right)

        elapsed = time.perf_counter() - t0
        if (
            i >= warmup_frames
        ):  # skip warmup: first calls pay one-time CUDA/model warmup cost
            times.append(elapsed)
        logger.debug(
            f"  frame {i + 1}/{total} ({'warmup' if i < warmup_frames else 'timed'}): {elapsed * 1000:.1f}ms"
        )

    avg_fps = len(times) / sum(times) if times else 0.0
    return {
        "device": device,
        "width": width,
        "height": height,
        "avg_fps": avg_fps,
        "avg_ms": (sum(times) / len(times) * 1000) if times else 0.0,
        "p50_ms": statistics.median(times) * 1000 if times else 0.0,
        "p95_ms": (statistics.quantiles(times, n=20)[18] * 1000)
        if len(times) >= 20
        else max(times, default=0) * 1000,
        "frames_timed": len(times),
    }


def _hardware_label(device: str) -> str:
    """Real hardware name, queried at runtime — not a placeholder, since
    torch/platform can tell us this directly."""
    if device == "cuda" and torch.cuda.is_available():
        return f"CUDA - {torch.cuda.get_device_name(0)}"
    return f"CPU - {platform.processor() or platform.machine()}"


def write_benchmarks_md(results: list[dict], path: str, video_path: str | None) -> None:
    lines = [
        "# Benchmarks",
        "",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        f"Source: {'real video — ' + video_path if video_path else 'synthetic random-noise frames (throughput only, not representative of real detection load)'}",
        "",
        "Each row times the full per-frame pipeline — YOLO detection, depth "
        "estimation, ByteTrack update, and both panel renders — matching "
        "what `main.py` actually does per frame, not an isolated model-only "
        "number.",
        "",
        "| Device | Resolution | Avg FPS | Avg (ms) | p50 (ms) | p95 (ms) | Frames timed |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {_hardware_label(r['device'])} | {r['width']}x{r['height']} | "
            f"{r['avg_fps']:.2f} | {r['avg_ms']:.1f} | {r['p50_ms']:.1f} | "
            f"{r['p95_ms']:.1f} | {r['frames_timed']} |"
        )
    lines.append("")
    lines.append(
        "_p95 with fewer than 20 timed frames falls back to the max observed "
        "time — increase `--frames` for a statistically meaningful p95._"
    )

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    logger.success(f"Wrote {path}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Benchmark the perception pipeline across resolutions/devices"
    )
    p.add_argument(
        "--model", type=str, required=True, help="Path to YOLOv8 .pt checkpoint"
    )
    p.add_argument(
        "--video",
        type=str,
        default=None,
        help="Real video to benchmark against; omit to use synthetic frames",
    )
    p.add_argument(
        "--resolutions",
        type=str,
        default="320,640,960",
        help="Comma-separated working frame widths to test",
    )
    p.add_argument(
        "--devices",
        type=str,
        default="cuda,cpu",
        help="Comma-separated devices to test (cuda entries are skipped if unavailable)",
    )
    p.add_argument(
        "--frames", type=int, default=60, help="Frames to time per configuration"
    )
    p.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Frames to run and discard before timing starts",
    )
    p.add_argument(
        "--output", type=str, default="BENCHMARKS.md", help="Output markdown file path"
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging (per-frame timings)",
    )
    args = p.parse_args()

    configure_logging("DEBUG" if args.verbose else None)

    resolutions = [int(w) for w in args.resolutions.split(",")]
    requested_devices = [d.strip() for d in args.devices.split(",")]

    results = []
    for device_req in requested_devices:
        device, dev_label = select_device(device_req)
        if device_req == "cuda" and device != "cuda":
            logger.warning(
                f"CUDA requested but unavailable — skipping device={device_req!r}"
            )
            continue
        logger.info(f"=== Device: {dev_label} ===")
        for width in resolutions:
            logger.info(
                f"  Resolution: {width}px wide, {args.frames} frames "
                f"({args.warmup} warmup, discarded)"
            )
            result = _run_one_config(
                args.model, device, width, args.video, args.frames, args.warmup
            )
            logger.success(
                f"  -> {result['avg_fps']:.2f} avg FPS ({result['avg_ms']:.1f}ms/frame)"
            )
            results.append(result)

    if not results:
        logger.error("No configurations ran — check --devices/--resolutions.")
        return

    write_benchmarks_md(results, args.output, args.video)


if __name__ == "__main__":
    main()
