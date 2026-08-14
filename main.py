#!/usr/bin/env python3
"""
main.py — CLI entry point for the Real-Time 3D Dashcam Perception System.

Usage:
    python main.py path/to/video.mp4 --model [PATH_TO_MODEL]/yolov8n.pt
"""

import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore")

from config import DEFAULT_CONF_THRESHOLD, YOLO_MODEL_PATH
from inference import process_video


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Real-Time 3D Dashcam Perception System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("input", type=str, help="Path to input dashcam MP4")
    p.add_argument(
        "--model",
        type=str,
        default=YOLO_MODEL_PATH,
        help="Path to YOLOv8 .pt checkpoint",
    )
    p.add_argument(
        "--conf",
        type=float,
        default=DEFAULT_CONF_THRESHOLD,
        help="YOLO confidence threshold",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Falls back to CPU automatically if CUDA isn't available",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for output video (defaults to config.OUTPUT_DIR)",
    )
    p.add_argument("--show", action="store_true", help="Live preview window")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not os.path.isfile(args.input):
        print(f"[ERROR] Input file not found: {args.input}")
        sys.exit(1)
    if not os.path.isfile(args.model):
        print(f"[ERROR] YOLO model checkpoint not found: {args.model}")
        print(
            "        Set --model or the YOLO_MODEL_PATH env var to a real yolov8n.pt path."
        )
        sys.exit(1)
    process_video(args.input, args, model_path=args.model)
