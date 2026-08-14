#!/usr/bin/env python3
"""
smoke_test.py — Verify the environment is correctly configured before
running a full inference pass.

Checks, in order:
  1. Required libraries import cleanly.
  2. Torch/device availability.
  3. The YOLOv8 checkpoint loads and runs a single forward pass on a
     synthetic image (no real video or GPU required).

Usage:
    python smoke_test.py --model [PATH_TO_MODEL]/yolov8n.pt
"""
import argparse
import os
import sys


def check_imports() -> bool:
    print("[1/3] Checking imports ...")
    required = ["cv2", "numpy", "torch", "torchvision", "ultralytics", "supervision", "transformers"]
    missing = []
    for mod in required:
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        print(f"  [FAIL] Missing packages: {', '.join(missing)}")
        print("         Run: pip install -r requirements.txt")
        return False
    print("  [OK] All required packages import cleanly.")
    return True


def check_device() -> str:
    print("[2/3] Checking device ...")
    import torch

    if torch.cuda.is_available():
        print(f"  [OK] CUDA available: {torch.cuda.get_device_name(0)}")
        return "cuda"
    print("  [OK] CUDA not available — falling back to CPU.")
    return "cpu"


def check_model_load(model_path: str, device: str) -> bool:
    print(f"[3/3] Loading YOLO checkpoint from {model_path} ...")
    if not os.path.isfile(model_path):
        print(f"  [FAIL] Checkpoint not found at: {model_path}")
        print("         Pass --model with a real path to your yolov8n.pt file.")
        return False

    import numpy as np
    from ultralytics import YOLO

    try:
        model = YOLO(model_path)
        model.to(device)
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        results = model(dummy, verbose=False, device=device)
        print(f"  [OK] Model loaded and ran a forward pass "
              f"({len(results[0].boxes)} detections on blank frame, as expected).")
        return True
    except Exception as e:
        print(f"  [FAIL] Model load/inference failed: {e}")
        return False


def main() -> None:
    p = argparse.ArgumentParser(description="Environment + model smoke test")
    p.add_argument("--model", type=str, default="[PATH_TO_MODEL]/yolov8n.pt",
                   help="Path to YOLOv8 .pt checkpoint")
    args = p.parse_args()

    ok = check_imports()
    if not ok:
        sys.exit(1)

    device = check_device()

    ok = check_model_load(args.model, device)
    if not ok:
        sys.exit(1)

    print("\n[OK] Smoke test passed — environment is ready for full inference.")


if __name__ == "__main__":
    main()
