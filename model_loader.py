"""
model_loader.py — Model initialization: device selection, YOLOv8 detector,
Depth Anything V2 depth estimator, and the ByteTrack-based object tracker.
"""

import collections

import cv2
import numpy as np
import supervision as sv
import torch
from ultralytics import YOLO

from config import DEPTH_MODEL_NAME, DEPTH_SMOOTHING_ALPHA, YOLO_MODEL_PATH


def select_device(requested: str = "cpu") -> tuple[str, str]:
    """Resolve the requested device against actual hardware availability."""
    if requested == "cuda" and torch.cuda.is_available():
        return "cuda", f"CUDA - {torch.cuda.get_device_name(0)}"
    return "cpu", "CPU"


def load_yolo(model_path: str = YOLO_MODEL_PATH, device: str = "cpu") -> YOLO:
    """Load a YOLOv8 model (Ultralytics) and move it to the target device.

    `model_path` should point at a local .pt checkpoint (e.g. yolov8n.pt).
    Replace the config placeholder with a real path before running.
    """
    print(f"[YOLO] Loading model from {model_path} ...")
    model = YOLO(model_path)
    model.to(device)
    print("[YOLO] Model ready ✓")
    return model


class DepthEstimator:
    """Wraps the Depth Anything V2 HF pipeline with temporal smoothing."""

    def __init__(
        self,
        device: str,
        model_name: str = DEPTH_MODEL_NAME,
        smoothing_alpha: float = DEPTH_SMOOTHING_ALPHA,
    ):
        self.device = device
        self.model_name = model_name
        self._alpha = smoothing_alpha
        self._prev = None
        self._pipe = None

    def load(self) -> None:
        print("[DEPTH] Loading Depth Anything V2 ...")
        from transformers import pipeline as hf_pipeline

        hf_device = 0 if (self.device == "cuda" and torch.cuda.is_available()) else -1
        self._pipe = hf_pipeline(
            task="depth-estimation", model=self.model_name, device=hf_device
        )
        print("[DEPTH] Model ready ✓")

    def estimate(self, bgr: np.ndarray) -> np.ndarray:
        if self._pipe is None:
            raise RuntimeError(
                "DepthEstimator.load() must be called before estimate()."
            )

        from PIL import Image as PILImage

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        result = self._pipe(PILImage.fromarray(rgb))
        depth = np.array(result["depth"], dtype=np.float32)
        depth = cv2.resize(
            depth, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_LINEAR
        )

        mn, mx = depth.min(), depth.max()
        if mx - mn > 1e-5:
            depth = (depth - mn) / (mx - mn)

        if self._prev is not None and self._prev.shape == depth.shape:
            depth = self._alpha * self._prev + (1 - self._alpha) * depth
        self._prev = depth.copy()
        return depth


class ObjectTracker:
    """Thin wrapper around ByteTrack that also keeps per-ID trajectories/velocity."""

    def __init__(self, max_history: int = 30):
        self.tracker = sv.ByteTrack()
        self.trajectories = collections.defaultdict(
            lambda: collections.deque(maxlen=max_history)
        )
        self.velocities = collections.defaultdict(lambda: (0.0, 0.0))
        self._prev_centers = {}

    def update(self, detections: sv.Detections) -> sv.Detections:
        tracked = self.tracker.update_with_detections(detections)
        if tracked.tracker_id is None:
            return tracked

        for i, tid in enumerate(tracked.tracker_id):
            box = tracked.xyxy[i]
            cx = int((box[0] + box[2]) / 2)
            cy = int((box[1] + box[3]) / 2)
            self.trajectories[tid].append((cx, cy))
            if tid in self._prev_centers:
                px, py = self._prev_centers[tid]
                self.velocities[tid] = (cx - px, cy - py)
            self._prev_centers[tid] = (cx, cy)

        return tracked
