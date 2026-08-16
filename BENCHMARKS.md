# Benchmarks

Generated: 2026-08-16 04:27 UTC

Source: real video — video.mp4

Each row times the full per-frame pipeline — YOLO detection, depth estimation, ByteTrack update, and both panel renders — matching what `main.py` actually does per frame, not an isolated model-only number.

| Device | Resolution | Avg FPS | Avg (ms) | p50 (ms) | p95 (ms) | Frames timed |
|---|---|---|---|---|---|---|
| CUDA - NVIDIA GeForce RTX 3050 Laptop GPU | 320x180 | 19.29 | 51.8 | 51.3 | 55.6 | 60 |
| CUDA - NVIDIA GeForce RTX 3050 Laptop GPU | 640x360 | 18.89 | 52.9 | 53.0 | 55.9 | 60 |
| CUDA - NVIDIA GeForce RTX 3050 Laptop GPU | 960x540 | 18.40 | 54.3 | 54.6 | 57.1 | 60 |
| CPU - x86_64 | 320x180 | 3.64 | 274.4 | 274.2 | 291.7 | 60 |
| CPU - x86_64 | 640x360 | 3.29 | 303.8 | 300.2 | 323.9 | 60 |
| CPU - x86_64 | 960x540 | 2.83 | 352.8 | 350.9 | 363.2 | 60 |

_p95 with fewer than 20 timed frames falls back to the max observed time — increase `--frames` for a statistically meaningful p95._
