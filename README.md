# Real-Time 3D Dashcam Perception System

Turns a regular dashcam video into a real-time 3D perception feed: YOLOv8 detects
vehicles, Depth Anything V2 estimates monocular depth, and the two are fused into
a collision-risk overlay plus a bird's-eye-view (BEV) panel.

> **Fork notice:** this project began as a fork of
> [tubakhxn/Real-Time-3D-Dashcam-Perception-System](https://github.com/tubakhxn/Real-Time-3D-Dashcam-Perception-System)
> (MIT License, © 2026 Tuba Khan). The original single-file prototype and core
> idea — YOLO + Depth Anything V2 + BEV collision risk — are Tuba Khan's. Since
> forking, the codebase has been substantially reworked: see
> [What changed from the original](#what-changed-from-the-original) below for
> specifics, including a correctness bug in the depth math that was found and
> fixed during that rework. The original `LICENSE` is preserved unchanged.

---

## What it does

- Detects vehicles (cars, motorcycles, buses, trucks, bicycles) with YOLOv8.
- Estimates per-pixel relative depth with Depth Anything V2 (small variant).
- Fuses depth + projective geometry (bounding-box width vs. focal length) into a
  per-vehicle distance estimate.
- Tracks vehicles across frames (ByteTrack) and draws trajectory + velocity.
- Renders a real-time BEV panel: a depth-colored point cloud plus per-vehicle
  markers, positioned by estimated distance.
- Flags each tracked vehicle's collision risk (`SAFE` / `CAUTION` / `WARNING` /
  `DANGER`) from distance and lateral offset.

## Project layout

```
Real-Time-3D-Dashcam-Perception-System/
├── pyproject.toml     # uv project + dependency config (mirrors requirements.txt)
├── requirements.txt   # pinned deps — see "Dependency notes" below
├── config.py          # all tunables: paths, thresholds, colors, geometry constants
├── utils.py           # pure functions: distance/risk math, pinhole unprojection, drawing
├── model_loader.py    # device selection, YOLO loader, DepthEstimator, ObjectTracker
├── inference.py        # panel rendering + the main process_video() pipeline
├── main.py            # CLI entry point
├── smoke_test.py       # environment + model-load check, no video required
├── tests/              # pytest unit tests for utils.py
└── yolov8n.pt          # YOLOv8 nano checkpoint
```

## Setup

Requires Python 3.12 (see `requirements.txt`/`pyproject.toml` — several
dependency versions are tied to this).

```bash
uv sync
```

or, without `uv`:

```bash
pip install -r requirements.txt
```

The default pulls CUDA 12.1 PyTorch wheels. On a CPU-only machine, install
PyTorch separately first:

```bash
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Verify the environment before running on real footage:

```bash
python smoke_test.py --model yolov8n.pt
```

This checks imports, GPU/CPU availability, and runs one forward pass of YOLO on
a synthetic frame — no video file needed.

## Running the tests

```bash
uv run pytest tests/ -v
```

Covers the distance/risk math and the pinhole-unprojection geometry in
`utils.py` — no model loading or GPU required, runs in well under a second.
Includes a regression test for the near/far depth-inversion bug described
below: it fails immediately if that formula is ever reintroduced.

## Usage

```bash
python main.py path/to/video.mp4 --model yolov8n.pt
```

Output is written to `./output/final_output.mp4` by default.

## Docker

Build the image:

```bash
docker build -t dashcam-perception .
```

Run with the current project directory mounted at `/data`:

```bash
docker run --rm --gpus all \
  -v "$(pwd):/data" \
  dashcam-perception \
  /data/video.mp4 --model /data/yolov8n.pt --output-dir /data/output
```

This assumes `video.mp4` and `yolov8n.pt` are in the project root.

| Flag | Default | Purpose |
|---|---|---|
| `--model` | `$YOLO_MODEL_PATH` or none | Path to a YOLOv8 `.pt` checkpoint |
| `--conf` | `0.35` | YOLO detection confidence threshold |
| `--device` | `cuda` | Falls back to CPU automatically if CUDA isn't available |
| `--output-dir` | `./output` | Where the output video and temp file are written |
| `--show` | off | Live preview window (press `q` to stop early) |

## What changed from the original

The original was a single ~500-line `app.py` that also auto-installed missing
packages at runtime via `pip`/`apt`/`brew` calls. Since forking:

- **Split into modules by responsibility** — `config.py` / `utils.py` /
  `model_loader.py` / `inference.py` / `main.py` — instead of one file mixing
  config, math, model loading, and the render loop together.
- **Removed the runtime auto-installer.** Shelling out to package managers on
  every launch is fragile and non-reproducible; `requirements.txt` now owns
  dependency resolution, installed once.
- **Pinned dependencies deliberately, with upper bounds where they matter.**
  `numpy<1.28` paired with `opencv-python<4.12.0` avoids a real conflict —
  `opencv-python>=4.12` requires NumPy 2, which would silently break the pin
  above it if left unbounded.
- **Fixed a near/far inversion bug in the depth math.** `estimate_distance()`
  and the BEV point-cloud placement (`project_pointcloud()`) both treated
  *higher* depth values as *farther* — backwards for Depth Anything V2's
  relative-depth output, where higher values mean *closer*. In practice this
  meant a close, genuinely dangerous vehicle could be computed as ~75m away
  and flagged `SAFE`. Confirmed by comparing BEV output before and after:

  Before (left): the point cloud is one undifferentiated fan with no depth
  contrast. After (right): distant terrain clusters near the vanishing point
  (top), the road surface correctly spreads toward the bottom (near the
  vehicle) — matching the actual scene.
- **FFmpeg encoder auto-detection.** The compression step used to hardcode
  `libx264`, which fails outright on ffmpeg builds compiled without it (common
  on some distro packages for licensing reasons). It now probes
  `ffmpeg -encoders` and picks whichever of `libx264` / `libopenh264` /
  `h264_nvenc` / `mpeg4` is actually available, falling back to an uncompressed
  output rather than crashing if none are.
- **No hardcoded paths.** Model and output paths are placeholders resolved via
  env vars or CLI flags (`--model`, `--output-dir`), with a real relative
  default (`./output`) for the output directory so a forgotten override can't
  silently create a directory literally named after the placeholder string.
- **CUDA device auto-detection with CPU fallback**, reported in the on-screen
  HUD so it's visible which device actually ran a given output.
- **Replaced the BEV panel's heuristic point placement with real pinhole-camera
  unprojection.** The original fan-shaped BEV placed points using tuned
  normalized-position formulas with no connection to actual camera geometry.
  It now back-projects each depth pixel through `X = (px - cx) × depth / f`
  into real-world meters, and renders them with a true orthographic (constant-
  scale, no vanishing point) top-down mapping — labeled range rings, lane-width
  reference lines, and vehicle markers sized to their real-world footprint
  regardless of distance:

  The wedge shape is now expected, not a bug: a single forward-facing camera's
  field of view genuinely covers more real-world width at greater distance, so
  a correct point cloud from monocular depth *should* fan out like this —
  the difference is that this fan is now a direct consequence of the pinhole
  geometry rather than an arbitrary tuned curve.
- **Added a pytest suite** (`tests/`) covering the distance/risk math and the
  unprojection geometry, including a regression test that fails immediately if
  the near/far depth-inversion bug above is ever reintroduced. Verified by
  temporarily reintroducing that exact bug and confirming the test suite
  catches it before restoring the fix.
- **Added `pyproject.toml`** for `uv`, mirroring the `requirements.txt` pins
  with `pytest` as a dev dependency.

## Known limitations

- **The point cloud is noisy, not the geometry.** The BEV panel now uses real
  pinhole unprojection (see above), but Depth Anything V2's per-pixel depth
  estimate isn't perfectly smooth — a flat road or building facade should
  unproject to a clean flat band, but you'll see visible rippling from the
  model's estimation noise instead. Read the near/far ordering and rough
  free-space shape as reliable; don't read individual ripples as real bumps
  or texture in the scene.
- **`FOCAL_LENGTH_PX` is an assumed constant, not a calibrated camera
  intrinsic.** The pinhole math is only as accurate as this value; a properly
  calibrated focal length (e.g. via OpenCV's checkerboard calibration) for
  your specific camera would tighten up both distance and lateral-position
  accuracy.
- **Single monocular camera, not stereo or LiDAR-fused** — there's no true
  ranging sensor here, so absolute distance accuracy is inherently limited
  compared to a system with real depth sensing.
- **YOLOv8n (nano) can miss small, distant vehicles**, especially after the
  pipeline's internal downscale to `MAX_WORK_WIDTH=640`. Lower `--conf` to
  surface borderline detections if this matters for your footage; a larger
  YOLOv8 variant (`s`/`m`) will also help at the cost of speed.
- **Distance estimates are relative, not metric-calibrated** — Depth Anything
  V2's relative-depth model output isn't in real-world units, so
  `estimate_distance()`'s depth term is a heuristic mapping, cross-checked
  against (but not replaced by) the projective-geometry term. Treat on-screen
  distances as approximate.

## Relevant links

- [YOLO (object detection)](https://en.wikipedia.org/wiki/You_Only_Look_Once)
- [Depth estimation](https://en.wikipedia.org/wiki/Depth_map)
- [Bird's-eye view](https://en.wikipedia.org/wiki/Bird%27s-eye_view)
- [Advanced driver-assistance systems](https://en.wikipedia.org/wiki/Advanced_driver-assistance_systems)
- [Depth Anything V2](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf)

## License

MIT — see [`LICENSE`](LICENSE). Original copyright © 2026 Tuba Khan, preserved
as required by the MIT license terms.
