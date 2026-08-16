# ============================================================
# Real-Time 3D Dashcam Perception System — Dockerfile
#
# Multi-stage build: dependency resolution happens in a throwaway
# "builder" stage; the final "runtime" image only gets the resulting
# virtual environment + app code, not the compilers/dev headers used to
# build it. This is the main size lever here — skipping it would ship
# every build-time apt package (gcc, headers, uv's own cache) in the
# final image for no runtime benefit.
#
# Prerequisite: run `uv lock` locally first so uv.lock exists — this
# Dockerfile intentionally does NOT fall back to resolving without a
# lockfile, so the image you build matches what you tested locally.
#
# Build:
#   uv lock                      # once, or whenever pyproject.toml changes
#   docker build -t dashcam-perception .
#
# Run (requires the NVIDIA Container Toolkit on the host for --gpus):
#   docker run --rm --gpus all \
#     -v "$(pwd):/data" \
#     dashcam-perception /data/video.mp4 --model /data/yolov8n.pt --output-dir /data/output
# ============================================================

# ---------- Builder stage: resolve + install dependencies with uv ----------
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    UV_LINK_MODE=copy \
    UV_PYTHON_PREFERENCE=only-system

# Python 3.12 isn't in Ubuntu 22.04's default repos (only 3.10/3.11) —
# deadsnakes provides it. --no-install-recommends keeps this stage from
# pulling in doc/suggests bloat that adds build time for no benefit
# (irrelevant to final size anyway, since this whole stage is discarded).
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common curl ca-certificates \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3.12-dev \
        libgl1 libglib2.0-0 libsm6 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# uv ships as a single static binary — copying it from its own official
# image is faster and more reproducible than `curl | sh` here.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy only the dependency manifests first so this layer (the slow part —
# resolving ~80+ packages plus CUDA torch wheels) is cached and skipped
# on rebuilds that only change application code, not dependencies.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Now copy the actual application code and finish the sync (installs the
# project itself into the venv; near-instant since deps are already
# resolved above).
COPY . .
RUN uv sync --frozen --no-dev

# Print what the venv's python symlink actually resolves to, and verify
# every heavy dependency imports — both catch a broken/dangling
# interpreter (e.g. from uv resolving to a self-managed Python build that
# won't survive the copy into the runtime stage below) at build time,
# with a clear error, instead of a bare ModuleNotFoundError later.
RUN readlink -f /app/.venv/bin/python
RUN /app/.venv/bin/python -c "import loguru, cv2, torch, ultralytics, supervision, transformers; print('dependency sanity check OK')"

# ---------- Runtime stage: copy only the venv + app, nothing else ----------
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    PYTHONUNBUFFERED=1

# Runtime-only system deps: python3.12 to run the venv's interpreter,
# ffmpeg for output encoding (inference.py probes for libx264 at
# runtime — the standard Ubuntu ffmpeg package includes it), and the
# handful of shared libraries opencv-python needs at import time even
# though it's not the -headless variant (libGL, libglib, libSM, libXext).
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common curl \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.12 ffmpeg \
        libgl1 libglib2.0-0 libsm6 libxext6 \
    && apt-get purge -y --auto-remove software-properties-common curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /app /app

ENV PATH="/app/.venv/bin:$PATH"

# No ENTRYPOINT argument here — main.py requires a positional video path,
# so it's supplied at `docker run` time, not baked into the image.
# Points directly at the venv's own interpreter by absolute path rather
# than relying on `python3.12` resolving correctly via PATH order — the
# runtime stage installs its own system python3.12 for other reasons, and
# a bare name here risked matching that one instead of the venv's.
ENTRYPOINT ["/app/.venv/bin/python", "main.py"]
CMD ["--help"]
