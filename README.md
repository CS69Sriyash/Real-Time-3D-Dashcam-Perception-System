## Dev/Creator: tubakhxn

# Real-Time 3D Dashcam Perception System

This project transforms a regular dashcam video into a real-time 3D map with collision risk analysis. It leverages YOLO for vehicle detection, Depth Anything V2 for depth estimation, and visualizes the results in a Bird’s Eye View (BEV) with risk analysis.

## What is this project about?
- Detects vehicles (cars, motorcycles, trucks) in dashcam videos using YOLO.
- Estimates accurate metric depth from a single camera using Depth Anything V2.
- Projects pixels into 3D XYZ coordinates and renders them in BEV.
- Tracks trajectories and monitors lateral intrusion and frontal distance.
- Provides instant risk level updates with color-coded detections.

## How to fork and run
1. Clone or fork this repository to your local machine.
2. Ensure you have Python 3.12 installed.
3. Create and activate a virtual environment:
   ```
   uv venv --python 3.12
   source .venv/bin/activate
   ```
4. Install dependencies:
   ```
   uv sync
   ```
5. Place your input video (e.g., `video.mp4`) in the project directory.
6. Run the main script:
   ```
   uv run --python 3.12 main.py video.mp4 --model yolov8n.pt
   ```
7. The output video will be saved in the `output/` directory.

## Relevant Wikipedia links
- [YOLO (object detection)](https://en.wikipedia.org/wiki/You_Only_Look_Once)
- [Depth estimation](https://en.wikipedia.org/wiki/Depth_map)
- [Bird's-eye view](https://en.wikipedia.org/wiki/Bird%27s-eye_view)
- [Advanced driver-assistance systems](https://en.wikipedia.org/wiki/Advanced_driver-assistance_systems)
- [Dashcam](https://en.wikipedia.org/wiki/Dashcam)
