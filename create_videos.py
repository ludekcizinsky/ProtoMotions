#!/usr/bin/env python3
from pathlib import Path
import subprocess
import sys

# Allow an optional root argument; default to ./output/renderings
root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("output/renderings")

if not root.exists():
    sys.exit(f"{root} not found")

for frames_dir in sorted(root.iterdir()):
    if not frames_dir.is_dir():
        continue
    if not any(frames_dir.glob("*.png")):
        continue
    video_path = frames_dir.with_suffix(".mp4")
    print(f"Encoding {frames_dir} -> {video_path}")
    subprocess.run(
        [
            "ffmpeg", "-y", "-framerate", "30",
            "-i", f"{frames_dir}/%04d.png",
            "-pix_fmt", "yuv420p", str(video_path),
        ],
        check=True,
    )