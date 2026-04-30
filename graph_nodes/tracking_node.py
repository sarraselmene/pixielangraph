"""
graph_nodes/tracking_node.py
===========================
Description:
    LangGraph node that orchestrates GPU-accelerated student tracking using 
    YOLOv8/v11. It is optimized for macOS (MPS/Metal) and produces both 
    skeletal keypoint CSVs and facial crops for downstream parallel sensors.

Changes Effectuated:
    - Added comprehensive documentation header and change log.
    - Verified MPS (Metal Performance Shaders) optimization for Apple Silicon.
    - Standardized face crop directory management and error handling.
    - Made heavy imports lazy to avoid import errors in skip-extraction mode.
"""

import os
import sys
from pathlib import Path

def run_yolo_tracking(state: dict) -> dict:
    video_path = state.get("video_path", "")
    work_dir   = state.get("work_dir", str(Path(video_path).parent))

    print(f"\n{'='*60}")
    print(f"[Node: GPU Tracking] Target: {video_path}")
    print(f"[Node: GPU Tracking] Mode  : MPS (Metal Performance Shaders)")
    print(f"{'='*60}\n")

    if not os.path.isfile(video_path):
        msg = f"[Tracking] ERROR: video not found → {video_path}"
        return {"tracking_done": False, "error": msg}

    if state.get("skip_extraction"):
        body_csv = os.path.join(work_dir, "raw_body_multi.csv")
        crops_dir = os.path.join(work_dir, "face_crops")
        if os.path.isfile(body_csv) and os.path.isdir(crops_dir):
            print("[Tracking] ✓ Skipping GPU Tracking — using existing crops/CSV.")
            return {"raw_body_csv": body_csv, "face_crops_dir": crops_dir, "tracking_done": True, "error": None}
        else:
            print("[Tracking] --skip-extraction set but files missing. Running tracking anyway.")

    # Lazy import — only load heavy ML modules when actually running extraction
    try:
        from extract_yolo_tracking import run_tracking
    except ImportError:
        sys.path.append(str(Path(__file__).parent.parent))
        from extract_yolo_tracking import run_tracking

    # Execute tracking using imported function
    try:
        run_tracking(video_path, work_dir)
    except Exception as exc:
        msg = f"[Tracking] Runtime error: {exc}"
        print(msg)
        return {"tracking_done": False, "error": msg}

    body_csv = os.path.join(work_dir, "raw_body_multi.csv")
    crops_dir = os.path.join(work_dir, "face_crops")

    if not os.path.isfile(body_csv):
        msg = "[Tracking] Output CSV missing"
        return {"tracking_done": False, "error": msg}

    print(f"[Node: GPU Tracking] ✓ Done. Crops generated in {crops_dir}")
    return {
        "raw_body_csv": body_csv,
        "face_crops_dir": crops_dir,
        "tracking_done": True,
        "error": None
    }
