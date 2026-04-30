"""
graph_nodes/head_pose_node.py
=============================
LangGraph node for parallelized 6DRepNet head pose extraction.
Consumes face crops and produces Head Pose orientation labels.

Changes:
    - Made heavy imports lazy to avoid import errors in skip-extraction mode.
"""

import os
import sys
from pathlib import Path

def run_headpose_extraction_parallel(state: dict) -> dict:
    work_dir = state.get("work_dir", ".")
    crops_dir = state.get("face_crops_dir", os.path.join(work_dir, "face_crops"))

    print(f"\n{'='*60}")
    print(f"[Node: HeadPose Parallel] Input: {crops_dir}")
    print(f"{'='*60}\n")

    if state.get("skip_extraction"):
        raw_head_pose_csv = os.path.join(work_dir, "raw_head_pose_multi.csv")
        if os.path.isfile(raw_head_pose_csv):
            print("[HeadPose] ✓ Skipping Parallel Extraction — using existing raw CSV.")
            return {"raw_head_pose_csv": raw_head_pose_csv, "error": None}

    if not os.path.isdir(crops_dir):
        msg = f"[HeadPose] Skipping: Crops directory not found."
        print(msg)
        return {"raw_head_pose_csv": os.path.join(work_dir, "raw_head_pose_multi.csv")}

    # Lazy import — only load when actually extracting
    try:
        from extract_headpose import run_headpose_parallel as run_hp_parallel
    except ImportError:
        sys.path.append(str(Path(__file__).parent.parent))
        from extract_headpose import run_headpose_parallel as run_hp_parallel

    # Execute using imported function
    try:
        run_hp_parallel(work_dir)
    except Exception as exc:
        msg = f"[HeadPose] Parallel Error: {exc}"
        print(msg)
        return {"error": msg}

    raw_head_pose_csv = os.path.join(work_dir, "raw_head_pose_multi.csv")

    return {
        "raw_head_pose_csv": raw_head_pose_csv,
        "error": None
    }
