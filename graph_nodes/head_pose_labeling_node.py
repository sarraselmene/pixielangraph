"""
graph_nodes/head_pose_labeling_node.py
======================================
LangGraph node wrapper for label_head_pose.py.
Reads raw_head_pose_multi.csv and produces labeled_head_pose_multi.csv.

Changes:
    - Made heavy imports lazy to avoid import errors in skip-extraction mode.
"""

import os
import sys
from pathlib import Path

def run_head_pose_labeling(state: dict) -> dict:
    raw_head_pose_csv = state.get("raw_head_pose_csv", "")
    work_dir          = state.get("work_dir", ".")

    print(f"\n{'='*60}")
    print(f"[Node: Head Pose Labeling] Input : {raw_head_pose_csv}")
    print(f"{'='*60}\n")

    if not os.path.isfile(raw_head_pose_csv):
        msg = f"[HeadPose] ERROR: raw head-pose CSV not found → {raw_head_pose_csv}"
        print(msg)
        return {"head_labeling_done": False, "error": msg}

    # Lazy import
    try:
        from label_head_pose import main as run_hp_labeling
    except ImportError:
        sys.path.append(str(Path(__file__).parent.parent))
        from label_head_pose import main as run_hp_labeling

    # Execute labeling
    try:
        run_hp_labeling()
    except Exception as exc:
        msg = f"[HeadPose] Labeling Runtime error: {exc}"
        print(msg)
        return {"head_labeling_done": False, "error": msg}

    output_csv = os.path.join(work_dir, "labeled_head_pose_multi.csv")
    print(f"[Node: Head Pose Labeling] ✓ Done → {output_csv}")
    return {
        "head_label_csv":    output_csv,
        "head_labeling_done": True,
        "error": None,
    }
